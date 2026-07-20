package storage

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"gorm.io/gorm"
)

// CreateExecutionUsage persists a batch of token/cost usage rows for an
// execution. A nil/empty slice is a no-op. On success the ID field of each
// input row is populated with the assigned primary key.
func (ls *LocalStorage) CreateExecutionUsage(ctx context.Context, rows []*types.ExecutionUsage) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create execution usage: %w", err)
	}
	if len(rows) == 0 {
		return nil
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	models := make([]*ExecutionUsageModel, 0, len(rows))
	for _, r := range rows {
		if r == nil {
			continue
		}
		models = append(models, executionUsageToModel(r))
	}
	if len(models) == 0 {
		return nil
	}

	if err := gormDB.Create(&models).Error; err != nil {
		return fmt.Errorf("failed to store execution usage: %w", err)
	}

	// Propagate assigned IDs back to the caller's rows (models preserves order).
	mi := 0
	for _, r := range rows {
		if r == nil {
			continue
		}
		if mi < len(models) {
			r.ID = models[mi].ID
			mi++
		}
	}
	return nil
}

func executionUsageToModel(r *types.ExecutionUsage) *ExecutionUsageModel {
	m := &ExecutionUsageModel{
		ExecutionID:         r.ExecutionID,
		WorkflowID:          r.WorkflowID,
		AgentNodeID:         r.AgentNodeID,
		Reasoner:            r.Reasoner,
		Source:              r.Source,
		Provider:            r.Provider,
		Model:               r.Model,
		Harness:             r.Harness,
		InputTokens:         r.InputTokens,
		OutputTokens:        r.OutputTokens,
		CacheReadTokens:     r.CacheReadTokens,
		CacheCreationTokens: r.CacheCreationTokens,
		TotalTokens:         r.TotalTokens,
		CostUSD:             r.CostUSD,
		CostSource:          r.CostSource,
	}
	if !r.CreatedAt.IsZero() {
		m.CreatedAt = r.CreatedAt
	}
	return m
}

// GetExecutionUsageTotals sums the token/cost usage rows for a single
// execution. Cost is nil when the execution has no usage rows or all rows
// report a null cost. Used to light up per-execution cost in detail views.
func (ls *LocalStorage) GetExecutionUsageTotals(ctx context.Context, executionID string) (*float64, int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, fmt.Errorf("context cancelled during get execution usage totals: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	var row struct {
		CostUSD     sql.NullFloat64
		TotalTokens sql.NullInt64
	}
	err = gormDB.Table("execution_usage").
		Select("SUM(cost_usd) AS cost_usd, COALESCE(SUM(total_tokens),0) AS total_tokens").
		Where("execution_id = ?", executionID).
		Scan(&row).Error
	if err != nil {
		return nil, 0, fmt.Errorf("failed to sum execution usage: %w", err)
	}

	var cost *float64
	if row.CostUSD.Valid {
		c := row.CostUSD.Float64
		cost = &c
	}
	return cost, row.TotalTokens.Int64, nil
}

// usageTotalsRow is the scan target for the aggregate totals query.
type usageTotalsRow struct {
	InputTokens         sql.NullInt64
	OutputTokens        sql.NullInt64
	CacheReadTokens     sql.NullInt64
	CacheCreationTokens sql.NullInt64
	TotalTokens         sql.NullInt64
	CostUSD             sql.NullFloat64
	ExecutionsWithUsage int64
}

// usageGroupRow is the scan target for grouped usage queries.
type usageGroupRow struct {
	GroupKey     string
	Provider     string
	CostUSD      sql.NullFloat64
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	Entries      int64
}

// GetUsageStats aggregates execution_usage rows over the window starting at
// since (inclusive). A nil since aggregates over all rows. Works on both the
// SQLite and PostgreSQL backends (all dialect-specific time math is done in Go;
// GORM handles placeholder binding).
func (ls *LocalStorage) GetUsageStats(ctx context.Context, since *time.Time) (*types.UsageStatsAggregation, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get usage stats: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	result := &types.UsageStatsAggregation{
		ByModel:    []types.UsageStatsGroup{},
		ByProvider: []types.UsageStatsGroup{},
		ByAgent:    []types.UsageStatsGroup{},
		ByHarness:  []types.UsageStatsGroup{},
	}

	// Totals
	totalsQuery := gormDB.Table("execution_usage").Select(
		"COALESCE(SUM(input_tokens),0) AS input_tokens, " +
			"COALESCE(SUM(output_tokens),0) AS output_tokens, " +
			"COALESCE(SUM(cache_read_tokens),0) AS cache_read_tokens, " +
			"COALESCE(SUM(cache_creation_tokens),0) AS cache_creation_tokens, " +
			"COALESCE(SUM(total_tokens),0) AS total_tokens, " +
			"SUM(cost_usd) AS cost_usd, " +
			"COUNT(DISTINCT execution_id) AS executions_with_usage")
	if since != nil {
		// Compare as unix epochs (usageEpochExpr), not as raw column values:
		// SQLite stores timestamps as text and compares them lexicographically,
		// so a row written with a non-UTC offset (e.g. "…10:03:47-04:00") would
		// never match "created_at >= <UTC time>" even when the instant is in
		// range. Epoch comparison is offset-proof and dialect-portable.
		totalsQuery = totalsQuery.Where(ls.usageEpochExpr()+" >= ?", since.Unix())
	}
	var totals usageTotalsRow
	if err := totalsQuery.Scan(&totals).Error; err != nil {
		return nil, fmt.Errorf("failed to aggregate usage totals: %w", err)
	}

	result.Totals = types.UsageStatsTotals{
		InputTokens:         totals.InputTokens.Int64,
		OutputTokens:        totals.OutputTokens.Int64,
		CacheReadTokens:     totals.CacheReadTokens.Int64,
		CacheCreationTokens: totals.CacheCreationTokens.Int64,
		TotalTokens:         totals.TotalTokens.Int64,
		ExecutionsWithUsage: totals.ExecutionsWithUsage,
	}
	if totals.CostUSD.Valid {
		cost := totals.CostUSD.Float64
		result.Totals.CostUSD = &cost
	}

	// Last updated: the most recent created_at in the window. Queried through
	// the typed model (rather than MAX()) so the driver returns a real
	// time.Time on both SQLite and PostgreSQL.
	// Order by epoch, not by the raw column: text timestamps with mixed UTC
	// offsets sort lexicographically, which can pick the wrong "latest" row.
	latestQuery := gormDB.Model(&ExecutionUsageModel{}).Select("created_at").Order(ls.usageEpochExpr() + " DESC").Limit(1)
	if since != nil {
		latestQuery = latestQuery.Where(ls.usageEpochExpr()+" >= ?", since.Unix())
	}
	var latestRows []ExecutionUsageModel
	if err := latestQuery.Find(&latestRows).Error; err != nil {
		return nil, fmt.Errorf("failed to query usage last_updated: %w", err)
	}
	if len(latestRows) == 1 && !latestRows[0].CreatedAt.IsZero() {
		lu := latestRows[0].CreatedAt.UTC()
		result.LastUpdated = &lu
	}

	// Groups
	if result.ByModel, err = ls.queryUsageGroups(ctx, "model", true, since); err != nil {
		return nil, err
	}
	if result.ByProvider, err = ls.queryUsageGroups(ctx, "provider", false, since); err != nil {
		return nil, err
	}
	if result.ByAgent, err = ls.queryUsageGroups(ctx, "agent_node_id", false, since); err != nil {
		return nil, err
	}
	if result.ByHarness, err = ls.queryUsageGroups(ctx, "harness", false, since); err != nil {
		return nil, err
	}

	return result, nil
}

// usageBucketRow is the scan target for the bucketed timeseries query.
type usageBucketRow struct {
	BucketIdx   int64
	TotalTokens int64
	CostUSD     sql.NullFloat64
}

// usageEpochExpr returns a dialect-specific SQL expression that yields the
// created_at column as integer unix seconds. SQLite uses strftime; PostgreSQL
// uses EXTRACT(EPOCH ...). Mirrors the ls.mode dialect switch used elsewhere in
// this package (see ensureVectorSchema / initializeVectorStore).
func (ls *LocalStorage) usageEpochExpr() string {
	switch ls.mode {
	case "postgres":
		return "CAST(FLOOR(EXTRACT(EPOCH FROM created_at)) AS BIGINT)"
	default:
		return "CAST(strftime('%s', created_at) AS INTEGER)"
	}
}

// GetUsageTimeseries returns a bucketed token/cost series over a window ending
// at now. The window is divided into exactly `buckets` equal buckets:
// bucket_seconds = windowSeconds/buckets (integer). When since is non-nil the
// window is [since, now]; when since is nil ("all" window) the window spans from
// the oldest row's created_at to now, falling back to 24h when the table is
// empty. The returned Points slice always has exactly `buckets` entries,
// ascending by start time, zero-filled: buckets with no rows get 0 tokens and a
// nil cost. Cost is also nil for a bucket whose rows all report a null cost.
// SQL grouping is dialect-portable (SQLite strftime vs PostgreSQL EXTRACT via
// usageEpochExpr); zero-filling is done here in Go, not in SQL.
func (ls *LocalStorage) GetUsageTimeseries(ctx context.Context, since *time.Time, now time.Time, buckets int) (*types.UsageTimeseries, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get usage timeseries: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	grid, err := ls.usageBucketGrid(gormDB, since, now, buckets)
	if err != nil {
		return nil, err
	}

	// Bucket index = floor((epoch(created_at) - seriesStartEpoch) / bucketSeconds).
	// The integer literals are computed here in Go (never user input), so they are
	// inlined into the grouped expression (GORM's Group takes no bind args).
	bucketExpr := grid.bucketExpr(ls)

	var rows []usageBucketRow
	if err := gormDB.Table("execution_usage").
		Select(bucketExpr+" AS bucket_idx, "+
			"COALESCE(SUM(total_tokens),0) AS total_tokens, "+
			"SUM(cost_usd) AS cost_usd").
		Where(ls.usageEpochExpr()+" >= ?", grid.seriesStart.Unix()).
		Group(bucketExpr).
		Scan(&rows).Error; err != nil {
		return nil, fmt.Errorf("failed to aggregate usage timeseries: %w", err)
	}

	// Zero-fill exactly `buckets` points in Go, ascending by start time.
	points := make([]types.UsageTimeseriesPoint, grid.buckets)
	for i := 0; i < grid.buckets; i++ {
		points[i] = types.UsageTimeseriesPoint{
			Start:       grid.bucketStart(i),
			TotalTokens: 0,
			CostUSD:     nil,
		}
	}
	for _, r := range rows {
		if r.BucketIdx < 0 || r.BucketIdx >= int64(grid.buckets) {
			continue // rows outside the aligned series window (e.g. exactly at now)
		}
		p := &points[r.BucketIdx]
		p.TotalTokens = r.TotalTokens
		if r.CostUSD.Valid {
			c := r.CostUSD.Float64
			p.CostUSD = &c
		}
	}

	return &types.UsageTimeseries{
		BucketSeconds: grid.bucketSeconds,
		Points:        points,
	}, nil
}

// usageBucketGridSpec is the aligned bucket grid shared by the plain and
// per-model timeseries queries: the same start, bucket_seconds, and count are
// derived once so both series land on an identical time axis.
type usageBucketGridSpec struct {
	seriesStart      time.Time
	seriesStartEpoch int64
	bucketSeconds    int64
	buckets          int
}

// usageBucketGrid computes the aligned bucket grid for the window ending at now.
// A nil since spans from the oldest row to now ("all" window), falling back to
// 24h when the table is empty. gormDB is reused for the oldest-row probe.
func (ls *LocalStorage) usageBucketGrid(gormDB *gorm.DB, since *time.Time, now time.Time, buckets int) (usageBucketGridSpec, error) {
	if buckets < 1 {
		buckets = 1
	}
	now = now.UTC()

	// Determine the window start.
	var windowStart time.Time
	if since != nil {
		windowStart = since.UTC()
	} else {
		// "all" window: span from the oldest row to now. Queried through the typed
		// model (rather than MIN()) so the driver returns a real time.Time on both
		// SQLite and PostgreSQL, matching GetUsageStats' last_updated pattern.
		var oldestRows []ExecutionUsageModel
		if err := gormDB.Model(&ExecutionUsageModel{}).
			Select("created_at").Order(ls.usageEpochExpr() + " ASC").Limit(1).
			Find(&oldestRows).Error; err != nil {
			return usageBucketGridSpec{}, fmt.Errorf("failed to query oldest usage row: %w", err)
		}
		if len(oldestRows) == 1 && !oldestRows[0].CreatedAt.IsZero() {
			windowStart = oldestRows[0].CreatedAt.UTC()
		} else {
			windowStart = now.Add(-24 * time.Hour) // empty-table fallback
		}
	}

	windowSeconds := int64(now.Sub(windowStart).Seconds())
	if windowSeconds < 0 {
		windowSeconds = 0
	}
	bucketSeconds := windowSeconds / int64(buckets)
	if bucketSeconds < 1 {
		bucketSeconds = 1
	}

	// The series spans buckets*bucketSeconds seconds ending at now. seriesStart is
	// the start of the first (oldest) bucket.
	seriesStart := now.Add(-time.Duration(int64(buckets)*bucketSeconds) * time.Second)
	return usageBucketGridSpec{
		seriesStart:      seriesStart,
		seriesStartEpoch: seriesStart.Unix(),
		bucketSeconds:    bucketSeconds,
		buckets:          buckets,
	}, nil
}

// bucketExpr returns the dialect-portable SQL expression yielding a row's
// zero-based bucket index within the grid.
func (g usageBucketGridSpec) bucketExpr(ls *LocalStorage) string {
	return fmt.Sprintf("((%s) - %d) / %d", ls.usageEpochExpr(), g.seriesStartEpoch, g.bucketSeconds)
}

// bucketStart returns the UTC start time of bucket i.
func (g usageBucketGridSpec) bucketStart(i int) time.Time {
	return g.seriesStart.Add(time.Duration(int64(i)*g.bucketSeconds) * time.Second).UTC()
}

// usageModelBucketRow is the scan target for the per-model bucketed query.
type usageModelBucketRow struct {
	Model       string
	BucketIdx   int64
	TotalTokens int64
}

// usageByModelTopN is the number of top models that get their own series before
// the remainder is rolled up into the "other" entry.
const usageByModelTopN = 3

// GetUsageTimeseriesByModel returns per-model bucketed token series over the
// same aligned grid as GetUsageTimeseries (identical start, bucket_seconds, and
// count). The top usageByModelTopN models by their in-window token total each
// get their own series (ordered by total descending, key ascending on ties);
// every remaining model is summed into a trailing "other" series, which is
// omitted entirely when there is no remainder. Each series is zero-filled with
// exactly `buckets` ascending points, tokens only (no cost). A single grouped
// SQL query (bucket x model) feeds the top-N/other rollup done here in Go. Rows
// with an empty model are excluded. Dialect-portable via usageEpochExpr.
func (ls *LocalStorage) GetUsageTimeseriesByModel(ctx context.Context, since *time.Time, now time.Time, buckets int) ([]types.UsageModelSeries, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get usage timeseries by model: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	grid, err := ls.usageBucketGrid(gormDB, since, now, buckets)
	if err != nil {
		return nil, err
	}

	bucketExpr := grid.bucketExpr(ls)

	var rows []usageModelBucketRow
	if err := gormDB.Table("execution_usage").
		Select("model AS model, "+bucketExpr+" AS bucket_idx, "+
			"COALESCE(SUM(total_tokens),0) AS total_tokens").
		Where(ls.usageEpochExpr()+" >= ?", grid.seriesStart.Unix()).
		Where("model <> ''").
		Group("model, " + bucketExpr).
		Scan(&rows).Error; err != nil {
		return nil, fmt.Errorf("failed to aggregate usage timeseries by model: %w", err)
	}

	// Accumulate per-model window totals (used for ranking) and keep only rows
	// that land inside the aligned series window.
	type modelBuckets struct {
		total   int64
		buckets map[int64]int64
	}
	perModel := make(map[string]*modelBuckets)
	for _, r := range rows {
		if r.BucketIdx < 0 || r.BucketIdx >= int64(grid.buckets) {
			continue // rows outside the aligned series window (e.g. exactly at now)
		}
		mb := perModel[r.Model]
		if mb == nil {
			mb = &modelBuckets{buckets: make(map[int64]int64)}
			perModel[r.Model] = mb
		}
		mb.total += r.TotalTokens
		mb.buckets[r.BucketIdx] += r.TotalTokens
	}

	if len(perModel) == 0 {
		return []types.UsageModelSeries{}, nil
	}

	// Rank models by window total descending, breaking ties by key ascending.
	keys := make([]string, 0, len(perModel))
	for k := range perModel {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		ti, tj := perModel[keys[i]].total, perModel[keys[j]].total
		if ti != tj {
			return ti > tj
		}
		return keys[i] < keys[j]
	})

	topN := usageByModelTopN
	if topN > len(keys) {
		topN = len(keys)
	}

	series := make([]types.UsageModelSeries, 0, topN+1)
	for _, k := range keys[:topN] {
		series = append(series, buildModelSeries(grid, k, perModel[k].buckets))
	}

	// Roll every remaining model into a single "other" series.
	if len(keys) > topN {
		otherBuckets := make(map[int64]int64)
		for _, k := range keys[topN:] {
			for idx, tok := range perModel[k].buckets {
				otherBuckets[idx] += tok
			}
		}
		series = append(series, buildModelSeries(grid, "other", otherBuckets))
	}

	return series, nil
}

// buildModelSeries zero-fills exactly grid.buckets ascending points for one
// series, filling token totals from the per-bucket map.
func buildModelSeries(grid usageBucketGridSpec, key string, bucketTokens map[int64]int64) types.UsageModelSeries {
	points := make([]types.UsageModelTimeseriesPoint, grid.buckets)
	for i := 0; i < grid.buckets; i++ {
		points[i] = types.UsageModelTimeseriesPoint{
			Start:       grid.bucketStart(i),
			TotalTokens: bucketTokens[int64(i)],
		}
	}
	return types.UsageModelSeries{Key: key, Points: points}
}

// queryUsageGroups groups execution_usage by keyCol (excluding empty keys),
// summing tokens/cost and counting entries, ordered by total tokens descending.
func (ls *LocalStorage) queryUsageGroups(ctx context.Context, keyCol string, includeProvider bool, since *time.Time) ([]types.UsageStatsGroup, error) {
	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	selectParts := []string{keyCol + " AS group_key"}
	groupParts := []string{keyCol}
	if includeProvider {
		selectParts = append(selectParts, "provider AS provider")
		groupParts = append(groupParts, "provider")
	}
	selectParts = append(selectParts,
		"COALESCE(SUM(input_tokens),0) AS input_tokens",
		"COALESCE(SUM(output_tokens),0) AS output_tokens",
		"COALESCE(SUM(total_tokens),0) AS total_tokens",
		"SUM(cost_usd) AS cost_usd",
		"COUNT(*) AS entries",
	)

	query := gormDB.Table("execution_usage").
		Select(strings.Join(selectParts, ", ")).
		Where(keyCol + " <> ''")
	if since != nil {
		query = query.Where(ls.usageEpochExpr()+" >= ?", since.Unix())
	}
	query = query.Group(strings.Join(groupParts, ", ")).Order("total_tokens DESC")

	var rows []usageGroupRow
	if err := query.Scan(&rows).Error; err != nil {
		return nil, fmt.Errorf("failed to aggregate usage by %s: %w", keyCol, err)
	}

	groups := make([]types.UsageStatsGroup, 0, len(rows))
	for _, r := range rows {
		g := types.UsageStatsGroup{
			Key:          r.GroupKey,
			InputTokens:  r.InputTokens,
			OutputTokens: r.OutputTokens,
			TotalTokens:  r.TotalTokens,
			Entries:      r.Entries,
		}
		if includeProvider {
			g.Provider = r.Provider
		}
		if r.CostUSD.Valid {
			cost := r.CostUSD.Float64
			g.CostUSD = &cost
		}
		groups = append(groups, g)
	}
	return groups, nil
}
