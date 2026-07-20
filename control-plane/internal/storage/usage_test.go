package storage

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func newUsageTestStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()
	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping usage storage test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() { _ = ls.Close(ctx) })
	return ls, ctx
}

func costPtr(v float64) *float64 { return &v }

func TestCreateAndAggregateExecutionUsage(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "exec_1", WorkflowID: "wf_1", AgentNodeID: "agent-a",
			Source: "llm", Provider: "anthropic", Model: "claude-opus-4-8",
			InputTokens: 100, OutputTokens: 50, TotalTokens: 150,
			CostUSD: costPtr(0.01), CostSource: "litellm", CreatedAt: now,
		},
		{
			ExecutionID: "exec_1", WorkflowID: "wf_1", AgentNodeID: "agent-a",
			Source: "harness", Harness: "claude_code", Model: "claude-opus-4-8", Provider: "anthropic",
			InputTokens: 10, OutputTokens: 5, TotalTokens: 15,
			CostUSD: nil, CreatedAt: now,
		},
		{
			ExecutionID: "exec_2", WorkflowID: "wf_2", AgentNodeID: "agent-b",
			Source: "llm", Provider: "openrouter", Model: "gpt-4o",
			InputTokens: 200, OutputTokens: 100, TotalTokens: 300,
			CostUSD: costPtr(0.05), CostSource: "provider", CreatedAt: now,
		},
	}

	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create execution usage: %v", err)
	}
	for i, r := range rows {
		if r.ID == 0 {
			t.Errorf("row %d did not get an assigned ID", i)
		}
	}

	agg, err := ls.GetUsageStats(ctx, nil)
	if err != nil {
		t.Fatalf("get usage stats: %v", err)
	}

	// Totals
	if agg.Totals.InputTokens != 310 {
		t.Errorf("input_tokens = %d, want 310", agg.Totals.InputTokens)
	}
	if agg.Totals.OutputTokens != 155 {
		t.Errorf("output_tokens = %d, want 155", agg.Totals.OutputTokens)
	}
	if agg.Totals.TotalTokens != 465 {
		t.Errorf("total_tokens = %d, want 465", agg.Totals.TotalTokens)
	}
	if agg.Totals.CostUSD == nil {
		t.Fatalf("totals cost_usd is nil, want ~0.06")
	}
	if *agg.Totals.CostUSD < 0.0599 || *agg.Totals.CostUSD > 0.0601 {
		t.Errorf("totals cost_usd = %f, want ~0.06", *agg.Totals.CostUSD)
	}
	if agg.Totals.ExecutionsWithUsage != 2 {
		t.Errorf("executions_with_usage = %d, want 2", agg.Totals.ExecutionsWithUsage)
	}
	if agg.LastUpdated == nil {
		t.Errorf("last_updated is nil, want a timestamp")
	}

	// by_model: claude-opus-4-8 (165 tokens) should sort before gpt-4o (300)?
	// gpt-4o has 300 tokens vs claude 150+15=165, so gpt-4o first.
	if len(agg.ByModel) != 2 {
		t.Fatalf("by_model has %d groups, want 2", len(agg.ByModel))
	}
	if agg.ByModel[0].Key != "gpt-4o" || agg.ByModel[0].TotalTokens != 300 {
		t.Errorf("by_model[0] = %+v, want gpt-4o/300", agg.ByModel[0])
	}
	if agg.ByModel[1].Key != "claude-opus-4-8" || agg.ByModel[1].TotalTokens != 165 {
		t.Errorf("by_model[1] = %+v, want claude-opus-4-8/165", agg.ByModel[1])
	}
	if agg.ByModel[1].Provider != "anthropic" {
		t.Errorf("by_model[1].Provider = %q, want anthropic", agg.ByModel[1].Provider)
	}
	if agg.ByModel[1].Entries != 2 {
		t.Errorf("by_model[1].Entries = %d, want 2", agg.ByModel[1].Entries)
	}

	// by_harness: only the harness=claude_code row has a non-empty harness.
	if len(agg.ByHarness) != 1 {
		t.Fatalf("by_harness has %d groups, want 1", len(agg.ByHarness))
	}
	if agg.ByHarness[0].Key != "claude_code" {
		t.Errorf("by_harness[0].Key = %q, want claude_code", agg.ByHarness[0].Key)
	}

	// by_agent
	if len(agg.ByAgent) != 2 {
		t.Fatalf("by_agent has %d groups, want 2", len(agg.ByAgent))
	}

	// by_provider
	if len(agg.ByProvider) != 2 {
		t.Fatalf("by_provider has %d groups, want 2", len(agg.ByProvider))
	}

	// per-execution totals
	cost, tokens, err := ls.GetExecutionUsageTotals(ctx, "exec_1")
	if err != nil {
		t.Fatalf("get execution usage totals: %v", err)
	}
	if tokens != 165 {
		t.Errorf("exec_1 tokens = %d, want 165", tokens)
	}
	if cost == nil || *cost < 0.0099 || *cost > 0.0101 {
		t.Errorf("exec_1 cost = %v, want ~0.01", cost)
	}
}

func TestGetUsageStatsWindowFiltering(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "recent", WorkflowID: "wf", AgentNodeID: "agent-a",
			Source: "llm", Model: "m1", InputTokens: 10, OutputTokens: 10, TotalTokens: 20,
			CreatedAt: now.Add(-30 * time.Minute),
		},
		{
			ExecutionID: "old", WorkflowID: "wf", AgentNodeID: "agent-a",
			Source: "llm", Model: "m1", InputTokens: 5, OutputTokens: 5, TotalTokens: 10,
			CreatedAt: now.Add(-48 * time.Hour),
		},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create execution usage: %v", err)
	}

	// 1h window: only the recent row.
	since1h := now.Add(-time.Hour)
	agg, err := ls.GetUsageStats(ctx, &since1h)
	if err != nil {
		t.Fatalf("get usage stats (1h): %v", err)
	}
	if agg.Totals.TotalTokens != 20 {
		t.Errorf("1h total_tokens = %d, want 20", agg.Totals.TotalTokens)
	}
	if agg.Totals.ExecutionsWithUsage != 1 {
		t.Errorf("1h executions_with_usage = %d, want 1", agg.Totals.ExecutionsWithUsage)
	}

	// all window: both rows.
	agg, err = ls.GetUsageStats(ctx, nil)
	if err != nil {
		t.Fatalf("get usage stats (all): %v", err)
	}
	if agg.Totals.TotalTokens != 30 {
		t.Errorf("all total_tokens = %d, want 30", agg.Totals.TotalTokens)
	}
}

// TestGetUsageStatsNonUTCOffsets pins the window filter against rows whose
// created_at carries a non-UTC offset — what a control plane running in a
// non-UTC timezone stores (GORM stamps time.Now() in server-local time).
// SQLite keeps timestamps as text and compares them lexicographically, so a
// naive "created_at >= <UTC since>" silently excludes such rows even when the
// instant is in range; all window comparisons must go through the epoch
// expression instead. Caught live on an EDT host: /usage/stats?window=1h
// returned zeros while window=all showed the rows.
func TestGetUsageStatsNonUTCOffsets(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()
	edt := time.FixedZone("EDT", -4*3600)

	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "recent-local", WorkflowID: "wf", AgentNodeID: "agent-a",
			Source: "llm", Model: "m1", InputTokens: 10, OutputTokens: 10, TotalTokens: 20,
			CreatedAt: now.Add(-30 * time.Minute).In(edt),
		},
		{
			ExecutionID: "old-local", WorkflowID: "wf", AgentNodeID: "agent-a",
			Source: "llm", Model: "m1", InputTokens: 5, OutputTokens: 5, TotalTokens: 10,
			CreatedAt: now.Add(-48 * time.Hour).In(edt),
		},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create execution usage: %v", err)
	}

	since1h := now.Add(-time.Hour)
	agg, err := ls.GetUsageStats(ctx, &since1h)
	if err != nil {
		t.Fatalf("get usage stats (1h): %v", err)
	}
	if agg.Totals.TotalTokens != 20 || agg.Totals.ExecutionsWithUsage != 1 {
		t.Errorf("1h window missed offset-zoned row: tokens=%d execs=%d, want 20/1",
			agg.Totals.TotalTokens, agg.Totals.ExecutionsWithUsage)
	}
	if len(agg.ByModel) != 1 || agg.ByModel[0].TotalTokens != 20 {
		t.Errorf("1h by_model missed offset-zoned row: %+v", agg.ByModel)
	}
	if agg.LastUpdated == nil {
		t.Error("1h last_updated missing for offset-zoned row")
	} else if diff := agg.LastUpdated.Sub(now.Add(-30 * time.Minute)); diff < -time.Minute || diff > time.Minute {
		t.Errorf("last_updated = %v, want ~%v", agg.LastUpdated, now.Add(-30*time.Minute))
	}

	// The recent offset-zoned row must land in the timeseries window too.
	series, err := ls.GetUsageTimeseries(ctx, &since1h, now, 4)
	if err != nil {
		t.Fatalf("get usage timeseries (1h): %v", err)
	}
	var seriesTotal int64
	for _, p := range series.Points {
		seriesTotal += p.TotalTokens
	}
	if seriesTotal != 20 {
		t.Errorf("1h timeseries missed offset-zoned row: total=%d, want 20", seriesTotal)
	}
}

func TestGetUsageTimeseriesBucketing(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	// 24h window, 24 buckets → bucket_seconds = 3600 (one hour each).
	// Place rows in distinct buckets to verify grouping and zero-fill.
	//  - two rows ~ 30min ago  → last bucket (index 23), costs 0.01 + nil
	//  - one row  ~ 90min ago  → bucket 22, cost nil (rows present, all null)
	//  - one row  ~ 5h ago     → bucket 19, cost 0.05
	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "e1", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", InputTokens: 10, OutputTokens: 10, TotalTokens: 20,
			CostUSD: costPtr(0.01), CreatedAt: now.Add(-30 * time.Minute),
		},
		{
			ExecutionID: "e2", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", InputTokens: 5, OutputTokens: 5, TotalTokens: 30,
			CostUSD: nil, CreatedAt: now.Add(-31 * time.Minute),
		},
		{
			ExecutionID: "e3", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", InputTokens: 5, OutputTokens: 5, TotalTokens: 7,
			CostUSD: nil, CreatedAt: now.Add(-90 * time.Minute),
		},
		{
			ExecutionID: "e4", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", InputTokens: 5, OutputTokens: 5, TotalTokens: 500,
			CostUSD: costPtr(0.05), CreatedAt: now.Add(-5 * time.Hour),
		},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create execution usage: %v", err)
	}

	since := now.Add(-24 * time.Hour)
	ts, err := ls.GetUsageTimeseries(ctx, &since, now, 24)
	if err != nil {
		t.Fatalf("get usage timeseries: %v", err)
	}

	if ts.BucketSeconds != 3600 {
		t.Errorf("bucket_seconds = %d, want 3600", ts.BucketSeconds)
	}
	if len(ts.Points) != 24 {
		t.Fatalf("points = %d, want exactly 24", len(ts.Points))
	}

	// Ascending order, contiguous by bucket_seconds.
	for i := 1; i < len(ts.Points); i++ {
		if !ts.Points[i].Start.After(ts.Points[i-1].Start) {
			t.Errorf("points not ascending at %d: %v <= %v", i, ts.Points[i].Start, ts.Points[i-1].Start)
		}
		gap := ts.Points[i].Start.Sub(ts.Points[i-1].Start)
		if gap != time.Duration(ts.BucketSeconds)*time.Second {
			t.Errorf("gap at %d = %v, want %ds", i, gap, ts.BucketSeconds)
		}
	}
	// Starts are UTC.
	if ts.Points[0].Start.Location() != time.UTC {
		t.Errorf("point start not UTC: %v", ts.Points[0].Start.Location())
	}

	// Last bucket (23): tokens 20+30=50, cost = 0.01 (only non-null summed).
	last := ts.Points[23]
	if last.TotalTokens != 50 {
		t.Errorf("bucket 23 total_tokens = %d, want 50", last.TotalTokens)
	}
	if last.CostUSD == nil || *last.CostUSD < 0.0099 || *last.CostUSD > 0.0101 {
		t.Errorf("bucket 23 cost_usd = %v, want ~0.01", last.CostUSD)
	}

	// Bucket 22: one row, cost all-null → tokens 7, cost nil.
	b22 := ts.Points[22]
	if b22.TotalTokens != 7 {
		t.Errorf("bucket 22 total_tokens = %d, want 7", b22.TotalTokens)
	}
	if b22.CostUSD != nil {
		t.Errorf("bucket 22 cost_usd = %v, want nil (rows present, all costs null)", b22.CostUSD)
	}

	// Bucket 19 (~5h ago): tokens 500, cost 0.05.
	b19 := ts.Points[19]
	if b19.TotalTokens != 500 {
		t.Errorf("bucket 19 total_tokens = %d, want 500", b19.TotalTokens)
	}
	if b19.CostUSD == nil || *b19.CostUSD < 0.0499 || *b19.CostUSD > 0.0501 {
		t.Errorf("bucket 19 cost_usd = %v, want ~0.05", b19.CostUSD)
	}

	// An empty bucket (e.g. index 0) is zero-filled with nil cost.
	if ts.Points[0].TotalTokens != 0 || ts.Points[0].CostUSD != nil {
		t.Errorf("bucket 0 = %+v, want zero tokens + nil cost", ts.Points[0])
	}
}

func TestGetUsageTimeseriesBucketCount(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()
	since := now.Add(-time.Hour)

	for _, n := range []int{1, 5, 60, 200} {
		ts, err := ls.GetUsageTimeseries(ctx, &since, now, n)
		if err != nil {
			t.Fatalf("timeseries n=%d: %v", n, err)
		}
		if len(ts.Points) != n {
			t.Errorf("n=%d: points = %d, want %d", n, len(ts.Points), n)
		}
		if ts.BucketSeconds != int64(3600/n) {
			t.Errorf("n=%d: bucket_seconds = %d, want %d", n, ts.BucketSeconds, 3600/n)
		}
	}
}

func TestGetUsageTimeseriesAllWindowFallback(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	// Empty table: "all" window (nil since) falls back to a 24h span.
	ts, err := ls.GetUsageTimeseries(ctx, nil, now, 24)
	if err != nil {
		t.Fatalf("timeseries (empty all): %v", err)
	}
	if len(ts.Points) != 24 {
		t.Fatalf("points = %d, want 24", len(ts.Points))
	}
	if ts.BucketSeconds != 3600 {
		t.Errorf("empty all bucket_seconds = %d, want 3600 (24h/24 fallback)", ts.BucketSeconds)
	}
	for _, p := range ts.Points {
		if p.TotalTokens != 0 || p.CostUSD != nil {
			t.Errorf("empty all should be zero-filled, got %+v", p)
		}
	}

	// With data: "all" spans from the oldest row (~10h ago) to now.
	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "old", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", TotalTokens: 100, CostUSD: costPtr(0.02),
			CreatedAt: now.Add(-10 * time.Hour),
		},
		{
			ExecutionID: "new", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", TotalTokens: 200, CostUSD: costPtr(0.03),
			CreatedAt: now.Add(-1 * time.Minute),
		},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create: %v", err)
	}

	ts, err = ls.GetUsageTimeseries(ctx, nil, now, 10)
	if err != nil {
		t.Fatalf("timeseries (all with data): %v", err)
	}
	if len(ts.Points) != 10 {
		t.Fatalf("points = %d, want 10", len(ts.Points))
	}
	// ~10h span / 10 buckets ≈ 3600s each.
	if ts.BucketSeconds < 3500 || ts.BucketSeconds > 3700 {
		t.Errorf("all-with-data bucket_seconds = %d, want ~3600", ts.BucketSeconds)
	}
	// All tokens present across the series should sum to 300.
	var total int64
	for _, p := range ts.Points {
		total += p.TotalTokens
	}
	if total != 300 {
		t.Errorf("summed tokens = %d, want 300", total)
	}
}

func TestGetUsageTimeseriesByModelBucketing(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	// 24h window, 24 buckets → bucket_seconds = 3600 (one hour each). Place rows
	// for a single model in distinct buckets to verify per-model placement and
	// zero-fill (tokens only, no cost carried).
	//  - model "m": two rows ~30/31min ago → bucket 23 (20+30=50), one ~5h ago → bucket 19 (500)
	rows := []*types.ExecutionUsage{
		{
			ExecutionID: "e1", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", TotalTokens: 20,
			CostUSD: costPtr(0.01), CreatedAt: now.Add(-30 * time.Minute),
		},
		{
			ExecutionID: "e2", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", TotalTokens: 30,
			CreatedAt: now.Add(-31 * time.Minute),
		},
		{
			ExecutionID: "e3", WorkflowID: "wf", AgentNodeID: "a",
			Source: "llm", Model: "m", TotalTokens: 500,
			CreatedAt: now.Add(-5 * time.Hour),
		},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create execution usage: %v", err)
	}

	since := now.Add(-24 * time.Hour)
	series, err := ls.GetUsageTimeseriesByModel(ctx, &since, now, 24)
	if err != nil {
		t.Fatalf("get usage timeseries by model: %v", err)
	}
	if len(series) != 1 {
		t.Fatalf("series has %d entries, want 1", len(series))
	}
	s := series[0]
	if s.Key != "m" {
		t.Errorf("series[0].Key = %q, want m", s.Key)
	}
	if len(s.Points) != 24 {
		t.Fatalf("series[0] points = %d, want exactly 24", len(s.Points))
	}
	// Ascending, contiguous by bucket_seconds, UTC.
	for i := 1; i < len(s.Points); i++ {
		if !s.Points[i].Start.After(s.Points[i-1].Start) {
			t.Errorf("points not ascending at %d: %v <= %v", i, s.Points[i].Start, s.Points[i-1].Start)
		}
		if gap := s.Points[i].Start.Sub(s.Points[i-1].Start); gap != 3600*time.Second {
			t.Errorf("gap at %d = %v, want 3600s", i, gap)
		}
	}
	if s.Points[0].Start.Location() != time.UTC {
		t.Errorf("point start not UTC: %v", s.Points[0].Start.Location())
	}
	if s.Points[23].TotalTokens != 50 {
		t.Errorf("bucket 23 tokens = %d, want 50", s.Points[23].TotalTokens)
	}
	if s.Points[19].TotalTokens != 500 {
		t.Errorf("bucket 19 tokens = %d, want 500", s.Points[19].TotalTokens)
	}
	if s.Points[0].TotalTokens != 0 {
		t.Errorf("bucket 0 tokens = %d, want 0 (zero-fill)", s.Points[0].TotalTokens)
	}
}

func TestGetUsageTimeseriesByModelTopThreeAndOther(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()
	at := now.Add(-30 * time.Minute)

	// Five models with descending window totals; top 3 keep their own series,
	// the remaining two roll into "other" (m4=40 + m5=30 = 70).
	rows := []*types.ExecutionUsage{
		{ExecutionID: "x1", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m1", TotalTokens: 100, CreatedAt: at},
		{ExecutionID: "x2", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m2", TotalTokens: 80, CreatedAt: at},
		{ExecutionID: "x3", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m3", TotalTokens: 60, CreatedAt: at},
		{ExecutionID: "x4", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m4", TotalTokens: 40, CreatedAt: at},
		{ExecutionID: "x5", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m5", TotalTokens: 30, CreatedAt: at},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create: %v", err)
	}

	since := now.Add(-time.Hour)
	series, err := ls.GetUsageTimeseriesByModel(ctx, &since, now, 12)
	if err != nil {
		t.Fatalf("by model: %v", err)
	}
	if len(series) != 4 {
		t.Fatalf("series has %d entries, want 4 (top 3 + other)", len(series))
	}
	wantKeys := []string{"m1", "m2", "m3", "other"}
	for i, want := range wantKeys {
		if series[i].Key != want {
			t.Errorf("series[%d].Key = %q, want %q", i, series[i].Key, want)
		}
		if len(series[i].Points) != 12 {
			t.Errorf("series[%d] points = %d, want 12", i, len(series[i].Points))
		}
	}
	sumTokens := func(s types.UsageModelSeries) int64 {
		var t int64
		for _, p := range s.Points {
			t += p.TotalTokens
		}
		return t
	}
	if got := sumTokens(series[0]); got != 100 {
		t.Errorf("m1 total = %d, want 100", got)
	}
	if got := sumTokens(series[3]); got != 70 {
		t.Errorf("other total = %d, want 70 (m4+m5)", got)
	}
}

func TestGetUsageTimeseriesByModelNoOtherWhenThreeOrFewer(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()
	at := now.Add(-10 * time.Minute)

	rows := []*types.ExecutionUsage{
		{ExecutionID: "x1", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m1", TotalTokens: 30, CreatedAt: at},
		{ExecutionID: "x2", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m2", TotalTokens: 20, CreatedAt: at},
		{ExecutionID: "x3", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m3", TotalTokens: 10, CreatedAt: at},
		// An empty-model row must be excluded entirely (not counted, no series).
		{ExecutionID: "x4", WorkflowID: "wf", AgentNodeID: "a", Source: "harness", Model: "", TotalTokens: 999, CreatedAt: at},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create: %v", err)
	}

	since := now.Add(-time.Hour)
	series, err := ls.GetUsageTimeseriesByModel(ctx, &since, now, 6)
	if err != nil {
		t.Fatalf("by model: %v", err)
	}
	if len(series) != 3 {
		t.Fatalf("series has %d entries, want 3 (no other)", len(series))
	}
	for _, s := range series {
		if s.Key == "other" {
			t.Errorf("unexpected 'other' series with only 3 models")
		}
		if s.Key == "" {
			t.Errorf("empty-model row leaked into series")
		}
	}
}

func TestGetUsageTimeseriesByModelEmpty(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()
	since := now.Add(-time.Hour)

	series, err := ls.GetUsageTimeseriesByModel(ctx, &since, now, 12)
	if err != nil {
		t.Fatalf("by model (empty): %v", err)
	}
	if len(series) != 0 {
		t.Errorf("empty table series = %d entries, want 0", len(series))
	}
}

func TestGetUsageTimeseriesByModelSameGridAsSeries(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)
	now := time.Now().UTC()

	rows := []*types.ExecutionUsage{
		{ExecutionID: "x1", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m1", TotalTokens: 40, CreatedAt: now.Add(-90 * time.Minute)},
		{ExecutionID: "x2", WorkflowID: "wf", AgentNodeID: "a", Source: "llm", Model: "m2", TotalTokens: 60, CreatedAt: now.Add(-5 * time.Hour)},
	}
	if err := ls.CreateExecutionUsage(ctx, rows); err != nil {
		t.Fatalf("create: %v", err)
	}

	since := now.Add(-24 * time.Hour)
	ts, err := ls.GetUsageTimeseries(ctx, &since, now, 24)
	if err != nil {
		t.Fatalf("timeseries: %v", err)
	}
	byModel, err := ls.GetUsageTimeseriesByModel(ctx, &since, now, 24)
	if err != nil {
		t.Fatalf("by model: %v", err)
	}
	if len(byModel) == 0 {
		t.Fatalf("expected per-model series")
	}
	// Every per-model series lands on exactly the same time axis as "series".
	for _, s := range byModel {
		if len(s.Points) != len(ts.Points) {
			t.Fatalf("series %q has %d points, want %d", s.Key, len(s.Points), len(ts.Points))
		}
		for i := range s.Points {
			if !s.Points[i].Start.Equal(ts.Points[i].Start) {
				t.Errorf("series %q point %d start = %v, want %v", s.Key, i, s.Points[i].Start, ts.Points[i].Start)
			}
		}
	}
	// Per-model tokens summed across models must equal the plain series tokens
	// bucket-for-bucket (all rows here have non-empty models).
	perBucket := make([]int64, len(ts.Points))
	for _, s := range byModel {
		for i, p := range s.Points {
			perBucket[i] += p.TotalTokens
		}
	}
	for i, p := range ts.Points {
		if perBucket[i] != p.TotalTokens {
			t.Errorf("bucket %d: per-model sum = %d, plain series = %d", i, perBucket[i], p.TotalTokens)
		}
	}
}

func TestGetUsageStatsEmpty(t *testing.T) {
	ls, ctx := newUsageTestStorage(t)

	agg, err := ls.GetUsageStats(ctx, nil)
	if err != nil {
		t.Fatalf("get usage stats: %v", err)
	}
	if agg.Totals.TotalTokens != 0 {
		t.Errorf("total_tokens = %d, want 0", agg.Totals.TotalTokens)
	}
	if agg.Totals.CostUSD != nil {
		t.Errorf("cost_usd = %v, want nil", agg.Totals.CostUSD)
	}
	if agg.LastUpdated != nil {
		t.Errorf("last_updated = %v, want nil", agg.LastUpdated)
	}
	if len(agg.ByModel) != 0 || len(agg.ByProvider) != 0 || len(agg.ByAgent) != 0 || len(agg.ByHarness) != 0 {
		t.Errorf("expected empty groups, got model=%d provider=%d agent=%d harness=%d",
			len(agg.ByModel), len(agg.ByProvider), len(agg.ByAgent), len(agg.ByHarness))
	}
}
