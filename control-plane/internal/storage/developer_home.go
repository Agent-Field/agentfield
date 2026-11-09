package storage

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// The Home aggregates are plain SQL over the executions table and its
// neighbours. Every figure is a COUNT, SUM or a rank picked with a window
// function, so none of them depends on how many rows a list query is allowed
// to return, and the same statements run on SQLite and PostgreSQL.

const (
	homeFailedStatuses   = "('failed','timeout')"
	homeTerminalStatuses = "('succeeded','failed','timeout','cancelled')"
	homeHeldStatus       = "'waiting'"
)

// homeEpochExpr yields a timestamp column as integer unix seconds.
func (ls *LocalStorage) homeEpochExpr(column string) string {
	if ls.mode == "postgres" {
		return "CAST(FLOOR(EXTRACT(EPOCH FROM " + column + ")) AS BIGINT)"
	}
	return "CAST(strftime('%s', " + column + ") AS INTEGER)"
}

// homeBucketExpr maps started_at to a zero-based bucket index. The two
// integers are computed in Go from the request window, never from input text.
func (ls *LocalStorage) homeBucketExpr(column string, startEpoch, bucketSeconds int64) string {
	return fmt.Sprintf("((%s - %d) / %d)", ls.homeEpochExpr(column), startEpoch, bucketSeconds)
}

// percentileRankSQL is the 1-based rank holding the p-th percentile of cnt
// sorted values (nearest-rank method, ceil(cnt*p/100)).
func percentileRankSQL(p int) string {
	return fmt.Sprintf("((cnt * %d + 99) / 100)", p)
}

// CountExecutionsByStatus returns exact counts of executions that started in
// [start, end). It replaces counting the rows of a capped list query.
func (ls *LocalStorage) CountExecutionsByStatus(ctx context.Context, start, end time.Time) (types.ExecutionStatusCounts, error) {
	var counts types.ExecutionStatusCounts
	if err := ctx.Err(); err != nil {
		return counts, err
	}
	db := ls.requireSQLDB()
	query := `SELECT COUNT(*),
		COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status IN ` + homeFailedStatuses + ` THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status IN ` + homeTerminalStatuses + ` THEN 1 ELSE 0 END), 0)
		FROM executions WHERE started_at >= ? AND started_at < ?`
	if err := db.QueryRowContext(ctx, query, start.UTC(), end.UTC()).Scan(
		&counts.Total, &counts.Succeeded, &counts.Failed, &counts.Cancelled, &counts.Terminal,
	); err != nil {
		return counts, fmt.Errorf("count executions by status: %w", err)
	}
	return counts, nil
}

// QueryRecentFailures returns the newest failed, timed-out or cancelled
// executions that started in [start, end), newest first.
func (ls *LocalStorage) QueryRecentFailures(ctx context.Context, start, end time.Time, limit int) ([]*types.Execution, error) {
	if limit <= 0 {
		limit = 10
	}
	db := ls.requireSQLDB()
	rows, err := db.QueryContext(ctx, `SELECT execution_id, run_id, agent_node_id, reasoner_id, status,
		error_message, started_at, completed_at, duration_ms
		FROM executions
		WHERE started_at >= ? AND started_at < ? AND status IN ('failed','timeout','cancelled')
		ORDER BY started_at DESC LIMIT ?`, start.UTC(), end.UTC(), limit)
	if err != nil {
		return nil, fmt.Errorf("query recent failures: %w", err)
	}
	defer rows.Close()
	var out []*types.Execution
	for rows.Next() {
		var (
			e                   types.Execution
			errMsg              sql.NullString
			startedRaw, doneRaw interface{}
			duration            sql.NullInt64
		)
		if err := rows.Scan(&e.ExecutionID, &e.RunID, &e.AgentNodeID, &e.ReasonerID, &e.Status,
			&errMsg, &startedRaw, &doneRaw, &duration); err != nil {
			return nil, fmt.Errorf("scan recent failure: %w", err)
		}
		if errMsg.Valid {
			msg := errMsg.String
			e.ErrorMessage = &msg
		}
		e.StartedAt, _ = parseDBTime(startedRaw)
		if done, _ := parseDBTime(doneRaw); !done.IsZero() {
			e.CompletedAt = &done
		}
		if duration.Valid {
			d := duration.Int64
			e.DurationMS = &d
		}
		out = append(out, &e)
	}
	return out, rows.Err()
}

// QueryDeveloperHome computes every aggregate the Home page shows for one
// window.
func (ls *LocalStorage) QueryDeveloperHome(ctx context.Context, q types.DeveloperHomeQuery) (*types.DeveloperHomeData, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !q.End.After(q.Start) {
		return nil, fmt.Errorf("developer home window must end after it starts")
	}
	if q.BucketSeconds <= 0 {
		q.BucketSeconds = int64(q.End.Sub(q.Start).Seconds()) / 12
		if q.BucketSeconds <= 0 {
			q.BucketSeconds = 1
		}
	}
	if q.Limit <= 0 {
		q.Limit = 12
	}
	q.Start = q.Start.UTC()
	q.End = q.End.UTC()

	out := &types.DeveloperHomeData{}
	steps := []func(context.Context, types.DeveloperHomeQuery, *types.DeveloperHomeData) error{
		ls.homeWindow,
		ls.homeBaseline,
		ls.homeProblems,
		ls.homeEdgesAndEndpoints,
		ls.homeApprovals,
		ls.homeSpend,
		ls.homeWebhooks,
		ls.homeTriggers,
	}
	for _, step := range steps {
		if err := step(ctx, q, out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// homePercentiles returns the p50 and p95 duration of finished executions
// that started in [start, end), picked by rank from one sort.
func (ls *LocalStorage) homePercentiles(ctx context.Context, start, end time.Time) (*int64, *int64, error) {
	db := ls.requireSQLDB()
	rows, err := db.QueryContext(ctx, `SELECT rn, cnt, d FROM (
			SELECT duration_ms AS d,
				ROW_NUMBER() OVER (ORDER BY duration_ms) AS rn,
				COUNT(*) OVER () AS cnt
			FROM executions
			WHERE started_at >= ? AND started_at < ? AND duration_ms IS NOT NULL
				AND status IN `+homeTerminalStatuses+`
		) ranked
		WHERE rn = `+percentileRankSQL(50)+` OR rn = `+percentileRankSQL(95),
		start.UTC(), end.UTC())
	if err != nil {
		return nil, nil, fmt.Errorf("query duration percentiles: %w", err)
	}
	defer rows.Close()
	var p50, p95 *int64
	for rows.Next() {
		var rn, cnt, d int64
		if err := rows.Scan(&rn, &cnt, &d); err != nil {
			return nil, nil, fmt.Errorf("scan duration percentile: %w", err)
		}
		v := d
		if rn == (cnt*50+99)/100 {
			p50 = &v
		}
		if rn == (cnt*95+99)/100 {
			v95 := d
			p95 = &v95
		}
	}
	return p50, p95, rows.Err()
}

func homeBucketCount(q types.DeveloperHomeQuery) int {
	span := int64(q.End.Sub(q.Start).Seconds())
	n := int((span + q.BucketSeconds - 1) / q.BucketSeconds)
	if n < 1 {
		n = 1
	}
	return n
}

// homeWindow fills the window totals and the sparkline series. One grouped
// scan gives the per-bucket counts (the totals are their sum), and one ranked
// scan gives the window's p50/p95 and each bucket's p95.
func (ls *LocalStorage) homeWindow(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	n := homeBucketCount(q)
	series := make([]types.DeveloperHomeBucket, n)
	for i := range series {
		series[i].Start = q.Start.Add(time.Duration(int64(i)*q.BucketSeconds) * time.Second)
	}
	bucket := ls.homeBucketExpr("started_at", q.Start.Unix(), q.BucketSeconds)

	rows, err := db.QueryContext(ctx, `SELECT `+bucket+` AS b, COUNT(*),
			COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status IN `+homeFailedStatuses+` THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status IN `+homeTerminalStatuses+` THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = `+homeHeldStatus+` THEN 1 ELSE 0 END), 0)
		FROM executions WHERE started_at >= ? AND started_at < ?
		GROUP BY `+bucket, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query call series: %w", err)
	}
	totals := &out.Totals
	for rows.Next() {
		var b, calls, succeeded, failed, cancelled, terminal, held int64
		if err := rows.Scan(&b, &calls, &succeeded, &failed, &cancelled, &terminal, &held); err != nil {
			rows.Close()
			return fmt.Errorf("scan call series: %w", err)
		}
		totals.Counts.Total += calls
		totals.Counts.Succeeded += succeeded
		totals.Counts.Failed += failed
		totals.Counts.Cancelled += cancelled
		totals.Counts.Terminal += terminal
		totals.Held += held
		if b < 0 || b >= int64(n) {
			continue
		}
		series[b].Calls, series[b].Failed, series[b].Terminal = calls, failed, terminal
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM executions WHERE status = 'running'`).
		Scan(&totals.RunningNow); err != nil {
		return fmt.Errorf("count running executions: %w", err)
	}

	prow, err := db.QueryContext(ctx, `SELECT b, d, rn_b, cnt_b, rn, cnt FROM (
			SELECT `+bucket+` AS b, duration_ms AS d,
				ROW_NUMBER() OVER (PARTITION BY `+bucket+` ORDER BY duration_ms) AS rn_b,
				COUNT(*) OVER (PARTITION BY `+bucket+`) AS cnt_b,
				ROW_NUMBER() OVER (ORDER BY duration_ms) AS rn,
				COUNT(*) OVER () AS cnt
			FROM executions
			WHERE started_at >= ? AND started_at < ? AND duration_ms IS NOT NULL
				AND status IN `+homeTerminalStatuses+`
		) ranked
		WHERE rn_b = ((cnt_b * 95 + 99) / 100) OR rn = `+percentileRankSQL(50)+` OR rn = `+percentileRankSQL(95),
		q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query duration percentiles: %w", err)
	}
	defer prow.Close()
	for prow.Next() {
		var b, d, rnB, cntB, rn, cnt int64
		if err := prow.Scan(&b, &d, &rnB, &cntB, &rn, &cnt); err != nil {
			return fmt.Errorf("scan duration percentile: %w", err)
		}
		if rnB == (cntB*95+99)/100 && b >= 0 && b < int64(n) {
			v := d
			series[b].P95Ms = &v
		}
		if rn == (cnt*50+99)/100 {
			v := d
			totals.P50Ms = &v
		}
		if rn == (cnt*95+99)/100 {
			v := d
			totals.P95Ms = &v
		}
	}
	out.Series = series
	return prow.Err()
}

func (ls *LocalStorage) homeBaseline(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	if q.BaselineStart.IsZero() || !q.Start.After(q.BaselineStart) {
		return nil
	}
	db := ls.requireSQLDB()
	bStart := q.BaselineStart.UTC()
	var (
		b        types.DeveloperHomeBaseline
		firstRaw interface{}
	)
	err := db.QueryRowContext(ctx, `SELECT COUNT(*),
			COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status IN `+homeFailedStatuses+` THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status IN `+homeTerminalStatuses+` THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status IN `+homeTerminalStatuses+` THEN COALESCE(duration_ms, 0) ELSE 0 END), 0),
			MIN(started_at)
		FROM executions WHERE started_at >= ? AND started_at < ?`, bStart, q.Start).Scan(
		&b.Counts.Total, &b.Counts.Succeeded, &b.Counts.Failed, &b.Counts.Cancelled,
		&b.Counts.Terminal, &b.BusyMs, &firstRaw)
	if err != nil {
		return fmt.Errorf("query baseline: %w", err)
	}
	if b.Counts.Total == 0 {
		return nil
	}
	first, _ := parseDBTime(firstRaw)
	spanStart := bStart
	if first.After(spanStart) {
		spanStart = first
	}
	b.Start, b.End = spanStart, q.Start
	b.Span = q.Start.Sub(spanStart).Seconds()
	if b.Span <= 0 {
		return nil
	}
	_, p95, err := ls.homePercentiles(ctx, bStart, q.Start)
	if err != nil {
		return err
	}
	b.P95Ms = p95
	out.Baseline = &b
	return nil
}

type homeProblemKey struct {
	agent, reasoner, status, message string
}

type homeProblemAcc struct {
	problem       types.DeveloperHomeProblem
	lastRaw       time.Time
	exampleExecID string
}

func (ls *LocalStorage) homeProblems(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	n := homeBucketCount(q)
	bucket := ls.homeBucketExpr("started_at", q.Start.Unix(), q.BucketSeconds)
	msgExpr := "COALESCE(NULLIF(error_message, ''), NULLIF(status_reason, ''), '')"
	rows, err := db.QueryContext(ctx, `SELECT agent_node_id, reasoner_id, status, `+msgExpr+` AS msg, `+bucket+` AS b,
			COUNT(*), MIN(started_at), MAX(started_at), MAX(execution_id)
		FROM executions
		WHERE started_at >= ? AND started_at < ? AND status IN `+homeFailedStatuses+`
		GROUP BY agent_node_id, reasoner_id, status, `+msgExpr+`, `+bucket, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query failure causes: %w", err)
	}
	groups := map[homeProblemKey]*homeProblemAcc{}
	for rows.Next() {
		var (
			agent, reasoner, status, msg, execID string
			b, count                             int64
			firstRaw, lastRaw                    interface{}
		)
		if err := rows.Scan(&agent, &reasoner, &status, &msg, &b, &count, &firstRaw, &lastRaw, &execID); err != nil {
			rows.Close()
			return fmt.Errorf("scan failure cause: %w", err)
		}
		key := homeProblemKey{agent, reasoner, status, NormalizeErrorMessage(msg)}
		acc, ok := groups[key]
		if !ok {
			acc = &homeProblemAcc{problem: types.DeveloperHomeProblem{
				AgentNodeID: agent, ReasonerID: reasoner, Status: status, Message: key.message,
				Buckets: make([]int64, n),
			}}
			groups[key] = acc
		}
		first, _ := parseDBTime(firstRaw)
		last, _ := parseDBTime(lastRaw)
		acc.problem.Count += count
		if acc.problem.FirstSeen.IsZero() || first.Before(acc.problem.FirstSeen) {
			acc.problem.FirstSeen = first
		}
		if last.After(acc.lastRaw) || acc.exampleExecID == "" {
			acc.lastRaw = last
			acc.problem.LastSeen = last
			acc.exampleExecID = execID
		}
		if b >= 0 && b < int64(n) {
			acc.problem.Buckets[b] += count
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	list := make([]*homeProblemAcc, 0, len(groups))
	for _, acc := range groups {
		list = append(list, acc)
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].problem.Count != list[j].problem.Count {
			return list[i].problem.Count > list[j].problem.Count
		}
		return list[i].problem.LastSeen.After(list[j].problem.LastSeen)
	})
	if len(list) > q.Limit {
		list = list[:q.Limit]
	}
	out.Problems = make([]types.DeveloperHomeProblem, 0, len(list))
	for _, acc := range list {
		acc.problem.ExampleExecutionID = acc.exampleExecID
		if err := db.QueryRowContext(ctx, `SELECT run_id FROM executions WHERE execution_id = ?`, acc.exampleExecID).
			Scan(&acc.problem.ExampleRunID); err != nil && err != sql.ErrNoRows {
			return fmt.Errorf("look up example run: %w", err)
		}
		out.Problems = append(out.Problems, acc.problem)
	}
	return nil
}

func (ls *LocalStorage) homeEdgesAndEndpoints(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	failedCase := "COALESCE(SUM(CASE WHEN c.status IN " + homeFailedStatuses + " THEN 1 ELSE 0 END), 0)"
	heldCase := "COALESCE(SUM(CASE WHEN c.status = " + homeHeldStatus + " THEN 1 ELSE 0 END), 0)"

	type epKey struct{ agent, reasoner string }
	endpoints := map[epKey]*types.DeveloperHomeEndpoint{}
	callerCalls := map[epKey]map[string]int64{}
	order := []epKey{}

	// Per-endpoint totals and p50/p95 from one ranked scan. Rows are ranked
	// within their endpoint with the finished, timed rows first by duration,
	// so the percentile ranks index those alone; rank 1 also carries the
	// endpoint's totals.
	timed := "(c.status IN " + homeTerminalStatuses + " AND c.duration_ms IS NOT NULL)"
	part := "PARTITION BY c.agent_node_id, c.reasoner_id"
	rows, err := db.QueryContext(ctx, `SELECT a, r, rn, calls, failed, terminal, cnt, d FROM (
			SELECT c.agent_node_id AS a, c.reasoner_id AS r, c.duration_ms AS d,
				ROW_NUMBER() OVER (`+part+` ORDER BY CASE WHEN `+timed+` THEN 0 ELSE 1 END, c.duration_ms) AS rn,
				COUNT(*) OVER (`+part+`) AS calls,
				SUM(CASE WHEN c.status IN `+homeFailedStatuses+` THEN 1 ELSE 0 END) OVER (`+part+`) AS failed,
				SUM(CASE WHEN c.status IN `+homeTerminalStatuses+` THEN 1 ELSE 0 END) OVER (`+part+`) AS terminal,
				SUM(CASE WHEN `+timed+` THEN 1 ELSE 0 END) OVER (`+part+`) AS cnt
			FROM executions c WHERE c.started_at >= ? AND c.started_at < ?
		) ranked
		WHERE rn = 1 OR rn = `+percentileRankSQL(50)+` OR rn = `+percentileRankSQL(95), q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query endpoint stats: %w", err)
	}
	for rows.Next() {
		var (
			a, r                             string
			rn, calls, failed, terminal, cnt int64
			d                                sql.NullInt64
		)
		if err := rows.Scan(&a, &r, &rn, &calls, &failed, &terminal, &cnt, &d); err != nil {
			rows.Close()
			return fmt.Errorf("scan endpoint stats: %w", err)
		}
		k := epKey{a, r}
		ep := endpoints[k]
		if ep == nil {
			ep = &types.DeveloperHomeEndpoint{AgentNodeID: a, ReasonerID: r, Calls: calls, Failed: failed,
				Terminal: terminal, CallerAgents: []string{}}
			endpoints[k] = ep
			order = append(order, k)
		}
		if cnt == 0 || !d.Valid {
			continue
		}
		if rn == (cnt*50+99)/100 {
			v := d.Int64
			ep.P50Ms = &v
		}
		if rn == (cnt*95+99)/100 {
			v := d.Int64
			ep.P95Ms = &v
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	// Agent-to-agent calls: each child execution joined to its parent.
	erows, err := db.QueryContext(ctx, `SELECT p.agent_node_id, c.agent_node_id, c.reasoner_id, COUNT(*),
			`+failedCase+`, `+heldCase+`
		FROM executions c
		JOIN executions p ON p.execution_id = c.parent_execution_id
		WHERE c.started_at >= ? AND c.started_at < ?
			AND c.parent_execution_id IS NOT NULL AND c.parent_execution_id <> ''
		GROUP BY p.agent_node_id, c.agent_node_id, c.reasoner_id`, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query call edges: %w", err)
	}
	type edgeKey struct{ caller, target string }
	edges := map[edgeKey]*types.DeveloperHomeEdge{}
	for erows.Next() {
		var (
			caller, target, reasoner string
			calls, failed, held      int64
		)
		if err := erows.Scan(&caller, &target, &reasoner, &calls, &failed, &held); err != nil {
			erows.Close()
			return fmt.Errorf("scan call edge: %w", err)
		}
		ek := edgeKey{caller, target}
		e := edges[ek]
		if e == nil {
			e = &types.DeveloperHomeEdge{Caller: caller, Target: target}
			edges[ek] = e
		}
		e.Calls += calls
		e.Failed += failed
		e.Held += held
		k := epKey{target, reasoner}
		if callerCalls[k] == nil {
			callerCalls[k] = map[string]int64{}
		}
		callerCalls[k][caller] += calls
	}
	erows.Close()
	if err := erows.Err(); err != nil {
		return err
	}

	// Calls that started a run, split into direct calls and trigger
	// dispatches. A trigger's run_id is the workflow id it recorded on the
	// inbound event.
	rrows, err := db.QueryContext(ctx, `SELECT c.agent_node_id, c.reasoner_id, COALESCE(ie.source_name, ''), COUNT(*),
			`+failedCase+`, `+heldCase+`
		FROM executions c
		LEFT JOIN inbound_events ie ON ie.dispatched_workflow_id = c.run_id
		WHERE c.started_at >= ? AND c.started_at < ?
			AND (c.parent_execution_id IS NULL OR c.parent_execution_id = '')
		GROUP BY c.agent_node_id, c.reasoner_id, COALESCE(ie.source_name, '')`, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query entry calls: %w", err)
	}
	type entryKey struct{ source, target string }
	entries := map[entryKey]*types.DeveloperHomeEntryEdge{}
	for rrows.Next() {
		var (
			agent, reasoner, source string
			calls, failed, held     int64
		)
		if err := rrows.Scan(&agent, &reasoner, &source, &calls, &failed, &held); err != nil {
			rrows.Close()
			return fmt.Errorf("scan entry call: %w", err)
		}
		if ep := endpoints[epKey{agent, reasoner}]; ep != nil {
			if source == "" {
				ep.DirectCalls += calls
			} else {
				ep.TriggerCalls += calls
			}
		}
		k := entryKey{source, agent}
		e := entries[k]
		if e == nil {
			e = &types.DeveloperHomeEntryEdge{Source: source, Target: agent}
			entries[k] = e
		}
		e.Calls += calls
		e.Failed += failed
		e.Held += held
	}
	rrows.Close()
	if err := rrows.Err(); err != nil {
		return err
	}

	for k, callers := range callerCalls {
		ep := endpoints[k]
		if ep == nil {
			continue
		}
		names := make([]string, 0, len(callers))
		for name := range callers {
			names = append(names, name)
		}
		sort.Slice(names, func(i, j int) bool {
			if callers[names[i]] != callers[names[j]] {
				return callers[names[i]] > callers[names[j]]
			}
			return names[i] < names[j]
		})
		ep.CallerAgents = names
	}

	out.Endpoints = make([]types.DeveloperHomeEndpoint, 0, len(order))
	for _, k := range order {
		out.Endpoints = append(out.Endpoints, *endpoints[k])
	}
	// Problems first (most failed calls), then the busiest.
	sort.SliceStable(out.Endpoints, func(i, j int) bool {
		a, b := out.Endpoints[i], out.Endpoints[j]
		if a.Failed != b.Failed {
			return a.Failed > b.Failed
		}
		if a.Calls != b.Calls {
			return a.Calls > b.Calls
		}
		return a.AgentNodeID+"."+a.ReasonerID < b.AgentNodeID+"."+b.ReasonerID
	})

	out.Edges = make([]types.DeveloperHomeEdge, 0, len(edges))
	for _, e := range edges {
		out.Edges = append(out.Edges, *e)
	}
	sort.Slice(out.Edges, func(i, j int) bool {
		if out.Edges[i].Caller != out.Edges[j].Caller {
			return out.Edges[i].Caller < out.Edges[j].Caller
		}
		return out.Edges[i].Target < out.Edges[j].Target
	})
	out.Entries = make([]types.DeveloperHomeEntryEdge, 0, len(entries))
	for _, e := range entries {
		out.Entries = append(out.Entries, *e)
	}
	sort.Slice(out.Entries, func(i, j int) bool {
		if out.Entries[i].Source != out.Entries[j].Source {
			return out.Entries[i].Source < out.Entries[j].Source
		}
		return out.Entries[i].Target < out.Entries[j].Target
	})
	return nil
}

func (ls *LocalStorage) homeApprovals(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	where := `FROM workflow_executions WHERE status = 'waiting' AND approval_status = 'pending'`
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) `+where).Scan(&out.Approvals.Total); err != nil {
		return fmt.Errorf("count pending approvals: %w", err)
	}
	out.Approvals.Items = []types.DeveloperHomeApproval{}
	if out.Approvals.Total == 0 {
		return nil
	}
	rows, err := db.QueryContext(ctx, `SELECT execution_id, COALESCE(run_id, workflow_id), agent_node_id, reasoner_id,
			COALESCE(approval_request_url, ''), approval_requested_at, approval_expires_at `+where+`
		ORDER BY approval_requested_at ASC LIMIT ?`, q.Limit)
	if err != nil {
		return fmt.Errorf("query pending approvals: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var (
			a                        types.DeveloperHomeApproval
			requestedRaw, expiresRaw interface{}
		)
		if err := rows.Scan(&a.ExecutionID, &a.RunID, &a.AgentNodeID, &a.ReasonerID, &a.RequestURL,
			&requestedRaw, &expiresRaw); err != nil {
			return fmt.Errorf("scan pending approval: %w", err)
		}
		if t, _ := parseDBTime(requestedRaw); !t.IsZero() {
			a.RequestedAt = &t
		}
		if t, _ := parseDBTime(expiresRaw); !t.IsZero() {
			a.ExpiresAt = &t
		}
		out.Approvals.Items = append(out.Approvals.Items, a)
	}
	return rows.Err()
}

func (ls *LocalStorage) homeSpend(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	query := `SELECT COUNT(*), SUM(cost_usd), COALESCE(SUM(total_tokens), 0)
		FROM execution_usage WHERE created_at >= ? AND created_at < ?`
	var (
		spend   types.DeveloperHomeSpend
		wCost   sql.NullFloat64
		bCost   sql.NullFloat64
		bTokens int64
	)
	if err := db.QueryRowContext(ctx, query, q.Start, q.End).Scan(&spend.WindowRows, &wCost, &spend.WindowTokens); err != nil {
		return fmt.Errorf("query window usage: %w", err)
	}
	if !q.BaselineStart.IsZero() && q.Start.After(q.BaselineStart) {
		if err := db.QueryRowContext(ctx, query, q.BaselineStart.UTC(), q.Start).Scan(&spend.BaselineRows, &bCost, &bTokens); err != nil {
			return fmt.Errorf("query baseline usage: %w", err)
		}
	}
	if spend.WindowRows == 0 && spend.BaselineRows == 0 {
		return nil
	}
	if wCost.Valid {
		v := wCost.Float64
		spend.WindowCostUSD = &v
	}
	if bCost.Valid {
		v := bCost.Float64
		spend.BaselineCostUSD = &v
	}
	out.Spend = &spend
	return nil
}

func (ls *LocalStorage) homeWebhooks(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	rows, err := db.QueryContext(ctx, `SELECT status, COUNT(*) FROM execution_webhooks
		WHERE created_at >= ? AND created_at < ? GROUP BY status`, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query result webhooks: %w", err)
	}
	var hooks types.DeveloperHomeWebhooks
	for rows.Next() {
		var (
			status string
			count  int64
		)
		if err := rows.Scan(&status, &count); err != nil {
			rows.Close()
			return fmt.Errorf("scan result webhooks: %w", err)
		}
		hooks.Registered += count
		switch status {
		case types.ExecutionWebhookStatusDelivered:
			hooks.Delivered += count
		case types.ExecutionWebhookStatusFailed:
			hooks.Failed += count
		default:
			hooks.Pending += count
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	if hooks.Registered == 0 {
		return nil
	}
	hooks.Recent = []types.DeveloperHomeWebhook{}
	if hooks.Failed > 0 {
		frows, err := db.QueryContext(ctx, `SELECT execution_id, url, attempt_count, COALESCE(last_error, ''), last_attempt_at
			FROM execution_webhooks
			WHERE status = 'failed' AND created_at >= ? AND created_at < ?
			ORDER BY updated_at DESC LIMIT 5`, q.Start, q.End)
		if err != nil {
			return fmt.Errorf("query failed webhooks: %w", err)
		}
		defer frows.Close()
		for frows.Next() {
			var (
				w       types.DeveloperHomeWebhook
				lastRaw interface{}
			)
			if err := frows.Scan(&w.ExecutionID, &w.URL, &w.AttemptCount, &w.LastError, &lastRaw); err != nil {
				return fmt.Errorf("scan failed webhook: %w", err)
			}
			if t, _ := parseDBTime(lastRaw); !t.IsZero() {
				w.LastAttemptAt = &t
			}
			hooks.Recent = append(hooks.Recent, w)
		}
		if err := frows.Err(); err != nil {
			return err
		}
	}
	out.Webhooks = &hooks
	return nil
}

func (ls *LocalStorage) homeTriggers(ctx context.Context, q types.DeveloperHomeQuery, out *types.DeveloperHomeData) error {
	db := ls.requireSQLDB()
	var trig types.DeveloperHomeTriggers
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*), COALESCE(SUM(CASE WHEN enabled IS TRUE THEN 1 ELSE 0 END), 0) FROM triggers`).
		Scan(&trig.Configured, &trig.Enabled); err != nil {
		return fmt.Errorf("count triggers: %w", err)
	}
	if trig.Configured == 0 {
		return nil
	}
	rows, err := db.QueryContext(ctx, `SELECT status, COUNT(*) FROM inbound_events
		WHERE received_at >= ? AND received_at < ? GROUP BY status`, q.Start, q.End)
	if err != nil {
		return fmt.Errorf("query inbound events: %w", err)
	}
	for rows.Next() {
		var (
			status string
			count  int64
		)
		if err := rows.Scan(&status, &count); err != nil {
			rows.Close()
			return fmt.Errorf("scan inbound events: %w", err)
		}
		trig.Events += count
		switch status {
		case types.InboundEventStatusDispatched:
			trig.Dispatched += count
		case types.InboundEventStatusFailed:
			trig.Failed += count
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM observability_dead_letter_queue WHERE kind = 'inbound_dispatch'`).
		Scan(&trig.DeadLetters); err != nil {
		return fmt.Errorf("count trigger dead letters: %w", err)
	}
	out.Triggers = &trig
	return nil
}

// CountForwarderDeadLetters returns how many observability events the
// forwarder could not deliver and parked in its dead-letter queue.
func (ls *LocalStorage) CountForwarderDeadLetters(ctx context.Context) (int64, error) {
	var n int64
	if err := ls.requireSQLDB().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM observability_dead_letter_queue WHERE kind = 'observability'`).Scan(&n); err != nil {
		return 0, fmt.Errorf("count forwarder dead letters: %w", err)
	}
	return n, nil
}

// QueryBlastRadius returns, for every run that had an execution on the agent
// in [start, end), the run's entry point (its root execution's agent and
// reasoner) and how many such runs there were.
func (ls *LocalStorage) QueryBlastRadius(ctx context.Context, agentNodeID string, start, end time.Time) ([]types.DeveloperHomeEntryPoint, error) {
	db := ls.requireSQLDB()
	rows, err := db.QueryContext(ctx, `SELECT r.agent_node_id, r.reasoner_id, COUNT(DISTINCT e.run_id),
			COUNT(DISTINCT CASE WHEN e.status IN `+homeFailedStatuses+` THEN e.run_id END)
		FROM executions e
		JOIN executions r ON r.run_id = e.run_id
			AND (r.parent_execution_id IS NULL OR r.parent_execution_id = '')
		WHERE e.agent_node_id = ? AND e.started_at >= ? AND e.started_at < ?
		GROUP BY r.agent_node_id, r.reasoner_id
		ORDER BY COUNT(DISTINCT e.run_id) DESC`, agentNodeID, start.UTC(), end.UTC())
	if err != nil {
		return nil, fmt.Errorf("query blast radius: %w", err)
	}
	defer rows.Close()
	out := []types.DeveloperHomeEntryPoint{}
	for rows.Next() {
		var ep types.DeveloperHomeEntryPoint
		if err := rows.Scan(&ep.AgentNodeID, &ep.ReasonerID, &ep.Runs, &ep.FailedRuns); err != nil {
			return nil, fmt.Errorf("scan blast radius: %w", err)
		}
		out = append(out, ep)
	}
	return out, rows.Err()
}

var (
	homeIDPattern   = regexp.MustCompile(`\b(exec|run|wf|req|evt|ev|sess|vc|apr)_[A-Za-z0-9_]{6,}`)
	homeUUIDPattern = regexp.MustCompile(`(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b`)
	homeHexPattern  = regexp.MustCompile(`(?i)\b[0-9a-f]{16,}\b`)
	homeTimePattern = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?`)
)

// NormalizeErrorMessage turns one execution's error text into the key its
// cause is grouped by: identifiers, timestamps and stand-alone numbers are
// replaced so that the same failure with different ids or amounts lands in
// one group. It never tries to read agent names out of the text.
func NormalizeErrorMessage(msg string) string {
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return ""
	}
	msg = homeTimePattern.ReplaceAllString(msg, "<time>")
	msg = homeUUIDPattern.ReplaceAllString(msg, "<id>")
	msg = homeIDPattern.ReplaceAllString(msg, "<id>")
	msg = homeHexPattern.ReplaceAllString(msg, "<id>")

	fields := strings.Fields(msg)
	for i, f := range fields {
		core := strings.TrimRightFunc(f, func(r rune) bool { return strings.ContainsRune(".,;:)]}\"'%", r) })
		lead := strings.TrimLeftFunc(core, func(r rune) bool { return strings.ContainsRune("([{\"'$#=", r) })
		if lead != "" && isPlainNumber(lead) {
			prefix := core[:len(core)-len(lead)]
			fields[i] = prefix + "N" + f[len(core):]
		}
	}
	msg = strings.Join(fields, " ")
	const maxLen = 240
	if r := []rune(msg); len(r) > maxLen {
		msg = string(r[:maxLen]) + "…"
	}
	return msg
}

func isPlainNumber(s string) bool {
	digits := 0
	dots := 0
	for _, r := range s {
		switch {
		case unicode.IsDigit(r):
			digits++
		case r == ',':
		case r == '.':
			dots++
		default:
			return false
		}
	}
	return digits > 0 && dots <= 1
}
