package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// referenceRunSummary is what the single grouped query QueryRunSummaries used
// before it split into a page query and a per-page aggregation. The reference
// below is that query verbatim, with run_id added as the tie-breaker so its
// order is defined, and it is what every page the two-phase path returns is
// compared against.
type referenceRunSummary struct {
	RunID           string
	EarliestStarted time.Time
	LatestActivity  time.Time
	Total           int
	Counts          map[string]int
	Active          int
	RootExecutionID string
	RootStatus      string
	RootErrorMsg    string
	RootAgent       string
	RootReasoner    string
	SessionID       string
	ActorID         string
}

func referenceRunSummaries(t *testing.T, ls *LocalStorage, ctx context.Context, filter types.ExecutionFilter) ([]referenceRunSummary, int) {
	t.Helper()
	var (
		where []string
		args  []interface{}
	)
	if filter.RunID != nil {
		where = append(where, "run_id = ?")
		args = append(args, *filter.RunID)
	}
	if filter.AgentNodeID != nil {
		where = append(where, "run_id IN (SELECT run_id FROM executions WHERE agent_node_id = ?)")
		args = append(args, *filter.AgentNodeID)
	}
	if filter.Status != nil {
		where = append(where, "status = ?")
		args = append(args, *filter.Status)
	}
	if filter.SessionID != nil {
		where = append(where, "run_id IN (SELECT run_id FROM executions WHERE session_id = ?)")
		args = append(args, *filter.SessionID)
	}
	if filter.ActorID != nil {
		where = append(where, "actor_id = ?")
		args = append(args, *filter.ActorID)
	}
	if filter.StartTime != nil {
		where = append(where, "started_at >= ?")
		args = append(args, filter.StartTime.UTC())
	}
	if filter.Search != nil {
		term := "%" + *filter.Search + "%"
		where = append(where, "(run_id LIKE ? OR agent_node_id LIKE ? OR reasoner_id LIKE ? OR "+runMetadataSearchPredicate(ls.mode)+")")
		args = append(args, term, term, term, term, term)
	}
	whereClause := ""
	if len(where) > 0 {
		whereClause = " WHERE " + strings.Join(where, " AND ")
	}
	having := ""
	if filter.ActiveOnly {
		having = " HAVING SUM(CASE WHEN LOWER(status) IN ('running','pending','queued','waiting','paused','unknown') THEN 1 ELSE 0 END) > 0"
	}
	dir := "DESC"
	if !filter.SortDescending {
		dir = "ASC"
	}
	limit := filter.Limit
	if limit <= 0 {
		limit = 20
	}
	db := ls.requireSQLDB()

	countQuery := "SELECT COUNT(DISTINCT run_id) FROM executions" + whereClause
	if filter.ActiveOnly {
		countQuery = "SELECT COUNT(*) FROM (SELECT run_id FROM executions" + whereClause + " GROUP BY run_id" + having + ") active_runs"
	}
	var total int
	require.NoError(t, db.QueryRowContext(ctx, countQuery, args...).Scan(&total))

	query := fmt.Sprintf(`
		SELECT
			run_id,
			MIN(started_at) AS earliest_started,
			MAX(COALESCE(updated_at, started_at)) AS latest_activity,
			COUNT(*) AS total_executions,
			SUM(CASE WHEN LOWER(status) = 'succeeded' THEN 1 ELSE 0 END) AS succeeded_count,
			SUM(CASE WHEN LOWER(status) = 'failed' THEN 1 ELSE 0 END) AS failed_count,
			SUM(CASE WHEN LOWER(status) = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count,
			SUM(CASE WHEN LOWER(status) = 'timeout' THEN 1 ELSE 0 END) AS timeout_count,
			SUM(CASE WHEN LOWER(status) = 'running' THEN 1 ELSE 0 END) AS running_count,
			SUM(CASE WHEN LOWER(status) = 'paused' THEN 1 ELSE 0 END) AS paused_count,
			SUM(CASE WHEN LOWER(status) = 'pending' THEN 1 ELSE 0 END) AS pending_count,
			SUM(CASE WHEN LOWER(status) = 'queued' THEN 1 ELSE 0 END) AS queued_count,
			SUM(CASE WHEN LOWER(status) = 'waiting' THEN 1 ELSE 0 END) AS waiting_count,
			SUM(CASE WHEN LOWER(status) IN ('running','pending','queued','waiting') THEN 1 ELSE 0 END) AS active_executions,
			MAX(CASE WHEN parent_execution_id IS NULL OR parent_execution_id = '' THEN execution_id END) AS root_execution_id,
			MAX(CASE WHEN parent_execution_id IS NULL OR parent_execution_id = '' THEN status END) AS root_status,
			MAX(CASE WHEN parent_execution_id IS NULL OR parent_execution_id = '' THEN error_message END) AS root_error_message,
			MAX(CASE WHEN parent_execution_id IS NULL OR parent_execution_id = '' THEN agent_node_id END) AS root_agent_node_id,
			MAX(CASE WHEN parent_execution_id IS NULL OR parent_execution_id = '' THEN reasoner_id END) AS root_reasoner_id,
			MAX(session_id) AS session_id,
			MAX(actor_id) AS actor_id,
			CASE
				WHEN SUM(CASE WHEN LOWER(status) IN ('failed','cancelled','timeout') THEN 1 ELSE 0 END) > 0 THEN 2
				WHEN SUM(CASE WHEN LOWER(status) IN ('running','pending','queued','waiting') THEN 1 ELSE 0 END) > 0 THEN 1
				ELSE 0
			END AS status_rank
		FROM executions
		%s
		GROUP BY run_id%s
		ORDER BY %s %s, run_id %s
		LIMIT %d OFFSET %d`,
		whereClause, having, mapRunSummarySortColumn(filter.SortBy), dir, dir, limit, filter.Offset)

	rows, err := db.QueryContext(ctx, query, args...)
	require.NoError(t, err)
	defer rows.Close()

	var out []referenceRunSummary
	for rows.Next() {
		var (
			r                                                       referenceRunSummary
			earliest, latest                                        interface{}
			succ, failed, canc, tout, run, paused, pend, que, wait  int
			rootID, rootStatus, rootErr, rootAgent, rootRsn, ss, ac sql.NullString
			rank                                                    int
		)
		require.NoError(t, rows.Scan(&r.RunID, &earliest, &latest, &r.Total,
			&succ, &failed, &canc, &tout, &run, &paused, &pend, &que, &wait, &r.Active,
			&rootID, &rootStatus, &rootErr, &rootAgent, &rootRsn, &ss, &ac, &rank))
		require.NoError(t, assignTimeValue(&r.EarliestStarted, earliest))
		require.NoError(t, assignTimeValue(&r.LatestActivity, latest))
		r.Counts = map[string]int{
			string(types.ExecutionStatusSucceeded): succ,
			string(types.ExecutionStatusFailed):    failed,
			string(types.ExecutionStatusCancelled): canc,
			string(types.ExecutionStatusTimeout):   tout,
			string(types.ExecutionStatusRunning):   run,
			string(types.ExecutionStatusPaused):    paused,
			string(types.ExecutionStatusPending):   pend,
			string(types.ExecutionStatusQueued):    que,
			string(types.ExecutionStatusWaiting):   wait,
		}
		r.RootExecutionID = rootID.String
		if rootStatus.String != "" {
			r.RootStatus = types.NormalizeExecutionStatus(rootStatus.String)
		}
		r.RootErrorMsg, r.RootAgent, r.RootReasoner = rootErr.String, rootAgent.String, rootRsn.String
		r.SessionID, r.ActorID = ss.String, ac.String
		out = append(out, r)
	}
	require.NoError(t, rows.Err())
	return out, total
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// seedRunSummaryFixture writes runs that exercise every sort key: ties on
// latest activity (two runs share one timestamp, so only run_id orders them),
// ties on status rank and step count, multiple root rows, mixed-case statuses,
// rows with and without session and actor, and run metadata for search. Each
// execution's updated_at is set explicitly after insert, because
// CreateExecutionRecord stamps it with the wall clock.
func seedRunSummaryFixture(t *testing.T, ls *LocalStorage, ctx context.Context, prefix string) {
	t.Helper()
	base := time.Date(2025, 3, 4, 10, 0, 0, 0, time.UTC)
	statuses := []string{"succeeded", "failed", "running", "queued", "waiting", "paused", "cancelled", "timeout", "pending", "SUCCEEDED"}
	db := ls.requireSQLDB()

	for r := 0; r < 37; r++ {
		runID := fmt.Sprintf("%srun-%02d", prefix, r)
		steps := 1 + (r*7)%9
		rootID := fmt.Sprintf("%s-root", runID)
		var session, actor *string
		if r%3 == 0 {
			s := fmt.Sprintf("session-%d", r%2)
			session = &s
		}
		if r%4 == 0 {
			a := fmt.Sprintf("actor-%d", r%3)
			actor = &a
		}
		for s := 0; s < steps; s++ {
			execID := fmt.Sprintf("%s-step-%02d", runID, s)
			var parent *string
			if s == 0 {
				execID = rootID
			} else if s%4 != 0 {
				p := rootID
				parent = &p
			} else if r%5 == 0 {
				// A second parentless row in some runs: the root columns are
				// each MAX over parentless rows, independently.
				parent = nil
			} else {
				p := fmt.Sprintf("%s-step-%02d", runID, s-1)
				parent = &p
			}
			status := statuses[(r+s*3)%len(statuses)]
			if r%6 == 0 {
				status = "succeeded"
			}
			var errMsg *string
			if status == "failed" {
				m := fmt.Sprintf("boom %d", r)
				errMsg = &m
			}
			started := base.Add(time.Duration(r*13%29)*time.Minute + time.Duration(s)*time.Second)
			agent := fmt.Sprintf("agent-%d", (r+s)%4)
			require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
				ExecutionID:       execID,
				RunID:             runID,
				ParentExecutionID: parent,
				AgentNodeID:       agent,
				ReasonerID:        fmt.Sprintf("reasoner.%d", (r+s)%5),
				NodeID:            agent,
				Status:            status,
				ErrorMessage:      errMsg,
				SessionID:         session,
				ActorID:           actor,
				StartedAt:         started,
			}))
			// Runs 10 and 11 end on the same instant, and so do 20..22, so
			// only the run_id tie-breaker orders them.
			updated := started.Add(time.Duration(s+1) * time.Second)
			switch r {
			case 10, 11:
				updated = base.Add(5 * time.Hour)
			case 20, 21, 22:
				updated = base.Add(6 * time.Hour)
			}
			_, err := db.ExecContext(ctx, "UPDATE executions SET updated_at = ? WHERE execution_id = ?", updated, execID)
			require.NoError(t, err)
		}
	}
	require.NoError(t, ls.UpdateWorkflowRunMetadata(ctx, prefix+"run-07", func(ns map[string]json.RawMessage) error {
		ns[types.RunMetadataNamespace] = json.RawMessage(`{"display_name":"Quarterly Close","labels":["finance"]}`)
		return nil
	}))
}

func TestQueryRunSummariesPagesMatchTheSingleGroupedQuery(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	seedRunSummaryFixture(t, ls, ctx, "")
	compareRunSummaryPages(t, ls, ctx, "")
}

// The same comparison on PostgreSQL, where the activity index is an
// expression index and ORDER BY ties fall to the database's collation.
func TestPostgresQueryRunSummariesPagesMatchTheSingleGroupedQuery(t *testing.T) {
	postgresURL := os.Getenv("POSTGRES_TEST_URL")
	if postgresURL == "" {
		t.Skip("POSTGRES_TEST_URL not set, skipping postgres tests")
	}
	ctx := context.Background()
	ls := NewPostgresStorage(PostgresStorageConfig{})
	require.NoError(t, ls.Initialize(ctx, StorageConfig{Mode: "postgres", Postgres: PostgresStorageConfig{DSN: postgresURL}}))
	defer ls.Close(ctx)

	prefix := fmt.Sprintf("pg%d-", time.Now().UnixNano())
	seedRunSummaryFixture(t, ls, ctx, prefix)
	compareRunSummaryPages(t, ls, ctx, prefix)

	var indexes int
	require.NoError(t, ls.requireSQLDB().QueryRowContext(ctx,
		"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_executions_run_activity'").Scan(&indexes))
	require.Equal(t, 1, indexes)
}

func compareRunSummaryPages(t *testing.T, ls *LocalStorage, ctx context.Context, prefix string) {
	t.Helper()
	str := func(s string) *string { return &s }
	since := time.Date(2025, 3, 4, 10, 10, 0, 0, time.UTC)
	filters := map[string]types.ExecutionFilter{
		"none":       {},
		"run":        {RunID: str(prefix + "run-07")},
		"agent":      {AgentNodeID: str("agent-2")},
		"status":     {Status: str("running")},
		"session":    {SessionID: str("session-1")},
		"actor":      {ActorID: str("actor-0")},
		"since":      {StartTime: &since},
		"search":     {Search: str("quarterly")},
		"search-id":  {Search: str(prefix + "run-1")},
		"activeOnly": {ActiveOnly: true},
	}
	if ls.mode == "postgres" {
		// Run search reads workflow_runs.metadata as JSONB, and a schema
		// created by this storage layer alone types it bytea; search is not
		// what this comparison is about, so it runs on SQLite only.
		delete(filters, "search")
		delete(filters, "search-id")
	}
	sorts := []string{"updated_at", "started_at", "status", "total_steps", "failed_steps", "active_executions"}
	pages := []struct{ limit, offset int }{{25, 0}, {5, 0}, {5, 5}, {7, 30}, {10, 40}}

	for name, base := range filters {
		for _, sortBy := range sorts {
			for _, desc := range []bool{true, false} {
				for _, pg := range pages {
					f := base
					f.SortBy = sortBy
					f.SortDescending = desc
					f.Limit = pg.limit
					f.Offset = pg.offset
					label := fmt.Sprintf("%s/%s/desc=%v/%d+%d", name, sortBy, desc, pg.offset, pg.limit)

					want, wantTotal := referenceRunSummaries(t, ls, ctx, f)
					got, gotTotal, err := ls.QueryRunSummaries(ctx, f)
					require.NoError(t, err, label)
					require.Equal(t, wantTotal, gotTotal, label)
					require.Len(t, got, len(want), label)
					for i := range want {
						w, g := want[i], got[i]
						require.Equal(t, w.RunID, g.RunID, label)
						require.Equal(t, w.Total, g.TotalExecutions, label)
						require.Equal(t, w.Counts, g.StatusCounts, label)
						require.Equal(t, w.Active, g.ActiveExecutions, label)
						require.True(t, w.EarliestStarted.Equal(g.EarliestStarted), label)
						require.True(t, w.LatestActivity.Equal(g.LatestStarted), label)
						require.Equal(t, w.RootExecutionID, deref(g.RootExecutionID), label)
						require.Equal(t, w.RootStatus, deref(g.RootStatus), label)
						require.Equal(t, w.RootErrorMsg, deref(g.RootErrorMessage), label)
						require.Equal(t, w.RootAgent, deref(g.RootAgentNodeID), label)
						require.Equal(t, w.RootReasoner, deref(g.RootReasonerID), label)
						require.Equal(t, w.SessionID, deref(g.SessionID), label)
						require.Equal(t, w.ActorID, deref(g.ActorID), label)
					}
				}
			}
		}
	}
}

// The unfiltered newest-first page is read from the activity index and widens
// its window when the newest rows all belong to a few runs. A tiny first
// window forces several widenings; the page must still be the grouped query's.
func TestRecentRunIDsByActivityWidensPastBusyRuns(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	seedRunSummaryFixture(t, ls, ctx, "")

	for _, pg := range []struct{ limit, offset int }{{1, 0}, {3, 2}, {25, 0}, {10, 30}, {5, 36}, {5, 37}, {5, 100}} {
		f := types.ExecutionFilter{SortBy: "updated_at", SortDescending: true, Limit: pg.limit, Offset: pg.offset}
		want, _ := referenceRunSummaries(t, ls, ctx, f)
		got, err := ls.recentRunIDsByActivity(ctx, pg.offset, pg.limit)
		require.NoError(t, err)
		wantIDs := make([]string, 0, len(want))
		for _, w := range want {
			wantIDs = append(wantIDs, w.RunID)
		}
		require.Equal(t, len(wantIDs), len(got), "offset %d limit %d", pg.offset, pg.limit)
		if len(wantIDs) > 0 {
			require.Equal(t, wantIDs, got, "offset %d limit %d", pg.offset, pg.limit)
		}
	}

	// Window smaller than one run's rows: the scan must widen, not stop.
	runIDs, rowsRead, err := scanDistinctRecentRunIDs(ctx, ls.requireSQLDB(), 10, 3)
	require.NoError(t, err)
	require.Equal(t, 3, rowsRead)
	require.Less(t, len(runIDs), 10)
}

func TestRunActivityIndexServesTheRecentRunsScan(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	rows, err := ls.requireSQLDB().QueryContext(ctx, `EXPLAIN QUERY PLAN
		SELECT run_id FROM executions
		ORDER BY COALESCE(updated_at, started_at) DESC, run_id DESC
		LIMIT 10`)
	require.NoError(t, err)
	defer rows.Close()
	var plan strings.Builder
	for rows.Next() {
		var id, parent, notused int
		var detail string
		require.NoError(t, rows.Scan(&id, &parent, &notused, &detail))
		plan.WriteString(detail + "\n")
	}
	require.NoError(t, rows.Err())
	require.Contains(t, plan.String(), "idx_executions_run_activity")
	require.NotContains(t, plan.String(), "TEMP B-TREE")
}
