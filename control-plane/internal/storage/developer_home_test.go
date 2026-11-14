package storage

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// homeStorages returns the storages the Home aggregates are checked on:
// always SQLite, and PostgreSQL in a fresh throwaway database when
// POSTGRES_TEST_URL is set.
func homeStorages(t *testing.T) map[string]*LocalStorage {
	t.Helper()
	out := map[string]*LocalStorage{}
	ls, _ := setupLocalStorage(t)
	out["sqlite"] = ls

	if raw := os.Getenv("POSTGRES_TEST_URL"); raw != "" {
		u, err := url.Parse(raw)
		require.NoError(t, err)
		u.Path = "/" + fmt.Sprintf("af_home_%d", time.Now().UnixNano())
		ctx := context.Background()
		pg := NewPostgresStorage(PostgresStorageConfig{})
		err = pg.Initialize(ctx, StorageConfig{Mode: "postgres", Postgres: PostgresStorageConfig{DSN: u.String()}})
		if err != nil && strings.Contains(err.Error(), "connection refused") {
			t.Logf("postgres not reachable, checking SQLite only: %v", err)
			return out
		}
		require.NoError(t, err)
		t.Cleanup(func() { _ = pg.Close(ctx) })
		out["postgres"] = pg
	}
	return out
}

type homeSeed struct {
	ls  *LocalStorage
	ctx context.Context
	t   *testing.T
}

func (s homeSeed) exec(id, run, parent, agent, reasoner, status string, started time.Time, durationMS int64, errMsg string) {
	s.t.Helper()
	e := &types.Execution{
		ExecutionID: id,
		RunID:       run,
		AgentNodeID: agent,
		ReasonerID:  reasoner,
		NodeID:      agent,
		Status:      status,
		StartedAt:   started,
	}
	if parent != "" {
		p := parent
		e.ParentExecutionID = &p
	}
	if types.IsTerminalExecutionStatus(status) {
		d := durationMS
		e.DurationMS = &d
		done := started.Add(time.Duration(durationMS) * time.Millisecond)
		e.CompletedAt = &done
	}
	if errMsg != "" {
		m := errMsg
		e.ErrorMessage = &m
	}
	require.NoError(s.t, s.ls.CreateExecutionRecord(s.ctx, e))
}

// seedHomeMesh records a small mesh: an orchestrator calling three agents,
// one of which fails with the same cause twice (different ids in the text),
// a call held for approval, a triggered run, and older traffic for the
// baseline.
func seedHomeMesh(t *testing.T, ls *LocalStorage, now time.Time) {
	ctx := context.Background()
	s := homeSeed{ls: ls, ctx: ctx, t: t}
	w := now.Add(-30 * time.Minute)

	// Run 1: direct call to orch.process_order.
	s.exec("e-r1", "run-1", "", "orch", "process_order", "failed", w, 900, "child failed")
	s.exec("e-r1-fraud", "run-1", "e-r1", "fraud", "score", "succeeded", w.Add(time.Second), 100, "")
	s.exec("e-r1-price", "run-1", "e-r1", "pricing", "quote", "failed", w.Add(2*time.Second), 200, "server unavailable for exec_20261109_aaaaaaaa after 3 attempts")
	s.exec("e-r1-pay", "run-1", "e-r1", "payments", "pay", "waiting", w.Add(3*time.Second), 0, "")

	// Run 2: the same pricing failure with a different id and count.
	s.exec("e-r2", "run-2", "", "orch", "process_order", "succeeded", w.Add(time.Minute), 400, "")
	s.exec("e-r2-price", "run-2", "e-r2", "pricing", "quote", "failed", w.Add(time.Minute+time.Second), 300, "server unavailable for exec_20261109_bbbbbbbb after 12 attempts")
	s.exec("e-r2-fraud", "run-2", "e-r2", "fraud", "score", "timeout", w.Add(time.Minute+2*time.Second), 5000, "")

	// Run 3: dispatched by a trigger.
	s.exec("e-r3", "run-3", "", "intake", "process_invoice", "succeeded", w.Add(2*time.Minute), 50, "")
	s.exec("e-run-now", "run-4", "", "intake", "process_invoice", "running", w.Add(3*time.Minute), 0, "")

	// Outside the window: baseline traffic a day earlier.
	for i := 0; i < 4; i++ {
		status := "succeeded"
		if i == 0 {
			status = "failed"
		}
		s.exec(fmt.Sprintf("e-old-%d", i), fmt.Sprintf("run-old-%d", i), "", "orch", "process_order", status,
			now.Add(-24*time.Hour+time.Duration(i)*time.Minute), 1000, "")
	}

	trig := &types.Trigger{ID: "trg-1", SourceName: "stripe", TargetNodeID: "intake", TargetReasoner: "process_invoice", Enabled: true, ManagedBy: types.ManagedByUI}
	require.NoError(t, ls.CreateTrigger(ctx, trig))
	require.NoError(t, ls.InsertInboundEvent(ctx, &types.InboundEvent{
		ID: "ev-1", TriggerID: "trg-1", SourceName: "stripe", EventType: "invoice.paid",
		Status: types.InboundEventStatusDispatched, ReceivedAt: w.Add(2 * time.Minute), DispatchedWorkflowID: "run-3",
	}))
	require.NoError(t, ls.InsertInboundEvent(ctx, &types.InboundEvent{
		ID: "ev-2", TriggerID: "trg-1", SourceName: "stripe", EventType: "invoice.paid", IdempotencyKey: "k2",
		Status: types.InboundEventStatusFailed, ReceivedAt: w.Add(4 * time.Minute),
	}))
	_, err := ls.requireSQLDB().ExecContext(ctx, `INSERT INTO observability_dead_letter_queue
		(kind, event_type, event_source, event_timestamp, payload, error_message, retry_count, created_at)
		VALUES ('inbound_dispatch', 'invoice.paid', 'stripe', ?, '{}', 'agent returned 500', 0, ?)`, w, w)
	require.NoError(t, err)

	// Approval pending on the held payment.
	runID := "run-1"
	approvalStatus := "pending"
	requested := w.Add(4 * time.Second)
	expires := requested.Add(72 * time.Hour)
	reqID := "apr-1"
	reqURL := "https://approvals.example/apr-1"
	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID: "run-1", ExecutionID: "e-r1-pay", AgentFieldRequestID: "req-1", RunID: &runID,
		AgentNodeID: "payments", ReasonerID: "pay", Status: string(types.ExecutionStatusWaiting),
		StartedAt: w.Add(3 * time.Second), ApprovalRequestID: &reqID, ApprovalRequestURL: &reqURL,
		ApprovalStatus: &approvalStatus, ApprovalRequestedAt: &requested, ApprovalExpiresAt: &expires,
	}))

	// A result webhook that ran out of attempts.
	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{ExecutionID: "e-r2", URL: "https://hooks.example/done"}))
	lastErr := "503 Service Unavailable"
	last := w.Add(2 * time.Minute)
	require.NoError(t, ls.UpdateExecutionWebhookState(ctx, "e-r2", types.ExecutionWebhookStateUpdate{
		Status: types.ExecutionWebhookStatusFailed, AttemptCount: 3, LastAttemptAt: &last, LastError: &lastErr,
	}))

	cost := 0.25
	require.NoError(t, ls.CreateExecutionUsage(ctx, []*types.ExecutionUsage{{
		ExecutionID: "e-r1", WorkflowID: "run-1", AgentNodeID: "orch", Reasoner: "process_order",
		Model: "m", TotalTokens: 1200, CostUSD: &cost, CreatedAt: w.Add(time.Second),
	}}))
}

func TestQueryDeveloperHome(t *testing.T) {
	for name, ls := range homeStorages(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			now := time.Now().UTC().Truncate(time.Second)
			seedHomeMesh(t, ls, now)

			data, err := ls.QueryDeveloperHome(ctx, types.DeveloperHomeQuery{
				Start: now.Add(-time.Hour), End: now.Add(time.Minute), BucketSeconds: 300,
				BaselineStart: now.Add(-7 * 24 * time.Hour), Limit: 10,
			})
			require.NoError(t, err)

			// Window totals: 9 executions, 3 failed or timed out, 7 finished.
			require.EqualValues(t, 9, data.Totals.Counts.Total)
			require.EqualValues(t, 4, data.Totals.Counts.Failed) // r1 failed, two pricing failures, the fraud timeout
			require.EqualValues(t, 3, data.Totals.Counts.Succeeded)
			require.EqualValues(t, 7, data.Totals.Counts.Terminal)
			require.EqualValues(t, 1, data.Totals.Held)
			require.EqualValues(t, 1, data.Totals.RunningNow)
			// Finished durations sorted: 50 100 200 300 400 900 5000 → p50 rank 4, p95 rank 7.
			require.NotNil(t, data.Totals.P50Ms)
			require.EqualValues(t, 300, *data.Totals.P50Ms)
			require.EqualValues(t, 5000, *data.Totals.P95Ms)

			// Series: 13 five-minute buckets, every call lands in one of them.
			require.Len(t, data.Series, 13)
			var seriesCalls int64
			for _, b := range data.Series {
				seriesCalls += b.Calls
			}
			require.EqualValues(t, 9, seriesCalls)

			// Baseline: the four calls a day earlier, spanning from the first one.
			require.NotNil(t, data.Baseline)
			require.EqualValues(t, 4, data.Baseline.Counts.Total)
			require.EqualValues(t, 1, data.Baseline.Counts.Failed)
			require.EqualValues(t, 4000, data.Baseline.BusyMs)
			require.InDelta(t, (23 * time.Hour).Seconds(), data.Baseline.Span, 1)
			require.EqualValues(t, 1000, *data.Baseline.P95Ms)

			// Problems: the two pricing failures are one cause.
			var pricing *types.DeveloperHomeProblem
			for i := range data.Problems {
				if data.Problems[i].AgentNodeID == "pricing" {
					pricing = &data.Problems[i]
				}
			}
			require.NotNil(t, pricing)
			require.EqualValues(t, 2, pricing.Count)
			require.Equal(t, "server unavailable for <id> after N attempts", pricing.Message)
			require.Equal(t, "e-r2-price", pricing.ExampleExecutionID)
			require.Equal(t, "run-2", pricing.ExampleRunID)
			require.Equal(t, pricing.Count, sumInt64s(pricing.Buckets))
			require.Equal(t, "pricing", data.Problems[0].AgentNodeID, "largest cause first")
			require.Len(t, data.Problems, 3)

			// Edges: orch → fraud (2 calls, 1 timeout), orch → pricing (2, 2 failed), orch → payments (1 held).
			edges := map[string]types.DeveloperHomeEdge{}
			for _, e := range data.Edges {
				edges[e.Caller+">"+e.Target] = e
			}
			require.Len(t, edges, 3)
			require.Equal(t, types.DeveloperHomeEdge{Caller: "orch", Target: "fraud", Calls: 2, Failed: 1}, edges["orch>fraud"])
			require.Equal(t, types.DeveloperHomeEdge{Caller: "orch", Target: "pricing", Calls: 2, Failed: 2}, edges["orch>pricing"])
			require.Equal(t, types.DeveloperHomeEdge{Caller: "orch", Target: "payments", Calls: 1, Held: 1}, edges["orch>payments"])

			entries := map[string]types.DeveloperHomeEntryEdge{}
			for _, e := range data.Entries {
				entries[e.Source+">"+e.Target] = e
			}
			require.Equal(t, types.DeveloperHomeEntryEdge{Source: "", Target: "orch", Calls: 2, Failed: 1}, entries[">orch"])
			require.Equal(t, types.DeveloperHomeEntryEdge{Source: "stripe", Target: "intake", Calls: 1}, entries["stripe>intake"])
			require.Equal(t, types.DeveloperHomeEntryEdge{Source: "", Target: "intake", Calls: 1}, entries[">intake"])

			// Endpoints: problems first.
			require.Equal(t, "pricing", data.Endpoints[0].AgentNodeID)
			eps := map[string]types.DeveloperHomeEndpoint{}
			for _, e := range data.Endpoints {
				eps[e.AgentNodeID+"."+e.ReasonerID] = e
			}
			require.Equal(t, []string{"orch"}, eps["pricing.quote"].CallerAgents)
			require.EqualValues(t, 300, *eps["pricing.quote"].P95Ms)
			require.EqualValues(t, 200, *eps["pricing.quote"].P50Ms)
			require.EqualValues(t, 2, eps["orch.process_order"].DirectCalls)
			require.EqualValues(t, 1, eps["intake.process_invoice"].TriggerCalls)
			require.EqualValues(t, 1, eps["intake.process_invoice"].DirectCalls)
			require.Empty(t, eps["orch.process_order"].CallerAgents)

			// Waiting on people.
			require.EqualValues(t, 1, data.Approvals.Total)
			require.Equal(t, "e-r1-pay", data.Approvals.Items[0].ExecutionID)
			require.Equal(t, "run-1", data.Approvals.Items[0].RunID)
			require.Equal(t, "https://approvals.example/apr-1", data.Approvals.Items[0].RequestURL)
			require.NotNil(t, data.Approvals.Items[0].ExpiresAt)

			// In and out.
			require.NotNil(t, data.Triggers)
			require.EqualValues(t, 1, data.Triggers.Configured)
			require.EqualValues(t, 2, data.Triggers.Events)
			require.EqualValues(t, 1, data.Triggers.Failed)
			require.EqualValues(t, 1, data.Triggers.DeadLetters)
			require.NotNil(t, data.Webhooks)
			require.EqualValues(t, 1, data.Webhooks.Failed)
			require.Equal(t, "https://hooks.example/done", data.Webhooks.Recent[0].URL)

			require.NotNil(t, data.Spend)
			require.EqualValues(t, 1, data.Spend.WindowRows)
			require.InDelta(t, 0.25, *data.Spend.WindowCostUSD, 1e-9)

			// Blast radius of fraud: both order runs passed through it, one failed there.
			radius, err := ls.QueryBlastRadius(ctx, "fraud", now.Add(-time.Hour), now.Add(time.Minute))
			require.NoError(t, err)
			require.Equal(t, []types.DeveloperHomeEntryPoint{{AgentNodeID: "orch", ReasonerID: "process_order", Runs: 2, FailedRuns: 1}}, radius)

			failures, err := ls.QueryRecentFailures(ctx, now.Add(-time.Hour), now.Add(time.Minute), 2)
			require.NoError(t, err)
			require.Len(t, failures, 2)
			require.Equal(t, "e-r2-fraud", failures[0].ExecutionID)

			dlq, err := ls.CountForwarderDeadLetters(ctx)
			require.NoError(t, err)
			require.EqualValues(t, 0, dlq)
		})
	}
}

func TestQueryDeveloperHomeEmptyInstall(t *testing.T) {
	for name, ls := range homeStorages(t) {
		t.Run(name, func(t *testing.T) {
			now := time.Now().UTC()
			data, err := ls.QueryDeveloperHome(context.Background(), types.DeveloperHomeQuery{
				Start: now.Add(-time.Hour), End: now, BaselineStart: now.Add(-7 * 24 * time.Hour),
			})
			require.NoError(t, err)
			require.Zero(t, data.Totals.Counts.Total)
			require.Nil(t, data.Totals.P95Ms, "no finished calls means no p95, not zero")
			require.Nil(t, data.Baseline)
			require.Nil(t, data.Spend)
			require.Nil(t, data.Webhooks)
			require.Nil(t, data.Triggers)
			require.Empty(t, data.Problems)
			require.Len(t, data.Series, 12)

			_, err = ls.QueryDeveloperHome(context.Background(), types.DeveloperHomeQuery{Start: now, End: now})
			require.Error(t, err)
		})
	}
}

func TestNormalizeErrorMessage(t *testing.T) {
	cases := map[string]string{
		"": "",
		"agent call failed: Post \"http://localhost:8306/reasoners/process_order\": dial tcp 127.0.0.1:8306: connect: connection refused": "agent call failed: Post \"http://localhost:8306/reasoners/process_order\": dial tcp 127.0.0.1:8306: connect: connection refused",
		"amount 52,000 exceeds limit of 50000.":                              "amount N exceeds limit of N.",
		"run run_20261109_133313_57doroae timed out at 2026-11-09T18:33:14Z": "run <id> timed out at <time>",
		"request 3f2a8c1e-1b2c-4d5e-8f90-123456789abc failed (code 503)":     "request <id> failed (code N)",
		"hash deadbeefdeadbeefdeadbeef mismatch":                             "hash <id> mismatch",
	}
	for in, want := range cases {
		require.Equal(t, want, NormalizeErrorMessage(in), in)
	}
	long := strings.Repeat("x", 400)
	require.Len(t, []rune(NormalizeErrorMessage(long)), 241)
}

func sumInt64s(xs []int64) int64 {
	var s int64
	for _, x := range xs {
		s += x
	}
	return s
}

// TestCountsAndRecentFailuresFollowTheRange checks the exact counts and the
// failure list the older dashboard endpoints now read, on both backends.
func TestCountsAndRecentFailuresFollowTheRange(t *testing.T) {
	for name, ls := range homeStorages(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			now := time.Now().UTC().Truncate(time.Second)
			seedHomeMesh(t, ls, now)

			counts, err := ls.CountExecutionsByStatus(ctx, now.Add(-time.Hour), now)
			require.NoError(t, err)
			require.Equal(t, int64(9), counts.Total)
			require.Equal(t, int64(3), counts.Succeeded)
			// failed and timeout both count as failed.
			require.Equal(t, int64(4), counts.Failed)
			require.Equal(t, int64(7), counts.Terminal)

			older, err := ls.CountExecutionsByStatus(ctx, now.Add(-48*time.Hour), now.Add(-time.Hour))
			require.NoError(t, err)
			require.Equal(t, int64(4), older.Total)
			require.Equal(t, int64(1), older.Failed)

			failures, err := ls.QueryRecentFailures(ctx, now.Add(-time.Hour), now, 3)
			require.NoError(t, err)
			require.Len(t, failures, 3)
			for i := 1; i < len(failures); i++ {
				require.False(t, failures[i].StartedAt.After(failures[i-1].StartedAt), "newest first")
			}
			require.Equal(t, "e-r2-fraud", failures[0].ExecutionID)

			none, err := ls.QueryRecentFailures(ctx, now.Add(-10*time.Hour), now.Add(-9*time.Hour), 0)
			require.NoError(t, err)
			require.Empty(t, none)

			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			_, err = ls.CountExecutionsByStatus(cancelled, now.Add(-time.Hour), now)
			require.Error(t, err)
		})
	}
}
