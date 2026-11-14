package ui

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// homeTestStorage opens a SQLite storage whose file the test can also write
// to directly, for seeding more rows than one-at-a-time inserts allow.
func homeTestStorage(t *testing.T) (*storage.LocalStorage, *sql.DB) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "agentfield.db")
	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: storage.LocalStorageConfig{
		DatabasePath: dbPath, KVStorePath: filepath.Join(dir, "agentfield.bolt"),
	}})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() { _ = ls.Close(ctx) })
	raw, err := sql.Open("sqlite3", dbPath+"?_busy_timeout=10000")
	require.NoError(t, err)
	t.Cleanup(func() { _ = raw.Close() })
	return ls, raw
}

// bulkInsertExecutions writes n executions in one transaction, starting at
// base and one millisecond apart. The last `failures` of them failed.
func bulkInsertExecutions(t *testing.T, raw *sql.DB, base time.Time, n, failures int) {
	t.Helper()
	tx, err := raw.Begin()
	require.NoError(t, err)
	stmt, err := tx.Prepare(`INSERT INTO executions
		(execution_id, run_id, agent_node_id, reasoner_id, node_id, status, error_message,
		 started_at, completed_at, duration_ms, notes, created_at, updated_at)
		VALUES (?, ?, 'bulk-agent', 'work', 'bulk-agent', ?, ?, ?, ?, 5, '[]', ?, ?)`)
	require.NoError(t, err)
	for i := 0; i < n; i++ {
		status := "succeeded"
		var errMsg interface{}
		if i >= n-failures {
			status = "failed"
			errMsg = fmt.Sprintf("upstream refused call %d", i)
		}
		started := base.Add(time.Duration(i) * time.Millisecond).UTC()
		_, err := stmt.Exec(fmt.Sprintf("bulk-%06d", i), fmt.Sprintf("bulk-run-%06d", i), status, errMsg,
			started, started.Add(5*time.Millisecond), started, started)
		require.NoError(t, err)
	}
	require.NoError(t, stmt.Close())
	require.NoError(t, tx.Commit())
}

const cappedTotal, cappedFailures = 50_010, 10

// seedPastTheCaps records 50,010 executions in the last few minutes (today),
// the last ten of them failed, and returns a router over both dashboard
// endpoints.
func seedPastTheCaps(t *testing.T) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)
	ls, raw := homeTestStorage(t)
	now := time.Now().UTC()
	base := now.Add(-2 * time.Minute)
	if dayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC); base.Before(dayStart) {
		base = dayStart
	}
	bulkInsertExecutions(t, raw, base, cappedTotal, cappedFailures)

	handler := NewDashboardHandler(ls, &stubDashboardAgentService{})
	router := gin.New()
	router.GET("/summary", handler.GetDashboardSummaryHandler)
	router.GET("/enhanced", handler.GetEnhancedDashboardSummaryHandler)
	return router
}

func getEnhanced(t *testing.T, router *gin.Engine, preset string) EnhancedDashboardResponse {
	t.Helper()
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/enhanced?preset="+preset, nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var enhanced EnhancedDashboardResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &enhanced))
	return enhanced
}

// On main the summary counted the rows of a list query capped at 10,000, so
// a busy day was reported as exactly 10,000 executions.
func TestDashboardSummaryCountIsNotCapped(t *testing.T) {
	router := seedPastTheCaps(t)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/summary", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var summary DashboardSummaryResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &summary))
	require.Equal(t, cappedTotal, summary.Executions.Today)
	require.InDelta(t, float64(cappedTotal-cappedFailures)/float64(cappedTotal)*100, summary.SuccessRate, 0.0001)
}

// On main the enhanced view counted a list capped at 50,000 and reported that
// one number as both its 24h and its 7d count.
func TestEnhancedDashboardCountsAreNotCapped(t *testing.T) {
	router := seedPastTheCaps(t)
	for _, preset := range []string{"24h", "7d"} {
		enhanced := getEnhanced(t, router, preset)
		require.Equal(t, cappedTotal, enhanced.Overview.ExecutionsLast24h, preset)
		require.Equal(t, cappedTotal, enhanced.Overview.ExecutionsLast7d, preset)
		require.True(t, enhanced.DetailSampled, preset)
	}
}

// On main incidents were picked from the OLDEST 50,000 executions of the
// requested range, so once a range held more than that, the failures at its
// end never showed.
func TestEnhancedDashboardIncidentsAreTheNewestInRange(t *testing.T) {
	router := seedPastTheCaps(t)
	enhanced := getEnhanced(t, router, "24h")
	require.Len(t, enhanced.Incidents, cappedFailures)
	require.Equal(t, fmt.Sprintf("bulk-%06d", cappedTotal-1), enhanced.Incidents[0].ExecutionID, "newest failure first")
	for _, inc := range enhanced.Incidents {
		require.False(t, inc.StartedAt.Before(enhanced.TimeRange.StartTime))
		require.False(t, inc.StartedAt.After(enhanced.TimeRange.EndTime))
	}
}

// On main an agent that heartbeats and reports ready was "offline" unless
// the control plane's own process manager had started it.
func TestDashboardAgentStatusFollowsRecordedHeartbeat(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ls, _ := homeTestStorage(t)
	ctx := context.Background()
	now := time.Now().UTC()
	require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{
		ID: "remote-agent", BaseURL: "http://remote", HealthStatus: types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now,
	}))
	require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{
		ID: "gone-agent", BaseURL: "http://gone", HealthStatus: types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline, LastHeartbeat: now.Add(-time.Hour),
	}))

	// The process manager knows neither agent.
	handler := NewDashboardHandler(ls, &stubDashboardAgentService{})
	router := gin.New()
	router.GET("/summary", handler.GetDashboardSummaryHandler)
	router.GET("/enhanced", handler.GetEnhancedDashboardSummaryHandler)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/summary", nil))
	var summary DashboardSummaryResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &summary))
	require.Equal(t, AgentsSummary{Running: 1, Total: 2}, summary.Agents)

	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/enhanced", nil))
	var enhanced EnhancedDashboardResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &enhanced))
	require.Equal(t, 1, enhanced.Overview.ActiveAgents)
	require.Equal(t, 1, enhanced.Overview.OfflineAgents)
	statuses := map[string]string{}
	for _, a := range enhanced.AgentHealth.Agents {
		statuses[a.ID] = a.Status
	}
	require.Equal(t, map[string]string{"remote-agent": "running", "gone-agent": "offline"}, statuses)
}

func TestRecordedAgentStatus(t *testing.T) {
	cases := []struct {
		health    types.HealthStatus
		lifecycle types.AgentLifecycleStatus
		want      string
	}{
		{types.HealthStatusActive, types.AgentStatusReady, agentStatusRunning},
		{types.HealthStatusUnknown, types.AgentStatusReady, agentStatusRunning},
		{types.HealthStatusActive, "", agentStatusRunning},
		{types.HealthStatusInactive, types.AgentStatusReady, agentStatusOffline},
		{types.HealthStatusActive, types.AgentStatusOffline, agentStatusOffline},
		{types.HealthStatusInactive, types.AgentStatusDegraded, agentStatusDegraded},
		{types.HealthStatusDegraded, types.AgentStatusReady, agentStatusDegraded},
		{types.HealthStatusUnknown, types.AgentStatusStarting, agentStatusStarting},
		{types.HealthStatusUnknown, types.AgentStatusPendingApproval, agentStatusPendingApproval},
		{types.HealthStatusUnknown, "", agentStatusUnknown},
	}
	for _, c := range cases {
		got := recordedAgentStatus(&types.AgentNode{HealthStatus: c.health, LifecycleStatus: c.lifecycle})
		require.Equal(t, c.want, got, "%s/%s", c.health, c.lifecycle)
	}
	require.Equal(t, agentStatusUnknown, recordedAgentStatus(nil))
}

func TestDeveloperHomeHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ls, _ := homeTestStorage(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Second)

	for _, a := range []*types.AgentNode{
		{ID: "orch", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now,
			Reasoners: []types.ReasonerDefinition{{ID: "process_order"}}},
		{ID: "fraud", HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline, LastHeartbeat: now.Add(-10 * time.Minute)},
		{ID: "payments", HealthStatus: types.HealthStatusUnknown, LifecycleStatus: types.AgentStatusPendingApproval, LastHeartbeat: now},
	} {
		a.BaseURL = "http://" + a.ID
		require.NoError(t, ls.RegisterAgent(ctx, a))
	}
	exec := func(id, run, parent, agent, reasoner, status string, started time.Time) {
		e := &types.Execution{ExecutionID: id, RunID: run, AgentNodeID: agent, ReasonerID: reasoner, NodeID: agent,
			Status: status, StartedAt: started}
		if parent != "" {
			e.ParentExecutionID = &parent
		}
		if types.IsTerminalExecutionStatus(status) {
			d := int64(120)
			e.DurationMS = &d
			done := started.Add(120 * time.Millisecond)
			e.CompletedAt = &done
		}
		if status == "failed" {
			msg := "server unavailable"
			e.ErrorMessage = &msg
		}
		require.NoError(t, ls.CreateExecutionRecord(ctx, e))
	}
	exec("r1", "run-1", "", "orch", "process_order", "succeeded", now.Add(-20*time.Minute))
	exec("r1-f", "run-1", "r1", "fraud", "score", "succeeded", now.Add(-20*time.Minute))
	exec("r2", "run-2", "", "orch", "process_order", "failed", now.Add(-5*time.Minute))
	exec("r2-f", "run-2", "r2", "fraud", "score", "failed", now.Add(-5*time.Minute))

	handler := NewDashboardHandler(ls, &stubDashboardAgentService{})
	handler.now = func() time.Time { return now }
	router := gin.New()
	router.GET("/home", handler.GetDeveloperHomeHandler)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home?window=1h", nil))
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	var body DeveloperHomeResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))

	require.Equal(t, "1h", body.Window.Preset)
	require.EqualValues(t, 300, body.Window.BucketSeconds)
	require.Equal(t, HomeAgentCounts{Registered: 3, Running: 1, Offline: 1, PendingApproval: 1}, body.AgentCounts)
	require.Len(t, body.Stopped, 1)
	require.Equal(t, "fraud", body.Stopped[0].ID)
	require.Equal(t, []types.DeveloperHomeEntryPoint{{AgentNodeID: "orch", ReasonerID: "process_order", Runs: 2, FailedRuns: 1}}, body.Stopped[0].EntryPoints)
	require.EqualValues(t, 4, body.Totals.Counts.Total)
	require.Len(t, body.Problems, 2)
	require.Nil(t, body.Forwarder, "no forwarder card without a configured webhook")

	// Served from cache within the TTL: a new execution does not show yet.
	exec("r3", "run-3", "", "orch", "process_order", "succeeded", now.Add(-time.Minute))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.EqualValues(t, 4, body.Totals.Counts.Total)

	// A configured forwarder gets its card, with its dead-letter depth.
	require.NoError(t, ls.SetObservabilityWebhook(ctx, &types.ObservabilityWebhookConfig{ID: "global", URL: "https://sink.example/events", Enabled: true}))
	require.NoError(t, ls.AddToDeadLetterQueue(ctx, &types.ObservabilityEvent{EventType: "execution.completed", EventSource: "execution", Timestamp: now.Format(time.RFC3339)}, "sink returned 500", 3))
	handler.now = func() time.Time { return now.Add(time.Minute) }
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home?window=24h", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	body = DeveloperHomeResponse{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.EqualValues(t, 5, body.Totals.Counts.Total)
	require.Equal(t, &HomeForwarder{Enabled: true, DeadLetters: 1}, body.Forwarder)

	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home?window=3d", nil))
	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestDeveloperHomeHandlerStorageErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	base, _ := homeTestStorage(t)

	cases := map[string]*homeErrStorage{
		"aggregates":   {StorageProvider: base, failHome: true},
		"agents":       {StorageProvider: base, failAgents: true},
		"blast radius": {StorageProvider: base, failRadius: true},
	}
	require.NoError(t, base.RegisterAgent(context.Background(), &types.AgentNode{
		ID: "gone", BaseURL: "http://gone", HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline,
	}))
	for name, store := range cases {
		t.Run(name, func(t *testing.T) {
			handler := NewDashboardHandler(store, &stubDashboardAgentService{statuses: map[string]*domain.AgentStatus{}})
			router := gin.New()
			router.GET("/home", handler.GetDeveloperHomeHandler)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home", nil))
			require.Equal(t, http.StatusInternalServerError, rec.Code)
		})
	}
}

type homeErrStorage struct {
	storage.StorageProvider
	failHome, failAgents, failRadius bool
}

func (s *homeErrStorage) QueryDeveloperHome(ctx context.Context, q types.DeveloperHomeQuery) (*types.DeveloperHomeData, error) {
	if s.failHome {
		return nil, fmt.Errorf("boom")
	}
	return s.StorageProvider.QueryDeveloperHome(ctx, q)
}

func (s *homeErrStorage) ListAgents(ctx context.Context, f types.AgentFilters) ([]*types.AgentNode, error) {
	if s.failAgents {
		return nil, fmt.Errorf("boom")
	}
	return s.StorageProvider.ListAgents(ctx, f)
}

func (s *homeErrStorage) QueryBlastRadius(ctx context.Context, id string, start, end time.Time) ([]types.DeveloperHomeEntryPoint, error) {
	if s.failRadius {
		return nil, fmt.Errorf("boom")
	}
	return s.StorageProvider.QueryBlastRadius(ctx, id, start, end)
}

func TestDeveloperHomeReusesTheBaselineAndKeepsSpend(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ls, _ := homeTestStorage(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Second)
	twoDaysAgo := now.Add(-48 * time.Hour)
	d := int64(250)
	done := twoDaysAgo.Add(250 * time.Millisecond)
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "old", RunID: "old", AgentNodeID: "a", ReasonerID: "r", NodeID: "a",
		Status: string(types.ExecutionStatusSucceeded), StartedAt: twoDaysAgo, DurationMS: &d, CompletedAt: &done,
	}))
	cost := 1.5
	require.NoError(t, ls.CreateExecutionUsage(ctx, []*types.ExecutionUsage{{
		ExecutionID: "old", WorkflowID: "old", AgentNodeID: "a", Reasoner: "r", TotalTokens: 10, CostUSD: &cost, CreatedAt: twoDaysAgo,
	}}))

	handler := NewDashboardHandler(ls, &stubDashboardAgentService{})
	router := gin.New()
	router.GET("/home", handler.GetDeveloperHomeHandler)
	get := func(at time.Time) DeveloperHomeResponse {
		handler.now = func() time.Time { return at }
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/home", nil))
		require.Equal(t, http.StatusOK, rec.Code)
		var body DeveloperHomeResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		return body
	}

	for _, at := range []time.Time{now, now.Add(homeCacheTTL + time.Second)} {
		body := get(at)
		require.NotNil(t, body.Baseline)
		require.EqualValues(t, 1, body.Baseline.Counts.Total)
		require.NotNil(t, body.Spend, "spend shows when only the baseline recorded usage")
		require.EqualValues(t, 0, body.Spend.WindowRows)
		require.EqualValues(t, 1, body.Spend.BaselineRows)
		require.InDelta(t, 1.5, *body.Spend.BaselineCostUSD, 1e-9)
	}
}
