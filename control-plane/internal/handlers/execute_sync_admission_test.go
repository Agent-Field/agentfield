package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type panicOnCreateExecutionStore struct {
	*testExecutionStorage
}

func (s *panicOnCreateExecutionStore) CreateExecutionRecord(context.Context, *types.Execution) error {
	panic("injected execution persistence panic")
}

func TestExecuteAdmission_RecoveredPersistencePanicReleasesSlot(t *testing.T) {
	for _, test := range []struct {
		name string
		run  func(t *testing.T, store *panicOnCreateExecutionStore) *httptest.ResponseRecorder
	}{
		{
			name: "sync",
			run: func(t *testing.T, store *panicOnCreateExecutionStore) *httptest.ResponseRecorder {
				router := gin.New()
				router.Use(gin.CustomRecoveryWithWriter(io.Discard, func(c *gin.Context, _ interface{}) { c.AbortWithStatus(http.StatusInternalServerError) }))
				router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
				req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)
				return resp
			},
		},
		{
			name: "restart",
			run: func(t *testing.T, store *panicOnCreateExecutionStore) *httptest.ResponseRecorder {
				useAsyncPoolForTest(t, newAsyncWorkerPool(0, 2))
				now := time.Now().UTC()
				source := &types.Execution{ExecutionID: "source", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now, CreatedAt: now, UpdatedAt: now}
				require.NoError(t, store.testExecutionStorage.CreateExecutionRecord(context.Background(), source))
				router := gin.New()
				router.Use(gin.CustomRecoveryWithWriter(io.Discard, func(c *gin.Context, _ interface{}) { c.AbortWithStatus(http.StatusInternalServerError) }))
				router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
				req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/source/restart", strings.NewReader(`{}`))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)
				return resp
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldLimiter := concurrencyLimiter
			concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
			t.Cleanup(func() { concurrencyLimiter = oldLimiter })
			store := &panicOnCreateExecutionStore{testExecutionStorage: newTestExecutionStorage(testRestartAgent("http://agent.example"))}
			resp := test.run(t, store)
			require.Equal(t, http.StatusInternalServerError, resp.Code)
			require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
		})
	}
}

func TestExecuteHandler_ConcurrencyRejectionHasNoPersistence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))
	t.Cleanup(func() { concurrencyLimiter = oldLimiter })

	agent := &types.AgentNode{ID: "node-1", BaseURL: "http://agent.example", Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store := newTestExecutionStorage(agent)
	payloadDir := t.TempDir()
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(payloadDir), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusTooManyRequests, resp.Code)
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "concurrency_limit", body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
	files, err := filepath.Glob(filepath.Join(payloadDir, "*"))
	require.NoError(t, err)
	require.Empty(t, files)
}

func TestExecuteHandler_LLMUnavailableRejectionHasNoPersistence(t *testing.T) {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	defer failing.Close()
	monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{Enabled: true, CheckInterval: 10 * time.Millisecond, CheckTimeout: 100 * time.Millisecond, FailureThreshold: 1, RecoveryTimeout: 30 * time.Second, Endpoints: []config.LLMEndpoint{{Name: "primary", URL: failing.URL}}}, nil)
	go monitor.Start()
	defer monitor.Stop()
	SetLLMHealthMonitor(monitor)
	defer SetLLMHealthMonitor(nil)
	require.Eventually(t, func() bool {
		s, ok := monitor.GetStatus("primary")
		return ok && s.CircuitState == services.CircuitOpen
	}, 2*time.Second, 10*time.Millisecond)

	store := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	payloadDir := t.TempDir()
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(payloadDir), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "llm_unavailable", body["error_category"])
	require.NotEmpty(t, resp.Header().Get("Retry-After"))
	require.Equal(t, fmt.Sprint(body["retry_after"]), resp.Header().Get("Retry-After"))
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
	files, err := filepath.Glob(filepath.Join(payloadDir, "*"))
	require.NoError(t, err)
	require.Empty(t, files)
}

func TestExecuteHandler_ReplayHitNotGatedBySaturatedAgent(t *testing.T) {
	for _, tc := range []struct {
		name, route string
		status      int
	}{{"sync", "/api/v1/execute/:target", http.StatusOK}, {"async", "/api/v1/execute/async/:target", http.StatusAccepted}} {
		t.Run(tc.name, func(t *testing.T) {
			oldLimiter := concurrencyLimiter
			concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
			require.NoError(t, concurrencyLimiter.Acquire("node-1"))
			t.Cleanup(func() { concurrencyLimiter = oldLimiter })
			useAsyncPoolForTest(t, newAsyncWorkerPool(1, 2))
			var calls int32
			agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { atomic.AddInt32(&calls, 1); w.WriteHeader(http.StatusOK) }))
			defer agentServer.Close()
			store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
			now := time.Now().UTC()
			seedExecutionRecord(t, store, &types.Execution{ExecutionID: "source", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusSucceeded, InputPayload: json.RawMessage(`{"input":{"foo":"bar"}}`), ResultPayload: json.RawMessage(`{"answer":42}`), StartedAt: now.Add(-time.Minute), CreatedAt: now.Add(-time.Minute), UpdatedAt: now.Add(-time.Minute)})
			seedExecutionRecord(t, store, &types.Execution{ExecutionID: "marker", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-b", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{}`), StartedAt: now, CreatedAt: now, UpdatedAt: now})
			router := gin.New()
			if tc.name == "sync" {
				router.POST(tc.route, ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
			} else {
				router.POST(tc.route, ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
			}
			req := httptest.NewRequest(http.MethodPost, strings.Replace(tc.route, ":target", "node-1.reasoner-a", 1), strings.NewReader(`{"input":{"foo":"bar"}}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Run-ID", "new-run")
			req.Header.Set("X-Parent-Execution-ID", "new-parent")
			req.Header.Set("X-AgentField-Replay-Source-Run-ID", "old-run")
			req.Header.Set("X-AgentField-Replay-Before-Execution-ID", "marker")
			req.Header.Set("X-AgentField-Replay-Mode", "succeeded-before")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, tc.status, resp.Code, resp.Body.String())
			require.Equal(t, "source", resp.Header().Get("X-AgentField-Replay-Hit"))
			require.EqualValues(t, 1, concurrencyLimiter.GetRunningCount("node-1"))
			require.Zero(t, atomic.LoadInt32(&calls))
		})
	}
}

func TestExecuteHandler_SlotBalancedAcrossOutcomes(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
	defer func() { concurrencyLimiter = oldLimiter }()
	var during int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.StoreInt32(&during, int32(concurrencyLimiter.GetRunningCount("node-1")))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	store := newTestExecutionStorage(testRestartAgent(server.URL))
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	request := func(r *gin.Engine) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		out := httptest.NewRecorder()
		r.ServeHTTP(out, req)
		return out
	}
	require.Equal(t, http.StatusOK, request(router).Code)
	require.EqualValues(t, 1, atomic.LoadInt32(&during))
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
	server.Close()
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusInternalServerError) }))
	defer errorServer.Close()
	errorRouter := gin.New()
	errorRouter.POST("/api/v1/execute/:target", ExecuteHandler(newTestExecutionStorage(testRestartAgent(errorServer.URL)), services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	require.Equal(t, http.StatusBadGateway, request(errorRouter).Code)
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
	inactive := testRestartAgent("http://agent.example")
	inactive.HealthStatus = types.HealthStatusInactive
	inactiveRouter := gin.New()
	inactiveRouter.POST("/api/v1/execute/:target", ExecuteHandler(newTestExecutionStorage(inactive), services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	require.Equal(t, http.StatusServiceUnavailable, request(inactiveRouter).Code)
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
}

func TestWriteExecutionError_RetryAfterPerCategory(t *testing.T) {
	tests := []struct {
		category ErrorCategory
		retry    int
	}{
		{ErrorCategoryConcurrencyLimit, 1},
		{ErrorCategoryNodeUnavailable, 1},
		{ErrorCategoryLLMUnavailable, 17},
		{ErrorCategoryControlPlaneShutdown, 1},
	}
	for _, test := range tests {
		t.Run(string(test.category), func(t *testing.T) {
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			err := &executionPreconditionError{code: http.StatusServiceUnavailable, message: "retry", category: test.category}
			if test.category == ErrorCategoryLLMUnavailable {
				err.retryAfter = test.retry
			}
			writeExecutionError(ctx, err)
			require.Equal(t, fmt.Sprint(test.retry), recorder.Header().Get("Retry-After"))
			var body map[string]any
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
			require.Equal(t, float64(test.retry), body["retry_after"])
		})
	}
	t.Run("body too large is not retryable", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		writeExecutionError(ctx, &http.MaxBytesError{})
		require.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
		require.Empty(t, recorder.Header().Get("Retry-After"))
		var body map[string]any
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
		require.NotContains(t, body, "retry_after")
	})
	t.Run("pending approval is not retryable", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		writeExecutionError(ctx, &executionPreconditionError{code: http.StatusServiceUnavailable, message: "pending", category: ErrorCategoryAgentError, errorCode: "agent_pending_approval"})
		require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
		require.Empty(t, recorder.Header().Get("Retry-After"))
		var body map[string]any
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
		require.NotContains(t, body, "retry_after")
	})
}

func TestRestartHandler_ConcurrencyRejectionHasNoPersistence(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))
	defer func() { concurrencyLimiter = oldLimiter }()
	useAsyncPoolForTest(t, newAsyncWorkerPool(0, 2))
	store := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{ExecutionID: "source", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{"input":{"foo":"bar"}}`), StartedAt: now, CreatedAt: now, UpdatedAt: now})
	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/source/restart", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusTooManyRequests, resp.Code, resp.Body.String())
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "concurrency_limit", body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 1)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
}

func TestRestartHandler_QueueFullCarriesRetryAfter(t *testing.T) {
	pool := newAsyncWorkerPool(0, 1)
	useAsyncPoolForTest(t, pool)
	require.True(t, pool.reserve())
	store := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{ExecutionID: "source", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now, CreatedAt: now, UpdatedAt: now})
	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/source/restart", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusServiceUnavailable, resp.Code, resp.Body.String())
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "concurrency_limit", body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 1)
}

// A restart admitted past reserve() but refused by submitReserved (the pool
// stopped in between) reports and persists one stable shutdown category.
func TestRestartHandler_PoolStoppedReturnsConsistentShutdownContract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(0, 4)
	useAsyncPoolForTest(t, pool)
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
	defer func() { concurrencyLimiter = oldLimiter }()

	base := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	now := time.Now().UTC()
	seedExecutionRecord(t, base, &types.Execution{
		ExecutionID: "source", RunID: "old-run", AgentNodeID: "node-1", NodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"foo":"bar"}}`),
		StartedAt:    now, CreatedAt: now, UpdatedAt: now,
	})
	reqCtx, cancel := context.WithCancel(context.Background())
	store := &stopPoolOnCreateStorage{testExecutionStorage: base, pool: pool, cancel: cancel}

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/source/restart", strings.NewReader(`{}`)).WithContext(reqCtx)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code, resp.Body.String())
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])

	records, err := base.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 2)
	var restarted *types.Execution
	for _, record := range records {
		if record.ExecutionID != "source" {
			restarted = record
		}
	}
	require.NotNil(t, restarted)
	require.Equal(t, types.ExecutionStatusFailed, restarted.Status)
	require.NotNil(t, restarted.StatusReason)
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), *restarted.StatusReason)
	workflows, err := base.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, workflows, 1)
	require.Equal(t, string(types.ExecutionStatusFailed), workflows[0].Status)
	require.NotNil(t, workflows[0].StatusReason)
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), *workflows[0].StatusReason)
	require.Equal(t, string(restarted.Status), workflows[0].Status)
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
}
