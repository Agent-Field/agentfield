package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestExecuteAsyncHandler_ReplaysMatchingSucceededChildCall(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var agentCalls int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&agentCalls, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:   "old-child",
		RunID:         "old-run",
		AgentNodeID:   "node-1",
		NodeID:        "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		InputPayload:  json.RawMessage(`{"input":{"foo":"bar"}}`),
		ResultPayload: json.RawMessage(`{"answer":42}`),
		StartedAt:     now.Add(-2 * time.Minute),
		CreatedAt:     now.Add(-2 * time.Minute),
		UpdatedAt:     now.Add(-2 * time.Minute),
	})
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-failed",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-b",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"step":"failed"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", "new-run")
	req.Header.Set("X-Parent-Execution-ID", "new-parent")
	req.Header.Set("X-AgentField-Replay-Source-Run-ID", "old-run")
	req.Header.Set("X-AgentField-Replay-Before-Execution-ID", "old-failed")
	req.Header.Set("X-AgentField-Replay-Mode", "succeeded-before")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)
	require.Equal(t, "old-child", resp.Header().Get("X-AgentField-Replay-Hit"))
	require.EqualValues(t, 0, atomic.LoadInt32(&agentCalls))

	var payload AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, string(types.ExecutionStatusSucceeded), payload.Status)

	record, err := store.GetExecutionRecord(context.Background(), payload.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, record)
	require.Equal(t, types.ExecutionStatusSucceeded, record.Status)
	require.JSONEq(t, `{"answer":42}`, string(record.ResultPayload))
	require.NotNil(t, record.StatusReason)
	require.Equal(t, "replayed_from_execution:old-child", *record.StatusReason)

	workflowRecord, err := store.GetWorkflowExecution(context.Background(), payload.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, workflowRecord)
	require.Equal(t, string(types.ExecutionStatusSucceeded), workflowRecord.Status)
}

func TestExecuteHandler_ReplaysMatchingSucceededChildCall(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var agentCalls int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&agentCalls, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:   "old-sync-child",
		RunID:         "old-run",
		AgentNodeID:   "node-1",
		NodeID:        "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		InputPayload:  json.RawMessage(`{"input":{"foo":"bar"}}`),
		ResultPayload: json.RawMessage(`{"answer":42}`),
		StartedAt:     now.Add(-2 * time.Minute),
		CreatedAt:     now.Add(-2 * time.Minute),
		UpdatedAt:     now.Add(-2 * time.Minute),
	})
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-sync-failed",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-b",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"step":"failed"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", "new-run")
	req.Header.Set("X-Parent-Execution-ID", "new-parent")
	req.Header.Set("X-AgentField-Replay-Source-Run-ID", "old-run")
	req.Header.Set("X-AgentField-Replay-Before-Execution-ID", "old-sync-failed")
	req.Header.Set("X-AgentField-Replay-Mode", "succeeded-before")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "old-sync-child", resp.Header().Get("X-AgentField-Replay-Hit"))
	require.EqualValues(t, 0, atomic.LoadInt32(&agentCalls))

	var payload ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, string(types.ExecutionStatusSucceeded), payload.Status)
	require.Equal(t, map[string]interface{}{"answer": float64(42)}, payload.Result)
	require.Zero(t, payload.DurationMS)
}

func TestExecuteHandler_DoesNotReplaySucceededChildAfterMarker(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var agentCalls int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&agentCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fresh":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-failed",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-b",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"step":"failed"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:   "old-child-late",
		RunID:         "old-run",
		AgentNodeID:   "node-1",
		NodeID:        "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		InputPayload:  json.RawMessage(`{"input":{"foo":"bar"}}`),
		ResultPayload: json.RawMessage(`{"answer":42}`),
		StartedAt:     now.Add(time.Minute),
		CreatedAt:     now.Add(time.Minute),
		UpdatedAt:     now.Add(time.Minute),
	})

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", "new-run")
	req.Header.Set("X-Parent-Execution-ID", "new-parent")
	req.Header.Set("X-AgentField-Replay-Source-Run-ID", "old-run")
	req.Header.Set("X-AgentField-Replay-Before-Execution-ID", "old-failed")
	req.Header.Set("X-AgentField-Replay-Mode", "succeeded-before")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Empty(t, resp.Header().Get("X-AgentField-Replay-Hit"))
	require.EqualValues(t, 1, atomic.LoadInt32(&agentCalls))
	var payload ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, resp.Header().Get("X-Execution-ID"), payload.ExecutionID)
	require.Equal(t, "new-run", payload.RunID)
	require.Equal(t, string(types.ExecutionStatusSucceeded), payload.Status)
	require.True(t, payload.DurationMS >= 0)
	require.NotEmpty(t, payload.FinishedAt)
	require.Equal(t, map[string]interface{}{"fresh": true}, payload.Result)
}

func TestExecuteHandler_AllSucceededReplaysMatchingChildAfterMarker(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var agentCalls int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&agentCalls, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-failed",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-b",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"step":"failed"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:   "old-child-late",
		RunID:         "old-run",
		AgentNodeID:   "node-1",
		NodeID:        "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		InputPayload:  json.RawMessage(`{"input":{"foo":"bar"}}`),
		ResultPayload: json.RawMessage(`{"answer":42}`),
		StartedAt:     now.Add(time.Minute),
		CreatedAt:     now.Add(time.Minute),
		UpdatedAt:     now.Add(time.Minute),
	})

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", "new-run")
	req.Header.Set("X-Parent-Execution-ID", "new-parent")
	req.Header.Set("X-AgentField-Replay-Source-Run-ID", "old-run")
	req.Header.Set("X-AgentField-Replay-Before-Execution-ID", "old-failed")
	req.Header.Set("X-AgentField-Replay-Mode", "all-succeeded")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "old-child-late", resp.Header().Get("X-AgentField-Replay-Hit"))
	require.EqualValues(t, 0, atomic.LoadInt32(&agentCalls))

	var payload ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, string(types.ExecutionStatusSucceeded), payload.Status)
	require.Equal(t, map[string]interface{}{"answer": float64(42)}, payload.Result)
}

func TestRestartExecutionHandler_ForwardsReplayHeadersToRestartedRoot(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var (
		mu      sync.Mutex
		headers http.Header
		body    string
	)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = r.Header.Clone()
		raw := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(raw)
		body = string(raw)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"root":"ok"}`))
	}))
	defer agentServer.Close()

	withTestAsyncPool(t)

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:   "old-root",
		RunID:         "old-run",
		AgentNodeID:   "node-1",
		NodeID:        "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		InputPayload:  json.RawMessage(`{"input":{"topic":"restart"},"context":{"priority":"high"}}`),
		ResultPayload: json.RawMessage(`{"root":"old"}`),
		StartedAt:     now.Add(-2 * time.Minute),
		CreatedAt:     now.Add(-2 * time.Minute),
		UpdatedAt:     now.Add(-2 * time.Minute),
	})
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:       "old-failed-child",
		RunID:             "old-run",
		ParentExecutionID: pointerString("old-root"),
		AgentNodeID:       "node-1",
		NodeID:            "node-1",
		ReasonerID:        "reasoner-b",
		Status:            types.ExecutionStatusFailed,
		InputPayload:      json.RawMessage(`{"input":{"child":"boom"}}`),
		StartedAt:         now,
		CreatedAt:         now,
		UpdatedAt:         now,
	})

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/old-failed-child/restart", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)
	var payload restartExecutionResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "old-failed-child", payload.SourceExecutionID)
	require.Equal(t, "old-root", payload.RestartedExecutionID)
	require.Equal(t, "succeeded-before", payload.ReplayMode)
	require.NotEqual(t, "old-run", payload.RunID)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return headers.Get("X-AgentField-Replay-Source-Run-ID") == "old-run"
	}, 2*time.Second, 20*time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, "old-failed-child", headers.Get("X-AgentField-Replay-Before-Execution-ID"))
	require.Equal(t, "succeeded-before", headers.Get("X-AgentField-Replay-Mode"))
	require.Equal(t, payload.RunID, headers.Get("X-Run-ID"))
	require.JSONEq(t, `{"topic":"restart"}`, body)
}

func TestRestartExecutionHandler_ReuseNoneDoesNotForwardReplayHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var (
		mu      sync.Mutex
		headers http.Header
	)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = r.Header.Clone()
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"root":"ok"}`))
	}))
	defer agentServer.Close()

	withTestAsyncPool(t)

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-root",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-a",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"topic":"restart"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/old-root/restart", strings.NewReader(`{"reuse":"none"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)
	var payload restartExecutionResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "none", payload.ReplayMode)
	require.Nil(t, payload.ReplayBeforeExecutionID)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return headers.Get("X-Run-ID") == payload.RunID
	}, 2*time.Second, 20*time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Empty(t, headers.Get("X-AgentField-Replay-Source-Run-ID"))
	require.Empty(t, headers.Get("X-AgentField-Replay-Before-Execution-ID"))
	require.Equal(t, "none", headers.Get("X-AgentField-Replay-Mode"))
}

func TestRestartExecutionHandler_ExecutionScopeUsesAllSucceededReplay(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var (
		mu      sync.Mutex
		headers http.Header
	)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = r.Header.Clone()
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"child":"ok"}`))
	}))
	defer agentServer.Close()

	withTestAsyncPool(t)

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID:  "old-child",
		RunID:        "old-run",
		AgentNodeID:  "node-1",
		NodeID:       "node-1",
		ReasonerID:   "reasoner-b",
		Status:       types.ExecutionStatusFailed,
		InputPayload: json.RawMessage(`{"input":{"child":"retry"}}`),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	})

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/old-child/restart", strings.NewReader(`{"scope":"execution"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)
	var payload restartExecutionResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "old-child", payload.RestartedExecutionID)
	require.Equal(t, "all-succeeded", payload.ReplayMode)
	require.Nil(t, payload.ReplayBeforeExecutionID)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return headers.Get("X-AgentField-Replay-Source-Run-ID") == "old-run"
	}, 2*time.Second, 20*time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Empty(t, headers.Get("X-AgentField-Replay-Before-Execution-ID"))
	require.Equal(t, "all-succeeded", headers.Get("X-AgentField-Replay-Mode"))
}

func TestRestartExecutionHandlerValidationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(testRestartAgent("http://agent.test"))
	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/restart", RestartExecutionHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))

	tests := []struct {
		name string
		body string
		want int
	}{
		{name: "invalid json", body: `{"scope":`, want: http.StatusBadRequest},
		{name: "invalid scope", body: `{"scope":"node"}`, want: http.StatusBadRequest},
		{name: "invalid reuse", body: `{"reuse":"cached"}`, want: http.StatusBadRequest},
		{name: "missing source execution", body: `{}`, want: http.StatusNotFound},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/missing/restart", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			require.Equal(t, tt.want, resp.Code)
			require.Contains(t, resp.Body.String(), "error")
		})
	}
}

func TestExecutionReuseInfoParsesReplayStatusReason(t *testing.T) {
	reason := "replayed_from_execution:source-exec"
	info := executionReuseInfo(&types.Execution{StatusReason: &reason})
	require.NotNil(t, info)
	require.True(t, info.Hit)
	require.Equal(t, "source-exec", info.SourceExecutionID)

	plain := "ordinary status"
	require.Nil(t, executionReuseInfo(&types.Execution{StatusReason: &plain}))
	require.Nil(t, executionReuseInfo(nil))
}

func testRestartAgent(baseURL string) *types.AgentNode {
	return &types.AgentNode{
		ID:              "node-1",
		BaseURL:         baseURL,
		Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}, {ID: "reasoner-b"}},
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
}

func seedExecutionRecord(t *testing.T, store *testExecutionStorage, exec *types.Execution) {
	t.Helper()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
}

func withTestAsyncPool(t *testing.T) {
	t.Helper()
	prevAsyncPool := asyncPool
	asyncPool = newAsyncWorkerPool(1, 8)
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	t.Cleanup(func() {
		asyncPool = prevAsyncPool
		asyncPoolOnce = sync.Once{}
	})
}
