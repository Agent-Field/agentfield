package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/services"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestExecuteHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		require.Equal(t, "/reasoners/reasoner-a", r.URL.Path)
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		defer r.Body.Close()

		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(body, &payload))
		require.Equal(t, map[string]interface{}{"foo": "bar"}, payload)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":42}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var envelope ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &envelope))
	require.Equal(t, types.ExecutionStatusSucceeded, envelope.Status)
	require.NotEmpty(t, envelope.ExecutionID)
	require.NotEmpty(t, envelope.RunID)
	require.GreaterOrEqual(t, envelope.DurationMS, int64(0))
	require.False(t, envelope.WebhookRegistered)

	result, ok := envelope.Result.(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, float64(42), result["answer"])

	record, err := store.GetExecutionRecord(context.Background(), envelope.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, record)
	require.Equal(t, types.ExecutionStatusSucceeded, record.Status)
	require.NotNil(t, record.ResultPayload)
	require.NotNil(t, record.ResultURI)
	require.Greater(t, len(record.ResultPayload), 0)

	require.Equal(t, int32(1), atomic.LoadInt32(&requestCount))
}

func TestExecuteHandler_AgentError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "agent error (500)")

	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 1)
	require.Equal(t, types.ExecutionStatusFailed, records[0].Status)
	require.NotNil(t, records[0].ErrorMessage)
	require.Contains(t, *records[0].ErrorMessage, "agent error (500)")
}

func TestExecuteHandler_TargetNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.unknown", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "target 'unknown' not found")

	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 0)
}

func TestExecuteAsyncHandler_ReturnsAccepted(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)

	var payload AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.NotEmpty(t, payload.ExecutionID)
	require.NotEmpty(t, payload.RunID)
	require.Equal(t, string(types.ExecutionStatusQueued), payload.Status)

	require.Eventually(t, func() bool {
		record, err := store.GetExecutionRecord(context.Background(), payload.ExecutionID)
		if err != nil || record == nil {
			return false
		}
		return record.Status == types.ExecutionStatusSucceeded
	}, 2*time.Second, 50*time.Millisecond)

	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&requestCount) > 0
	}, time.Second, 50*time.Millisecond)
}

func TestExecuteAsyncHandler_InvalidJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "node-1"})
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader("not-json"))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestGetExecutionStatusHandler_ReturnsResult(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	now := time.Now().UTC()
	result := json.RawMessage(`{"ok":true}`)

	execution := &types.Execution{
		ExecutionID:   "exec-1",
		RunID:         "run-1",
		AgentNodeID:   "node-1",
		ReasonerID:    "reasoner-a",
		Status:        types.ExecutionStatusSucceeded,
		ResultPayload: result,
		ResultURI:     ptrString("payload://result"),
		StartedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.GET("/api/v1/executions/:execution_id", GetExecutionStatusHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "exec-1", payload.ExecutionID)
	require.Equal(t, types.ExecutionStatusSucceeded, payload.Status)

	resultMap, ok := payload.Result.(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, true, resultMap["ok"])
}

func TestBatchExecutionStatusHandler_MixedResults(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-ok",
		RunID:       "run-1",
		Status:      types.ExecutionStatusSucceeded,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/executions/batch-status", BatchExecutionStatusHandler(store))

	body := `{"execution_ids":["exec-ok","exec-missing"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/batch-status", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload BatchStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, types.ExecutionStatusSucceeded, payload["exec-ok"].Status)
	require.Equal(t, "not_found", payload["exec-missing"].Status)
}

func ptrString(value string) *string {
	return &value
}
