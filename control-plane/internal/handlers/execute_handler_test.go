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

	"github.com/Agent-Field/agentfield/control-plane/internal/ard"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

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
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

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

func TestExecuteHandlerWithARD_ExternalCallableBinding(t *testing.T) {
	gin.SetMode(gin.TestMode)
	services.SetWebhookAllowedHosts([]string{"127.0.0.1"})
	t.Cleanup(func() { services.SetWebhookAllowedHosts(nil) })

	externalServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/invoke", r.URL.Path)
		require.Equal(t, "external.vendor.review_contract", r.Header.Get("X-AgentField-ARD-Target"))
		var payload map[string]map[string]interface{}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		require.Equal(t, "review", payload["input"]["operation"])
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"result":{"decision":"approved"}}`))
	}))
	defer externalServer.Close()

	store := newTestExecutionStorage(nil)
	state := ard.State{
		Imports: []ard.ExternalEntry{{
			ID:          "ext_1",
			Identifier:  "urn:ai:vendor.example:agent:review",
			Type:        "application/a2a-agent-card+json",
			DisplayName: "Vendor Review",
			URL:         externalServer.URL + "/invoke",
		}},
		Bindings: map[string]ard.ExternalBinding{
			"ext_1": {
				ExternalEntryID:   "ext_1",
				Callable:          true,
				LocalTarget:       "external.vendor.review_contract",
				Adapter:           "a2a",
				TimeoutMS:         30000,
				AllowedOperations: []string{"review"},
			},
		},
	}
	rawState, err := json.Marshal(state)
	require.NoError(t, err)
	store.config[ard.StateConfigKey] = string(rawState)

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandlerWithARD(store, nil, nil, 90*time.Second, "", func() config.ARDConfig {
		return config.ARDConfig{External: config.ARDExternalConfig{InvocationEnabled: true}}
	}))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/external.vendor.review_contract", strings.NewReader(`{"input":{"operation":"review","text":"msa"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
	var envelope ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &envelope))
	require.Equal(t, types.ExecutionStatusSucceeded, envelope.Status)
	require.Equal(t, map[string]interface{}{"decision": "approved"}, envelope.Result)
	require.NotEmpty(t, resp.Header().Get("X-Execution-ID"))
	record, err := store.GetExecutionRecord(context.Background(), envelope.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, record)
	require.Equal(t, "external", record.AgentNodeID)
	require.Equal(t, "vendor.review_contract", record.ReasonerID)
	require.Equal(t, types.ExecutionStatusSucceeded, record.Status)
	require.NotNil(t, record.CompletedAt)
	require.NotNil(t, record.ResultPayload)
}

func TestExecuteHandlerWithARD_ExternalCallableBindingGates(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newTestExecutionStorage(nil)
	state := ard.State{
		Imports: []ard.ExternalEntry{{
			ID:          "ext_1",
			Identifier:  "urn:ai:vendor.example:agent:review",
			Type:        "application/a2a-agent-card+json",
			DisplayName: "Vendor Review",
			URL:         "https://vendor.example/invoke",
		}},
		Bindings: map[string]ard.ExternalBinding{
			"ext_1": {
				ExternalEntryID:   "ext_1",
				Callable:          true,
				LocalTarget:       "external.vendor.review_contract",
				Adapter:           "a2a",
				TimeoutMS:         30000,
				AllowedOperations: []string{"review"},
			},
		},
	}
	rawState, err := json.Marshal(state)
	require.NoError(t, err)
	store.config[ard.StateConfigKey] = string(rawState)

	for _, tc := range []struct {
		name       string
		cfg        config.ARDConfig
		body       string
		wantStatus int
	}{
		{
			name:       "external invocation disabled",
			cfg:        config.ARDConfig{External: config.ARDExternalConfig{InvocationEnabled: false}},
			body:       `{"input":{"operation":"review"}}`,
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "operation not allowed",
			cfg:        config.ARDConfig{External: config.ARDExternalConfig{InvocationEnabled: true}},
			body:       `{"input":{"operation":"delete"}}`,
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "conflicting operation fields",
			cfg:        config.ARDConfig{External: config.ARDExternalConfig{InvocationEnabled: true}},
			body:       `{"input":{"operation":"delete"},"context":{"operation":"review"}}`,
			wantStatus: http.StatusBadRequest,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/api/v1/execute/:target", ExecuteHandlerWithARD(store, nil, nil, 90*time.Second, "", func() config.ARDConfig {
				return tc.cfg
			}))
			req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/external.vendor.review_contract", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, tc.wantStatus, resp.Code, resp.Body.String())
		})
	}
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
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	// Agent returned 500 → control plane returns 502 Bad Gateway with structured error details.
	require.Equal(t, http.StatusBadGateway, resp.Code)

	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "agent error (500)")
	require.Equal(t, "failed", payload["status"])
	// The agent's JSON response body is preserved as error_details.
	require.NotNil(t, payload["error_details"])

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
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

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
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil, 90*time.Second, ""))

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
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader("not-json"))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestExecuteHandler_PendingApprovalAgent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:              "node-1",
		BaseURL:         "http://agent.example",
		Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
		LifecycleStatus: types.AgentStatusPendingApproval,
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)

	// Response contract (matches reasoners.go / skills.go / permission middleware):
	//   { "error": "agent_pending_approval", "message": "<human text>", "error_category": "agent_error" }
	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "agent_pending_approval", payload["error"])
	require.Contains(t, payload["message"], "awaiting tag approval")
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
