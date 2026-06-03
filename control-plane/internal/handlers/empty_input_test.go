package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type reasonerTestStorage struct {
	storage.StorageProvider
	agent      *types.AgentNode
	executions []*types.WorkflowExecution
}

func (s *reasonerTestStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.agent != nil && s.agent.ID == id {
		return s.agent, nil
	}
	return nil, nil
}

func (s *reasonerTestStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	if execution != nil {
		copy := *execution
		s.executions = append(s.executions, &copy)
	}
	return nil
}

// TestExecuteHandler_EmptyInput verifies that calling the execute endpoint with
// an empty input object ({"input":{}}) succeeds instead of returning 400.
// Reproduction for https://github.com/Agent-Field/agentfield/issues/196.
func TestExecuteHandler_EmptyInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		defer r.Body.Close()
		require.JSONEq(t, `{}`, string(body))

		// Agent receives the (empty) input and returns success
		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(body, &payload))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "ping"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

	// Empty input object — should be accepted for parameterless skills
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.ping",
		strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code, "empty input {} should be accepted, got: %s", resp.Body.String())

	var envelope ExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &envelope))
	require.Equal(t, types.ExecutionStatusSucceeded, envelope.Status)
}

// TestExecuteHandler_NilInput verifies that calling the execute endpoint with
// no input field at all ({}) succeeds — the handler should default to empty map.
func TestExecuteHandler_NilInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		defer r.Body.Close()
		require.JSONEq(t, `{}`, string(body))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "ping"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

	// No input field at all — should default to empty map
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.ping",
		strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code, "missing input field should be accepted, got: %s", resp.Body.String())
}

// TestExecuteRequest_BindingAcceptsEmptyInput directly tests that the ExecuteRequest
// struct's binding tags accept empty and nil input (unit-level binding test).
func TestExecuteRequest_BindingAcceptsEmptyInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name string
		body string
	}{
		{"empty input object", `{"input":{}}`},
		{"no input field", `{}`},
		{"input with context", `{"input":{},"context":{"session":"abc"}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/", strings.NewReader(tt.body))
			c.Request.Header.Set("Content-Type", "application/json")

			var req ExecuteRequest
			err := c.ShouldBindJSON(&req)
			require.NoError(t, err, "binding should accept %s", tt.name)
		})
	}
}

// TestExecuteReasonerRequest_BindingAcceptsEmptyInput directly tests that the
// ExecuteReasonerRequest struct's binding tags accept empty and nil input.
func TestExecuteReasonerRequest_BindingAcceptsEmptyInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name string
		body string
	}{
		{"empty input object", `{"input":{}}`},
		{"no input field", `{}`},
		{"input with context", `{"input":{},"context":{"session":"abc"}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/", strings.NewReader(tt.body))
			c.Request.Header.Set("Content-Type", "application/json")

			var req ExecuteReasonerRequest
			err := c.ShouldBindJSON(&req)
			require.NoError(t, err, "binding should accept %s", tt.name)
		})
	}
}

// TestExecuteRequest_BindingStillRequiresJSON verifies that completely invalid
// input is still rejected (the fix doesn't break validation entirely).
func TestExecuteRequest_BindingStillRequiresJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/", strings.NewReader("not-json"))
	c.Request.Header.Set("Content-Type", "application/json")

	var req ExecuteRequest
	err := c.ShouldBindJSON(&req)
	require.Error(t, err, "invalid JSON should still be rejected")
}

func TestExecuteReasonerHandler_EmptyAndNilInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name string
		body string
	}{
		{name: "empty input object", body: `{"input":{}}`},
		{name: "nil input object", body: `{}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/reasoners/ping", r.URL.Path)
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				defer r.Body.Close()
				require.JSONEq(t, `{}`, string(body))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"ok":true}`))
			}))
			defer agentServer.Close()

			store := &reasonerTestStorage{agent: &types.AgentNode{
				ID:              "node-1",
				BaseURL:         agentServer.URL,
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusReady,
				Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
			}}

			router := gin.New()
			router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

			req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
			require.NotEmpty(t, store.executions)
			require.JSONEq(t, `{}`, string(store.executions[0].InputData))
		})
	}
}

func TestExecuteSkillHandler_EmptyAndNilInput(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name string
		body string
	}{
		{name: "empty input object", body: `{"input":{}}`},
		{name: "nil input object", body: `{}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/skills/list_items", r.URL.Path)
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				defer r.Body.Close()
				require.JSONEq(t, `{}`, string(body))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"ok":true}`))
			}))
			defer agentServer.Close()

			store := &reasonerTestStorage{agent: &types.AgentNode{
				ID:              "node-1",
				BaseURL:         agentServer.URL,
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusReady,
				Skills:          []types.SkillDefinition{{ID: "list_items"}},
			}}

			router := gin.New()
			router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

			req := httptest.NewRequest(http.MethodPost, "/skills/node-1.list_items", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
			require.NotEmpty(t, store.executions)
			require.JSONEq(t, `{}`, string(store.executions[0].InputData))
		})
	}
}
