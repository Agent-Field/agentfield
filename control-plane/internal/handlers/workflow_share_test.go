package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGetWorkflowShareHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)
	provider, ctx := setupTestStorage(t)

	runID := "run-share-1"
	rootID := "exec-root"
	childID := "exec-child"
	start := time.Now().UTC()
	completed := start.Add(2 * time.Second)
	dur := int64(2000)

	require.NoError(t, provider.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:   rootID,
		RunID:         runID,
		AgentNodeID:   "orchestrator",
		ReasonerID:    "run_audit",
		Status:        "succeeded",
		StartedAt:     start,
		CompletedAt:   &completed,
		DurationMS:    &dur,
		InputPayload:  json.RawMessage(`{"target":"acme"}`),
		ResultPayload: json.RawMessage(`{"findings":3}`),
	}))
	childParent := rootID
	childErr := "boom"
	require.NoError(t, provider.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:       childID,
		RunID:             runID,
		ParentExecutionID: &childParent,
		AgentNodeID:       "hunter",
		ReasonerID:        "sql_injection",
		Status:            "failed",
		StartedAt:         start.Add(1 * time.Second),
		InputPayload:      json.RawMessage(`{"endpoints":18}`),
		ErrorMessage:      &childErr,
	}))

	router := gin.New()
	router.GET(
		"/api/ui/v1/workflows/:workflowId/share",
		GetWorkflowShareHandler(provider, services.NewFilePayloadStore(t.TempDir())),
	)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflows/"+runID+"/share", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Contains(t, resp.Header().Get("Content-Type"), "text/html")
	require.Equal(t, `attachment; filename="run-share-1.html"`, resp.Header().Get("Content-Disposition"))

	body := resp.Body.String()
	require.Contains(t, body, "<html", "response should be an HTML document")
	// The bundle JSON is inlined into the template; the previews and the run
	// structure must be present in the artifact.
	require.Contains(t, body, "run_audit")
	require.Contains(t, body, "sql_injection")
	require.Contains(t, body, "findings")
	require.Contains(t, body, "boom", "error message should be inlined")
}

func TestGetWorkflowShareHandler_Redact(t *testing.T) {
	gin.SetMode(gin.TestMode)
	provider, ctx := setupTestStorage(t)

	runID := "run-share-redact"
	start := time.Now().UTC()
	require.NoError(t, provider.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:   "exec-1",
		RunID:         runID,
		AgentNodeID:   "orchestrator",
		ReasonerID:    "run_audit",
		Status:        "succeeded",
		StartedAt:     start,
		InputPayload:  json.RawMessage(`{"secret":"topsecretvalue"}`),
		ResultPayload: json.RawMessage(`{"leaked":"anothersecret"}`),
	}))

	router := gin.New()
	router.GET(
		"/api/ui/v1/workflows/:workflowId/share",
		GetWorkflowShareHandler(provider, nil),
	)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflows/"+runID+"/share?redact=1", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	body := resp.Body.String()
	require.Contains(t, body, "[redacted]")
	require.NotContains(t, body, "topsecretvalue")
	require.NotContains(t, body, "anothersecret")
}

func TestGetWorkflowShareHandler_NotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	provider, _ := setupTestStorage(t)

	router := gin.New()
	router.GET(
		"/api/ui/v1/workflows/:workflowId/share",
		GetWorkflowShareHandler(provider, nil),
	)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflows/does-not-exist/share", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusNotFound, resp.Code)
	require.Contains(t, resp.Header().Get("Content-Type"), "application/json")

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.NotEmpty(t, payload["error"])
}

func TestShareFilename(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"run-abc123", "run-abc123.html"},
		{"abc123", "run-abc123.html"},
		{"run-a/b:c d", "run-a-b-c-d.html"},
		{"", "run-run.html"},
	}
	for _, tt := range tests {
		require.Equal(t, tt.want, shareFilename(tt.in))
	}
}
