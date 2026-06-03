package ui

import (
	"context"
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

func TestDIDHelpersCoverage(t *testing.T) {
	require.NotNil(t, NewDIDHandler(nil, nil, nil, nil))
	require.True(t, contains("did:agent-1", "did"))
	require.True(t, contains("abc", "abc"))
	require.False(t, contains("", "abc"))
	require.True(t, parseTime("2026-04-08T12:00:00Z").Equal(time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)))
	require.True(t, parseTime("").IsZero())
	require.True(t, hasFailedVCs([]types.ExecutionVC{{Status: "failed"}}))
	require.False(t, hasFailedVCs([]types.ExecutionVC{{Status: "completed"}}))
	require.Equal(t, "issuer", getDIDRole("did:issuer", types.ExecutionVC{IssuerDID: "did:issuer"}))
	require.Equal(t, "target", getDIDRole("did:target", types.ExecutionVC{TargetDID: "did:target"}))
	require.Equal(t, "caller", getDIDRole("did:caller", types.ExecutionVC{CallerDID: "did:caller"}))
	require.Equal(t, "unknown", getDIDRole("did:none", types.ExecutionVC{}))
	require.NotContains(t, sanitizeDIDForFilename(`did:web:example.com/path?x=1`), "/")
	require.LessOrEqual(t, len(sanitizeDIDForFilename(string(make([]byte, 150)))), 100)
}

func TestDIDHandlerNilServiceCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := setupTestStorage(t)
	handler := NewDIDHandler(store, nil, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/nodes/:nodeId/did", handler.GetNodeDIDHandler)
	router.GET("/api/ui/v1/nodes/:nodeId/vc-status", handler.GetNodeVCStatusHandler)
	router.GET("/api/ui/v1/executions/:executionId/vc-status", handler.GetExecutionVCStatusHandler)
	router.GET("/api/ui/v1/executions/:executionId/vc", handler.GetExecutionVCHandler)
	router.POST("/api/ui/v1/workflows/vc-status", handler.GetWorkflowVCStatusBatchHandler)
	router.GET("/api/ui/v1/workflows/:workflowId/vc-chain", handler.GetWorkflowVCChainHandler)
	router.GET("/api/ui/v1/vc/:vc_id/download", handler.DownloadVCHandler)
	router.POST("/api/ui/v1/vc/verify", handler.VerifyVCHandler)
	router.POST("/api/ui/v1/executions/:executionId/verify-vc", handler.VerifyExecutionVCComprehensiveHandler)
	router.POST("/api/ui/v1/workflows/:workflowId/verify-vc", handler.VerifyWorkflowVCComprehensiveHandler)
	router.GET("/api/ui/v1/did/export/vcs", handler.ExportVCsHandler)
	router.GET("/api/ui/v1/did/status", handler.GetDIDSystemStatusHandler)
	router.GET("/api/ui/v1/did/:did/resolution-bundle", handler.GetDIDResolutionBundleHandler)
	router.GET("/api/ui/v1/did/:did/resolution-bundle/download", handler.DownloadDIDResolutionBundleHandler)

	rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/did", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/vc-status", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-1/vc-status", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-1/vc", nil)
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)

	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/workflows/vc-status", map[string]any{})
	require.Equal(t, http.StatusBadRequest, rec.Code)
	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/workflows/vc-status", map[string]any{"workflow_ids": []string{"wf-1", "wf-2"}})
	require.Equal(t, http.StatusOK, rec.Code)
	var batch struct {
		Summaries []types.WorkflowVCStatusSummary `json:"summaries"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &batch))
	require.Len(t, batch.Summaries, 2)

	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/workflows/wf-1/vc-chain", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/vc/test-vc/download", nil)
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)
	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/vc/verify", map[string]any{"vc_document": map[string]any{"id": "vc-1"}})
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)
	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/exec-1/verify-vc", nil)
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)
	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/workflows/wf-1/verify-vc", nil)
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/export/vcs", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/status", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/did:agent:alpha/resolution-bundle", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/did:agent:alpha/resolution-bundle/download", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Contains(t, rec.Header().Get("Content-Disposition"), "attachment;")
}

func TestDIDHandlerDirectContextValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := NewDIDHandler(nil, nil, nil, nil)

	tests := []struct {
		name   string
		call   func(*gin.Context)
		status int
	}{
		{name: "missing node id", call: handler.GetNodeDIDHandler, status: http.StatusBadRequest},
		{name: "missing node vc id", call: handler.GetNodeVCStatusHandler, status: http.StatusBadRequest},
		{name: "missing execution status id", call: handler.GetExecutionVCStatusHandler, status: http.StatusBadRequest},
		{name: "missing execution vc id", call: handler.GetExecutionVCHandler, status: http.StatusBadRequest},
		{name: "missing workflow chain id", call: handler.GetWorkflowVCChainHandler, status: http.StatusBadRequest},
		{name: "missing vc download id", call: handler.DownloadVCHandler, status: http.StatusBadRequest},
		{name: "verify vc invalid body", call: handler.VerifyVCHandler, status: http.StatusBadRequest},
		{name: "missing execution verify id", call: handler.VerifyExecutionVCComprehensiveHandler, status: http.StatusBadRequest},
		{name: "missing workflow verify id", call: handler.VerifyWorkflowVCComprehensiveHandler, status: http.StatusBadRequest},
		{name: "missing did bundle id", call: handler.GetDIDResolutionBundleHandler, status: http.StatusBadRequest},
		{name: "missing did download id", call: handler.DownloadDIDResolutionBundleHandler, status: http.StatusBadRequest},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(rec)
			ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
			if tc.name == "verify vc invalid body" {
				ctx.Request = httptest.NewRequest(http.MethodPost, "/", httptest.NewRecorder().Body)
			}
			tc.call(ctx)
			require.Equal(t, tc.status, rec.Code)
		})
	}
}

func TestDIDResolutionBundleResolvedCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := setupTestStorage(t)
	ctx := context.Background()
	require.NoError(t, store.StoreAgentFieldServerDID(ctx, "did:af:server", "did:root:server", []byte("seed"), time.Now().UTC(), time.Now().UTC()))
	require.NoError(t, store.StoreAgentDID(ctx, "agent-alpha", "did:af:agent-alpha", "did:af:server", `{"kty":"OKP"}`, 1))
	require.NoError(t, store.StoreComponentDID(ctx, "reasoner.alpha", "did:af:reasoner-alpha", "did:af:agent-alpha", "reasoner", "alpha", 2))
	require.NoError(t, store.StoreComponentDID(ctx, "skill.alpha", "did:af:skill-alpha", "did:af:agent-alpha", "skill", "alpha-skill", 3))
	require.NoError(t, store.StoreExecutionVC(ctx,
		"vc-1", "exec-1", "wf-1", "session-1",
		"did:af:agent-alpha", "did:target", "did:caller",
		"in", "out", "completed", []byte(`{"id":"vc-1"}`), "sig", "", 16,
	))

	handler := NewDIDHandler(store, &services.DIDService{}, nil, nil)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/did:af:agent-alpha/resolution-bundle", nil)
	c.Params = gin.Params{{Key: "did", Value: "did:af:agent-alpha"}}
	handler.GetDIDResolutionBundleHandler(c)
	require.Equal(t, http.StatusOK, rec.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, "resolved", body["resolution_status"])
	require.NotNil(t, body["did_document"])

	rec = httptest.NewRecorder()
	c, _ = gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/did:af:agent-alpha/resolution-bundle/download", nil)
	c.Params = gin.Params{{Key: "did", Value: "did:af:agent-alpha"}}
	handler.DownloadDIDResolutionBundleHandler(c)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Contains(t, rec.Header().Get("Content-Disposition"), "did-resolution-bundle-")
}
