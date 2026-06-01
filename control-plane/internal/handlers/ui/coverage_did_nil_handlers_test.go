package ui

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestDIDHandlerNilServiceCoverageAdditional(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := &DIDHandler{}
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

	tests := []struct {
		name string
		req  *http.Request
		want int
	}{
		{name: "node did", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/node-1/did", nil), want: http.StatusOK},
		{name: "node vc status", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/node-1/vc-status", nil), want: http.StatusOK},
		{name: "execution vc status", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/vc-status", nil), want: http.StatusOK},
		{name: "execution vc unavailable", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/vc", nil), want: http.StatusServiceUnavailable},
		{name: "workflow vc batch defaults", req: httptest.NewRequest(http.MethodPost, "/api/ui/v1/workflows/vc-status", bytes.NewBufferString(`{"workflow_ids":["wf-1","wf-2"]}`)), want: http.StatusOK},
		{name: "workflow vc chain", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflows/wf-1/vc-chain", nil), want: http.StatusOK},
		{name: "download vc unavailable", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/vc/vc-1/download", nil), want: http.StatusServiceUnavailable},
		{name: "verify vc unavailable", req: httptest.NewRequest(http.MethodPost, "/api/ui/v1/vc/verify", bytes.NewBufferString(`{"vc_document":{"id":"vc-1"}}`)), want: http.StatusServiceUnavailable},
		{name: "verify execution vc unavailable", req: httptest.NewRequest(http.MethodPost, "/api/ui/v1/executions/exec-1/verify-vc", nil), want: http.StatusServiceUnavailable},
		{name: "verify workflow vc unavailable", req: httptest.NewRequest(http.MethodPost, "/api/ui/v1/workflows/wf-1/verify-vc", nil), want: http.StatusServiceUnavailable},
		{name: "export vcs", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/export/vcs?limit=5&offset=1", nil), want: http.StatusOK},
		{name: "did status", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/status", nil), want: http.StatusOK},
		{name: "resolution bundle inactive", req: httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/did:example:123/resolution-bundle", nil), want: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.req.Method == http.MethodPost {
				tt.req.Header.Set("Content-Type", "application/json")
			}
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, tt.req)
			require.Equal(t, tt.want, resp.Code)
		})
	}

	t.Run("verify vc invalid body", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/vc/verify", bytes.NewBufferString(`{`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)
	})

	t.Run("workflow vc batch invalid body", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/workflows/vc-status", bytes.NewBufferString(`{"workflow_ids":[]}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)
	})

	t.Run("response payload shapes", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/did/export/vcs", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		var body map[string]interface{}
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Equal(t, float64(0), body["total_count"])
	})
}
