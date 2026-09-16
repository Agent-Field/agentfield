package ui

import (
	"net/http"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// wf-1 in seedDIDHandlerData references caller/target DIDs that are not
// registered anywhere, so the real VC service cannot bundle their keys.
func TestUIWorkflowVCChainHandlerReportsUnresolvedDIDs(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ls, registry, didService, vcService, didWebService, ctx := setupDIDHandlerServices(t)
	seedDIDHandlerData(t, ls, registry, vcService, didWebService, ctx)

	handler := NewDIDHandler(ls, didService, vcService, didWebService)
	router := gin.New()
	router.GET("/api/ui/v1/workflows/:workflowId/vc-chain", handler.GetWorkflowVCChainHandler)

	rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/workflows/wf-1/vc-chain", nil)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "false", rec.Header().Get(handlers.VCChainVerifiableHeader))

	body := decodeJSONResponse[map[string]any](t, rec)
	require.Equal(t, "wf-1", body["workflow_id"])
	require.Equal(t, false, body["verifiable"])
	unresolved, ok := body["unresolved_dids"].([]any)
	require.True(t, ok, "unresolved_dids should be a list")
	require.Contains(t, unresolved, "did:caller:client")
	require.Contains(t, unresolved, "did:target:node-1")
}
