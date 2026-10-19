package ui

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
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

// C1/F1 uses the real service and both API routes, then the offline CLI verifier.
func TestC1F1HistoricalWorkflowChainRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, kind := range []string{"active-issuer", "retired-component", "inactive-registry", "revoked-agent"} {
		t.Run(kind, func(t *testing.T) {
			ls, registry, didService, vcService, didWeb, _ := setupDIDHandlerServices(t)
			first, err := didService.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
			require.NoError(t, err)
			require.True(t, first.Success, first.Error)
			issuer := first.IdentityPackage.ReasonerDIDs["r"].DID
			exec, err := vcService.GenerateExecutionVC(&types.ExecutionContext{
				ExecutionID: "exec", WorkflowID: "wf", SessionID: "session", CallerDID: issuer,
				AgentNodeDID: first.IdentityPackage.AgentDID.DID, Timestamp: time.Now(),
			}, []byte(`{}`), []byte(`{}`), "succeeded", nil, 1)
			require.NoError(t, err)
			require.NotNil(t, exec)
			switch kind {
			case "retired-component":
				removed, err := didService.DeregisterComponents(&types.ComponentDeregistrationRequest{AgentNodeID: "node", ReasonerIDsToRemove: []string{"r"}})
				require.NoError(t, err)
				require.True(t, removed.Success, removed.Error)
			case "inactive-registry", "revoked-agent":
				if kind == "revoked-agent" {
					require.NoError(t, registry.UpdateAgentStatus("srv-ui", "node", types.AgentDIDStatusRevoked))
				}
				reg := services.NewDIDRegistryWithStorage(ls)
				require.NoError(t, reg.Initialize())
				serverID := "srv-ui"
				if kind == "inactive-registry" {
					serverID = "other-server"
				}
				// Reuse the configuration and keystore contract of the setup.
				cfg := &config.DIDConfig{Enabled: true}
				didService = services.NewDIDService(cfg, nil, reg)
				require.NoError(t, didService.Initialize(serverID))
				vcService = services.NewVCService(cfg, didService, ls)
			}
			wantIssuer := issuer
			if kind != "active-issuer" {
				wantIssuer, err = didService.GetControlPlaneIssuerDID()
				require.NoError(t, err)
			}
			router := gin.New()
			router.GET("/api/v1/did/workflow/:workflow_id/vc-chain", handlers.NewDIDHandlers(didService, vcService).GetWorkflowVCChain)
			router.GET("/api/ui/v1/workflows/:workflowId/vc-chain", NewDIDHandler(ls, didService, vcService, didWeb).GetWorkflowVCChainHandler)
			for _, path := range []string{"/api/v1/did/workflow/wf/vc-chain", "/api/ui/v1/workflows/wf/vc-chain"} {
				rec := performJSONRequest(router, http.MethodGet, path, nil)
				require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
				var chain types.WorkflowVCChainResponse
				require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &chain))
				require.True(t, chain.Verifiable)
				require.Empty(t, chain.UnresolvedDIDs)
				require.Equal(t, wantIssuer, chain.WorkflowVC.IssuerDID)
				require.Contains(t, chain.DIDResolutionBundle, wantIssuer)
				require.Len(t, chain.ComponentVCs, 1)
				require.Equal(t, issuer, chain.ComponentVCs[0].IssuerDID)
				require.JSONEq(t, string(exec.VCDocument), string(chain.ComponentVCs[0].VCDocument))
				var doc types.WorkflowVCDocument
				require.NoError(t, json.Unmarshal(chain.WorkflowVC.VCDocument, &doc))
				require.Equal(t, wantIssuer, doc.Issuer)
				require.Equal(t, wantIssuer+"#key-1", doc.Proof.VerificationMethod)
				result := cli.VerifyProvenanceJSON(rec.Body.Bytes(), cli.VerifyOptions{})
				require.True(t, result.Valid, "%+v", result.Comprehensive)
				require.Len(t, result.ComponentResults, 1)
				require.True(t, result.ComponentResults[0].SignatureValid)
				require.NotNil(t, result.Comprehensive.WorkflowVerification)
				require.True(t, result.Comprehensive.WorkflowVerification.SignatureValid)
			}
		})
	}
}
