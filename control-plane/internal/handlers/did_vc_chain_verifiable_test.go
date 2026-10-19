package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGetWorkflowVCChainHandlerReportsVerifiability(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		chain      types.WorkflowVCChainResponse
		wantHeader string
	}{
		{
			name: "unverifiable chain",
			chain: types.WorkflowVCChainResponse{
				UnresolvedDIDs: []string{"did:web:example.com:agents:gone"},
				Verifiable:     false,
			},
			wantHeader: "false",
		},
		{
			name:       "verifiable chain",
			chain:      types.WorkflowVCChainResponse{Verifiable: true},
			wantHeader: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{
				workflowChainFn: func(id string) (*types.WorkflowVCChainResponse, error) {
					chain := tc.chain
					chain.WorkflowID = id
					return &chain, nil
				},
			})
			router := gin.New()
			router.GET("/api/v1/did/workflow/:workflow_id/vc-chain", handler.GetWorkflowVCChain)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/v1/did/workflow/wf-1/vc-chain", nil))

			require.Equal(t, http.StatusOK, resp.Code)
			require.Equal(t, tc.wantHeader, resp.Header().Get(VCChainVerifiableHeader))

			var body map[string]any
			require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
			require.Equal(t, tc.chain.Verifiable, body["verifiable"])
			if len(tc.chain.UnresolvedDIDs) > 0 {
				require.Equal(t, []any{"did:web:example.com:agents:gone"}, body["unresolved_dids"])
			} else {
				require.NotContains(t, body, "unresolved_dids")
			}
		})
	}
}
