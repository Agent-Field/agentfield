package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type fakeDIDService struct {
	registerFn func(*types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error)
	resolveFn  func(string) (*types.DIDIdentity, error)
	listFn     func() ([]string, error)
}

func (f *fakeDIDService) RegisterAgent(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	if f.registerFn != nil {
		return f.registerFn(req)
	}
	return &types.DIDRegistrationResponse{
		Success: true,
		IdentityPackage: types.DIDIdentityPackage{
			AgentDID: types.DIDIdentity{DID: "did:example:agent"},
		},
		Message: "registered",
	}, nil
}

func (f *fakeDIDService) ResolveDID(did string) (*types.DIDIdentity, error) {
	if f.resolveFn != nil {
		return f.resolveFn(did)
	}
	return &types.DIDIdentity{DID: did, PublicKeyJWK: "{\"kty\":\"OKP\"}"}, nil
}

func (f *fakeDIDService) ListAllAgentDIDs() ([]string, error) {
	if f.listFn != nil {
		return f.listFn()
	}
	return []string{"did:example:agent"}, nil
}

type fakeVCService struct {
	verifyFn          func(json.RawMessage) (*types.VCVerificationResponse, error)
	workflowChainFn   func(string) (*types.WorkflowVCChainResponse, error)
	createWorkflowFn  func(string, string, []string) (*types.WorkflowVC, error)
	generateExecFn    func(*types.ExecutionContext, []byte, []byte, string, *string, int) (*types.ExecutionVC, error)
	queryExecsFn      func(*types.VCFilters) ([]types.ExecutionVC, error)
	listWorkflowVCsFn func() ([]*types.WorkflowVC, error)
}

func (f *fakeVCService) VerifyVC(doc json.RawMessage) (*types.VCVerificationResponse, error) {
	if f.verifyFn != nil {
		return f.verifyFn(doc)
	}
	return &types.VCVerificationResponse{Valid: true}, nil
}

func (f *fakeVCService) GetWorkflowVCChain(workflowID string) (*types.WorkflowVCChainResponse, error) {
	if f.workflowChainFn != nil {
		return f.workflowChainFn(workflowID)
	}
	return &types.WorkflowVCChainResponse{WorkflowID: workflowID}, nil
}

func (f *fakeVCService) CreateWorkflowVC(workflowID, sessionID string, executionVCIDs []string) (*types.WorkflowVC, error) {
	if f.createWorkflowFn != nil {
		return f.createWorkflowFn(workflowID, sessionID, executionVCIDs)
	}
	return &types.WorkflowVC{WorkflowVCID: "workflow-vc"}, nil
}

func (f *fakeVCService) GenerateExecutionVC(ctx *types.ExecutionContext, inputData, outputData []byte, status string, errorMessage *string, durationMS int) (*types.ExecutionVC, error) {
	if f.generateExecFn != nil {
		return f.generateExecFn(ctx, inputData, outputData, status, errorMessage, durationMS)
	}
	return &types.ExecutionVC{
		VCID:        "vc-1",
		ExecutionID: ctx.ExecutionID,
		WorkflowID:  ctx.WorkflowID,
		SessionID:   ctx.SessionID,
		IssuerDID:   "did:issuer",
		TargetDID:   ctx.TargetDID,
		CallerDID:   ctx.CallerDID,
		InputHash:   "hash-in",
		OutputHash:  "hash-out",
		Status:      status,
		CreatedAt:   time.Now(),
		VCDocument:  json.RawMessage(`{"vc":true}`),
		Signature:   "sig",
	}, nil
}

func (f *fakeVCService) QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error) {
	if f.queryExecsFn != nil {
		return f.queryExecsFn(filters)
	}
	return []types.ExecutionVC{}, nil
}

func (f *fakeVCService) GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error) {
	if f.queryExecsFn != nil {
		filters := &types.VCFilters{ExecutionID: &executionID, Limit: 1}
		results, err := f.queryExecsFn(filters)
		if err != nil {
			return nil, err
		}
		if len(results) == 0 {
			return nil, fmt.Errorf("not found")
		}
		return &results[0], nil
	}
	return &types.ExecutionVC{VCID: "vc-test", ExecutionID: executionID}, nil
}

func (f *fakeVCService) ListWorkflowVCs() ([]*types.WorkflowVC, error) {
	if f.listWorkflowVCsFn != nil {
		return f.listWorkflowVCsFn()
	}
	return []*types.WorkflowVC{}, nil
}

func (f *fakeVCService) ListAgentTagVCs() ([]*types.AgentTagVCRecord, error) {
	return []*types.AgentTagVCRecord{}, nil
}

func TestRegisterAgentHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})

	router := gin.New()
	router.POST("/api/v1/did/register", handler.RegisterAgent)

	reqBody := `{"agent_node_id":"node-1"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/did/register", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload types.DIDRegistrationResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.True(t, payload.Success)
	require.Equal(t, "did:example:agent", payload.IdentityPackage.AgentDID.DID)
}

func TestVerifyVCHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})

	router := gin.New()
	router.POST("/api/v1/did/verify", handler.VerifyVC)

	reqBody := `{"vc_document": {"issuer": "did:example:issuer"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/did/verify", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload types.VCVerificationResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.True(t, payload.Valid)
}

func TestResolveDIDHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	router := gin.New()
	router.GET("/api/v1/did/resolve/:did", handler.ResolveDID)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/resolve/did:example:123", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "did:example:123", payload["did"])
}

type fakeDIDWebService struct {
	resolveFn func(ctx context.Context, did string) (*types.DIDResolutionResult, error)
}

func (f *fakeDIDWebService) ResolveDID(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
	if f.resolveFn != nil {
		return f.resolveFn(ctx, did)
	}
	return &types.DIDResolutionResult{
		DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "notFound"},
	}, nil
}

func TestResolveDIDHandler_DIDWeb(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	handler.SetDIDWebService(&fakeDIDWebService{
		resolveFn: func(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
			return &types.DIDResolutionResult{
				DIDDocument: &types.DIDWebDocument{
					Context: []string{"https://www.w3.org/ns/did/v1"},
					ID:      did,
					VerificationMethod: []types.VerificationMethod{{
						ID:           did + "#key-1",
						Type:         "Ed25519VerificationKey2020",
						Controller:   did,
						PublicKeyJwk: json.RawMessage(`{"kty":"OKP","crv":"Ed25519","x":"abc"}`),
					}},
					Authentication: []string{did + "#key-1"},
				},
				DIDResolutionMetadata: types.DIDResolutionMetadata{ContentType: "application/did+ld+json"},
			}, nil
		},
	})

	router := gin.New()
	router.GET("/api/v1/did/resolve/:did", handler.ResolveDID)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/resolve/did:web:localhost%3A8080:agents:test-agent", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["did"], "did:web:")
	require.NotNil(t, payload["did_document"])
}

func TestResolveDIDHandler_DIDWebRevoked(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	handler.SetDIDWebService(&fakeDIDWebService{
		resolveFn: func(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
			return &types.DIDResolutionResult{
				DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "deactivated"},
				DIDDocumentMetadata:   types.DIDDocumentMetadata{Deactivated: true},
			}, nil
		},
	})

	router := gin.New()
	router.GET("/api/v1/did/resolve/:did", handler.ResolveDID)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/resolve/did:web:localhost%3A8080:agents:revoked-agent", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusGone, resp.Code)
}

func TestGetWorkflowVCChainHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{
		workflowChainFn: func(id string) (*types.WorkflowVCChainResponse, error) {
			return &types.WorkflowVCChainResponse{WorkflowID: id, TotalSteps: 3}, nil
		},
	})

	router := gin.New()
	router.GET("/api/v1/did/workflow/:workflow_id/vc-chain", handler.GetWorkflowVCChain)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/workflow/wf-1/vc-chain", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload types.WorkflowVCChainResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "wf-1", payload.WorkflowID)
	require.Equal(t, 3, payload.TotalSteps)
}

func TestCreateWorkflowVCHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{
		createWorkflowFn: func(workflowID, sessionID string, execIDs []string) (*types.WorkflowVC, error) {
			return &types.WorkflowVC{WorkflowVCID: "workflow-vc", WorkflowID: workflowID}, nil
		},
	})

	router := gin.New()
	router.POST("/api/v1/did/workflow/:workflow_id/vc", handler.CreateWorkflowVC)

	body := `{"session_id":"sess-1","execution_vc_ids":["vc-a"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/did/workflow/wf-2/vc", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload types.WorkflowVC
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "workflow-vc", payload.WorkflowVCID)
}

func TestExportVCsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{
		queryExecsFn: func(filters *types.VCFilters) ([]types.ExecutionVC, error) {
			return []types.ExecutionVC{{VCID: "vc-1", ExecutionID: "exec-1", WorkflowID: "wf-1", CreatedAt: time.Now()}}, nil
		},
		listWorkflowVCsFn: func() ([]*types.WorkflowVC, error) {
			return []*types.WorkflowVC{{WorkflowVCID: "wvc-1", WorkflowID: "wf-1"}}, nil
		},
	})

	router := gin.New()
	router.GET("/api/v1/did/export/vcs", handler.ExportVCs)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/export/vcs", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, float64(2), payload["total_count"])
	// D11 fix: Verify agent_tag_vcs field is present in export
	require.Contains(t, payload, "agent_tag_vcs", "export should include agent_tag_vcs field")
}

func TestGetDIDStatusHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	router := gin.New()
	router.GET("/api/v1/did/status", handler.GetDIDStatus)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
}

func TestGetDIDDocumentHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	router := gin.New()
	router.GET("/api/v1/did/document/:did", handler.GetDIDDocument)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/document/did:example:doc", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "did:example:doc", payload["id"])
}

func TestCreateExecutionVC_ReturnsVCInfo(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})

	router := gin.New()
	router.POST("/api/v1/execution/vc", handler.CreateExecutionVC)

	reqBody := `{
        "execution_context": {
            "execution_id": "exec-1",
            "workflow_id": "wf-1",
            "session_id": "sess-1",
            "caller_did": "did:caller",
            "target_did": "did:target",
            "agent_node_did": "did:agent",
            "timestamp": "2023-01-02T15:04:05Z"
        },
        "input_data": "aW5wdXQ=",
        "output_data": "b3V0cHV0",
        "status": "succeeded",
        "duration_ms": 123
    }`

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execution/vc", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "vc-1", payload["vc_id"])
	require.Equal(t, "exec-1", payload["execution_id"])
}
