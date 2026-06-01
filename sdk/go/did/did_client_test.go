package did

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	c := NewClient("http://localhost:8080")
	assert.NotNil(t, c)
	assert.Equal(t, "http://localhost:8080", c.baseURL)
	assert.NotNil(t, c.httpClient)
}

func TestNewClient_TrimsTrailingSlash(t *testing.T) {
	c := NewClient("http://localhost:8080/")
	assert.Equal(t, "http://localhost:8080", c.baseURL)
}

func TestClient_RegisterAgent_Success(t *testing.T) {
	identityPkg := DIDIdentityPackage{
		AgentDID: DIDIdentity{
			DID:           "did:web:localhost:agents:test-agent",
			PrivateKeyJWK: `{"kty":"OKP","crv":"Ed25519","d":"dGVzdC1wcml2YXRlLWtleS1zZWVkMDAwMDAwMA","x":"dGVzdC1wdWJsaWMta2V5LXZhbHVl"}`,
			PublicKeyJWK:  `{"kty":"OKP","crv":"Ed25519","x":"dGVzdC1wdWJsaWMta2V5LXZhbHVl"}`,
			ComponentType: "agent",
		},
		ReasonerDIDs: map[string]DIDIdentity{
			"greet": {
				DID:           "did:web:localhost:agents:test-agent:reasoners:greet",
				ComponentType: "reasoner",
				FunctionName:  "greet",
			},
		},
		SkillDIDs:          map[string]DIDIdentity{},
		AgentFieldServerID: "localhost:8080",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/did/register", r.URL.Path)
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req RegistrationRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "test-agent", req.AgentNodeID)
		assert.Len(t, req.Reasoners, 1)
		assert.Equal(t, "greet", req.Reasoners[0].ID)

		resp := RegistrationResponse{
			Success:         true,
			IdentityPackage: identityPkg,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	c := NewClient(server.URL)
	resp, err := c.RegisterAgent(context.Background(), RegistrationRequest{
		AgentNodeID: "test-agent",
		Reasoners:   []FunctionDef{{ID: "greet"}},
		Skills:      []FunctionDef{},
	})

	require.NoError(t, err)
	assert.True(t, resp.Success)
	assert.Equal(t, "did:web:localhost:agents:test-agent", resp.IdentityPackage.AgentDID.DID)
	assert.Contains(t, resp.IdentityPackage.ReasonerDIDs, "greet")
}

func TestClient_RegisterAgent_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer server.Close()

	c := NewClient(server.URL)
	resp, err := c.RegisterAgent(context.Background(), RegistrationRequest{
		AgentNodeID: "test-agent",
	})

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "500")
}

func TestClient_RegisterAgent_SuccessFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := RegistrationResponse{
			Success: false,
			Error:   "agent already registered with conflicting config",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	c := NewClient(server.URL)
	resp, err := c.RegisterAgent(context.Background(), RegistrationRequest{
		AgentNodeID: "test-agent",
	})

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "agent already registered")
}

func TestClient_GenerateExecutionVC_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/execution/vc", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req VCGenerationRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "exec-123", req.ExecutionContext.ExecutionID)
		assert.Equal(t, "succeeded", req.Status)

		vc := ExecutionVC{
			VCID:        "vc-456",
			ExecutionID: "exec-123",
			WorkflowID:  "wf-789",
			IssuerDID:   "did:web:localhost:agents:caller",
			TargetDID:   "did:web:localhost:agents:target",
			Status:      "completed",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vc)
	}))
	defer server.Close()

	c := NewClient(server.URL)
	vc, err := c.GenerateExecutionVC(context.Background(), VCGenerationRequest{
		ExecutionContext: ExecutionContext{
			ExecutionID: "exec-123",
			WorkflowID:  "wf-789",
		},
		Status: "succeeded",
	})

	require.NoError(t, err)
	assert.Equal(t, "vc-456", vc.VCID)
	assert.Equal(t, "exec-123", vc.ExecutionID)
}

func TestClient_GenerateExecutionVC_WithDIDAuth(t *testing.T) {
	var receivedDID string
	var receivedSig string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedDID = r.Header.Get("X-Caller-DID")
		receivedSig = r.Header.Get("X-DID-Signature")

		vc := ExecutionVC{VCID: "vc-signed"}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vc)
	}))
	defer server.Close()

	c := NewClient(server.URL)
	c.SetSignFunc(func(body []byte) map[string]string {
		return map[string]string{
			"X-Caller-DID":    "did:web:test-agent",
			"X-DID-Signature": "test-signature",
			"X-DID-Timestamp": "1234567890",
		}
	})

	vc, err := c.GenerateExecutionVC(context.Background(), VCGenerationRequest{
		ExecutionContext: ExecutionContext{ExecutionID: "exec-1"},
		Status:           "succeeded",
	})

	require.NoError(t, err)
	assert.Equal(t, "vc-signed", vc.VCID)
	assert.Equal(t, "did:web:test-agent", receivedDID)
	assert.Equal(t, "test-signature", receivedSig)
}

func TestClient_ExportWorkflowVCChain_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/did/workflow/wf-123/vc-chain", r.URL.Path)
		assert.Equal(t, "GET", r.Method)

		chain := WorkflowVCChain{
			WorkflowID: "wf-123",
			ExecutionVCs: []ExecutionVC{
				{VCID: "vc-1", ExecutionID: "exec-1"},
				{VCID: "vc-2", ExecutionID: "exec-2"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chain)
	}))
	defer server.Close()

	c := NewClient(server.URL)
	chain, err := c.ExportWorkflowVCChain(context.Background(), "wf-123")

	require.NoError(t, err)
	assert.Equal(t, "wf-123", chain.WorkflowID)
	assert.Len(t, chain.ExecutionVCs, 2)
}

func TestClient_WithBearerToken(t *testing.T) {
	var receivedAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		resp := RegistrationResponse{
			Success: true,
			IdentityPackage: DIDIdentityPackage{
				AgentDID: DIDIdentity{DID: "did:web:test"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	c := NewClient(server.URL, WithToken("my-secret-token"))
	_, err := c.RegisterAgent(context.Background(), RegistrationRequest{AgentNodeID: "test"})

	require.NoError(t, err)
	assert.Equal(t, "Bearer my-secret-token", receivedAuth)
}

func TestClient_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never respond — client should cancel
		select {}
	}))
	defer server.Close()

	c := NewClient(server.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := c.RegisterAgent(ctx, RegistrationRequest{AgentNodeID: "test"})
	assert.Error(t, err)
}
