package did

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestLogger() *log.Logger {
	return log.New(os.Stdout, "[test] ", log.LstdFlags)
}

func newTestIdentityPackage() DIDIdentityPackage {
	return DIDIdentityPackage{
		AgentDID: DIDIdentity{
			DID:            "did:web:localhost:agents:test-agent",
			PrivateKeyJWK:  `{"kty":"OKP","crv":"Ed25519","d":"dGVzdC1wcml2YXRlLWtleS1zZWVkMDAwMDAwMA","x":"dGVzdC1wdWJsaWMta2V5LXZhbHVl"}`,
			PublicKeyJWK:   `{"kty":"OKP","crv":"Ed25519","x":"dGVzdC1wdWJsaWMta2V5LXZhbHVl"}`,
			DerivationPath: "m/44'/0'/0'",
			ComponentType:  "agent",
		},
		ReasonerDIDs: map[string]DIDIdentity{
			"greet": {
				DID:           "did:web:localhost:agents:test-agent:reasoners:greet",
				ComponentType: "reasoner",
				FunctionName:  "greet",
			},
			"analyze": {
				DID:           "did:web:localhost:agents:test-agent:reasoners:analyze",
				ComponentType: "reasoner",
				FunctionName:  "analyze",
			},
		},
		SkillDIDs: map[string]DIDIdentity{
			"format": {
				DID:           "did:web:localhost:agents:test-agent:skills:format",
				ComponentType: "skill",
				FunctionName:  "format",
			},
		},
		AgentFieldServerID: "localhost:8080",
	}
}

func TestManager_RegisterAgent_Success(t *testing.T) {
	identityPkg := newTestIdentityPackage()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req RegistrationRequest
		json.NewDecoder(r.Body).Decode(&req)

		assert.Equal(t, "test-agent", req.AgentNodeID)
		assert.Len(t, req.Reasoners, 2)
		assert.Len(t, req.Skills, 0)

		resp := RegistrationResponse{
			Success:         true,
			IdentityPackage: identityPkg,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	mgr := NewManager(NewClient(server.URL), newTestLogger())

	err := mgr.RegisterAgent(context.Background(), "test-agent", []string{"greet", "analyze"}, nil)
	require.NoError(t, err)

	assert.True(t, mgr.IsRegistered())
	assert.Equal(t, "did:web:localhost:agents:test-agent", mgr.GetAgentDID())
	assert.NotEmpty(t, mgr.GetAgentPrivateKeyJWK())
}

func TestManager_RegisterAgent_Failure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "server error"}`))
	}))
	defer server.Close()

	mgr := NewManager(NewClient(server.URL), newTestLogger())

	err := mgr.RegisterAgent(context.Background(), "test-agent", []string{"greet"}, nil)
	assert.Error(t, err)
	assert.False(t, mgr.IsRegistered())
	assert.Empty(t, mgr.GetAgentDID())
}

func TestManager_GetFunctionDID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := RegistrationResponse{
			Success:         true,
			IdentityPackage: newTestIdentityPackage(),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	mgr := NewManager(NewClient(server.URL), newTestLogger())
	err := mgr.RegisterAgent(context.Background(), "test-agent", []string{"greet", "analyze"}, nil)
	require.NoError(t, err)

	// Reasoner DID
	assert.Equal(t, "did:web:localhost:agents:test-agent:reasoners:greet", mgr.GetFunctionDID("greet"))
	assert.Equal(t, "did:web:localhost:agents:test-agent:reasoners:analyze", mgr.GetFunctionDID("analyze"))

	// Skill DID
	assert.Equal(t, "did:web:localhost:agents:test-agent:skills:format", mgr.GetFunctionDID("format"))

	// Unknown function falls back to agent DID
	assert.Equal(t, "did:web:localhost:agents:test-agent", mgr.GetFunctionDID("unknown"))
}

func TestManager_IsRegistered_BeforeRegistration(t *testing.T) {
	mgr := NewManager(NewClient("http://localhost:8080"), newTestLogger())
	assert.False(t, mgr.IsRegistered())
	assert.Empty(t, mgr.GetAgentDID())
	assert.Empty(t, mgr.GetAgentPrivateKeyJWK())
	assert.Empty(t, mgr.GetFunctionDID("anything"))
	assert.Nil(t, mgr.GetIdentityPackage())
}

func TestManager_SetIdentityFromCredentials(t *testing.T) {
	mgr := NewManager(NewClient("http://localhost:8080"), newTestLogger())

	assert.False(t, mgr.IsRegistered())

	mgr.SetIdentityFromCredentials("did:web:test", `{"kty":"OKP","crv":"Ed25519","d":"test"}`)

	assert.True(t, mgr.IsRegistered())
	assert.Equal(t, "did:web:test", mgr.GetAgentDID())
	assert.Equal(t, `{"kty":"OKP","crv":"Ed25519","d":"test"}`, mgr.GetAgentPrivateKeyJWK())

	// Function DID falls back to agent DID when no per-function DIDs are set.
	assert.Equal(t, "did:web:test", mgr.GetFunctionDID("any-reasoner"))
}

func TestManager_GetIdentityPackage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := RegistrationResponse{
			Success:         true,
			IdentityPackage: newTestIdentityPackage(),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	mgr := NewManager(NewClient(server.URL), newTestLogger())
	err := mgr.RegisterAgent(context.Background(), "test-agent", nil, nil)
	require.NoError(t, err)

	pkg := mgr.GetIdentityPackage()
	require.NotNil(t, pkg)
	assert.Equal(t, "did:web:localhost:agents:test-agent", pkg.AgentDID.DID)
	assert.Equal(t, "localhost:8080", pkg.AgentFieldServerID)
	assert.Len(t, pkg.ReasonerDIDs, 2)
	assert.Len(t, pkg.SkillDIDs, 1)
}

func TestManager_RegisterAgent_WithSkills(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req RegistrationRequest
		json.NewDecoder(r.Body).Decode(&req)

		assert.Equal(t, "agent-with-skills", req.AgentNodeID)
		assert.Len(t, req.Skills, 2)
		assert.Equal(t, "format", req.Skills[0].ID)
		assert.Equal(t, "validate", req.Skills[1].ID)

		resp := RegistrationResponse{
			Success:         true,
			IdentityPackage: newTestIdentityPackage(),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	mgr := NewManager(NewClient(server.URL), newTestLogger())
	err := mgr.RegisterAgent(context.Background(), "agent-with-skills", nil, []string{"format", "validate"})
	require.NoError(t, err)
	assert.True(t, mgr.IsRegistered())
}
