package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/did"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInitializeDIDSystem_UsesTokenAndWrapsRegistrationError(t *testing.T) {
	var authHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The DID client should forward the configured bearer token.
		authHeader = r.Header.Get("Authorization")
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		EnableDID:     true,
		Token:         "secret",
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(context.Context, map[string]any) (any, error) { return nil, nil })

	err = a.initializeDIDSystem(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "DID registration:")
	assert.Equal(t, "Bearer secret", authHeader)
}

func TestInitializeDIDSystem_InvalidRegisteredCredentialsReturnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/did/register", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		// Return a malformed key so SetDIDCredentials fails after registration.
		require.NoError(t, json.NewEncoder(w).Encode(did.RegistrationResponse{
			Success: true,
			IdentityPackage: did.DIDIdentityPackage{
				AgentDID: did.DIDIdentity{
					DID:           "did:web:example.com:agents:test-agent",
					PrivateKeyJWK: "{invalid",
				},
			},
		}))
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		EnableDID:     true,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(context.Context, map[string]any) (any, error) { return nil, nil })

	err = a.initializeDIDSystem(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "set DID credentials:")
}

func TestDIDHelpers_GuardsAndOverrides(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	assert.Nil(t, a.DIDManager())
	assert.Nil(t, a.VCGenerator())
	assert.False(t, a.shouldGenerateVC(nil))

	// fillDIDContext should not overwrite an already-populated DID.
	ec := ExecutionContext{AgentNodeDID: "existing"}
	a.fillDIDContext(&ec)
	assert.Equal(t, "existing", ec.AgentNodeDID)

	a.didManager = did.NewManager(nil, log.New(io.Discard, "", 0))
	a.fillDIDContext(&ec)
	assert.Equal(t, "existing", ec.AgentNodeDID)

	a.maybeGenerateVC(ExecutionContext{ExecutionID: "exec-1"}, nil, nil, "succeeded", "", 0, nil)
}
