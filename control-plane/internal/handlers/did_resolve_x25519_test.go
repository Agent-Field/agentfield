package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestResolveDIDHandler_X25519KeyAgreement exercises the full HTTP + registration
// plumbing for the X25519 keyAgreement key: it registers an agent against a REAL
// DIDService, then resolves that agent's did:key over the GET
// /api/v1/did/resolve/:did handler. It asserts the JSON response carries a
// `key_agreement` object that is a valid X25519 public JWK (crv == "X25519",
// non-empty `x`) and — critically — does NOT leak the private scalar `d`.
//
// This is requirement (2) of the X25519 plumbing contract.
func TestResolveDIDHandler_X25519KeyAgreement(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Build a real DIDService so the resolve handler returns a genuine,
	// derived X25519 keyAgreement key (the fake service would bypass the
	// derivation/serialization plumbing under test).
	provider, _ := setupTestStorage(t)
	registry := services.NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := services.NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	didService := services.NewDIDService(cfg, ks, registry)
	require.NoError(t, didService.Initialize("agentfield-handler-x25519"))

	// Register an agent and capture its did:key.
	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-resolve-x25519",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)
	agentDID := regResp.IdentityPackage.AgentDID.DID
	require.NotEmpty(t, agentDID)

	handler := NewDIDHandlers(didService, &fakeVCService{})
	router := gin.New()
	router.GET("/api/v1/did/resolve/:did", handler.ResolveDID)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/resolve/"+agentDID, nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, agentDID, payload["did"])

	// The resolve response must expose the keyAgreement public key.
	keyAgreementRaw, ok := payload["key_agreement"]
	require.True(t, ok, "resolve response must include a key_agreement object")
	keyAgreement, ok := keyAgreementRaw.(map[string]any)
	require.True(t, ok, "key_agreement must be a JSON object")

	require.Equal(t, "X25519", keyAgreement["crv"], "key_agreement crv must be X25519")
	x, ok := keyAgreement["x"].(string)
	require.True(t, ok, "key_agreement must have a string `x`")
	require.NotEmpty(t, x, "key_agreement `x` (public key) must be non-empty")

	// The resolve endpoint must NEVER leak the private scalar `d`.
	_, hasD := keyAgreement["d"]
	require.False(t, hasD, "key_agreement must NOT contain the private `d` component")
}
