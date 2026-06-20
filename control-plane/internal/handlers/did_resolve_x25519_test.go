package handlers

import (
	"bytes"
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

// TestRotateX25519KeyHandler_RoundTrip exercises the rotation endpoint against a
// REAL DIDService: it registers an agent, captures its original keyAgreement
// public key via GET resolve, POSTs to /api/v1/did/key-agreement/rotate, and
// asserts the response carries a NEW X25519 public JWK (crv X25519, non-empty
// `x`, NO private `d`). It then resolves again and asserts the SAME new public
// key is returned — proving the rotation persisted.
func TestRotateX25519KeyHandler_RoundTrip(t *testing.T) {
	gin.SetMode(gin.TestMode)

	provider, _ := setupTestStorage(t)
	registry := services.NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := services.NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	didService := services.NewDIDService(cfg, ks, registry)
	require.NoError(t, didService.Initialize("agentfield-handler-rotate"))

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-rotate-http",
		Reasoners:   []types.ReasonerDefinition{},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)
	agentDID := regResp.IdentityPackage.AgentDID.DID
	require.NotEmpty(t, agentDID)

	handler := NewDIDHandlers(didService, &fakeVCService{})
	router := gin.New()
	router.GET("/api/v1/did/resolve/:did", handler.ResolveDID)
	router.POST("/api/v1/did/key-agreement/rotate", handler.RotateX25519Key)

	// Capture the original public key via resolve.
	origX := resolveKeyAgreementX(t, router, agentDID)
	require.NotEmpty(t, origX)

	// Rotate.
	body, _ := json.Marshal(map[string]string{"did": agentDID})
	rotReq := httptest.NewRequest(http.MethodPost, "/api/v1/did/key-agreement/rotate", bytes.NewReader(body))
	rotReq.Header.Set("Content-Type", "application/json")
	rotResp := httptest.NewRecorder()
	router.ServeHTTP(rotResp, rotReq)
	require.Equal(t, http.StatusOK, rotResp.Code, rotResp.Body.String())

	var rotPayload map[string]any
	require.NoError(t, json.Unmarshal(rotResp.Body.Bytes(), &rotPayload))
	require.Equal(t, agentDID, rotPayload["did"])
	require.EqualValues(t, 1, rotPayload["epoch"], "rotation must report epoch 1")

	ka, ok := rotPayload["x25519_public_key_jwk"].(map[string]any)
	require.True(t, ok, "rotate response must include x25519_public_key_jwk object")
	require.Equal(t, "X25519", ka["crv"])
	newX, ok := ka["x"].(string)
	require.True(t, ok)
	require.NotEmpty(t, newX)
	_, hasD := ka["d"]
	require.False(t, hasD, "rotated key must NOT contain the private `d` component")
	require.NotEqual(t, origX, newX, "rotated public key must differ from the original")

	// A subsequent resolve must return the SAME new public key (persisted).
	require.Equal(t, newX, resolveKeyAgreementX(t, router, agentDID),
		"resolve after rotation must return the rotated public key")
}

// resolveKeyAgreementX GETs the resolve endpoint for did and returns the
// base64url `x` of its key_agreement public JWK.
func resolveKeyAgreementX(t *testing.T, router *gin.Engine, did string) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/resolve/"+did, nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	ka, ok := payload["key_agreement"].(map[string]any)
	require.True(t, ok, "resolve response must include key_agreement")
	x, ok := ka["x"].(string)
	require.True(t, ok)
	return x
}
