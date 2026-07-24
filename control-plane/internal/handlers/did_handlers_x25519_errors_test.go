package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestRotateX25519KeyHandler_InvalidBody returns 400 on a body that is not valid
// JSON.
func TestRotateX25519KeyHandler_InvalidBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	router := gin.New()
	router.POST("/rotate", handler.RotateX25519Key)

	req := httptest.NewRequest(http.MethodPost, "/rotate", bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "Invalid request body")
}

// TestRotateX25519KeyHandler_MissingDID returns 400 when the did field is empty.
func TestRotateX25519KeyHandler_MissingDID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{})
	router := gin.New()
	router.POST("/rotate", handler.RotateX25519Key)

	body, _ := json.Marshal(map[string]string{"did": ""})
	req := httptest.NewRequest(http.MethodPost, "/rotate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "did is required")
}

// TestRotateX25519KeyHandler_ServiceError maps a service rotation failure to a
// 400 with the underlying error surfaced in details.
func TestRotateX25519KeyHandler_ServiceError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := &fakeDIDService{
		rotateFn: func(string) (string, int, error) {
			return "", 0, fmt.Errorf("cannot rotate keyAgreement key for reasoner DID: unsupported")
		},
	}
	handler := NewDIDHandlers(svc, &fakeVCService{})
	router := gin.New()
	router.POST("/rotate", handler.RotateX25519Key)

	body, _ := json.Marshal(map[string]string{"did": "did:key:zReasoner"})
	req := httptest.NewRequest(http.MethodPost, "/rotate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "Failed to rotate keyAgreement key")
	require.Contains(t, resp.Body.String(), "unsupported")
}

// TestRotateX25519KeyHandler_MalformedNewKey covers the fallback branch: when the
// service returns a NEW public key that is not valid JSON, the handler still
// returns 200 but surfaces the raw string rather than a parsed object.
func TestRotateX25519KeyHandler_MalformedNewKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := &fakeDIDService{
		rotateFn: func(string) (string, int, error) {
			return "this-is-not-json", 2, nil
		},
	}
	handler := NewDIDHandlers(svc, &fakeVCService{})
	router := gin.New()
	router.POST("/rotate", handler.RotateX25519Key)

	body, _ := json.Marshal(map[string]string{"did": "did:key:zAgent"})
	req := httptest.NewRequest(http.MethodPost, "/rotate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.EqualValues(t, 2, payload["epoch"])
	// Malformed key falls back to the raw string value.
	require.Equal(t, "this-is-not-json", payload["x25519_public_key_jwk"])
}

// TestResolveDIDHandler_MalformedX25519Key covers the warn-and-skip branch in
// ResolveDID: when the identity carries a malformed X25519 JWK, the handler must
// still return 200 and simply omit key_agreement.
func TestResolveDIDHandler_MalformedX25519Key(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := &fakeDIDService{
		resolveFn: func(did string) (*types.DIDIdentity, error) {
			return &types.DIDIdentity{
				DID:                did,
				PublicKeyJWK:       `{"kty":"OKP"}`,
				X25519PublicKeyJWK: "not-json",
				ComponentType:      "agent",
			}, nil
		},
	}
	handler := NewDIDHandlers(svc, &fakeVCService{})
	router := gin.New()
	router.GET("/resolve/:did", handler.ResolveDID)

	req := httptest.NewRequest(http.MethodGet, "/resolve/did:key:zAgent", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	_, hasKA := payload["key_agreement"]
	require.False(t, hasKA, "malformed X25519 JWK must be omitted from the response, not emitted")
}
