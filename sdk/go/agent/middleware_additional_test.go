package agent

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func signedMiddlewareRequest(t *testing.T, path string, body []byte, priv ed25519.PrivateKey, callerDID string) *http.Request {
	t.Helper()

	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	bodyHash := sha256.Sum256(body)
	payload := []byte(timestamp + ":" + fmt.Sprintf("%x", bodyHash))
	signature := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, payload))

	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("X-Caller-DID", callerDID)
	req.Header.Set("X-DID-Signature", signature)
	req.Header.Set("X-DID-Timestamp", timestamp)
	return req
}

func signedDidKey(t *testing.T) (string, ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	return "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, pub...)), pub, priv
}

func TestOriginAuthMiddleware_ProtectsOnlyExecutionEndpoints(t *testing.T) {
	a := &Agent{}
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	t.Run("exempt endpoints bypass auth", func(t *testing.T) {
		for _, path := range []string{"/health", "/discover", "/agentfield/v1/logs"} {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			resp := httptest.NewRecorder()
			a.originAuthMiddleware(next, "secret").ServeHTTP(resp, req)
			assert.Equal(t, http.StatusNoContent, resp.Code, path)
		}
	})

	t.Run("protected endpoints require bearer token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/execute/demo", nil)
		resp := httptest.NewRecorder()
		a.originAuthMiddleware(next, "secret").ServeHTTP(resp, req)
		assert.Equal(t, http.StatusUnauthorized, resp.Code)

		req.Header.Set("Authorization", "Bearer secret")
		resp = httptest.NewRecorder()
		a.originAuthMiddleware(next, "secret").ServeHTTP(resp, req)
		assert.Equal(t, http.StatusNoContent, resp.Code)
	})
}

func TestLocalVerificationMiddleware_CoversFailureAndSuccessPaths(t *testing.T) {
	callerDID, pub, priv := signedDidKey(t)

	refreshServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/v1/policies":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"policies": []map[string]any{{
					"name":            "allow-demo",
					"allow_functions": []string{"demo"},
					"action":          "allow",
					"priority":        10,
				}},
			})
		case "/api/v1/revocations":
			_, _ = io.WriteString(w, `{"revoked_dids":[]}`)
		case "/api/v1/registered-dids":
			_ = json.NewEncoder(w).Encode(map[string]any{"registered_dids": []string{callerDID}})
		case "/api/v1/admin/public-key":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer_did": "did:example:issuer",
				"public_key_jwk": map[string]any{
					"kty": "OKP",
					"crv": "Ed25519",
					"x":   base64.RawURLEncoding.EncodeToString(pub),
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer refreshServer.Close()

	newAgent := func() *Agent {
		return &Agent{
			cfg: Config{
				NodeID:        "node-1",
				Token:         "cp-token",
				InternalToken: "internal-token",
				Tags:          []string{"target"},
			},
			logger:                      log.New(io.Discard, "", 0),
			localVerifier:               NewLocalVerifier(refreshServer.URL, time.Minute, ""),
			realtimeValidationFunctions: map[string]struct{}{},
		}
	}

	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_, _ = w.Write(body)
	})

	t.Run("exempt and realtime-validated paths bypass verification", func(t *testing.T) {
		a := newAgent()
		a.realtimeValidationFunctions["demo"] = struct{}{}

		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		resp := httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader([]byte(`{"ok":true}`)))
		resp = httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)
		assert.JSONEq(t, `{"ok":true}`, resp.Body.String())
	})

	t.Run("internal token bypasses DID verification", func(t *testing.T) {
		a := newAgent()
		req := httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader([]byte(`{"ok":true}`)))
		req.Header.Set("Authorization", "Bearer internal-token")
		resp := httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("missing DID and missing signature are rejected", func(t *testing.T) {
		a := newAgent()

		req := httptest.NewRequest(http.MethodPost, "/execute/demo", nil)
		resp := httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusUnauthorized, resp.Code)
		assert.Contains(t, resp.Body.String(), "did_auth_required")

		req = httptest.NewRequest(http.MethodPost, "/execute/demo", nil)
		req.Header.Set("X-Caller-DID", callerDID)
		resp = httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusUnauthorized, resp.Code)
		assert.Contains(t, resp.Body.String(), "signature_required")
	})

	t.Run("revoked and unregistered callers are rejected", func(t *testing.T) {
		a := newAgent()
		a.localVerifier.lastRefresh = time.Now()
		a.localVerifier.revokedDIDs[callerDID] = struct{}{}
		a.localVerifier.registeredDIDs[callerDID] = struct{}{}

		req := httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader([]byte(`{}`)))
		req.Header.Set("X-Caller-DID", callerDID)
		req.Header.Set("X-DID-Signature", "irrelevant")
		req.Header.Set("X-DID-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))
		resp := httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusForbidden, resp.Code)
		assert.Contains(t, resp.Body.String(), "did_revoked")

		a = newAgent()
		a.localVerifier.lastRefresh = time.Now()
		a.localVerifier.registeredDIDs["did:example:someone-else"] = struct{}{}
		req = httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader([]byte(`{}`)))
		req.Header.Set("X-Caller-DID", callerDID)
		req.Header.Set("X-DID-Signature", "irrelevant")
		req.Header.Set("X-DID-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))
		resp = httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusForbidden, resp.Code)
		assert.Contains(t, resp.Body.String(), "did_not_registered")
	})

	t.Run("invalid signature and denied policy fail after body buffering", func(t *testing.T) {
		a := newAgent()
		a.localVerifier.lastRefresh = time.Now()
		a.localVerifier.registeredDIDs[callerDID] = struct{}{}
		a.localVerifier.policies = []PolicyEntry{{DenyFunctions: []string{"demo"}, Action: "deny", Priority: 10}}
		a.localVerifier.adminPublicKey = pub

		req := httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader([]byte(`{"value":1}`)))
		req.Header.Set("X-Caller-DID", callerDID)
		req.Header.Set("X-DID-Signature", "bad")
		req.Header.Set("X-DID-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))
		resp := httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusUnauthorized, resp.Code)
		assert.Contains(t, resp.Body.String(), "signature_invalid")

		body := []byte(`{"value":2}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		hash := sha256.Sum256(body)
		payload := []byte(timestamp + ":" + fmt.Sprintf("%x", hash))
		signature := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, payload))
		req = httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader(body))
		req.Header.Set("X-Caller-DID", callerDID)
		req.Header.Set("X-DID-Signature", signature)
		req.Header.Set("X-DID-Timestamp", timestamp)
		resp = httptest.NewRecorder()
		a.localVerificationMiddleware(next).ServeHTTP(resp, req)
		assert.Equal(t, http.StatusForbidden, resp.Code)
		assert.Contains(t, resp.Body.String(), "policy_denied")
	})

	t.Run("stale verifier refreshes and valid signed request passes", func(t *testing.T) {
		a := newAgent()
		a.localVerifier.lastRefresh = time.Now().Add(-2 * time.Hour)

		body := []byte(`{"value":3}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		hash := sha256.Sum256(body)
		payload := []byte(timestamp + ":" + fmt.Sprintf("%x", hash))
		signature := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, payload))

		req := httptest.NewRequest(http.MethodPost, "/execute/demo", bytes.NewReader(body))
		req.Header.Set("X-Caller-DID", callerDID)
		req.Header.Set("X-DID-Signature", signature)
		req.Header.Set("X-DID-Timestamp", timestamp)
		resp := httptest.NewRecorder()

		a.localVerificationMiddleware(next).ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)
		assert.JSONEq(t, `{"value":3}`, resp.Body.String())
		assert.True(t, a.localVerifier.initialized)
	})
}
