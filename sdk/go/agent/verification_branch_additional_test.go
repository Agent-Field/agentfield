package agent

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalVerifier_FetchHelpersAdditionalErrors(t *testing.T) {
	t.Run("fetch revocations and registrations handle non-200 and bad json", func(t *testing.T) {
		statusServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/revocations":
				http.Error(w, "nope", http.StatusBadGateway)
			case "/api/v1/registered-dids":
				http.Error(w, "nope", http.StatusForbidden)
			default:
				http.NotFound(w, r)
			}
		}))
		defer statusServer.Close()

		v := NewLocalVerifier(statusServer.URL, 0, "")
		client := statusServer.Client()
		_, err := v.fetchRevocations(client)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "HTTP 502")

		_, err = v.fetchRegisteredDIDs(client)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "HTTP 403")

		badJSONServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/revocations", "/api/v1/registered-dids":
				_, _ = w.Write([]byte(`{`))
			default:
				http.NotFound(w, r)
			}
		}))
		defer badJSONServer.Close()

		v = NewLocalVerifier(badJSONServer.URL, 0, "")
		client = badJSONServer.Client()
		_, err = v.fetchRevocations(client)
		require.Error(t, err)

		_, err = v.fetchRegisteredDIDs(client)
		require.Error(t, err)
	})

	t.Run("fetch admin public key validates response body", func(t *testing.T) {
		validKey := make(ed25519.PublicKey, ed25519.PublicKeySize)
		validKey[0] = 1
		validX := base64.RawURLEncoding.EncodeToString(validKey)

		newServer := func(handler func(http.ResponseWriter, *http.Request)) *httptest.Server {
			return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/v1/admin/public-key" {
					http.NotFound(w, r)
					return
				}
				handler(w, r)
			}))
		}

		httpServer := newServer(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "bad gateway", http.StatusBadGateway)
		})
		defer httpServer.Close()
		v := NewLocalVerifier(httpServer.URL, 0, "")
		_, _, err := v.fetchAdminPublicKey(httpServer.Client())
		require.Error(t, err)

		missingXServer := newServer(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer_did": "did:issuer", "public_key_jwk": map[string]any{}})
		})
		defer missingXServer.Close()
		v = NewLocalVerifier(missingXServer.URL, 0, "")
		_, _, err = v.fetchAdminPublicKey(missingXServer.Client())
		require.Error(t, err)

		badBase64Server := newServer(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer_did": "did:issuer", "public_key_jwk": map[string]any{"x": "%%%"}})
		})
		defer badBase64Server.Close()
		v = NewLocalVerifier(badBase64Server.URL, 0, "")
		_, _, err = v.fetchAdminPublicKey(badBase64Server.Client())
		require.Error(t, err)

		badSizeServer := newServer(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer_did": "did:issuer", "public_key_jwk": map[string]any{"x": base64.RawURLEncoding.EncodeToString([]byte{1, 2, 3})}})
		})
		defer badSizeServer.Close()
		v = NewLocalVerifier(badSizeServer.URL, 0, "")
		_, _, err = v.fetchAdminPublicKey(badSizeServer.Client())
		require.Error(t, err)

		successServer := newServer(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer_did": "did:issuer", "public_key_jwk": map[string]any{"x": validX}})
		})
		defer successServer.Close()
		v = NewLocalVerifier(successServer.URL, 0, "")
		client := successServer.Client()
		pub, issuer, err := v.fetchAdminPublicKey(client)
		require.NoError(t, err)
		assert.Equal(t, ed25519.PublicKey(validKey), pub)
		assert.Equal(t, "did:issuer", issuer)
	})
}

func TestEvaluateConstraints_AdditionalOperators(t *testing.T) {
	assert.True(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "<=", Value: 3}}, "agent.read", map[string]any{"value": 3}))
	assert.False(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "<=", Value: 3}}, "agent.read", map[string]any{"value": 4}))
	assert.True(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "<", Value: 3}}, "agent.read", map[string]any{"value": 2}))
	assert.False(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "<", Value: 3}}, "agent.read", map[string]any{"value": 3}))
	assert.True(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: ">=", Value: 3}}, "agent.read", map[string]any{"value": 3}))
	assert.False(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: ">", Value: 3}}, "agent.read", map[string]any{"value": 3}))
	assert.True(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "unknown", Value: 3}}, "agent.read", map[string]any{"value": 3}))
}
