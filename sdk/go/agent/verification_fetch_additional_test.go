package agent

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalVerifier_FetchAdminPublicKeyErrors(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	t.Run("non-200 includes response body", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, "upstream failed")
		}))
		defer server.Close()

		v := NewLocalVerifier(server.URL, time.Minute, "")
		_, _, err := v.fetchAdminPublicKey(server.Client())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "HTTP 502")
		assert.Contains(t, err.Error(), "upstream failed")
	})

	t.Run("missing or malformed x values fail", func(t *testing.T) {
		for _, body := range []string{
			`{"issuer_did":"did:example:issuer","public_key_jwk":{"kty":"OKP"}}`,
			`{"issuer_did":"did:example:issuer","public_key_jwk":{"x":"%%%bad"}}`,
			`{"issuer_did":"did:example:issuer","public_key_jwk":{"x":"` + base64.RawURLEncoding.EncodeToString(pub[:4]) + `"}}`,
		} {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			}))

			v := NewLocalVerifier(server.URL, time.Minute, "")
			_, _, err := v.fetchAdminPublicKey(server.Client())
			require.Error(t, err)
			server.Close()
		}
	})
}
