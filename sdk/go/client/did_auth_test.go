package client

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testKeyPair generates a real Ed25519 key pair and returns the
// public key, private key, and JWK JSON string for the private key.
func testKeyPair(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey, string) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	seed := priv.Seed()
	jwkJSON, err := json.Marshal(jwk{
		Kty: "OKP",
		Crv: "Ed25519",
		D:   base64.RawURLEncoding.EncodeToString(seed),
		X:   base64.RawURLEncoding.EncodeToString(pub),
	})
	require.NoError(t, err)

	return pub, priv, string(jwkJSON)
}

// =====================================================
// NewDIDAuthenticator Tests
// =====================================================

func TestDIDNewDIDAuthenticator(t *testing.T) {
	pub, _, jwkStr := testKeyPair(t)
	_ = pub

	tests := []struct {
		name          string
		did           string
		privateKeyJWK string
		wantNil       bool
		wantErr       bool
	}{
		{
			name:          "valid credentials",
			did:           "did:web:example.com:agents:test-agent",
			privateKeyJWK: jwkStr,
			wantNil:       false,
			wantErr:       false,
		},
		{
			name:          "empty DID returns nil authenticator",
			did:           "",
			privateKeyJWK: jwkStr,
			wantNil:       true,
			wantErr:       false,
		},
		{
			name:          "empty JWK returns nil authenticator",
			did:           "did:web:example.com:agents:test-agent",
			privateKeyJWK: "",
			wantNil:       true,
			wantErr:       false,
		},
		{
			name:          "both empty returns nil authenticator",
			did:           "",
			privateKeyJWK: "",
			wantNil:       true,
			wantErr:       false,
		},
		{
			name:          "invalid JWK JSON",
			did:           "did:web:example.com:agents:test-agent",
			privateKeyJWK: `{not valid json`,
			wantNil:       false,
			wantErr:       true,
		},
		{
			name:          "wrong kty in JWK",
			did:           "did:web:example.com:agents:test-agent",
			privateKeyJWK: `{"kty":"RSA","crv":"Ed25519","d":"AAAA"}`,
			wantNil:       false,
			wantErr:       true,
		},
		{
			name:          "wrong crv in JWK",
			did:           "did:web:example.com:agents:test-agent",
			privateKeyJWK: `{"kty":"OKP","crv":"P-256","d":"AAAA"}`,
			wantNil:       false,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			auth, err := NewDIDAuthenticator(tt.did, tt.privateKeyJWK)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, auth)
			} else {
				assert.NoError(t, err)
				if tt.wantNil {
					assert.Nil(t, auth)
				} else {
					assert.NotNil(t, auth)
					assert.Equal(t, tt.did, auth.DID())
				}
			}
		})
	}
}

// =====================================================
// IsConfigured Tests
// =====================================================

func TestDIDIsConfigured(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)

	t.Run("configured authenticator returns true", func(t *testing.T) {
		auth, err := NewDIDAuthenticator("did:web:example.com:agents:test", jwkStr)
		require.NoError(t, err)
		require.NotNil(t, auth)
		assert.True(t, auth.IsConfigured())
	})

	t.Run("nil authenticator returns false", func(t *testing.T) {
		var auth *DIDAuthenticator
		assert.False(t, auth.IsConfigured())
	})

	t.Run("nil from empty credentials returns false", func(t *testing.T) {
		auth, err := NewDIDAuthenticator("", "")
		assert.NoError(t, err)
		assert.Nil(t, auth)
		// Calling IsConfigured on nil should be safe and return false
		assert.False(t, auth.IsConfigured())
	})
}

// =====================================================
// DID() accessor Tests
// =====================================================

func TestDIDAccessor(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)

	t.Run("returns DID when configured", func(t *testing.T) {
		auth, err := NewDIDAuthenticator("did:web:example.com:agents:my-agent", jwkStr)
		require.NoError(t, err)
		assert.Equal(t, "did:web:example.com:agents:my-agent", auth.DID())
	})

	t.Run("returns empty on nil authenticator", func(t *testing.T) {
		var auth *DIDAuthenticator
		assert.Equal(t, "", auth.DID())
	})
}

// =====================================================
// SignRequest Tests
// =====================================================

func TestDIDSignRequest(t *testing.T) {
	pub, _, jwkStr := testKeyPair(t)
	testDID := "did:web:example.com:agents:signer"

	t.Run("produces correct headers", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		body := []byte(`{"action":"test"}`)
		headers := auth.SignRequest(body)

		require.NotNil(t, headers)
		assert.Equal(t, testDID, headers[HeaderCallerDID])
		assert.NotEmpty(t, headers[HeaderDIDSignature])
		assert.NotEmpty(t, headers[HeaderDIDTimestamp])
		assert.NotEmpty(t, headers[HeaderDIDNonce])

		// Verify exactly four headers are returned
		assert.Len(t, headers, 4)
	})

	t.Run("timestamp is a valid unix timestamp", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		before := time.Now().Unix()
		headers := auth.SignRequest([]byte("test"))
		after := time.Now().Unix()

		ts, err := strconv.ParseInt(headers[HeaderDIDTimestamp], 10, 64)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, ts, before)
		assert.LessOrEqual(t, ts, after)
	})

	t.Run("signature is valid Ed25519 signature", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		body := []byte(`{"data":"hello world"}`)
		headers := auth.SignRequest(body)

		// Decode signature
		sigBytes, err := base64.StdEncoding.DecodeString(headers[HeaderDIDSignature])
		require.NoError(t, err)
		assert.Len(t, sigBytes, ed25519.SignatureSize)

		// Reconstruct the payload: "{timestamp}:{nonce}:{sha256_hex_hash}"
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%s:%x", headers[HeaderDIDTimestamp], headers[HeaderDIDNonce], bodyHash)

		// Verify with the public key
		assert.True(t, ed25519.Verify(pub, []byte(payload), sigBytes),
			"Ed25519 signature verification failed")
	})

	t.Run("payload format is timestamp:nonce:sha256hex", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		body := []byte("specific body content")
		headers := auth.SignRequest(body)

		// Manually compute expected hash
		expectedHash := sha256.Sum256(body)
		expectedPayload := fmt.Sprintf("%s:%s:%x", headers[HeaderDIDTimestamp], headers[HeaderDIDNonce], expectedHash)

		// Decode signature and verify it was signed over the expected payload
		sigBytes, err := base64.StdEncoding.DecodeString(headers[HeaderDIDSignature])
		require.NoError(t, err)
		assert.True(t, ed25519.Verify(pub, []byte(expectedPayload), sigBytes))
	})

	t.Run("different bodies produce different signatures", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		headers1 := auth.SignRequest([]byte("body one"))
		headers2 := auth.SignRequest([]byte("body two"))

		assert.NotEqual(t, headers1[HeaderDIDSignature], headers2[HeaderDIDSignature])
	})

	t.Run("same body produces different signatures via nonce", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		body := []byte(`{"same":"body"}`)
		headers1 := auth.SignRequest(body)
		headers2 := auth.SignRequest(body)

		// Nonces must differ
		assert.NotEqual(t, headers1[HeaderDIDNonce], headers2[HeaderDIDNonce])
		// Signatures must differ (even with same body and potentially same timestamp)
		assert.NotEqual(t, headers1[HeaderDIDSignature], headers2[HeaderDIDSignature])
	})

	t.Run("empty body is signed correctly", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		headers := auth.SignRequest([]byte{})
		require.NotNil(t, headers)

		sigBytes, err := base64.StdEncoding.DecodeString(headers[HeaderDIDSignature])
		require.NoError(t, err)

		bodyHash := sha256.Sum256([]byte{})
		payload := fmt.Sprintf("%s:%s:%x", headers[HeaderDIDTimestamp], headers[HeaderDIDNonce], bodyHash)
		assert.True(t, ed25519.Verify(pub, []byte(payload), sigBytes))
	})

	t.Run("nil body is signed correctly", func(t *testing.T) {
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)

		headers := auth.SignRequest(nil)
		require.NotNil(t, headers)

		sigBytes, err := base64.StdEncoding.DecodeString(headers[HeaderDIDSignature])
		require.NoError(t, err)

		// sha256.Sum256(nil) produces the hash of zero-length input
		bodyHash := sha256.Sum256(nil)
		payload := fmt.Sprintf("%s:%s:%x", headers[HeaderDIDTimestamp], headers[HeaderDIDNonce], bodyHash)
		assert.True(t, ed25519.Verify(pub, []byte(payload), sigBytes))
	})

	t.Run("returns nil when not configured", func(t *testing.T) {
		var auth *DIDAuthenticator
		headers := auth.SignRequest([]byte("test"))
		assert.Nil(t, headers)
	})
}

// =====================================================
// parsePrivateKeyJWK Tests
// =====================================================

func TestDIDParsePrivateKeyJWK(t *testing.T) {
	t.Run("valid JWK", func(t *testing.T) {
		pub, priv, jwkStr := testKeyPair(t)

		parsed, err := parsePrivateKeyJWK(jwkStr)
		require.NoError(t, err)
		require.NotNil(t, parsed)

		// Verify the parsed key matches the original
		assert.Equal(t, priv.Seed(), parsed.Seed())
		assert.Equal(t, ed25519.PublicKey(pub), parsed.Public().(ed25519.PublicKey))

		// Verify signing with parsed key produces verifiable signatures
		msg := []byte("test message")
		sig := ed25519.Sign(parsed, msg)
		assert.True(t, ed25519.Verify(pub, msg, sig))
	})

	t.Run("wrong kty", func(t *testing.T) {
		_, _, err := ed25519.GenerateKey(rand.Reader)
		require.NoError(t, err)

		jwkJSON := `{"kty":"RSA","crv":"Ed25519","d":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}`
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid key type")
	})

	t.Run("wrong crv", func(t *testing.T) {
		jwkJSON := `{"kty":"OKP","crv":"X25519","d":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}`
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid key type")
	})

	t.Run("missing d field", func(t *testing.T) {
		jwkJSON := `{"kty":"OKP","crv":"Ed25519","x":"AAAA"}`
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "missing 'd'")
	})

	t.Run("invalid base64 in d field", func(t *testing.T) {
		jwkJSON := `{"kty":"OKP","crv":"Ed25519","d":"!!!not-valid-base64!!!"}`
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid private key encoding")
	})

	t.Run("wrong key length - too short", func(t *testing.T) {
		shortKey := base64.RawURLEncoding.EncodeToString([]byte("tooshort"))
		jwkJSON := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","d":"%s"}`, shortKey)
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid private key length")
	})

	t.Run("wrong key length - too long", func(t *testing.T) {
		longKey := make([]byte, 64)
		_, err := rand.Read(longKey)
		require.NoError(t, err)
		encoded := base64.RawURLEncoding.EncodeToString(longKey)
		jwkJSON := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","d":"%s"}`, encoded)
		key, err := parsePrivateKeyJWK(jwkJSON)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid private key length")
	})

	t.Run("invalid JSON", func(t *testing.T) {
		key, err := parsePrivateKeyJWK(`{broken json`)
		assert.Error(t, err)
		assert.Nil(t, key)
		assert.Contains(t, err.Error(), "invalid JWK format")
	})

	t.Run("empty string", func(t *testing.T) {
		key, err := parsePrivateKeyJWK("")
		assert.Error(t, err)
		assert.Nil(t, key)
	})
}

// =====================================================
// Client.SignHTTPRequest Tests
// =====================================================

func TestDIDSignHTTPRequest(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)
	testDID := "did:web:example.com:agents:http-signer"

	t.Run("applies DID headers to http.Request", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth(testDID, jwkStr))
		require.NoError(t, err)

		body := []byte(`{"key":"value"}`)
		req := httptest.NewRequest(http.MethodPost, "/test", strings.NewReader(string(body)))

		c.SignHTTPRequest(req, body)

		assert.Equal(t, testDID, req.Header.Get(HeaderCallerDID))
		assert.NotEmpty(t, req.Header.Get(HeaderDIDSignature))
		assert.NotEmpty(t, req.Header.Get(HeaderDIDTimestamp))
		assert.NotEmpty(t, req.Header.Get(HeaderDIDNonce))
	})

	t.Run("no-op when DID auth not configured", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)

		body := []byte(`{"key":"value"}`)
		req := httptest.NewRequest(http.MethodPost, "/test", strings.NewReader(string(body)))

		c.SignHTTPRequest(req, body)

		assert.Empty(t, req.Header.Get(HeaderCallerDID))
		assert.Empty(t, req.Header.Get(HeaderDIDSignature))
		assert.Empty(t, req.Header.Get(HeaderDIDTimestamp))
	})

	t.Run("no-op on nil client", func(t *testing.T) {
		var c *Client
		req := httptest.NewRequest(http.MethodGet, "/test", nil)

		// Should not panic
		c.SignHTTPRequest(req, nil)

		assert.Empty(t, req.Header.Get(HeaderCallerDID))
	})
}

// =====================================================
// Client DID credential management Tests
// =====================================================

func TestDIDClientSetDIDCredentials(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)
	testDID := "did:web:example.com:agents:setter"

	t.Run("set valid credentials after creation", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)
		assert.False(t, c.DIDAuthConfigured())

		err = c.SetDIDCredentials(testDID, jwkStr)
		assert.NoError(t, err)
		assert.True(t, c.DIDAuthConfigured())
		assert.Equal(t, testDID, c.DID())
	})

	t.Run("set invalid credentials returns error", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)

		err = c.SetDIDCredentials(testDID, `{invalid json}`)
		assert.Error(t, err)
		assert.False(t, c.DIDAuthConfigured())
	})

	t.Run("set empty credentials clears auth", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth(testDID, jwkStr))
		require.NoError(t, err)
		assert.True(t, c.DIDAuthConfigured())

		err = c.SetDIDCredentials("", "")
		assert.NoError(t, err)
		// Empty credentials produce nil authenticator
		assert.False(t, c.DIDAuthConfigured())
	})
}

func TestDIDClientDIDAuthConfigured(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)

	t.Run("true when configured via option", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth("did:web:example.com:agents:test", jwkStr))
		require.NoError(t, err)
		assert.True(t, c.DIDAuthConfigured())
	})

	t.Run("false when not configured", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)
		assert.False(t, c.DIDAuthConfigured())
	})
}

func TestDIDClientDID(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)

	t.Run("returns DID when configured", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth("did:web:example.com:agents:test", jwkStr))
		require.NoError(t, err)
		assert.Equal(t, "did:web:example.com:agents:test", c.DID())
	})

	t.Run("returns empty when not configured", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)
		assert.Equal(t, "", c.DID())
	})
}

func TestDIDClientSignBody(t *testing.T) {
	testDID := "did:web:example.com:agents:test"
	_, _, jwkStr := testKeyPair(t)

	t.Run("returns nil without DID auth", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)
		assert.Nil(t, c.SignBody([]byte(`{"ok":true}`)))
	})

	t.Run("signs request body when configured", func(t *testing.T) {
		c, err := New("http://localhost:8080")
		require.NoError(t, err)
		auth, err := NewDIDAuthenticator(testDID, jwkStr)
		require.NoError(t, err)
		c.didAuthenticator = auth

		headers := c.SignBody([]byte(`{"ok":true}`))
		assert.Equal(t, testDID, headers[HeaderCallerDID])
		assert.NotEmpty(t, headers[HeaderDIDSignature])
		assert.NotEmpty(t, headers[HeaderDIDTimestamp])
		assert.NotEmpty(t, headers[HeaderDIDNonce])
	})
}

// =====================================================
// WithDIDAuth Option Tests
// =====================================================

func TestDIDWithDIDAuthOption(t *testing.T) {
	_, _, jwkStr := testKeyPair(t)

	t.Run("valid DID auth option", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth("did:web:example.com:agents:test", jwkStr))
		require.NoError(t, err)
		assert.True(t, c.DIDAuthConfigured())
	})

	t.Run("invalid JWK silently disables DID auth", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth("did:web:example.com:agents:test", `{bad json}`))
		require.NoError(t, err)
		// WithDIDAuth logs a warning but doesn't fail
		assert.False(t, c.DIDAuthConfigured())
	})

	t.Run("empty credentials produce no authenticator", func(t *testing.T) {
		c, err := New("http://localhost:8080", WithDIDAuth("", ""))
		require.NoError(t, err)
		assert.False(t, c.DIDAuthConfigured())
	})
}

// =====================================================
// Integration: DID headers in do() method
// =====================================================

func TestDIDHeadersInDoMethod(t *testing.T) {
	pub, _, jwkStr := testKeyPair(t)
	testDID := "did:web:example.com:agents:integration"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify DID headers are present
		callerDID := r.Header.Get(HeaderCallerDID)
		assert.Equal(t, testDID, callerDID)

		sig := r.Header.Get(HeaderDIDSignature)
		assert.NotEmpty(t, sig)

		ts := r.Header.Get(HeaderDIDTimestamp)
		assert.NotEmpty(t, ts)

		nonce := r.Header.Get(HeaderDIDNonce)
		assert.NotEmpty(t, nonce)

		// Verify the signature is valid
		sigBytes, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)

		// The do() method serializes the body to JSON, so we need to read and
		// reconstruct the expected hash. The body sent was {"msg":"hello"}.
		bodyHash := sha256.Sum256([]byte(`{"msg":"hello"}`))
		payload := fmt.Sprintf("%s:%s:%x", ts, nonce, bodyHash)
		assert.True(t, ed25519.Verify(pub, []byte(payload), sigBytes),
			"server-side signature verification failed")

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer server.Close()

	c, err := New(server.URL, WithDIDAuth(testDID, jwkStr))
	require.NoError(t, err)

	var resp map[string]interface{}
	err = c.do(nil, http.MethodPost, "/test", map[string]string{"msg": "hello"}, &resp)
	// context.Background() is nil-safe in newer Go, but let's use a real context
	// Actually, http.NewRequestWithContext with nil context will panic. Re-test:
	assert.Error(t, err) // nil context causes error
}

func TestDIDHeadersInDoMethodWithContext(t *testing.T) {
	pub, _, jwkStr := testKeyPair(t)
	testDID := "did:web:example.com:agents:integration"

	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify DID headers are present
		callerDID := r.Header.Get(HeaderCallerDID)
		assert.Equal(t, testDID, callerDID)

		sig := r.Header.Get(HeaderDIDSignature)
		assert.NotEmpty(t, sig)

		ts := r.Header.Get(HeaderDIDTimestamp)
		assert.NotEmpty(t, ts)

		nonce := r.Header.Get(HeaderDIDNonce)
		assert.NotEmpty(t, nonce)

		// Decode signature
		sigBytes, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)

		// Reconstruct payload. The body is json-marshaled by do().
		bodyHash := sha256.Sum256(capturedBody)
		payload := fmt.Sprintf("%s:%s:%x", ts, nonce, bodyHash)
		assert.True(t, ed25519.Verify(pub, []byte(payload), sigBytes),
			"server-side signature verification failed")

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer server.Close()

	c, err := New(server.URL, WithDIDAuth(testDID, jwkStr))
	require.NoError(t, err)

	requestBody := map[string]string{"msg": "hello"}
	// Pre-compute what the do() method will marshal
	capturedBody, err = json.Marshal(requestBody)
	require.NoError(t, err)

	var resp map[string]interface{}
	err = c.do(context.Background(), http.MethodPost, "/test", requestBody, &resp)
	assert.NoError(t, err)
	assert.Equal(t, "ok", resp["status"])
}
