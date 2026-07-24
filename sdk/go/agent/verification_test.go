package agent

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func verifierTestServer(t *testing.T, failPolicies *atomic.Bool, pubKey ed25519.PublicKey) *httptest.Server {
	t.Helper()

	jwkX := base64.RawURLEncoding.EncodeToString(pubKey)

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch r.URL.Path {
		case "/api/v1/policies":
			if failPolicies != nil && failPolicies.Load() {
				http.Error(w, "boom", http.StatusInternalServerError)
				return
			}
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"policies": []map[string]any{
					{
						"name":            "allow-read",
						"allow_functions": []string{"read"},
						"action":          "allow",
						"priority":        10,
					},
				},
			}))
		case "/api/v1/revocations":
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"revoked_dids": []string{"did:example:revoked"},
			}))
		case "/api/v1/registered-dids":
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"registered_dids": []string{"did:example:registered"},
			}))
		case "/api/v1/admin/public-key":
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"issuer_did": "did:example:issuer",
				"public_key_jwk": map[string]any{
					"kty": "OKP",
					"crv": "Ed25519",
					"x":   jwkX,
				},
			}))
		default:
			t.Fatalf("unexpected verifier request %s %s", r.Method, r.URL.Path)
		}
	}))
}

func TestLocalVerifier_RefreshPopulatesCaches(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	server := verifierTestServer(t, nil, pub)
	defer server.Close()

	v := NewLocalVerifier(server.URL, time.Minute, "api-key")

	err = v.Refresh()
	require.NoError(t, err)

	require.Len(t, v.policies, 1)
	assert.Equal(t, "allow-read", v.policies[0].Name)
	assert.True(t, v.CheckRevocation("did:example:revoked"))
	assert.False(t, v.CheckRevocation("did:example:unknown"))
	_, registered := v.registeredDIDs["did:example:registered"]
	assert.True(t, registered)
	assert.Equal(t, pub, v.adminPublicKey)
	assert.Equal(t, "did:example:issuer", v.issuerDID)
	assert.True(t, v.initialized)
	assert.False(t, v.lastRefresh.IsZero())
}

func TestLocalVerifier_RefreshFailureLeavesPreviousCacheIntact(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	var failPolicies atomic.Bool
	server := verifierTestServer(t, &failPolicies, pub)
	defer server.Close()

	v := NewLocalVerifier(server.URL, time.Minute, "")
	require.NoError(t, v.Refresh())

	wantPolicies := append([]PolicyEntry(nil), v.policies...)
	wantLastRefresh := v.lastRefresh
	wantIssuer := v.issuerDID
	wantAdminKey := append(ed25519.PublicKey(nil), v.adminPublicKey...)

	failPolicies.Store(true)

	// Refresh itself returns the error and keeps the previous cache untouched.
	err = v.Refresh()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fetch policies")
	assert.Equal(t, wantPolicies, v.policies)
	assert.Equal(t, wantLastRefresh, v.lastRefresh)
	assert.Equal(t, wantIssuer, v.issuerDID)
	assert.Equal(t, wantAdminKey, v.adminPublicKey)
	assert.True(t, v.CheckRevocation("did:example:revoked"))
}

func TestLocalVerifier_NeedsRefreshTracksLastRefresh(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	server := verifierTestServer(t, nil, pub)
	defer server.Close()

	v := NewLocalVerifier(server.URL, 50*time.Millisecond, "")
	require.NoError(t, v.Refresh())
	assert.False(t, v.NeedsRefresh())

	v.mu.Lock()
	v.lastRefresh = time.Now().Add(-2 * v.refreshInterval)
	v.mu.Unlock()

	assert.True(t, v.NeedsRefresh())
}

func TestLocalVerifier_ConcurrentRefreshAndCheckRevocation(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	server := verifierTestServer(t, nil, pub)
	defer server.Close()

	v := NewLocalVerifier(server.URL, time.Minute, "")
	require.NoError(t, v.Refresh())

	// 4 refreshers * 10 calls each = 40 sends; channel must hold them all
	// because the consumer drains only after wg.Wait().
	errCh := make(chan error, 40)
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				errCh <- v.Refresh()
			}
		}()
	}

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				_ = v.CheckRevocation("did:example:revoked")
				_ = v.CheckRevocation("did:example:unknown")
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		assert.NoError(t, err)
	}
}

func TestLocalVerifier_CheckRegistrationBeforeFirstRefreshAllowsCaller(t *testing.T) {
	v := NewLocalVerifier("http://example.invalid", time.Minute, "")

	assert.True(t, v.CheckRegistration("did:example:anything"))
	assert.False(t, v.CheckRevocation("did:example:anything"))
}

func TestLocalVerifier_ResolvePublicKeyForDidKeyAndMalformedInput(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	v := NewLocalVerifier("http://example.invalid", time.Minute, "")

	encoded := base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, pub...))
	resolved := v.resolvePublicKey("did:key:z" + encoded)
	require.NotNil(t, resolved)
	assert.Equal(t, pub, resolved)

	var malformed ed25519.PublicKey
	assert.NotPanics(t, func() {
		malformed = v.resolvePublicKey("did:key:z%%%")
	})
	assert.Nil(t, malformed)
}

func TestLocalVerifier_RefreshUsesProvidedAPIKey(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	var seenAPIKey atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAPIKey.Store(r.Header.Get("X-API-Key"))
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/v1/policies":
			_, _ = io.WriteString(w, `{"policies":[]}`)
		case "/api/v1/revocations":
			_, _ = io.WriteString(w, `{"revoked_dids":[]}`)
		case "/api/v1/registered-dids":
			_, _ = io.WriteString(w, `{"registered_dids":[]}`)
		case "/api/v1/admin/public-key":
			_, _ = io.WriteString(w, `{"issuer_did":"did:example:issuer","public_key_jwk":{"kty":"OKP","crv":"Ed25519","x":"`+base64.RawURLEncoding.EncodeToString(pub)+`"}}`)
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	v := NewLocalVerifier(server.URL, time.Minute, "secret-key")
	require.NoError(t, v.Refresh())
	assert.Equal(t, "secret-key", seenAPIKey.Load())
}

func TestLocalVerifier_VerifySignature(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	v := NewLocalVerifier("http://example.invalid", time.Minute, "")
	v.adminPublicKey = pub
	v.timestampWindow = 300

	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	body := []byte(`{"hello":"world"}`)
	bodyHash := sha256.Sum256(body)
	nonce := "nonce-123"
	payload := []byte(timestamp + ":" + nonce + ":" + fmt.Sprintf("%x", bodyHash))
	signature := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, payload))

	callerDID := "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, pub...))
	assert.True(t, v.VerifySignature(callerDID, signature, timestamp, body, nonce))
	assert.False(t, v.VerifySignature(callerDID, "not-base64", timestamp, body, nonce))
	assert.False(t, v.VerifySignature(callerDID, signature, "bad-ts", body, nonce))
	assert.False(t, v.VerifySignature(callerDID, signature, strconv.FormatInt(time.Now().Add(-10*time.Minute).Unix(), 10), body, nonce))

	noNoncePayload := []byte(timestamp + ":" + fmt.Sprintf("%x", bodyHash))
	noNonceSig := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, noNoncePayload))
	assert.True(t, v.VerifySignature("did:example:caller", noNonceSig, timestamp, body, ""))
	assert.False(t, v.VerifySignature("did:key:zbad", signature, timestamp, body, nonce))
}

func TestLocalVerifier_EvaluatePolicyAndHelpers(t *testing.T) {
	v := NewLocalVerifier("http://example.invalid", time.Minute, "")
	assert.False(t, v.EvaluatePolicy(nil, nil, "agent.read", nil))

	disabled := false
	v.policies = []PolicyEntry{
		{
			Name:           "disabled-deny",
			DenyFunctions:  []string{"agent.*"},
			Priority:       100,
			Enabled:        &disabled,
		},
		{
			Name:           "allow-read",
			CallerTags:     []string{"internal"},
			TargetTags:     []string{"finance"},
			AllowFunctions: []string{"agent.read*"},
			Constraints: map[string]ConstraintEntry{
				"limit": {Operator: "<=", Value: 5},
			},
			Action:   "allow",
			Priority: 10,
		},
		{
			Name:          "deny-write",
			DenyFunctions: []string{"agent.write"},
			Priority:      5,
		},
	}

	assert.True(t, v.EvaluatePolicy([]string{"internal"}, []string{"finance"}, "agent.read.summary", map[string]any{"limit": 3}))
	assert.False(t, v.EvaluatePolicy([]string{"internal"}, []string{"finance"}, "agent.read.summary", map[string]any{"limit": 8}))
	assert.False(t, v.EvaluatePolicy([]string{"internal"}, []string{"finance"}, "agent.read.summary", map[string]any{}))
	assert.False(t, v.EvaluatePolicy([]string{"internal"}, []string{"finance"}, "agent.write", map[string]any{"limit": 1}))
	assert.True(t, v.EvaluatePolicy([]string{"other"}, []string{"other"}, "agent.other", map[string]any{"limit": 1}))

	assert.True(t, anyTagMatch([]string{"finance", "ops"}, []string{"ops"}))
	assert.False(t, anyTagMatch([]string{"finance"}, []string{"eng"}))
	assert.True(t, functionMatches("agent.read.summary", []string{"agent.read*"}))
	assert.True(t, matchWildcard("agent.read.summary", "*summary"))
	assert.True(t, matchWildcard("anything", "*"))
	assert.False(t, matchWildcard("agent.read", "agent.write"))

	assert.True(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: ">", Value: 1}}, "agent.read", map[string]any{"value": 2}))
	assert.False(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: "==", Value: 1}}, "agent.read", map[string]any{"value": 2}))
	assert.False(t, evaluateConstraints(map[string]ConstraintEntry{"value": {Operator: ">=", Value: 1}}, "agent.read", map[string]any{"value": "bad"}))

	for _, tc := range []struct {
		name string
		in   any
		want float64
	}{
		{name: "float64", in: 1.5, want: 1.5},
		{name: "float32", in: float32(2.5), want: 2.5},
		{name: "int", in: 3, want: 3},
		{name: "int64", in: int64(4), want: 4},
		{name: "json-number", in: json.Number("5.5"), want: 5.5},
		{name: "string", in: "6.5", want: 6.5},
	} {
		got, err := toFloat64(tc.in)
		require.NoError(t, err, tc.name)
		assert.InDelta(t, tc.want, got, 1e-9, tc.name)
	}

	_, err := toFloat64(struct{}{})
	assert.Error(t, err)
	assert.Equal(t, int64(5), abs64(-5))
	assert.Equal(t, int64(5), abs64(5))
	assert.Equal(t, int64(math.MaxInt64), abs64(math.MinInt64))
}
