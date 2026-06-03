package handlers

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests exercise the SSRF hardening added for the
// parseServerlessDiscoveryURL / RegisterServerlessAgentHandler pair:
// the discovery URL must be reconstructed from validated components, so that
// scheme, host and path are all known-safe before an outbound HTTP request
// is issued (see CodeQL go/request-forgery, CWE-918).

func TestParseServerlessDiscoveryURL_RebuildsFromValidatedComponents(t *testing.T) {
	t.Parallel()

	safe, err := parseServerlessDiscoveryURL("https://agents.internal/invoke", []string{"agents.internal"})
	require.NoError(t, err)

	assert.Equal(t, "https", safe.Scheme)
	assert.Equal(t, "agents.internal", safe.Host)
	assert.Equal(t, "/invoke", safe.Path)
	assert.Nil(t, safe.User)
	assert.Equal(t, "", safe.RawQuery)
	assert.Equal(t, "", safe.Fragment)
	assert.Equal(t, "", safe.Opaque)
}

func TestNormalizeServerlessDiscoveryURL_CleansPathTraversal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		rawURL  string
		allowed []string
		want    string
	}{
		{
			name:    "dot segments collapsed",
			rawURL:  "http://localhost:9000/a/./b/../invoke",
			allowed: nil,
			want:    "http://localhost:9000/a/invoke",
		},
		{
			name:    "parent traversal clamped to root",
			rawURL:  "http://localhost:9000/../../../etc/passwd",
			allowed: nil,
			want:    "http://localhost:9000/etc/passwd",
		},
		{
			name:    "trailing slash stripped after cleaning",
			rawURL:  "http://localhost:9000/invoke/",
			allowed: nil,
			want:    "http://localhost:9000/invoke",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeServerlessDiscoveryURL(tc.rawURL, tc.allowed)
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestNormalizeServerlessDiscoveryURL_RejectsOpaqueURL(t *testing.T) {
	t.Parallel()

	_, err := normalizeServerlessDiscoveryURL("http:foo.example.com/path", []string{"foo.example.com"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must not be opaque")
}

func TestNormalizeServerlessDiscoveryURL_NormalizesIPv6ZeroHost(t *testing.T) {
	t.Parallel()

	got, err := normalizeServerlessDiscoveryURL("http://[::]:9000/invoke", nil)
	require.NoError(t, err)
	assert.Equal(t, "http://localhost:9000/invoke", got)
}

func TestNormalizeServerlessDiscoveryURL_IgnoresQueryAndFragmentInPath(t *testing.T) {
	t.Parallel()

	// A valid-shaped URL whose path tries to carry a query-like suffix must
	// still succeed — the query is part of the path, not a real query string.
	got, err := normalizeServerlessDiscoveryURL("http://localhost:7000/inv%3Fstrange", nil)
	require.NoError(t, err)
	assert.Equal(t, "http://localhost:7000/inv%3Fstrange", got)
}

// TestRegisterServerlessAgentHandler_DiscoverySanitizedFunctional is the SSRF
// functional test: it verifies that (1) URLs whose host is not loopback and
// not on the allowlist are rejected before any HTTP request goes out, and
// (2) when a valid-but-path-traversal-shaped URL is submitted the handler's
// outbound request lands on the allowlisted server at a sanitized path —
// both of which are the properties CodeQL rule go/request-forgery asks the
// code to uphold.
func TestRegisterServerlessAgentHandler_DiscoverySanitizedFunctional(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var allowedHits int32
	sawSanitizedPath := make(chan string, 4)

	discoveryPayload := `{
		"node_id":"serverless-ok",
		"version":"v1",
		"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"}}],
		"skills":[{"id":"s1","input_schema":{"type":"object"}}]
	}`

	allowedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&allowedHits, 1)
		select {
		case sawSanitizedPath <- r.URL.Path:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(discoveryPayload))
	}))
	defer allowedServer.Close()

	allowedURL, err := url.Parse(allowedServer.URL)
	require.NoError(t, err)
	_, _, err = net.SplitHostPort(allowedURL.Host)
	require.NoError(t, err)

	// Allowlist the loopback (default) plus a fake external host we can
	// assert against without risking real network egress.
	allowedHosts := []string{"127.0.0.1", "localhost"}

	t.Run("rejects non-loopback host outside allowlist", func(t *testing.T) {
		router := gin.New()
		router.POST("/serverless/register",
			RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, allowedHosts),
		)

		// evil.example.com is neither a loopback address nor in the allow
		// list, so the handler must reject this synchronously — no outbound
		// request, no DNS, no socket.
		body := `{"invocation_url":"https://evil.example.com/invoke"}`
		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "Invalid invocation_url")
		assert.Contains(t, resp.Body.String(), "not allowlisted")
	})

	t.Run("outbound request uses only validated components", func(t *testing.T) {
		atomic.StoreInt32(&allowedHits, 0)

		// The attacker-shaped path contains ".." traversal segments. After
		// normalization the handler must contact the allowlisted server at
		// the collapsed path — specifically, "/discover" (since "/a/./b/../.."
		// collapses to "/" and "/" + "/discover" cleans to "/discover").
		body := fmt.Sprintf(`{"invocation_url":"http://%s/a/./b/../.."}`, allowedURL.Host)
		router := gin.New()
		router.POST("/serverless/register",
			RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, allowedHosts),
		)
		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.GreaterOrEqual(t, atomic.LoadInt32(&allowedHits), int32(1),
			"allowed host should receive the discovery request; response was: %s", resp.Body.String())

		select {
		case got := <-sawSanitizedPath:
			// The path seen by the allowlisted server must be a literal,
			// sanitized "/discover" — never "/a/./b/../../discover" and never
			// anything outside the "/discover" endpoint.
			assert.Equal(t, "/discover", got,
				"discovery request must target the sanitized /discover path")
		default:
			t.Fatalf("expected an inbound request on the allowlisted discovery server")
		}
	})

	t.Run("rejects opaque URL before any network call", func(t *testing.T) {
		atomic.StoreInt32(&allowedHits, 0)

		router := gin.New()
		router.POST("/serverless/register",
			RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, allowedHosts),
		)

		// "http:foo" is parsed as an opaque URL (no "//", no host).
		req := httptest.NewRequest(http.MethodPost, "/serverless/register",
			strings.NewReader(`{"invocation_url":"http:opaque/content"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "Invalid invocation_url")
		assert.Equal(t, int32(0), atomic.LoadInt32(&allowedHits))
	})
}
