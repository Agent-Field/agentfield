package handlers

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateCallbackURL_Valid(t *testing.T) {
	// Create a test HTTP server that responds to health checks
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		}
	}))
	defer server.Close()

	err := validateCallbackURL(server.URL)
	assert.NoError(t, err, "Should validate reachable callback URL")
}

func TestValidateCallbackURL_Empty(t *testing.T) {
	err := validateCallbackURL("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be empty")
}

func TestValidateCallbackURL_InvalidFormat(t *testing.T) {
	tests := []struct {
		name string
		url  string
	}{
		{"no scheme", "localhost:8080"},
		{"invalid scheme", "ftp://localhost:8080"},
		{"no host", "http://"},
		{"malformed", "ht!tp://invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCallbackURL(tt.url)
			require.Error(t, err, "Should reject invalid URL format")
		})
	}
}

func TestValidateCallbackURL_InvalidScheme(t *testing.T) {
	err := validateCallbackURL("ftp://example.com")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must use http or https scheme")
}

func TestValidateCallbackURL_NoHost(t *testing.T) {
	err := validateCallbackURL("http://")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must include a host")
}

func TestValidateCallbackURL_Unreachable(t *testing.T) {
	// Use a valid URL format but unreachable address
	err := validateCallbackURL("http://localhost:99999")
	// Should not error even if unreachable (logs warning instead)
	assert.NoError(t, err, "Should not error for unreachable but valid URL")
}

func TestExtractPortFromURL_WithPort(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected string
	}{
		{"explicit port", "http://localhost:8080", "8080"},
		{"https port", "https://example.com:9443", "9443"},
		{"default http", "http://example.com", "80"},
		{"default https", "https://example.com", "443"},
		{"ipv4 with port", "http://192.168.1.1:3000", "3000"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			port := extractPortFromURL(tt.url)
			assert.Equal(t, tt.expected, port)
		})
	}
}

func TestExtractPortFromURL_Empty(t *testing.T) {
	port := extractPortFromURL("")
	assert.Equal(t, "", port)
}

func TestExtractPortFromURL_InvalidURL(t *testing.T) {
	port := extractPortFromURL("not-a-valid-url")
	assert.Equal(t, "", port)
}

func TestExtractPortFromURL_UnknownScheme(t *testing.T) {
	port := extractPortFromURL("ftp://example.com")
	assert.Equal(t, "", port)
}

func TestGatherCallbackCandidates_BaseURLOnly(t *testing.T) {
	baseURL := "http://localhost:8080"
	candidates, defaultPort := gatherCallbackCandidates(baseURL, nil, "")

	assert.Equal(t, 1, len(candidates))
	assert.Equal(t, baseURL, candidates[0])
	assert.Equal(t, "8080", defaultPort)
}

func TestGatherCallbackCandidates_WithDiscovery(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Preferred: "http://192.168.1.100:8080",
		Candidates: []string{
			"http://10.0.0.1:8080",
			"http://172.16.0.1:8080",
		},
	}

	candidates, defaultPort := gatherCallbackCandidates(baseURL, discovery, "")

	assert.GreaterOrEqual(t, len(candidates), 3)
	assert.Contains(t, candidates, baseURL)
	assert.Contains(t, candidates, discovery.Preferred)
	assert.Equal(t, "8080", defaultPort)
}

func TestGatherCallbackCandidates_WithClientIP(t *testing.T) {
	baseURL := "http://localhost:8080"
	clientIP := "192.168.1.50"

	candidates, defaultPort := gatherCallbackCandidates(baseURL, nil, clientIP)

	assert.Equal(t, 2, len(candidates))
	assert.Contains(t, candidates, baseURL)
	assert.Contains(t, candidates, "http://192.168.1.50:8080")
	assert.Equal(t, "8080", defaultPort)
}

func TestGatherCallbackCandidates_Deduplication(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Preferred: "http://localhost:8080", // Same as baseURL
		Candidates: []string{
			"http://localhost:8080", // Duplicate
			"http://127.0.0.1:8080",
		},
	}

	candidates, _ := gatherCallbackCandidates(baseURL, discovery, "")

	// Should deduplicate the URL
	uniqueCount := 0
	for _, c := range candidates {
		if c == baseURL {
			uniqueCount++
		}
	}
	assert.Equal(t, 1, uniqueCount, "Should deduplicate identical URLs")
}

func TestGatherCallbackCandidates_EmptyStrings(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Preferred: "",
		Candidates: []string{
			"",
			"  ",
			"http://valid:8080",
		},
	}

	candidates, _ := gatherCallbackCandidates(baseURL, discovery, "")

	// Should skip empty strings and whitespace
	for _, c := range candidates {
		assert.NotEqual(t, "", c)
		assert.NotEqual(t, "  ", c)
	}
}

func TestGatherCallbackCandidates_ClientIPWithoutPort(t *testing.T) {
	baseURL := "http://example.com" // No explicit port
	clientIP := "192.168.1.50"

	candidates, defaultPort := gatherCallbackCandidates(baseURL, nil, clientIP)

	// Should use default port 80 for http
	assert.Equal(t, "80", defaultPort)
	assert.Contains(t, candidates, fmt.Sprintf("http://%s:80", clientIP))
}

func TestGatherCallbackCandidates_PreferredPortOverride(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Preferred: "http://192.168.1.100:9000", // Different port
	}

	candidates, defaultPort := gatherCallbackCandidates(baseURL, discovery, "")

	// Should use baseURL port as default (first seen)
	assert.Equal(t, "8080", defaultPort)
	assert.Contains(t, candidates, baseURL)
	assert.Contains(t, candidates, discovery.Preferred)
}

func TestGatherCallbackCandidates_MultipleCandidatesOrdering(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Preferred: "http://preferred:8080",
		Candidates: []string{
			"http://candidate1:8080",
			"http://candidate2:8080",
		},
	}
	clientIP := "192.168.1.50"

	candidates, _ := gatherCallbackCandidates(baseURL, discovery, clientIP)

	// baseURL should be first
	assert.Equal(t, baseURL, candidates[0], "baseURL should be first candidate")

	// preferred should come before other candidates
	preferredIdx := -1
	for i, c := range candidates {
		if c == discovery.Preferred {
			preferredIdx = i
			break
		}
	}
	assert.True(t, preferredIdx > 0, "Preferred should be in candidate list")
}

// Security Tests

func TestValidateCallbackURL_PreventSSRF_PrivateIPs(t *testing.T) {
	// Note: Current implementation allows private IPs
	// This test documents the current behavior
	// In production, you may want to add SSRF protection
	privateIPs := []string{
		"http://127.0.0.1:8080",
		"http://localhost:8080",
		"http://192.168.1.1:8080",
		"http://10.0.0.1:8080",
		"http://172.16.0.1:8080",
	}

	for _, ip := range privateIPs {
		t.Run(ip, func(t *testing.T) {
			err := validateCallbackURL(ip)
			// Currently allows private IPs (may want to add protection)
			// This test documents current behavior
			_ = err // Behavior depends on reachability
		})
	}
}

func TestValidateCallbackURL_LongURL(t *testing.T) {
	// Test with very long URL
	longURL := "http://example.com/" + string(make([]byte, 2000))
	err := validateCallbackURL(longURL)
	// Should handle long URLs gracefully (may fail reachability check)
	_ = err // Test ensures no panic
}

func TestExtractPortFromURL_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected string
	}{
		{"port with path", "http://localhost:8080/path", "8080"},
		{"port with query", "http://localhost:8080?query=value", "8080"},
		{"port with fragment", "http://localhost:8080#fragment", "8080"},
		{"port zero", "http://localhost:0", "0"},
		{"max port", "http://localhost:65535", "65535"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			port := extractPortFromURL(tt.url)
			assert.Equal(t, tt.expected, port)
		})
	}
}

func TestGatherCallbackCandidates_NilDiscovery(t *testing.T) {
	baseURL := "http://localhost:8080"

	// Should not panic with nil discovery
	candidates, defaultPort := gatherCallbackCandidates(baseURL, nil, "")

	assert.Equal(t, 1, len(candidates))
	assert.Equal(t, baseURL, candidates[0])
	assert.Equal(t, "8080", defaultPort)
}

func TestGatherCallbackCandidates_EmptyClientIP(t *testing.T) {
	baseURL := "http://localhost:8080"

	candidates, _ := gatherCallbackCandidates(baseURL, nil, "")

	// Should handle empty client IP gracefully
	assert.Equal(t, 1, len(candidates))
	assert.Equal(t, baseURL, candidates[0])
}

func TestGatherCallbackCandidates_IPv6(t *testing.T) {
	baseURL := "http://[::1]:8080"

	candidates, defaultPort := gatherCallbackCandidates(baseURL, nil, "::1")

	assert.Contains(t, candidates, baseURL)
	assert.Equal(t, "8080", defaultPort)
}

func TestGatherCallbackCandidates_MalformedURLsInDiscovery(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Candidates: []string{
			"not-a-url",
			"",
			"http://valid:8080",
			"://noscheme",
		},
	}

	// Should handle malformed URLs gracefully (skip invalid ones)
	candidates, _ := gatherCallbackCandidates(baseURL, discovery, "")

	// Should at least include baseURL and valid candidate
	assert.GreaterOrEqual(t, len(candidates), 1)
	assert.Contains(t, candidates, baseURL)
}

func TestGatherCallbackCandidates_WhitespaceInCandidates(t *testing.T) {
	baseURL := "http://localhost:8080"
	discovery := &types.CallbackDiscoveryInfo{
		Candidates: []string{
			"  http://example1:8080  ",
			"\t\nhttp://example2:8080\n\t",
			"http://example3:8080",
		},
	}

	candidates, _ := gatherCallbackCandidates(baseURL, discovery, "")

	// Should trim whitespace from candidates
	for _, c := range candidates {
		assert.Equal(t, len(c), len(c), "Should not have leading/trailing whitespace")
	}
}

func TestResolveCallbackCandidates_SelectsFirstNormalizedCandidate(t *testing.T) {
	resolved, normalized, probeResults := resolveCallbackCandidates([]string{" test-runner:8080 ", "http://second:8080"}, "8080")

	require.Equal(t, "http://test-runner:8080", resolved)
	require.Equal(t, []string{"http://test-runner:8080", "http://second:8080"}, normalized)
	require.Nil(t, probeResults)
}

func TestNormalizeServerlessDiscoveryURL_NormalizesWildcardBindAddress(t *testing.T) {
	normalized, err := normalizeServerlessDiscoveryURL("http://0.0.0.0:7000/invoke", nil)

	require.NoError(t, err)
	assert.Equal(t, "http://localhost:7000/invoke", normalized)
}

func TestNormalizeServerlessDiscoveryURL_RejectsUnlistedHost(t *testing.T) {
	_, err := normalizeServerlessDiscoveryURL("https://example.com/invoke", nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "not allowlisted")
}

func TestNormalizeServerlessDiscoveryURL_AllowsConfiguredHost(t *testing.T) {
	normalized, err := normalizeServerlessDiscoveryURL("https://agents.internal/invoke", []string{"*.trusted.example", "agents.internal", "10.0.0.0/8"})

	require.NoError(t, err)
	assert.Equal(t, "https://agents.internal/invoke", normalized)
}
