package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupPrivilegedRouter(config AuthConfig) *gin.Engine {
	router := gin.New()
	router.Use(PrivilegedAccess(config))
	router.POST("/api/ui/v1/agents/packages/install", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "installed"})
	})
	return router
}

func privilegedRequest(t *testing.T, router *gin.Engine, remoteAddr string, mutate func(*http.Request)) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/agents/packages/install", nil)
	req.RemoteAddr = remoteAddr
	if mutate != nil {
		mutate(req)
	}
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

// Without a configured key the endpoint is local-only: the default developer
// setup keeps working, but nothing else on the network can reach it.
func TestPrivilegedAccess_NoAPIKey(t *testing.T) {
	router := setupPrivilegedRouter(AuthConfig{APIKey: ""})

	tests := []struct {
		name       string
		remoteAddr string
		wantStatus int
	}{
		{"IPv4 loopback allowed", "127.0.0.1:54321", http.StatusOK},
		{"IPv4 loopback range allowed", "127.0.1.7:54321", http.StatusOK},
		{"IPv6 loopback allowed", "[::1]:54321", http.StatusOK},
		// Dual-stack hosts report an IPv4 client this way; it is genuinely local.
		{"IPv4-mapped IPv6 loopback allowed", "[::ffff:127.0.0.1]:54321", http.StatusOK},
		{"private LAN address rejected", "192.168.2.16:54321", http.StatusUnauthorized},
		{"other private range rejected", "10.4.1.9:54321", http.StatusUnauthorized},
		{"public address rejected", "203.0.113.9:54321", http.StatusUnauthorized},
		{"unparseable address rejected", "not-an-address", http.StatusUnauthorized},
		{"empty address rejected", "", http.StatusUnauthorized},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := privilegedRequest(t, router, tt.remoteAddr, nil)
			assert.Equal(t, tt.wantStatus, w.Code)
		})
	}
}

// gin trusts every proxy by default, so ClientIP() reflects caller-supplied
// forwarding headers. The guard must key off the real TCP peer instead, or a
// remote host can simply announce itself as 127.0.0.1.
func TestPrivilegedAccess_ForwardedHeadersCannotForgeLoopback(t *testing.T) {
	router := setupPrivilegedRouter(AuthConfig{APIKey: ""})

	spoofHeaders := []struct {
		name  string
		key   string
		value string
	}{
		{"X-Forwarded-For", "X-Forwarded-For", "127.0.0.1"},
		{"X-Forwarded-For chain", "X-Forwarded-For", "127.0.0.1, 10.0.0.1"},
		{"X-Real-IP", "X-Real-IP", "127.0.0.1"},
		{"X-Forwarded-For IPv6 loopback", "X-Forwarded-For", "::1"},
		{"X-Forwarded-For IPv4-mapped loopback", "X-Forwarded-For", "::ffff:127.0.0.1"},
		{"Forwarded (RFC 7239)", "Forwarded", "for=127.0.0.1"},
	}

	for _, tt := range spoofHeaders {
		t.Run(tt.name, func(t *testing.T) {
			w := privilegedRequest(t, router, "192.168.2.16:54321", func(r *http.Request) {
				r.Header.Set(tt.key, tt.value)
			})
			assert.Equal(t, http.StatusUnauthorized, w.Code,
				"forwarded headers must not grant privileged access")
		})
	}
}

// Once a key is configured it is required everywhere, including from the local
// host: enabling authentication should not leave a silent local bypass.
func TestPrivilegedAccess_WithAPIKey(t *testing.T) {
	const key = "s3cret-key"
	router := setupPrivilegedRouter(AuthConfig{APIKey: key})

	tests := []struct {
		name       string
		remoteAddr string
		mutate     func(*http.Request)
		wantStatus int
	}{
		{
			name:       "remote caller with header key",
			remoteAddr: "192.168.2.16:54321",
			mutate:     func(r *http.Request) { r.Header.Set("X-API-Key", key) },
			wantStatus: http.StatusOK,
		},
		{
			name:       "remote caller with bearer token",
			remoteAddr: "192.168.2.16:54321",
			mutate:     func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+key) },
			wantStatus: http.StatusOK,
		},
		{
			name:       "remote caller with wrong key",
			remoteAddr: "192.168.2.16:54321",
			mutate:     func(r *http.Request) { r.Header.Set("X-API-Key", "wrong") },
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "remote caller with no key",
			remoteAddr: "192.168.2.16:54321",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "loopback caller still needs the key",
			remoteAddr: "127.0.0.1:54321",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "loopback caller with key",
			remoteAddr: "127.0.0.1:54321",
			mutate:     func(r *http.Request) { r.Header.Set("X-API-Key", key) },
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := privilegedRequest(t, router, tt.remoteAddr, tt.mutate)
			assert.Equal(t, tt.wantStatus, w.Code)
		})
	}
}

// The query-string fallback exists for browser EventSource and WebSocket
// clients, which never call these routes; accepting it here would leak the key
// into access logs.
func TestPrivilegedAccess_RejectsQueryStringKey(t *testing.T) {
	const key = "s3cret-key"
	router := gin.New()
	router.Use(PrivilegedAccess(AuthConfig{APIKey: key}))
	router.POST("/api/ui/v1/agents/packages/install", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "installed"})
	})

	req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/agents/packages/install?api_key="+key, nil)
	req.RemoteAddr = "192.168.2.16:54321"
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

// A rejected operator needs to know how to fix it, in both rejection modes.
func TestPrivilegedAccess_RejectionExplainsRemedy(t *testing.T) {
	t.Run("no api key configured", func(t *testing.T) {
		router := setupPrivilegedRouter(AuthConfig{APIKey: ""})
		w := privilegedRequest(t, router, "192.168.2.16:54321", nil)

		require.Equal(t, http.StatusUnauthorized, w.Code)
		var resp struct {
			Error   string            `json:"error"`
			Message string            `json:"message"`
			Help    map[string]string `json:"help"`
		}
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, "unauthorized", resp.Error)
		assert.NotEmpty(t, resp.Message)
		assert.Contains(t, resp.Help["enable_auth"], "AGENTFIELD_API_KEY")
		assert.Contains(t, resp.Help["cli"], "af auth login")
	})

	t.Run("api key configured", func(t *testing.T) {
		router := setupPrivilegedRouter(AuthConfig{APIKey: "s3cret-key"})
		w := privilegedRequest(t, router, "192.168.2.16:54321", nil)

		require.Equal(t, http.StatusUnauthorized, w.Code)
		var resp struct {
			Error string            `json:"error"`
			Help  map[string]string `json:"help"`
		}
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, "unauthorized", resp.Error)
		assert.Contains(t, resp.Help["cli"], "af auth login")
	})
}

// Downstream agentic discovery filters on auth_level; a rejected caller must
// not be advertised as privileged.
func TestPrivilegedAccess_SetsAuthLevel(t *testing.T) {
	tests := []struct {
		name       string
		config     AuthConfig
		remoteAddr string
		mutate     func(*http.Request)
		wantLevel  string
	}{
		{"authenticated caller", AuthConfig{APIKey: "k"}, "192.168.2.16:1", func(r *http.Request) { r.Header.Set("X-API-Key", "k") }, "api_key"},
		{"rejected caller", AuthConfig{APIKey: "k"}, "192.168.2.16:1", nil, "public"},
		{"rejected remote in no-auth mode", AuthConfig{APIKey: ""}, "192.168.2.16:1", nil, "public"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/agents/packages/install", nil)
			req.RemoteAddr = tt.remoteAddr
			if tt.mutate != nil {
				tt.mutate(req)
			}
			c.Request = req

			PrivilegedAccess(tt.config)(c)

			level, _ := c.Get("auth_level")
			assert.Equal(t, tt.wantLevel, level)
		})
	}
}
