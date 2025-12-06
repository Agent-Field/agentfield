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

func init() {
	gin.SetMode(gin.TestMode)
}

func setupRouter(config AuthConfig) *gin.Engine {
	router := gin.New()
	router.Use(APIKeyAuth(config))
	router.GET("/api/v1/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "success"})
	})
	router.GET("/api/v1/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})
	router.GET("/metrics", func(c *gin.Context) {
		c.String(http.StatusOK, "metrics_data")
	})
	router.GET("/ui/index.html", func(c *gin.Context) {
		c.String(http.StatusOK, "<html>UI</html>")
	})
	router.GET("/custom/skip", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "skipped"})
	})
	return router
}

func TestAPIKeyAuth_NoAuthConfigured(t *testing.T) {
	// When no API key is configured, all requests should be allowed
	router := setupRouter(AuthConfig{APIKey: ""})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]string
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, "success", resp["message"])
}

func TestAPIKeyAuth_ValidXAPIKeyHeader(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
	req.Header.Set("X-API-Key", "secret-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_ValidBearerToken(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_ValidQueryParam(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test?api_key=secret-key", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_InvalidKey(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	tests := []struct {
		name        string
		headerKey   string
		headerValue string
		queryParam  string
	}{
		{
			name:        "wrong X-API-Key",
			headerKey:   "X-API-Key",
			headerValue: "wrong-key",
		},
		{
			name:        "wrong bearer token",
			headerKey:   "Authorization",
			headerValue: "Bearer wrong-key",
		},
		{
			name:       "wrong query param",
			queryParam: "wrong-key",
		},
		{
			name: "no auth at all",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := "/api/v1/test"
			if tt.queryParam != "" {
				url += "?api_key=" + tt.queryParam
			}

			req := httptest.NewRequest(http.MethodGet, url, nil)
			if tt.headerKey != "" {
				req.Header.Set(tt.headerKey, tt.headerValue)
			}
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusUnauthorized, w.Code)

			var resp map[string]string
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			require.NoError(t, err)
			assert.Equal(t, "unauthorized", resp["error"])
			assert.Contains(t, resp["message"], "invalid or missing API key")
		})
	}
}

func TestAPIKeyAuth_SkipHealthEndpoint(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	// Health endpoint should be accessible without auth
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_SkipHealthSubpaths(t *testing.T) {
	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/health/ready", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health/ready", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_SkipMetricsEndpoint(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_SkipUIPath(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/ui/index.html", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_CustomSkipPaths(t *testing.T) {
	router := setupRouter(AuthConfig{
		APIKey:    "secret-key",
		SkipPaths: []string{"/custom/skip"},
	})

	// Custom skip path should be accessible without auth
	req := httptest.NewRequest(http.MethodGet, "/custom/skip", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_XAPIKeyTakesPrecedence(t *testing.T) {
	// If X-API-Key is set, it should be checked first
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
	// Valid X-API-Key should succeed even with invalid bearer
	req.Header.Set("X-API-Key", "secret-key")
	req.Header.Set("Authorization", "Bearer wrong-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_BearerFallback(t *testing.T) {
	// If X-API-Key is empty, should fall back to Bearer token
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
	req.Header.Set("X-API-Key", "") // Empty, not missing
	req.Header.Set("Authorization", "Bearer secret-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAPIKeyAuth_InvalidBearerFormat(t *testing.T) {
	router := setupRouter(AuthConfig{APIKey: "secret-key"})

	tests := []struct {
		name   string
		header string
	}{
		{"no Bearer prefix", "secret-key"},
		{"Basic auth instead", "Basic secret-key"},
		{"malformed Bearer", "Bearersecret-key"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/v1/test", nil)
			req.Header.Set("Authorization", tt.header)
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusUnauthorized, w.Code)
		})
	}
}

func TestAPIKeyAuth_MultipleSkipPaths(t *testing.T) {
	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{
		APIKey:    "secret-key",
		SkipPaths: []string{"/public/a", "/public/b", "/public/c"},
	}))
	router.GET("/public/a", func(c *gin.Context) { c.String(http.StatusOK, "a") })
	router.GET("/public/b", func(c *gin.Context) { c.String(http.StatusOK, "b") })
	router.GET("/public/c", func(c *gin.Context) { c.String(http.StatusOK, "c") })
	router.GET("/private", func(c *gin.Context) { c.String(http.StatusOK, "private") })

	// All public paths should be accessible
	for _, path := range []string{"/public/a", "/public/b", "/public/c"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code, "path %s should be accessible", path)
	}

	// Private path should require auth
	req := httptest.NewRequest(http.MethodGet, "/private", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
}
