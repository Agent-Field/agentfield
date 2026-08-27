package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestRateLimiterStore_Allow(t *testing.T) {
	store := NewRateLimiterStore(1, 2, time.Minute) // 1 req/s, burst 2
	defer store.Stop()

	// Burst of 2 should be allowed
	assert.True(t, store.Allow("key-1"))
	assert.True(t, store.Allow("key-1"))

	// Third request exceeds burst
	assert.False(t, store.Allow("key-1"))

	// Different key has its own limit
	assert.True(t, store.Allow("key-2"))
}

func TestRateLimiterStore_Eviction(t *testing.T) {
	store := NewRateLimiterStore(10, 10, 50*time.Millisecond)
	defer store.Stop()

	store.Allow("evict-me")
	require.Equal(t, 1, store.Len())

	// Wait for eviction
	time.Sleep(100 * time.Millisecond)

	// Trigger eviction cycle
	store.evict()
	assert.Equal(t, 0, store.Len())
}

func TestRateLimiterStore_ActiveKeyNotEvicted(t *testing.T) {
	store := NewRateLimiterStore(10, 10, 200*time.Millisecond)
	defer store.Stop()

	store.Allow("active-key")

	// Keep it active
	time.Sleep(50 * time.Millisecond)
	store.Allow("active-key")

	// Evict — key should survive because lastSeen is recent
	store.evict()
	assert.Equal(t, 1, store.Len())
}

func TestRateLimit_Middleware_AllowsNormalTraffic(t *testing.T) {
	store := NewRateLimiterStore(10, 20, time.Minute)
	defer store.Stop()

	router := gin.New()
	router.Use(RateLimit(store))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "192.168.1.1:12345"
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusOK, resp.Code)
}

func TestRateLimit_Middleware_Returns429WhenExceeded(t *testing.T) {
	store := NewRateLimiterStore(1, 2, time.Minute) // 1 req/s, burst 2
	defer store.Stop()

	router := gin.New()
	router.Use(RateLimit(store))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	// Exhaust burst
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		req.RemoteAddr = "10.0.0.1:9999"
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)
	}

	// This should be rate-limited
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "10.0.0.1:9999"
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusTooManyRequests, resp.Code)

	// Verify Retry-After header
	assert.NotEmpty(t, resp.Header().Get("Retry-After"))

	// Verify response body
	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	assert.Equal(t, "rate limit exceeded", body["error"])
	assert.Equal(t, "rate_limit", body["error_category"])
	assert.NotEmpty(t, body["retry_after"])
}

func TestRateLimit_Middleware_PerKeyIsolation(t *testing.T) {
	store := NewRateLimiterStore(1, 1, time.Minute) // 1 req/s, burst 1
	defer store.Stop()

	router := gin.New()
	router.Use(RateLimit(store))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	// First client exhausts its limit
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "10.0.0.1:1234"
	resp1 := httptest.NewRecorder()
	router.ServeHTTP(resp1, req1)
	assert.Equal(t, http.StatusOK, resp1.Code)

	// First client is now rate-limited
	req1b := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1b.RemoteAddr = "10.0.0.1:1234"
	resp1b := httptest.NewRecorder()
	router.ServeHTTP(resp1b, req1b)
	assert.Equal(t, http.StatusTooManyRequests, resp1b.Code)

	// Second client is NOT affected
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "10.0.0.2:1234"
	resp2 := httptest.NewRecorder()
	router.ServeHTTP(resp2, req2)
	assert.Equal(t, http.StatusOK, resp2.Code)
}

func TestRateLimit_Middleware_UsesAPIKeyWhenAvailable(t *testing.T) {
	store := NewRateLimiterStore(1, 1, time.Minute)
	defer store.Stop()

	router := gin.New()
	// Simulate auth middleware setting api_key_identity
	router.Use(func(c *gin.Context) {
		c.Set("api_key_identity", "my-api-key")
		c.Next()
	})
	router.Use(RateLimit(store))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	// First request from "IP A" succeeds (keyed by API key, not IP)
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "10.0.0.1:1111"
	resp1 := httptest.NewRecorder()
	router.ServeHTTP(resp1, req1)
	assert.Equal(t, http.StatusOK, resp1.Code)

	// Second request from "IP B" but same API key is rate-limited
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "10.0.0.2:2222"
	resp2 := httptest.NewRecorder()
	router.ServeHTTP(resp2, req2)
	assert.Equal(t, http.StatusTooManyRequests, resp2.Code)
}

func TestRateLimitConfig_IsEnabled(t *testing.T) {
	t.Run("nil enabled with rps", func(t *testing.T) {
		cfg := RateLimitConfig{RequestsPerSecond: 10, BurstSize: 20}
		assert.True(t, cfg.IsEnabled())
	})

	t.Run("nil enabled without rps", func(t *testing.T) {
		cfg := RateLimitConfig{}
		assert.False(t, cfg.IsEnabled())
	})

	t.Run("explicitly enabled", func(t *testing.T) {
		enabled := true
		cfg := RateLimitConfig{Enabled: &enabled}
		assert.True(t, cfg.IsEnabled())
	})

	t.Run("explicitly disabled", func(t *testing.T) {
		disabled := false
		cfg := RateLimitConfig{Enabled: &disabled, RequestsPerSecond: 50}
		assert.False(t, cfg.IsEnabled())
	})
}

func TestDefaultRateLimitConfig(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	assert.Greater(t, cfg.Execute.RequestsPerSecond, 0.0)
	assert.Greater(t, cfg.Execute.BurstSize, 0)
	assert.Greater(t, cfg.Discovery.RequestsPerSecond, 0.0)
	assert.Greater(t, cfg.BulkStatus.RequestsPerSecond, 0.0)
	assert.Greater(t, cfg.Global.RequestsPerSecond, 0.0)
}
