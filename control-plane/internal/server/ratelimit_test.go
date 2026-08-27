package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInitRateLimiters_Disabled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &AgentFieldServer{
		config: &config.Config{
			AgentField: config.AgentFieldConfig{
				RateLimit: config.RateLimitConfig{Enabled: false},
			},
		},
	}

	s.initRateLimiters()

	assert.Nil(t, s.rateLimitExecute)
	assert.Nil(t, s.rateLimitDiscovery)
	assert.Nil(t, s.rateLimitBulkStatus)
	assert.Nil(t, s.rateLimitGlobal)
}

func TestInitRateLimiters_Enabled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &AgentFieldServer{
		config: &config.Config{
			AgentField: config.AgentFieldConfig{
				RateLimit: config.RateLimitConfig{
					Enabled:         true,
					ExecuteRPS:      10,
					ExecuteBurst:    20,
					DiscoveryRPS:    5,
					DiscoveryBurst:  10,
					BulkStatusRPS:   8,
					BulkStatusBurst: 16,
					GlobalRPS:       100,
					GlobalBurst:     200,
				},
			},
		},
	}

	s.initRateLimiters()
	defer func() {
		s.rateLimitExecute.Stop()
		s.rateLimitDiscovery.Stop()
		s.rateLimitBulkStatus.Stop()
		s.rateLimitGlobal.Stop()
	}()

	require.NotNil(t, s.rateLimitExecute)
	require.NotNil(t, s.rateLimitDiscovery)
	require.NotNil(t, s.rateLimitBulkStatus)
	require.NotNil(t, s.rateLimitGlobal)
}

func TestInitRateLimiters_DefaultsApplied(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &AgentFieldServer{
		config: &config.Config{
			AgentField: config.AgentFieldConfig{
				RateLimit: config.RateLimitConfig{
					Enabled: true,
					// All zeros — defaults should be applied
				},
			},
		},
	}

	s.initRateLimiters()
	defer func() {
		s.rateLimitExecute.Stop()
		s.rateLimitDiscovery.Stop()
		s.rateLimitBulkStatus.Stop()
		s.rateLimitGlobal.Stop()
	}()

	require.NotNil(t, s.rateLimitExecute)
	require.NotNil(t, s.rateLimitDiscovery)
	require.NotNil(t, s.rateLimitBulkStatus)
	require.NotNil(t, s.rateLimitGlobal)
}

func TestWithBulkStatusRateLimit_NilStore(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &AgentFieldServer{}

	handler := func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	}

	wrapped := s.withBulkStatusRateLimit(handler)

	router := gin.New()
	router.POST("/test", wrapped)

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusOK, resp.Code)
}

func TestWithBulkStatusRateLimit_Enforced(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := middleware.NewRateLimiterStore(1, 1, time.Minute)
	defer store.Stop()

	s := &AgentFieldServer{
		rateLimitBulkStatus: store,
	}

	handler := func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	}

	wrapped := s.withBulkStatusRateLimit(handler)

	router := gin.New()
	router.POST("/test", wrapped)

	// First request succeeds
	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	req.RemoteAddr = "10.0.0.1:1234"
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	assert.Equal(t, http.StatusOK, resp.Code)

	// Second request is rate-limited
	req2 := httptest.NewRequest(http.MethodPost, "/test", nil)
	req2.RemoteAddr = "10.0.0.1:1234"
	resp2 := httptest.NewRecorder()
	router.ServeHTTP(resp2, req2)
	assert.Equal(t, http.StatusTooManyRequests, resp2.Code)
	assert.NotEmpty(t, resp2.Header().Get("Retry-After"))
}

func TestRateLimitKeyFromContext_IP(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	var key string
	router.GET("/test", func(c *gin.Context) {
		key = rateLimitKeyFromContext(c)
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "192.168.1.100:54321"
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, "ip:192.168.1.100", key)
}

func TestRateLimitKeyFromContext_APIKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	var key string
	router.Use(func(c *gin.Context) {
		c.Set("api_key_identity", "my-key-123")
		c.Next()
	})
	router.GET("/test", func(c *gin.Context) {
		key = rateLimitKeyFromContext(c)
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "10.0.0.1:1234"
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, "key:my-key-123", key)
}

func TestStopRateLimiters(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &AgentFieldServer{
		config: &config.Config{
			AgentField: config.AgentFieldConfig{
				RateLimit: config.RateLimitConfig{Enabled: true},
			},
		},
	}

	s.initRateLimiters()
	require.NotNil(t, s.rateLimitExecute)

	// Stopping should not panic
	s.rateLimitExecute.Stop()
	s.rateLimitDiscovery.Stop()
	s.rateLimitBulkStatus.Stop()
	s.rateLimitGlobal.Stop()
}
