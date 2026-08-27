package middleware

import (
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// RateLimitConfig defines per-endpoint rate limiting parameters.
type RateLimitConfig struct {
	// RequestsPerSecond is the sustained rate of allowed requests per key.
	RequestsPerSecond float64 `yaml:"requests_per_second" mapstructure:"requests_per_second"`
	// BurstSize is the maximum burst of requests allowed above the sustained rate.
	BurstSize int `yaml:"burst_size" mapstructure:"burst_size"`
	// Enabled controls whether rate limiting is active. Defaults to true when
	// any rate limit config is present.
	Enabled *bool `yaml:"enabled,omitempty" mapstructure:"enabled"`
}

// IsEnabled returns whether this rate limit config is active.
func (c RateLimitConfig) IsEnabled() bool {
	if c.Enabled != nil {
		return *c.Enabled
	}
	// Enabled by default if any non-zero config is provided.
	return c.RequestsPerSecond > 0 || c.BurstSize > 0
}

// DefaultRateLimitConfig returns sensible defaults for production use.
// Execute: 50 req/s burst 100 (per key) — generous for normal agent traffic.
// Discovery: 20 req/s burst 40 — read-heavy but cacheable.
// BulkStatus: 30 req/s burst 60 — moderate fan-out.
// Global: 200 req/s burst 400 — per-IP fallback for unauthenticated paths.
func DefaultRateLimitConfig() RateLimitsConfig {
	return RateLimitsConfig{
		Execute:    RateLimitConfig{RequestsPerSecond: 50, BurstSize: 100},
		Discovery:  RateLimitConfig{RequestsPerSecond: 20, BurstSize: 40},
		BulkStatus: RateLimitConfig{RequestsPerSecond: 30, BurstSize: 60},
		Global:     RateLimitConfig{RequestsPerSecond: 200, BurstSize: 400},
	}
}

// RateLimitsConfig holds rate limit settings for different endpoint groups.
type RateLimitsConfig struct {
	Execute    RateLimitConfig `yaml:"execute" mapstructure:"execute"`
	Discovery  RateLimitConfig `yaml:"discovery" mapstructure:"discovery"`
	BulkStatus RateLimitConfig `yaml:"bulk_status" mapstructure:"bulk_status"`
	Global     RateLimitConfig `yaml:"global" mapstructure:"global"`
}

// rateLimiterEntry holds a limiter and the last time it was accessed.
type rateLimiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// RateLimiterStore manages per-key rate limiters with idle eviction.
type RateLimiterStore struct {
	mu       sync.Mutex
	limiters map[string]*rateLimiterEntry
	rps      rate.Limit
	burst    int
	ttl      time.Duration
	stopOnce sync.Once
	stopCh   chan struct{}
}

// NewRateLimiterStore creates a store for per-key rate limiters.
// Keys idle for longer than ttl are evicted by a background goroutine.
func NewRateLimiterStore(rps float64, burst int, ttl time.Duration) *RateLimiterStore {
	s := &RateLimiterStore{
		limiters: make(map[string]*rateLimiterEntry),
		rps:      rate.Limit(rps),
		burst:    burst,
		ttl:      ttl,
		stopCh:   make(chan struct{}),
	}
	go s.evictLoop()
	return s
}

// Allow checks whether the given key is within its rate limit.
func (s *RateLimiterStore) Allow(key string) bool {
	s.mu.Lock()
	entry, ok := s.limiters[key]
	if !ok {
		entry = &rateLimiterEntry{
			limiter:  rate.NewLimiter(s.rps, s.burst),
			lastSeen: time.Now(),
		}
		s.limiters[key] = entry
	} else {
		entry.lastSeen = time.Now()
	}
	s.mu.Unlock()
	return entry.limiter.Allow()
}

// Stop halts the background eviction goroutine.
func (s *RateLimiterStore) Stop() {
	s.stopOnce.Do(func() {
		close(s.stopCh)
	})
}

// Len returns the number of tracked keys (for testing/metrics).
func (s *RateLimiterStore) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.limiters)
}

func (s *RateLimiterStore) evictLoop() {
	ticker := time.NewTicker(s.ttl / 2)
	defer ticker.Stop()
	for {
		select {
		case <-s.stopCh:
			return
		case <-ticker.C:
			s.evict()
		}
	}
}

func (s *RateLimiterStore) evict() {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for key, entry := range s.limiters {
		if now.Sub(entry.lastSeen) > s.ttl {
			delete(s.limiters, key)
		}
	}
}

// RateLimit returns a Gin middleware that enforces per-key rate limiting.
// The key is derived from the API key (if authenticated) or the client IP.
// When the limit is exceeded, the middleware responds with HTTP 429 and a
// Retry-After header.
func RateLimit(store *RateLimiterStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := rateLimitKey(c)
		if !store.Allow(key) {
			retryAfter := fmt.Sprintf("%.0f", 1.0/float64(store.rps))
			c.Header("Retry-After", retryAfter)
			c.Header("X-RateLimit-Limit", fmt.Sprintf("%.0f", float64(store.rps)))
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error":          "rate limit exceeded",
				"error_category": "rate_limit",
				"retry_after":    retryAfter,
			})
			return
		}
		c.Next()
	}
}

// rateLimitKey extracts the rate-limiting key from the request context.
// Prefers API key (set by auth middleware) over client IP.
func rateLimitKey(c *gin.Context) string {
	// Check if auth middleware set an API key identity.
	if apiKey, exists := c.Get("api_key_identity"); exists {
		if key, ok := apiKey.(string); ok && key != "" {
			return "key:" + key
		}
	}

	// Fall back to client IP.
	ip := clientIP(c)
	return "ip:" + ip
}

// clientIP extracts the real client IP, respecting X-Forwarded-For and
// X-Real-IP headers (common behind reverse proxies).
func clientIP(c *gin.Context) string {
	// Gin's ClientIP already handles trusted proxies.
	ip := c.ClientIP()
	if ip != "" {
		return ip
	}

	// Fallback: parse RemoteAddr.
	host, _, err := net.SplitHostPort(c.Request.RemoteAddr)
	if err != nil {
		return strings.TrimSpace(c.Request.RemoteAddr)
	}
	return host
}
