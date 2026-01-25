package ai

import (
	"errors"
	"os"
	"time"
)

// Config holds AI/LLM configuration for making API calls.
type Config struct {
	// API Key for OpenAI or OpenRouter
	APIKey string

	// BaseURL can be either OpenAI or OpenRouter endpoint
	// Default: https://api.openai.com/v1
	// OpenRouter: https://openrouter.ai/api/v1
	BaseURL string

	// Default model to use (e.g., "gpt-4o", "openai/gpt-4o" for OpenRouter)
	Model string

	// Default temperature for responses (0.0 to 2.0)
	Temperature float64

	// Default max tokens for responses
	MaxTokens int

	// HTTP timeout for requests
	Timeout time.Duration

	// Optional: Site URL for OpenRouter rankings
	SiteURL string

	// Optional: Site name for OpenRouter rankings
	SiteName string

	// Rate Limiter Configuration
	RateLimitMaxRetries         int           // Maximum number of retry attempts (default: 5)
	RateLimitBaseDelay          time.Duration // Base delay for exponential backoff (default: 1s)
	RateLimitMaxDelay           time.Duration // Maximum delay between retries (default: 30s)
	RateLimitJitterFactor       float64       // Jitter factor 0.0-1.0 (default: 0.1)
	CircuitBreakerThreshold     int           // Consecutive failures before opening circuit (default: 5)
	CircuitBreakerTimeout       time.Duration // Time before attempting to close circuit (default: 60s)
	DisableRateLimiter          bool          // Disable rate limiting completely (default: false)
}

// DefaultConfig returns a Config with sensible defaults.
// It reads from environment variables:
// - OPENAI_API_KEY or OPENROUTER_API_KEY
// - AI_BASE_URL (defaults to OpenAI)
// - AI_MODEL (defaults to gpt-4o)
func DefaultConfig() *Config {
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := "https://api.openai.com/v1"

	// Check for OpenRouter configuration
	if routerKey := os.Getenv("OPENROUTER_API_KEY"); routerKey != "" {
		apiKey = routerKey
		baseURL = "https://openrouter.ai/api/v1"
	}

	// Allow override via AI_BASE_URL
	if customURL := os.Getenv("AI_BASE_URL"); customURL != "" {
		baseURL = customURL
	}

	model := os.Getenv("AI_MODEL")
	if model == "" {
		model = "gpt-4o"
	}

	return &Config{
		APIKey:      apiKey,
		BaseURL:     baseURL,
		Model:       model,
		Temperature: 0.7,
		MaxTokens:   4096,
		Timeout:     30 * time.Second,

		// Rate Limiter Defaults
		RateLimitMaxRetries:         5,
		RateLimitBaseDelay:          time.Second,
		RateLimitMaxDelay:           30 * time.Second,
		RateLimitJitterFactor:       0.1,
		CircuitBreakerThreshold:     5,
		CircuitBreakerTimeout:       60 * time.Second,
		DisableRateLimiter:          false,
	}
}

// Validate ensures the configuration is valid.
func (c *Config) Validate() error {
	if c.APIKey == "" {
		return errors.New("API key is required")
	}
	if c.BaseURL == "" {
		return errors.New("base URL is required")
	}
	if c.Model == "" {
		return errors.New("model is required")
	}
	return nil
}

// IsOpenRouter returns true if the base URL is for OpenRouter.
func (c *Config) IsOpenRouter() bool {
	return c.BaseURL == "https://openrouter.ai/api/v1" ||
		c.BaseURL == "https://openrouter.ai/api/v1/"
}
