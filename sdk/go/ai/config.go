package ai

import (
	"errors"
	"os"
	"strings"
	"time"
)

// defaultInfronBaseURL is the Infron gateway's OpenAI-compatible endpoint.
// onerouter.pro is the domain Infron serves its gateway from; the two names
// refer to the same service, so grepping for either one should land here.
const defaultInfronBaseURL = "https://llm.onerouter.pro/v1"

// defaultIonetBaseURL is IO Intelligence (io.net)'s OpenAI-compatible
// Chat Completions endpoint. Model ids are Hugging Face-style `org/name`
// strings, e.g. "meta-llama/Llama-3.3-70B-Instruct".
const defaultIonetBaseURL = "https://api.intelligence.io.solutions/api/v1"

// Config holds AI/LLM configuration for making API calls.
type Config struct {
	// API Key for the selected provider: OpenAI, Infron,
	// IO Intelligence (io.net), or OpenRouter
	APIKey string

	// BaseURL can be either OpenAI or OpenRouter endpoint
	// Default: https://api.openai.com/v1
	// OpenRouter: https://openrouter.ai/api/v1
	// Infron: https://llm.onerouter.pro/v1
	// IO Intelligence (io.net): https://api.intelligence.io.solutions/api/v1
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

	// Rate limiting configuration. When RateLimitMaxRetries > 0, AI calls are
	// automatically retried on rate-limit responses (HTTP 429/503) with
	// exponential backoff and jitter. Zero values fall back to sensible
	// defaults inside the rate limiter.
	RateLimitMaxRetries   int
	RateLimitBaseDelay    time.Duration
	RateLimitMaxDelay     time.Duration
	RateLimitJitterFactor float64

	// Circuit breaker configuration. When CircuitBreakerThreshold > 0, the
	// client stops issuing requests after that many consecutive rate-limit
	// failures until CircuitBreakerTimeout elapses.
	CircuitBreakerThreshold int
	CircuitBreakerTimeout   time.Duration
}

// DefaultConfig returns a Config with sensible defaults.
// It reads from environment variables:
// - OPENAI_API_KEY or OPENROUTER_API_KEY
// - INFRON_API_KEY
// - IONET_API_KEY
// - AI_BASE_URL (defaults to OpenAI)
// - AI_MODEL (defaults to gpt-4o, or to an io.net model for IO Intelligence)
//
// A provider key that was already honored keeps precedence, so adding a
// gateway key (Infron, IO Intelligence) to an existing environment never
// silently reroutes it.
func DefaultConfig() *Config {
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := "https://api.openai.com/v1"

	// Check for Infron configuration. Only when no direct-provider key is
	// already set: an existing OPENAI_API_KEY keeps precedence so that adding
	// INFRON_API_KEY to a configured environment cannot silently move traffic
	// (and the credential) to a different gateway.
	if infronKey := os.Getenv("INFRON_API_KEY"); infronKey != "" && apiKey == "" {
		apiKey = infronKey
		baseURL = defaultInfronBaseURL
	}

	// Check for IO Intelligence (io.net) configuration, with the same
	// precedence rule: a key from a provider already configured in the
	// environment keeps precedence, so adding IONET_API_KEY never silently
	// reroutes an existing deployment.
	if ionetKey := os.Getenv("IONET_API_KEY"); ionetKey != "" && apiKey == "" {
		apiKey = ionetKey
		baseURL = defaultIonetBaseURL
	}

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
	modelDefaulted := model == ""
	if modelDefaulted {
		model = "gpt-4o"
	}

	cfg := &Config{
		APIKey:      apiKey,
		BaseURL:     baseURL,
		Model:       model,
		Temperature: 0.7,
		MaxTokens:   4096,
		Timeout:     30 * time.Second,
	}
	switch {
	case cfg.IsOpenRouter():
		if attr, ok := resolveOpenRouterAttribution("", ""); ok {
			cfg.SiteURL = attr.siteURL
			cfg.SiteName = attr.appName
		}
	case cfg.IsInfron():
		cfg.SiteURL, cfg.SiteName, _ = resolveInfronAttribution("", "")
	}
	if modelDefaulted && cfg.IsIonet() {
		// The catalog behind the io.net endpoint serves no gpt-4o (its ids
		// are HF-style org/name, e.g. openai/gpt-oss-120b), so the global
		// fallback would 404 there. IsIonet() is the same predicate the
		// request path uses, so a base URL that merely spells the endpoint
		// differently (trailing slash, case) still gets the io.net default.
		// An explicit AI_MODEL always wins.
		cfg.Model = defaultIonetModel
	}
	return cfg
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
	return strings.Contains(strings.ToLower(c.BaseURL), "openrouter.ai") ||
		strings.HasPrefix(strings.ToLower(c.Model), "openrouter/")
}

// IsInfron returns true if the base URL is for the Infron gateway.
//
// This is checked last everywhere it is used: the gateways this package
// already supported serve the same `<provider>/<model>` ids, so an explicit
// "infron/" prefix is the only thing that distinguishes Infron by model alone,
// and a config that matches both keeps its previous meaning.
func (c *Config) IsInfron() bool {
	return strings.Contains(strings.ToLower(c.BaseURL), "onerouter.pro") ||
		strings.HasPrefix(strings.ToLower(c.Model), infronModelPrefix)
}

// IsIonet returns true if the base URL is for IO Intelligence (io.net).
//
// IO Intelligence serves bare Hugging Face-style `org/name` model ids, so an
// explicit "ionet/" prefix is the only thing that distinguishes it by model
// alone, and a config that matches a gateway already supported keeps its
// previous meaning.
func (c *Config) IsIonet() bool {
	return strings.Contains(strings.ToLower(c.BaseURL), "intelligence.io.solutions") ||
		strings.HasPrefix(strings.ToLower(c.Model), ionetModelPrefix)
}

// RateLimitEnabled reports whether automatic rate-limit retries are configured.
func (c *Config) RateLimitEnabled() bool {
	return c.RateLimitMaxRetries > 0
}

// rateLimiterConfig maps the public Config fields onto a RateLimiterConfig.
func (c *Config) rateLimiterConfig() RateLimiterConfig {
	return RateLimiterConfig{
		MaxRetries:              c.RateLimitMaxRetries,
		BaseDelay:               c.RateLimitBaseDelay,
		MaxDelay:                c.RateLimitMaxDelay,
		JitterFactor:            c.RateLimitJitterFactor,
		CircuitBreakerThreshold: c.CircuitBreakerThreshold,
		CircuitBreakerTimeout:   c.CircuitBreakerTimeout,
	}
}
