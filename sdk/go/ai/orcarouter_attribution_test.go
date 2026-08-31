package ai

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsOrcaRouter(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{
			name:     "OrcaRouter URL without trailing slash",
			baseURL:  "https://api.orcarouter.ai/v1",
			expected: true,
		},
		{
			name:     "OrcaRouter URL with trailing slash",
			baseURL:  "https://api.orcarouter.ai/v1/",
			expected: true,
		},
		{
			name:     "OpenAI URL",
			baseURL:  "https://api.openai.com/v1",
			expected: false,
		},
		{
			name:     "another gateway URL",
			baseURL:  "https://openrouter.ai/api/v1",
			expected: false,
		},
		{
			name:     "empty URL",
			baseURL:  "",
			expected: false,
		},
		{
			name:     "OrcaRouter model prefix",
			baseURL:  "https://api.openai.com/v1",
			model:    "orcarouter/auto",
			expected: true,
		},
		{
			name:     "bare shared model id is not OrcaRouter",
			baseURL:  "https://api.openai.com/v1",
			model:    "auto",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsOrcaRouter())
		})
	}
}

func TestDefaultConfigOrcaRouterKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("ORCAROUTER_API_KEY", "orcarouter-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "orcarouter-key", cfg.APIKey)
	assert.Equal(t, defaultOrcaRouterBaseURL, cfg.BaseURL)
	assert.True(t, cfg.IsOrcaRouter())
	assert.Equal(t, defaultOrcaRouterSiteURL, cfg.SiteURL)
	assert.Equal(t, defaultOrcaRouterAppName, cfg.SiteName)
}

// An OrcaRouter key must never move an existing deployment off the gateway it
// already resolves to — the same backwards-compatibility guarantee the Infron
// integration established.
func TestDefaultConfigExistingGatewayWinsOverOrcaRouter(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "existing-gateway-key")
	t.Setenv("ORCAROUTER_API_KEY", "orcarouter-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "existing-gateway-key", cfg.APIKey)
	assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
	assert.True(t, cfg.IsOpenRouter())
	assert.False(t, cfg.IsOrcaRouter())
}

func TestDefaultConfigExistingOpenAIKeyWinsOverOrcaRouter(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "existing-openai-key")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("ORCAROUTER_API_KEY", "orcarouter-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "existing-openai-key", cfg.APIKey)
	assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
	assert.False(t, cfg.IsOrcaRouter())
}

func TestDefaultConfigOrcaRouterAppliesWhenNoDirectKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "")
	t.Setenv("ORCAROUTER_API_KEY", "orcarouter-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "orcarouter-key", cfg.APIKey)
	assert.Equal(t, defaultOrcaRouterBaseURL, cfg.BaseURL)
	assert.True(t, cfg.IsOrcaRouter())
}

func TestDefaultConfigOrcaRouterAttributionEnvOverrides(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("ORCAROUTER_API_KEY", "orcarouter-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_SITE_URL", "https://custom.example")
	t.Setenv("AGENTFIELD_ORCAROUTER_APP_NAME", "Custom App")

	cfg := DefaultConfig()

	assert.Equal(t, "https://custom.example", cfg.SiteURL)
	assert.Equal(t, "Custom App", cfg.SiteName)
}

// A deployment that already declared its identity keeps it after switching
// gateways, so nobody has to re-declare to move.
func TestOrcaRouterAttributionFallsBackToExistingVars(t *testing.T) {
	t.Setenv("AGENTFIELD_ORCAROUTER_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "https://legacy.example")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "Legacy App")

	header := http.Header{}
	applyOrcaRouterAttributionHeaders(header, "", "")

	assert.Equal(t, "https://legacy.example", header.Get("HTTP-Referer"))
	assert.Equal(t, "Legacy App", header.Get("X-Title"))
}

// The opt-out travels with the inherited values: attribution a deployment
// suppressed for OpenRouter — often because the site URL names an internal
// host — must not be sent to a different vendor either.
func TestOrcaRouterAttributionDoesNotInheritOptedOutValues(t *testing.T) {
	t.Setenv("AGENTFIELD_ORCAROUTER_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_ATTRIBUTION", "false")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "https://internal-tools.corp.example")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "Internal Risk Engine")
	t.Setenv("OR_SITE_URL", "")
	t.Setenv("OR_APP_NAME", "")

	header := http.Header{}
	applyOrcaRouterAttributionHeaders(header, "", "")

	assert.Equal(t, defaultOrcaRouterSiteURL, header.Get("HTTP-Referer"))
	assert.Equal(t, defaultOrcaRouterAppName, header.Get("X-Title"))
}

func TestApplyOrcaRouterAttributionHeadersDisabled(t *testing.T) {
	t.Setenv("AGENTFIELD_ORCAROUTER_ATTRIBUTION", "false")

	header := http.Header{}
	applyOrcaRouterAttributionHeaders(header, "https://example.com", "Example")

	assert.Empty(t, header.Get("HTTP-Referer"))
	assert.Empty(t, header.Get("X-Title"))
}

func TestApplyOrcaRouterAttributionHeadersDefaults(t *testing.T) {
	t.Setenv("AGENTFIELD_ORCAROUTER_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_ORCAROUTER_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "")
	t.Setenv("OR_SITE_URL", "")
	t.Setenv("OR_APP_NAME", "")

	header := http.Header{}
	applyOrcaRouterAttributionHeaders(header, "", "")

	assert.Equal(t, defaultOrcaRouterSiteURL, header.Get("HTTP-Referer"))
	assert.Equal(t, defaultOrcaRouterAppName, header.Get("X-Title"))
}

// OrcaRouter requests must carry the usage opt-in so responses report cost,
// mirroring the OpenRouter and Infron behavior.
func TestMarshalRequestAddsUsageIncludeForOrcaRouter(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultOrcaRouterBaseURL,
		Model:   "orcarouter/auto",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "orcarouter/auto"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	usage, ok := wire["usage"].(map[string]any)
	require.True(t, ok, "usage opt-in missing: %s", body)
	assert.Equal(t, true, usage["include"])
}

// The "orcarouter/" prefix is part of the published model id on OrcaRouter (the
// gateway rejects the bare id), so unlike Infron it must stay on the wire.
func TestMarshalRequestKeepsOrcaRouterPrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultOrcaRouterBaseURL,
		Model:   "orcarouter/auto",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "orcarouter/auto"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "orcarouter/auto", wire["model"])
}
