package ai

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsInfron(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{
			name:     "Infron URL without trailing slash",
			baseURL:  "https://llm.onerouter.pro/v1",
			expected: true,
		},
		{
			name:     "Infron URL with trailing slash",
			baseURL:  "https://llm.onerouter.pro/v1/",
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
			name:     "Infron model prefix",
			baseURL:  "https://api.openai.com/v1",
			model:    "infron/moonshotai/kimi-k2.6",
			expected: true,
		},
		{
			// Gateways serve the same `<provider>/<model>` ids, so a bare id
			// must not be attributed to any one of them on its own.
			name:     "bare shared model id is not Infron",
			baseURL:  "https://api.openai.com/v1",
			model:    "moonshotai/kimi-k2.6",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsInfron())
		})
	}
}

func TestDefaultConfigInfronKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "infron-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "infron-key", cfg.APIKey)
	assert.Equal(t, defaultInfronBaseURL, cfg.BaseURL)
	assert.True(t, cfg.IsInfron())
	assert.Equal(t, defaultInfronSiteURL, cfg.SiteURL)
	assert.Equal(t, defaultInfronAppName, cfg.SiteName)
}

// An Infron key must never move an existing deployment off the gateway it
// already resolves to — that is the backwards-compatibility guarantee of this
// change.
func TestDefaultConfigExistingGatewayWinsOverInfron(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "existing-gateway-key")
	t.Setenv("INFRON_API_KEY", "infron-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "existing-gateway-key", cfg.APIKey)
	assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
	assert.True(t, cfg.IsOpenRouter())
}

// The guarantee in DefaultConfig's doc comment is that adding INFRON_API_KEY to
// an already-configured environment never reroutes it. OPENAI_API_KEY is such
// an environment, and it is the case the OpenRouter test above cannot cover
// because it clears the key. Agent processes inherit the parent environment, so
// a single exported INFRON_API_KEY reaching this branch would move every Go
// agent's traffic — and its credential — to a different gateway.
func TestDefaultConfigExistingOpenAIKeyWinsOverInfron(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "existing-openai-key")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "infron-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "existing-openai-key", cfg.APIKey)
	assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
	assert.False(t, cfg.IsInfron())
}

// Infron still applies when it is the only gateway key set.
func TestDefaultConfigInfronAppliesWhenNoDirectKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "infron-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "infron-key", cfg.APIKey)
	assert.Equal(t, defaultInfronBaseURL, cfg.BaseURL)
	assert.True(t, cfg.IsInfron())
}

func TestDefaultConfigInfronAttributionEnvOverrides(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "infron-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")
	t.Setenv("AGENTFIELD_INFRON_SITE_URL", "https://custom.example")
	t.Setenv("AGENTFIELD_INFRON_APP_NAME", "Custom App")

	cfg := DefaultConfig()

	assert.Equal(t, "https://custom.example", cfg.SiteURL)
	assert.Equal(t, "Custom App", cfg.SiteName)
}

// A deployment that already declared its identity keeps it after switching
// gateways, so nobody has to re-declare to move.
func TestInfronAttributionFallsBackToExistingVars(t *testing.T) {
	t.Setenv("AGENTFIELD_INFRON_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_INFRON_SITE_URL", "")
	t.Setenv("AGENTFIELD_INFRON_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "https://legacy.example")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "Legacy App")

	header := http.Header{}
	applyInfronAttributionHeaders(header, "", "")

	assert.Equal(t, "https://legacy.example", header.Get("HTTP-Referer"))
	assert.Equal(t, "Legacy App", header.Get("X-Title"))
}

// The opt-out travels with the inherited values: attribution a deployment
// suppressed for OpenRouter — often because the site URL names an internal
// host — must not be sent to a different vendor either. The Infron defaults
// are used instead.
func TestInfronAttributionDoesNotInheritOptedOutValues(t *testing.T) {
	t.Setenv("AGENTFIELD_INFRON_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_INFRON_SITE_URL", "")
	t.Setenv("AGENTFIELD_INFRON_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_ATTRIBUTION", "false")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "https://internal-tools.corp.example")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "Internal Risk Engine")
	t.Setenv("OR_SITE_URL", "")
	t.Setenv("OR_APP_NAME", "")

	header := http.Header{}
	applyInfronAttributionHeaders(header, "", "")

	assert.Equal(t, defaultInfronSiteURL, header.Get("HTTP-Referer"))
	assert.Equal(t, defaultInfronAppName, header.Get("X-Title"))
}

func TestApplyInfronAttributionHeadersDisabled(t *testing.T) {
	t.Setenv("AGENTFIELD_INFRON_ATTRIBUTION", "false")

	header := http.Header{}
	applyInfronAttributionHeaders(header, "https://example.com", "Example")

	assert.Empty(t, header.Get("HTTP-Referer"))
	assert.Empty(t, header.Get("X-Title"))
}

func TestApplyInfronAttributionHeadersDefaults(t *testing.T) {
	t.Setenv("AGENTFIELD_INFRON_ATTRIBUTION", "")
	t.Setenv("AGENTFIELD_INFRON_SITE_URL", "")
	t.Setenv("AGENTFIELD_INFRON_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "")
	t.Setenv("OR_SITE_URL", "")
	t.Setenv("OR_APP_NAME", "")

	header := http.Header{}
	applyInfronAttributionHeaders(header, "", "")

	assert.Equal(t, defaultInfronSiteURL, header.Get("HTTP-Referer"))
	assert.Equal(t, defaultInfronAppName, header.Get("X-Title"))
}

// ---------------------------------------------------------------------------
// Model prefix stripping
// ---------------------------------------------------------------------------

func TestStripInfronPrefix(t *testing.T) {
	tests := []struct{ in, want string }{
		{"infron/moonshotai/kimi-k2.6", "moonshotai/kimi-k2.6"},
		{"INFRON/deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"},
		{"moonshotai/kimi-k2.6", "moonshotai/kimi-k2.6"},
		{"openrouter/moonshotai/kimi-k2.6", "openrouter/moonshotai/kimi-k2.6"},
		{"", ""},
		{"infron/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripInfronPrefix(tt.in), tt.in)
	}
}

// The prefix is a routing marker only. If it reaches the gateway the request
// is rejected with "No available providers for model infron/...".
func TestMarshalRequestStripsInfronPrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultInfronBaseURL,
		Model:   "infron/moonshotai/kimi-k2.6",
	})
	require.NoError(t, err)

	req := &Request{Model: "infron/moonshotai/kimi-k2.6"}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "moonshotai/kimi-k2.6", wire["model"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "infron/moonshotai/kimi-k2.6", req.Model)
}

func TestMarshalRequestLeavesBareModelAlone(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultInfronBaseURL,
		Model:   "moonshotai/kimi-k2.6",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "moonshotai/kimi-k2.6"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "moonshotai/kimi-k2.6", wire["model"])
}

// Infron requests must carry the usage opt-in so responses report cost.
func TestMarshalRequestAddsUsageIncludeForInfron(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultInfronBaseURL,
		Model:   "moonshotai/kimi-k2.6",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "moonshotai/kimi-k2.6"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	usage, ok := wire["usage"].(map[string]any)
	require.True(t, ok, "usage opt-in missing: %s", body)
	assert.Equal(t, true, usage["include"])
}

// ---------------------------------------------------------------------------
// Top-level native cost normalization
// ---------------------------------------------------------------------------

// Infron reports the native cost at the top level of the body rather than
// nested under usage. Both shapes must land in Usage.Cost so the cost tracker
// records cost_source "provider" either way.
func TestResponseNormalizesTopLevelCost(t *testing.T) {
	body := `{
		"id": "chatcmpl-1",
		"model": "deepseek/deepseek-v4-flash",
		"cost": 0.000002,
		"choices": [],
		"usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}
	}`

	var resp Response
	require.NoError(t, json.Unmarshal([]byte(body), &resp))
	resp.normalizeNativeCost()

	require.NotNil(t, resp.Usage)
	require.NotNil(t, resp.Usage.Cost)
	assert.InDelta(t, 0.000002, *resp.Usage.Cost, 1e-12)
}

// A body carrying cost but no usage block must not have one fabricated for it:
// a synthesized Usage carries zero token counts that downstream consumers read
// as authoritative.
func TestResponseNormalizeDoesNotFabricateUsage(t *testing.T) {
	var resp Response
	require.NoError(t, json.Unmarshal([]byte(`{"cost": 0.5}`), &resp))
	resp.normalizeNativeCost()

	assert.Nil(t, resp.Usage)
}

// An explicit usage.cost is authoritative and must not be overwritten.
func TestResponseNormalizeDoesNotClobberUsageCost(t *testing.T) {
	body := `{"cost": 9.99, "usage": {"cost": 0.25}}`

	var resp Response
	require.NoError(t, json.Unmarshal([]byte(body), &resp))
	resp.normalizeNativeCost()

	require.NotNil(t, resp.Usage.Cost)
	assert.InDelta(t, 0.25, *resp.Usage.Cost, 1e-12)
}

func TestResponseNormalizeNoopWithoutCost(t *testing.T) {
	var resp Response
	require.NoError(t, json.Unmarshal([]byte(`{"usage": {"prompt_tokens": 3}}`), &resp))
	resp.normalizeNativeCost()

	require.NotNil(t, resp.Usage)
	assert.Nil(t, resp.Usage.Cost, "nil cost means unknown, not free")
}

// The streaming path carries the same shape on the final chunk.
func TestStreamChunkNormalizesTopLevelCost(t *testing.T) {
	body := `{
		"id": "chatcmpl-2",
		"object": "chat.completion.chunk",
		"cost": 0.000003,
		"choices": [],
		"usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
	}`

	var chunk StreamChunk
	require.NoError(t, json.Unmarshal([]byte(body), &chunk))
	chunk.normalizeNativeCost()

	require.NotNil(t, chunk.Usage)
	require.NotNil(t, chunk.Usage.Cost)
	assert.InDelta(t, 0.000003, *chunk.Usage.Cost, 1e-12)
}

func TestStreamChunkNormalizeNoopWithoutCost(t *testing.T) {
	var chunk StreamChunk
	require.NoError(t, json.Unmarshal([]byte(`{"choices": []}`), &chunk))
	chunk.normalizeNativeCost()

	assert.Nil(t, chunk.Usage)
}

// A cost-only chunk must not grow a fabricated zero-token Usage. Stream
// consumers accumulate last-usage-wins, so a synthesized Usage arriving after
// the real usage chunk would erase the real token counts.
func TestStreamChunkCostOnlyDoesNotFabricateUsage(t *testing.T) {
	var usageChunk StreamChunk
	require.NoError(t, json.Unmarshal(
		[]byte(`{"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}`),
		&usageChunk))
	usageChunk.normalizeNativeCost()

	var costChunk StreamChunk
	require.NoError(t, json.Unmarshal([]byte(`{"cost": 0.00042, "choices": []}`), &costChunk))
	costChunk.normalizeNativeCost()

	assert.Nil(t, costChunk.Usage, "cost-only chunk must not synthesize usage")

	// The last-usage-wins accumulation every stream consumer performs.
	var accumulated *Usage
	for _, chunk := range []StreamChunk{usageChunk, costChunk} {
		if chunk.Usage != nil {
			accumulated = chunk.Usage
		}
	}
	require.NotNil(t, accumulated)
	assert.Equal(t, 10, accumulated.PromptTokens)
	assert.Equal(t, 5, accumulated.CompletionTokens)
}
