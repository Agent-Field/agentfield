package ai

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// runcrateEnvVars are the provider-selection variables DefaultConfig reads;
// each subtest saves and restores all of them so ambient credentials cannot
// leak between cases.
func runcrateEnvVars() map[string]string {
	return map[string]string{
		"OPENAI_API_KEY":     os.Getenv("OPENAI_API_KEY"),
		"OPENROUTER_API_KEY": os.Getenv("OPENROUTER_API_KEY"),
		"INFRON_API_KEY":     os.Getenv("INFRON_API_KEY"),
		"IONET_API_KEY":      os.Getenv("IONET_API_KEY"),
		"RUNCRATE_API_KEY":   os.Getenv("RUNCRATE_API_KEY"),
		"AI_BASE_URL":        os.Getenv("AI_BASE_URL"),
		"AI_MODEL":           os.Getenv("AI_MODEL"),
	}
}

func TestDefaultConfigRuncrate(t *testing.T) {
	saved := runcrateEnvVars()
	defer restoreEnv(t, saved)

	for _, k := range []string{"OPENAI_API_KEY", "OPENROUTER_API_KEY", "INFRON_API_KEY", "IONET_API_KEY", "RUNCRATE_API_KEY", "AI_BASE_URL", "AI_MODEL"} {
		os.Unsetenv(k)
	}

	t.Run("runcrate only", func(t *testing.T) {
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-runcrate-key", cfg.APIKey)
		assert.Equal(t, defaultRuncrateBaseURL, cfg.BaseURL)
		// Runcrate serves no gpt-4o, so the default must be a Runcrate id.
		assert.Equal(t, defaultRuncrateModel, cfg.Model)
	})

	t.Run("runcrate only, AI_MODEL wins over the Runcrate default", func(t *testing.T) {
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")
		t.Setenv("AI_MODEL", "zai-org/GLM-5.3")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, defaultRuncrateBaseURL, cfg.BaseURL)
		assert.Equal(t, "zai-org/GLM-5.3", cfg.Model)
	})

	t.Run("openai keeps precedence over runcrate", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "openai-key")
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openai-key", cfg.APIKey)
		assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
	})

	t.Run("infron keeps precedence over runcrate", func(t *testing.T) {
		t.Setenv("INFRON_API_KEY", "infron-key")
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "infron-key", cfg.APIKey)
		assert.Equal(t, defaultInfronBaseURL, cfg.BaseURL)
	})

	t.Run("ionet keeps precedence over runcrate", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "ionet-key")
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "ionet-key", cfg.APIKey)
		assert.Equal(t, defaultIonetBaseURL, cfg.BaseURL)
	})

	t.Run("openrouter keeps precedence over runcrate", func(t *testing.T) {
		t.Setenv("OPENROUTER_API_KEY", "openrouter-key")
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openrouter-key", cfg.APIKey)
		assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
	})

	t.Run("ai base url overrides runcrate default", func(t *testing.T) {
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")
		t.Setenv("AI_BASE_URL", "https://custom.example.com/v1")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-runcrate-key", cfg.APIKey)
		assert.Equal(t, "https://custom.example.com/v1", cfg.BaseURL)
		// A custom base URL the SDK cannot recognize as Runcrate keeps the
		// global default; route through a proxy and set AI_MODEL yourself.
		assert.False(t, cfg.IsRuncrate())
		assert.Equal(t, "gpt-4o", cfg.Model)
	})

	t.Run("Runcrate base url with trailing slash still gets the Runcrate default", func(t *testing.T) {
		t.Setenv("RUNCRATE_API_KEY", "test-runcrate-key")
		t.Setenv("AI_BASE_URL", defaultRuncrateBaseURL+"/")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.True(t, cfg.IsRuncrate())
		// Same detection as the request path: the model default must not
		// depend on spelling the URL byte-identically.
		assert.Equal(t, defaultRuncrateModel, cfg.Model)
	})
}

func TestIsRuncrate(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{"Runcrate base URL", defaultRuncrateBaseURL, "deepseek-ai/DeepSeek-V4.1-Flash", true},
		{"Runcrate base URL case-insensitive", "https://API.Runcrate.AI/v1", "gpt-4o", true},
		{"runcrate model prefix", "https://api.openai.com/v1", "runcrate/deepseek-ai/DeepSeek-V4.1-Flash", true},
		{"RUNCRATE model prefix case-insensitive", "https://api.openai.com/v1", "RUNCRATE/deepseek-ai/DeepSeek-V4.1-Flash", true},
		{"unrelated endpoint", "https://api.openai.com/v1", "gpt-4o", false},
		{"openrouter stays itself", "https://openrouter.ai/api/v1", "openai/gpt-4o", false},
		{"infron stays itself", defaultInfronBaseURL, "moonshotai/kimi-k2.6", false},
		{"ionet stays itself", defaultIonetBaseURL, "deepseek-ai/DeepSeek-V4.1-Flash", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsRuncrate())
		})
	}
}

func TestStripRuncratePrefix(t *testing.T) {
	tests := []struct{ in, want string }{
		{"runcrate/deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"},
		{"RUNCRATE/zai-org/GLM-5.3", "zai-org/GLM-5.3"},
		{"deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"},
		{"openrouter/deepseek-ai/DeepSeek-V4.1-Flash", "openrouter/deepseek-ai/DeepSeek-V4.1-Flash"},
		{"", ""},
		{"runcrate/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripRuncratePrefix(tt.in), tt.in)
	}
}

// The prefix is a routing marker only; the endpoint serves the bare id, so the
// marker must not reach the wire.
func TestMarshalRequestStripsRuncratePrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultRuncrateBaseURL,
		Model:   "deepseek-ai/DeepSeek-V4.1-Flash",
	})
	require.NoError(t, err)

	req := &Request{Model: "runcrate/deepseek-ai/DeepSeek-V4.1-Flash"}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-ai/DeepSeek-V4.1-Flash", wire["model"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "runcrate/deepseek-ai/DeepSeek-V4.1-Flash", req.Model)
}

func TestMarshalRequestLeavesRuncrateBareModelAlone(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultRuncrateBaseURL,
		Model:   "deepseek-ai/DeepSeek-V4.1-Flash",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "deepseek-ai/DeepSeek-V4.1-Flash"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-ai/DeepSeek-V4.1-Flash", wire["model"])
}

// Tool fields are orthogonal to the Runcrate routing marker: they must reach
// the wire unchanged alongside the stripped model id.
func TestMarshalRequestRuncrateKeepsTools(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultRuncrateBaseURL,
		Model:   "deepseek-ai/DeepSeek-V4.1-Flash",
	})
	require.NoError(t, err)

	tools := []ToolDefinition{{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters:  map[string]interface{}{"type": "object"},
		},
	}}
	req := &Request{
		Model:      "runcrate/deepseek-ai/DeepSeek-V4.1-Flash",
		Tools:      tools,
		ToolChoice: "auto",
	}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-ai/DeepSeek-V4.1-Flash", wire["model"])
	assert.Len(t, wire["tools"], 1)
	assert.Equal(t, "auto", wire["tool_choice"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "runcrate/deepseek-ai/DeepSeek-V4.1-Flash", req.Model)
}

// Runcrate reports usage in the standard Chat Completions shape, so no gateway
// usage opt-in may be added to the wire body.
func TestMarshalRequestAddsNoUsageIncludeForRuncrate(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultRuncrateBaseURL,
		Model:   "deepseek-ai/DeepSeek-V4.1-Flash",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "deepseek-ai/DeepSeek-V4.1-Flash"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	_, hasUsage := wire["usage"]
	assert.False(t, hasUsage, "usage opt-in leaked onto Runcrate request: %s", body)
}

// Runcrate is not a vouched max_completion_tokens endpoint, so max_tokens is
// kept even for ids (openai/gpt-oss-*) that the legacy-OpenAI heuristics
// would otherwise rewrite.
func TestMarshalRequestRuncrateKeepsMaxTokens(t *testing.T) {
	for _, model := range []string{"deepseek-ai/DeepSeek-V4.1-Flash", "openai/gpt-oss-120b"} {
		t.Run(model, func(t *testing.T) {
			client, err := NewClient(&Config{
				APIKey:  "k",
				BaseURL: defaultRuncrateBaseURL,
				Model:   model,
			})
			require.NoError(t, err)

			maxTokens := 512
			body, err := client.marshalRequest(&Request{Model: model, MaxTokens: &maxTokens})
			require.NoError(t, err)

			var wire map[string]any
			require.NoError(t, json.Unmarshal(body, &wire))
			assert.Equal(t, float64(512), wire["max_tokens"])
			_, hasRewritten := wire["max_completion_tokens"]
			assert.False(t, hasRewritten)
		})
	}
}
