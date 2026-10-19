package ai

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ionetEnvVars are the provider-selection variables DefaultConfig reads;
// each subtest saves and restores all of them so ambient credentials cannot
// leak between cases.
func ionetEnvVars() map[string]string {
	return map[string]string{
		"OPENAI_API_KEY":     os.Getenv("OPENAI_API_KEY"),
		"OPENROUTER_API_KEY": os.Getenv("OPENROUTER_API_KEY"),
		"INFRON_API_KEY":     os.Getenv("INFRON_API_KEY"),
		"IONET_API_KEY":      os.Getenv("IONET_API_KEY"),
		"AI_BASE_URL":        os.Getenv("AI_BASE_URL"),
		"AI_MODEL":           os.Getenv("AI_MODEL"),
	}
}

func restoreEnv(t *testing.T, saved map[string]string) {
	t.Helper()
	for k, v := range saved {
		if v == "" {
			os.Unsetenv(k)
		} else {
			os.Setenv(k, v)
		}
	}
}

func TestDefaultConfigIonet(t *testing.T) {
	saved := ionetEnvVars()
	defer restoreEnv(t, saved)

	for _, k := range []string{"OPENAI_API_KEY", "OPENROUTER_API_KEY", "INFRON_API_KEY", "IONET_API_KEY", "AI_BASE_URL", "AI_MODEL"} {
		os.Unsetenv(k)
	}

	t.Run("ionet only", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "test-ionet-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-ionet-key", cfg.APIKey)
		assert.Equal(t, defaultIonetBaseURL, cfg.BaseURL)
		// io.net serves no gpt-4o, so the default must be an io.net id.
		assert.Equal(t, defaultIonetModel, cfg.Model)
	})

	t.Run("ionet only, AI_MODEL wins over the io.net default", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "test-ionet-key")
		t.Setenv("AI_MODEL", "deepseek-ai/DeepSeek-R1-0528")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, defaultIonetBaseURL, cfg.BaseURL)
		assert.Equal(t, "deepseek-ai/DeepSeek-R1-0528", cfg.Model)
	})

	t.Run("openai keeps precedence over ionet", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "openai-key")
		t.Setenv("IONET_API_KEY", "test-ionet-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openai-key", cfg.APIKey)
		assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
	})

	t.Run("infron keeps precedence over ionet", func(t *testing.T) {
		t.Setenv("INFRON_API_KEY", "infron-key")
		t.Setenv("IONET_API_KEY", "test-ionet-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "infron-key", cfg.APIKey)
		assert.Equal(t, defaultInfronBaseURL, cfg.BaseURL)
	})

	t.Run("openrouter keeps precedence over ionet", func(t *testing.T) {
		t.Setenv("OPENROUTER_API_KEY", "openrouter-key")
		t.Setenv("IONET_API_KEY", "test-ionet-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openrouter-key", cfg.APIKey)
		assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
	})

	t.Run("ai base url overrides ionet default", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "test-ionet-key")
		t.Setenv("AI_BASE_URL", "https://custom.example.com/v1")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-ionet-key", cfg.APIKey)
		assert.Equal(t, "https://custom.example.com/v1", cfg.BaseURL)
		// A custom base URL the SDK cannot recognize as io.net keeps the
		// global default; route through a proxy and set AI_MODEL yourself.
		assert.False(t, cfg.IsIonet())
		assert.Equal(t, "gpt-4o", cfg.Model)
	})

	t.Run("io.net base url with trailing slash still gets the io.net default", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "test-ionet-key")
		t.Setenv("AI_BASE_URL", defaultIonetBaseURL+"/")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.True(t, cfg.IsIonet())
		// Same detection as the request path: the model default must not
		// depend on spelling the URL byte-identically.
		assert.Equal(t, defaultIonetModel, cfg.Model)
	})
}

func TestIsIonet(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{"io.net base URL", defaultIonetBaseURL, "meta-llama/Llama-3.3-70B-Instruct", true},
		{"io.net base URL case-insensitive", "https://API.Intelligence.IO.Solutions/api/v1", "gpt-4o", true},
		{"ionet model prefix", "https://api.openai.com/v1", "ionet/meta-llama/Llama-3.3-70B-Instruct", true},
		{"IONET model prefix case-insensitive", "https://api.openai.com/v1", "IONET/meta-llama/Llama-3.3-70B-Instruct", true},
		{"unrelated endpoint", "https://api.openai.com/v1", "gpt-4o", false},
		{"openrouter stays itself", "https://openrouter.ai/api/v1", "openai/gpt-4o", false},
		{"infron stays itself", defaultInfronBaseURL, "moonshotai/kimi-k2.6", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsIonet())
		})
	}
}

func TestStripIonetPrefix(t *testing.T) {
	tests := []struct{ in, want string }{
		{"ionet/meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"},
		{"IONET/deepseek-ai/DeepSeek-R1-0528", "deepseek-ai/DeepSeek-R1-0528"},
		{"meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"},
		{"openrouter/meta-llama/Llama-3.3-70B-Instruct", "openrouter/meta-llama/Llama-3.3-70B-Instruct"},
		{"", ""},
		{"ionet/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripIonetPrefix(tt.in), tt.in)
	}
}

// The prefix is a routing marker only; the endpoint serves the bare id, so the
// marker must not reach the wire.
func TestMarshalRequestStripsIonetPrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultIonetBaseURL,
		Model:   "meta-llama/Llama-3.3-70B-Instruct",
	})
	require.NoError(t, err)

	req := &Request{Model: "ionet/meta-llama/Llama-3.3-70B-Instruct"}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "meta-llama/Llama-3.3-70B-Instruct", wire["model"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "ionet/meta-llama/Llama-3.3-70B-Instruct", req.Model)
}

func TestMarshalRequestLeavesIonetBareModelAlone(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultIonetBaseURL,
		Model:   "meta-llama/Llama-3.3-70B-Instruct",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "meta-llama/Llama-3.3-70B-Instruct"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "meta-llama/Llama-3.3-70B-Instruct", wire["model"])
}

// Tool fields are orthogonal to the io.net routing marker: they must reach
// the wire unchanged alongside the stripped model id.
func TestMarshalRequestIonetKeepsTools(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultIonetBaseURL,
		Model:   "meta-llama/Llama-3.3-70B-Instruct",
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
		Model:      "ionet/meta-llama/Llama-3.3-70B-Instruct",
		Tools:      tools,
		ToolChoice: "auto",
	}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "meta-llama/Llama-3.3-70B-Instruct", wire["model"])
	assert.Len(t, wire["tools"], 1)
	assert.Equal(t, "auto", wire["tool_choice"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "ionet/meta-llama/Llama-3.3-70B-Instruct", req.Model)
}

// io.net reports usage in the standard Chat Completions shape, so no gateway
// usage opt-in may be added to the wire body.
func TestMarshalRequestAddsNoUsageIncludeForIonet(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultIonetBaseURL,
		Model:   "meta-llama/Llama-3.3-70B-Instruct",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "meta-llama/Llama-3.3-70B-Instruct"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	_, hasUsage := wire["usage"]
	assert.False(t, hasUsage, "usage opt-in leaked onto io.net request: %s", body)
}

// io.net is not a vouched max_completion_tokens endpoint, so max_tokens is
// kept even for ids (openai/gpt-oss-*) that the legacy-OpenAI heuristics
// would otherwise rewrite.
func TestMarshalRequestIonetKeepsMaxTokens(t *testing.T) {
	for _, model := range []string{"meta-llama/Llama-3.3-70B-Instruct", "openai/gpt-oss-120b"} {
		t.Run(model, func(t *testing.T) {
			client, err := NewClient(&Config{
				APIKey:  "k",
				BaseURL: defaultIonetBaseURL,
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
