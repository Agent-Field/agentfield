package ai

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cloudgpuEnvVars are the provider-selection variables DefaultConfig reads;
// each subtest saves and restores all of them so ambient credentials cannot
// leak between cases.
func cloudgpuEnvVars() map[string]string {
	return map[string]string{
		"OPENAI_API_KEY":     os.Getenv("OPENAI_API_KEY"),
		"OPENROUTER_API_KEY": os.Getenv("OPENROUTER_API_KEY"),
		"INFRON_API_KEY":     os.Getenv("INFRON_API_KEY"),
		"IONET_API_KEY":      os.Getenv("IONET_API_KEY"),
		"RUNCRATE_API_KEY":   os.Getenv("RUNCRATE_API_KEY"),
		"CLOUDGPU_API_KEY":   os.Getenv("CLOUDGPU_API_KEY"),
		"AI_BASE_URL":        os.Getenv("AI_BASE_URL"),
		"AI_MODEL":           os.Getenv("AI_MODEL"),
	}
}

func TestDefaultConfigCloudGPU(t *testing.T) {
	saved := cloudgpuEnvVars()
	defer restoreEnv(t, saved)

	for _, k := range []string{"OPENAI_API_KEY", "OPENROUTER_API_KEY", "INFRON_API_KEY", "IONET_API_KEY", "RUNCRATE_API_KEY", "CLOUDGPU_API_KEY", "AI_BASE_URL", "AI_MODEL"} {
		os.Unsetenv(k)
	}

	t.Run("cloudgpu only", func(t *testing.T) {
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-cloudgpu-key", cfg.APIKey)
		assert.Equal(t, defaultCloudGPUBaseURL, cfg.BaseURL)
		assert.Equal(t, defaultCloudGPUModel, cfg.Model)
	})

	t.Run("cloudgpu only, AI_MODEL wins over the CloudGPU default", func(t *testing.T) {
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")
		t.Setenv("AI_MODEL", "glm-5.3")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, defaultCloudGPUBaseURL, cfg.BaseURL)
		assert.Equal(t, "glm-5.3", cfg.Model)
	})

	t.Run("openai keeps precedence over cloudgpu", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "openai-key")
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openai-key", cfg.APIKey)
		assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
	})

	t.Run("infron keeps precedence over cloudgpu", func(t *testing.T) {
		t.Setenv("INFRON_API_KEY", "infron-key")
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "infron-key", cfg.APIKey)
		assert.Equal(t, defaultInfronBaseURL, cfg.BaseURL)
	})

	t.Run("ionet keeps precedence over cloudgpu", func(t *testing.T) {
		t.Setenv("IONET_API_KEY", "ionet-key")
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "ionet-key", cfg.APIKey)
		assert.Equal(t, defaultIonetBaseURL, cfg.BaseURL)
	})

	t.Run("runcrate keeps precedence over cloudgpu", func(t *testing.T) {
		t.Setenv("RUNCRATE_API_KEY", "runcrate-key")
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "runcrate-key", cfg.APIKey)
		assert.Equal(t, defaultRuncrateBaseURL, cfg.BaseURL)
	})

	t.Run("openrouter keeps precedence over cloudgpu", func(t *testing.T) {
		t.Setenv("OPENROUTER_API_KEY", "openrouter-key")
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "openrouter-key", cfg.APIKey)
		assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
	})

	t.Run("ai base url overrides cloudgpu default", func(t *testing.T) {
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")
		t.Setenv("AI_BASE_URL", "https://custom.example.com/v1")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.Equal(t, "test-cloudgpu-key", cfg.APIKey)
		assert.Equal(t, "https://custom.example.com/v1", cfg.BaseURL)
		// A custom base URL the SDK cannot recognize as CloudGPU keeps the
		// global default; route through a proxy and set AI_MODEL yourself.
		assert.False(t, cfg.IsCloudGPU())
		assert.NotEqual(t, defaultCloudGPUModel, cfg.Model)
	})

	t.Run("CloudGPU base url with trailing slash still gets the CloudGPU default", func(t *testing.T) {
		t.Setenv("CLOUDGPU_API_KEY", "test-cloudgpu-key")
		t.Setenv("AI_BASE_URL", defaultCloudGPUBaseURL+"/")

		cfg := DefaultConfig()
		require.NotNil(t, cfg)
		assert.True(t, cfg.IsCloudGPU())
		assert.Equal(t, defaultCloudGPUModel, cfg.Model)
	})
}

func TestIsCloudGPU(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{"CloudGPU base URL", defaultCloudGPUBaseURL, "deepseek-v4-flash", true},
		{"CloudGPU base URL case-insensitive", "https://CloudGPU.App/v1", "deepseek-v4-flash", true},
		{"cloudgpu model prefix", "https://api.openai.com/v1", "cloudgpu/deepseek-v4-flash", true},
		{"CLOUDGPU model prefix case-insensitive", "https://api.openai.com/v1", "CLOUDGPU/deepseek-v4-flash", true},
		{"unrelated endpoint", "https://api.openai.com/v1", "deepseek-v4-flash", false},
		{"openrouter stays itself", "https://openrouter.ai/api/v1", "deepseek-v4-flash", false},
		{"infron stays itself", defaultInfronBaseURL, "kimi-k2.6", false},
		{"ionet stays itself", defaultIonetBaseURL, "deepseek-v4-flash", false},
		{"runcrate stays itself", defaultRuncrateBaseURL, "deepseek-v4-flash", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsCloudGPU())
		})
	}
}

func TestStripCloudGPUPrefix(t *testing.T) {
	tests := []struct{ in, want string }{
		{"cloudgpu/deepseek-v4-flash", "deepseek-v4-flash"},
		{"CLOUDGPU/glm-5.3", "glm-5.3"},
		{"deepseek-v4-flash", "deepseek-v4-flash"},
		{"openrouter/deepseek-v4-flash", "openrouter/deepseek-v4-flash"},
		{"", ""},
		{"cloudgpu/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripCloudGPUPrefix(tt.in), tt.in)
	}
}

// The prefix is a routing marker only; the endpoint serves the bare id, so the
// marker must not reach the wire.
func TestMarshalRequestStripsCloudGPUPrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCloudGPUBaseURL,
		Model:   "deepseek-v4-flash",
	})
	require.NoError(t, err)

	req := &Request{Model: "cloudgpu/deepseek-v4-flash"}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-v4-flash", wire["model"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "cloudgpu/deepseek-v4-flash", req.Model)
}

func TestMarshalRequestLeavesCloudGPUBareModelAlone(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCloudGPUBaseURL,
		Model:   "deepseek-v4-flash",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "deepseek-v4-flash"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-v4-flash", wire["model"])
}

// Tool fields are orthogonal to the CloudGPU routing marker: they must reach
// the wire unchanged alongside the stripped model id.
func TestMarshalRequestCloudGPUKeepsTools(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCloudGPUBaseURL,
		Model:   "deepseek-v4-flash",
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
		Model:      "cloudgpu/deepseek-v4-flash",
		Tools:      tools,
		ToolChoice: "auto",
	}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "deepseek-v4-flash", wire["model"])
	assert.Len(t, wire["tools"], 1)
	assert.Equal(t, "auto", wire["tool_choice"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "cloudgpu/deepseek-v4-flash", req.Model)
}

// CloudGPU reports usage in the standard Chat Completions shape, so no gateway
// usage opt-in may be added to the wire body.
func TestMarshalRequestAddsNoUsageIncludeForCloudGPU(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCloudGPUBaseURL,
		Model:   "deepseek-v4-flash",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "deepseek-v4-flash"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	_, hasUsage := wire["usage"]
	assert.False(t, hasUsage, "usage opt-in leaked onto CloudGPU request: %s", body)
}

// CloudGPU is not a vouched max_completion_tokens endpoint, so max_tokens is
// sent as-is.
func TestMarshalRequestCloudGPUKeepsMaxTokens(t *testing.T) {
	for _, model := range []string{"deepseek-v4-flash", "kimi-k3"} {
		t.Run(model, func(t *testing.T) {
			client, err := NewClient(&Config{
				APIKey:  "k",
				BaseURL: defaultCloudGPUBaseURL,
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
