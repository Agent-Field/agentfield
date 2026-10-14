package ai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsCheaperInference(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected bool
	}{
		{
			name:     "Cheaper Inference URL",
			baseURL:  "https://api.cheaperinference.com/v1",
			expected: true,
		},
		{
			name:     "Cheaper Inference URL with trailing slash",
			baseURL:  "https://api.cheaperinference.com/v1/",
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
			name:     "Cheaper Inference model prefix",
			baseURL:  "https://api.openai.com/v1",
			model:    "cheaperinference/gpt-5.4-mini",
			expected: true,
		},
		{
			name:     "bare model id is not Cheaper Inference",
			baseURL:  "https://api.openai.com/v1",
			model:    "gpt-5.4-mini",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL, Model: tt.model}
			assert.Equal(t, tt.expected, cfg.IsCheaperInference())
		})
	}
}

func TestDefaultConfigCheaperInferenceKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("INFRON_API_KEY", "")
	t.Setenv("CHEAPER_INFERENCE_API_KEY", "ci-key")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("AI_MODEL", "")

	cfg := DefaultConfig()

	assert.Equal(t, "ci-key", cfg.APIKey)
	assert.Equal(t, defaultCheaperInferenceBaseURL, cfg.BaseURL)
	assert.True(t, cfg.IsCheaperInference())
	assert.False(t, cfg.IsOpenRouter())
	assert.False(t, cfg.IsInfron())
}

// A Cheaper Inference key must never move an existing deployment off the
// endpoint it already resolves to.
func TestDefaultConfigExistingKeysWinOverCheaperInference(t *testing.T) {
	tests := []struct {
		name        string
		env         map[string]string
		wantKey     string
		wantBaseURL string
	}{
		{
			name:        "OPENAI_API_KEY",
			env:         map[string]string{"OPENAI_API_KEY": "existing-openai-key"},
			wantKey:     "existing-openai-key",
			wantBaseURL: "https://api.openai.com/v1",
		},
		{
			name:        "OPENROUTER_API_KEY",
			env:         map[string]string{"OPENROUTER_API_KEY": "existing-gateway-key"},
			wantKey:     "existing-gateway-key",
			wantBaseURL: "https://openrouter.ai/api/v1",
		},
		{
			name:        "INFRON_API_KEY",
			env:         map[string]string{"INFRON_API_KEY": "infron-key"},
			wantKey:     "infron-key",
			wantBaseURL: defaultInfronBaseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OPENAI_API_KEY", "")
			t.Setenv("OPENROUTER_API_KEY", "")
			t.Setenv("INFRON_API_KEY", "")
			t.Setenv("AI_BASE_URL", "")
			t.Setenv("AI_MODEL", "")
			t.Setenv("CHEAPER_INFERENCE_API_KEY", "ci-key")
			for k, v := range tt.env {
				t.Setenv(k, v)
			}

			cfg := DefaultConfig()

			assert.Equal(t, tt.wantKey, cfg.APIKey)
			assert.Equal(t, tt.wantBaseURL, cfg.BaseURL)
			assert.False(t, cfg.IsCheaperInference())
		})
	}
}

func TestStripCheaperInferencePrefix(t *testing.T) {
	tests := []struct{ in, want string }{
		{"cheaperinference/gpt-5.4-mini", "gpt-5.4-mini"},
		{"CheaperInference/claude-sonnet-5", "claude-sonnet-5"},
		{"gpt-5.4-mini", "gpt-5.4-mini"},
		{"openrouter/openai/gpt-5.4-mini", "openrouter/openai/gpt-5.4-mini"},
		{"", ""},
		{"cheaperinference/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripCheaperInferencePrefix(tt.in), tt.in)
	}
}

// The prefix is a routing marker only. The gateway serves the bare model id.
func TestMarshalRequestStripsCheaperInferencePrefix(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCheaperInferenceBaseURL,
		Model:   "cheaperinference/gpt-5.4-mini",
	})
	require.NoError(t, err)

	req := &Request{Model: "cheaperinference/gpt-5.4-mini"}
	body, err := client.marshalRequest(req)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "gpt-5.4-mini", wire["model"])

	// The caller's Request must not be mutated.
	assert.Equal(t, "cheaperinference/gpt-5.4-mini", req.Model)
}

func TestMarshalRequestLeavesBareCheaperInferenceModelAlone(t *testing.T) {
	client, err := NewClient(&Config{
		APIKey:  "k",
		BaseURL: defaultCheaperInferenceBaseURL,
		Model:   "gpt-5.4-mini",
	})
	require.NoError(t, err)

	body, err := client.marshalRequest(&Request{Model: "gpt-5.4-mini"})
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(body, &wire))
	assert.Equal(t, "gpt-5.4-mini", wire["model"])
}
