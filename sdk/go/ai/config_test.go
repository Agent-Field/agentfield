package ai

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	// Save original env vars
	originalOpenAIKey := os.Getenv("OPENAI_API_KEY")
	originalOpenRouterKey := os.Getenv("OPENROUTER_API_KEY")
	originalBaseURL := os.Getenv("AI_BASE_URL")
	originalModel := os.Getenv("AI_MODEL")

	// Clean up after test
	defer func() {
		if originalOpenAIKey != "" {
			os.Setenv("OPENAI_API_KEY", originalOpenAIKey)
		} else {
			os.Unsetenv("OPENAI_API_KEY")
		}
		if originalOpenRouterKey != "" {
			os.Setenv("OPENROUTER_API_KEY", originalOpenRouterKey)
		} else {
			os.Unsetenv("OPENROUTER_API_KEY")
		}
		if originalBaseURL != "" {
			os.Setenv("AI_BASE_URL", originalBaseURL)
		} else {
			os.Unsetenv("AI_BASE_URL")
		}
		if originalModel != "" {
			os.Setenv("AI_MODEL", originalModel)
		} else {
			os.Unsetenv("AI_MODEL")
		}
	}()

	tests := []struct {
		name           string
		setupEnv       func()
		checkConfig    func(t *testing.T, cfg *Config)
	}{
		{
			name: "default OpenAI config",
			setupEnv: func() {
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("OPENROUTER_API_KEY")
				os.Unsetenv("AI_BASE_URL")
				os.Unsetenv("AI_MODEL")
				os.Setenv("OPENAI_API_KEY", "test-openai-key")
			},
			checkConfig: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "test-openai-key", cfg.APIKey)
				assert.Equal(t, "https://api.openai.com/v1", cfg.BaseURL)
				assert.Equal(t, "gpt-4o", cfg.Model)
				assert.Equal(t, 0.7, cfg.Temperature)
				assert.Equal(t, 4096, cfg.MaxTokens)
				assert.Equal(t, 30*time.Second, cfg.Timeout)
			},
		},
		{
			name: "OpenRouter config takes precedence",
			setupEnv: func() {
				os.Setenv("OPENAI_API_KEY", "openai-key")
				os.Setenv("OPENROUTER_API_KEY", "openrouter-key")
				os.Unsetenv("AI_BASE_URL")
				os.Unsetenv("AI_MODEL")
			},
			checkConfig: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "openrouter-key", cfg.APIKey)
				assert.Equal(t, "https://openrouter.ai/api/v1", cfg.BaseURL)
			},
		},
		{
			name: "custom base URL override",
			setupEnv: func() {
				os.Setenv("OPENAI_API_KEY", "test-key")
				os.Setenv("AI_BASE_URL", "https://custom.example.com/v1")
				os.Unsetenv("AI_MODEL")
			},
			checkConfig: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "https://custom.example.com/v1", cfg.BaseURL)
			},
		},
		{
			name: "custom model override",
			setupEnv: func() {
				os.Setenv("OPENAI_API_KEY", "test-key")
				os.Unsetenv("AI_BASE_URL")
				os.Setenv("AI_MODEL", "gpt-3.5-turbo")
			},
			checkConfig: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "gpt-3.5-turbo", cfg.Model)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up env first
			os.Unsetenv("OPENAI_API_KEY")
			os.Unsetenv("OPENROUTER_API_KEY")
			os.Unsetenv("AI_BASE_URL")
			os.Unsetenv("AI_MODEL")

			tt.setupEnv()

			cfg := DefaultConfig()
			require.NotNil(t, cfg)
			tt.checkConfig(t, cfg)
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				APIKey:  "test-key",
				BaseURL: "https://api.example.com/v1",
				Model:   "gpt-4o",
			},
			wantErr: false,
		},
		{
			name: "missing API key",
			config: &Config{
				APIKey:  "",
				BaseURL: "https://api.example.com/v1",
				Model:   "gpt-4o",
			},
			wantErr: true,
		},
		{
			name: "missing base URL",
			config: &Config{
				APIKey:  "test-key",
				BaseURL: "",
				Model:   "gpt-4o",
			},
			wantErr: true,
		},
		{
			name: "missing model",
			config: &Config{
				APIKey:  "test-key",
				BaseURL: "https://api.example.com/v1",
				Model:   "",
			},
			wantErr: true,
		},
		{
			name: "all fields missing",
			config: &Config{
				APIKey:  "",
				BaseURL: "",
				Model:   "",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestIsOpenRouter(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		expected bool
	}{
		{
			name:     "OpenRouter URL without trailing slash",
			baseURL:  "https://openrouter.ai/api/v1",
			expected: true,
		},
		{
			name:     "OpenRouter URL with trailing slash",
			baseURL:  "https://openrouter.ai/api/v1/",
			expected: true,
		},
		{
			name:     "OpenAI URL",
			baseURL:  "https://api.openai.com/v1",
			expected: false,
		},
		{
			name:     "custom URL",
			baseURL:  "https://custom.example.com/v1",
			expected: false,
		},
		{
			name:     "empty URL",
			baseURL:  "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BaseURL: tt.baseURL}
			assert.Equal(t, tt.expected, cfg.IsOpenRouter())
		})
	}
}
