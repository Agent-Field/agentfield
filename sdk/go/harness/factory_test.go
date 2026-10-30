package harness

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveProviderName(t *testing.T) {
	tests := []struct {
		name     string
		explicit string
		env      string
		want     string
	}{
		{name: "default", want: ProviderCodeAF},
		{name: "environment", env: ProviderCodex, want: ProviderCodex},
		{name: "explicit wins", explicit: ProviderGemini, env: ProviderCodex, want: ProviderGemini},
		{name: "blank explicit", explicit: "   ", want: ProviderCodeAF},
		{name: "blank environment", env: "  ", want: ProviderCodeAF},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(ProviderEnvVar, tt.env)
			assert.Equal(t, tt.want, ResolveProviderName(tt.explicit))
		})
	}
}

func TestBuildProvider_DefaultsToCodeAF(t *testing.T) {
	t.Setenv(ProviderEnvVar, "")
	provider, err := BuildProvider("", "")
	require.NoError(t, err)
	require.NotNil(t, provider)
	_, ok := provider.(*CodeAFProvider)
	assert.True(t, ok)
}

func TestBuildProvider_ExplicitProviders(t *testing.T) {
	tests := []struct {
		name string
		want any
	}{
		{name: ProviderCodeAF, want: &CodeAFProvider{}},
		{name: ProviderAforge, want: &AforgeProvider{}},
		{name: ProviderClaudeCode, want: &ClaudeCodeProvider{}},
		{name: ProviderOpenCode, want: &OpenCodeProvider{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := BuildProvider(tt.name, "test-bin")
			require.NoError(t, err)
			require.NotNil(t, provider)
			assert.IsType(t, tt.want, provider)
		})
	}
}

func TestBuildProvider_RejectsUnknownName(t *testing.T) {
	provider, err := BuildProvider("nope", "")
	assert.Nil(t, provider)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nope")
	assert.True(t, strings.Contains(err.Error(), "supported: codeaf, aforge, claude-code, codex, gemini, opencode, pi, omp, cursor"))
}
