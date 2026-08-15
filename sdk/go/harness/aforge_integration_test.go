//go:build integration

package harness

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAforgeProvider_Integration drives the real aforge binary and its default
// `exec` JSON envelope. Set AFORGE_INTEGRATION=1 so TestMain does not shadow it.
func TestAforgeProvider_Integration(t *testing.T) {
	binPath, err := exec.LookPath("aforge")
	if err != nil {
		t.Skip("aforge binary not installed")
	}
	if os.Getenv("OPENROUTER_API_KEY") == "" && os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY or OPENAI_API_KEY is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	raw, err := NewAforgeProvider(binPath).Execute(ctx, "Reply with exactly: HELLO_AGENTFIELD", Options{Timeout: 300})
	require.NoError(t, err)
	require.NotNil(t, raw)
	assert.False(t, raw.IsError, raw.ErrorMessage)
	assert.Contains(t, raw.Result, "HELLO_AGENTFIELD")
	assert.NotEmpty(t, raw.Messages)
	assert.Greater(t, raw.Metrics.NumTurns, 0)
	assert.Greater(t, raw.Metrics.InputTokens, 0)
	assert.Greater(t, raw.Metrics.OutputTokens, 0)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.Greater(t, *raw.Metrics.CostUSD, 0.0)
	t.Logf(
		"aforge metrics: duration_ms=%d calls=%d input_tokens=%d output_tokens=%d cache_read_tokens=%d cost_usd=%.10f",
		raw.Metrics.DurationAPIMS, raw.Metrics.NumTurns, raw.Metrics.InputTokens,
		raw.Metrics.OutputTokens, raw.Metrics.CacheReadTokens, *raw.Metrics.CostUSD,
	)
}

func TestAforgeRunner_Integration_Schema(t *testing.T) {
	if _, err := exec.LookPath("aforge"); err != nil {
		t.Skip("aforge binary not installed")
	}
	if os.Getenv("OPENROUTER_API_KEY") == "" && os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY or OPENAI_API_KEY is required")
	}

	workDir := t.TempDir()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"greeting": map[string]any{"type": "string"},
			"number":   map[string]any{"type": "integer"},
		},
		"required": []string{"greeting", "number"},
	}

	var parsed map[string]any
	result, err := NewRunner(Options{Provider: ProviderAforge}).Run(
		context.Background(),
		`Return greeting="Hello from Aforge" and number=42. Follow the OUTPUT REQUIREMENTS precisely.`,
		schema,
		&parsed,
		Options{Cwd: workDir, MaxRetries: 1, Timeout: 300},
	)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.IsError, result.ErrorMessage)
	assert.Equal(t, "Hello from Aforge", parsed["greeting"])
	assert.EqualValues(t, 42, parsed["number"])
	require.NotNil(t, result.CostUSD)
	t.Logf(
		"aforge schema metrics: duration_ms=%d calls=%d input_tokens=%d output_tokens=%d cache_read_tokens=%d cost_usd=%.10f",
		result.DurationMS, result.NumTurns, result.InputTokens, result.OutputTokens,
		result.CacheReadTokens, *result.CostUSD,
	)
	matches, err := filepath.Glob(filepath.Join(workDir, ".agentfield-out-*"))
	require.NoError(t, err)
	assert.Empty(t, matches)
}
