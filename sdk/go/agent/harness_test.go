package agent

import (
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestAgentForHarness(t *testing.T) *Agent {
	t.Helper()
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}
	a, err := New(cfg)
	require.NoError(t, err)
	return a
}

func TestHarnessRunner_LazyInitialization(t *testing.T) {
	a := newTestAgentForHarness(t)

	// Runner should not be initialized before first call
	assert.Nil(t, a.harnessRunner)

	runner := a.HarnessRunner()
	assert.NotNil(t, runner)

	// Subsequent calls return the same runner (singleton)
	runner2 := a.HarnessRunner()
	assert.Same(t, runner, runner2)
}

func TestHarnessRunner_DefaultOptions(t *testing.T) {
	// Agent with no HarnessConfig — runner gets zero-value Options; the
	// provider is resolved at dispatch time (explicit > env > "codeaf").
	a := newTestAgentForHarness(t)

	runner := a.HarnessRunner()
	assert.NotNil(t, runner)
	assert.Equal(t, "", runner.DefaultOptions.Provider)
	assert.Equal(t, "", runner.DefaultOptions.Model)
}

func TestHarnessRunner_WithHarnessConfig(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{
			Provider:         "claude-code",
			Model:            "claude-sonnet",
			MaxTurns:         10,
			PermissionMode:   "auto",
			Env:              map[string]string{"FOO": "bar"},
			BinPath:          "/usr/local/bin/claude",
			Timeout:          300,
			MaxRetries:       5,
			SchemaMaxRetries: 3,
		},
	}

	a, err := New(cfg)
	require.NoError(t, err)

	runner := a.HarnessRunner()
	require.NotNil(t, runner)

	opts := runner.DefaultOptions
	assert.Equal(t, "claude-code", opts.Provider)
	assert.Equal(t, "claude-sonnet", opts.Model)
	assert.Equal(t, 10, opts.MaxTurns)
	assert.Equal(t, "auto", opts.PermissionMode)
	assert.Equal(t, map[string]string{"FOO": "bar"}, opts.Env)
	assert.Equal(t, "/usr/local/bin/claude", opts.BinPath)
	assert.Equal(t, 300, opts.Timeout)
	assert.Equal(t, 5, opts.MaxRetries)
	assert.Equal(t, 3, opts.SchemaMaxRetries)
}

func TestHarnessRunner_ConcurrentAccess(t *testing.T) {
	a := newTestAgentForHarness(t)

	// Spawn multiple goroutines that call HarnessRunner concurrently.
	// This verifies the sync.Mutex in HarnessRunner() prevents a data race.
	const goroutines = 20
	results := make(chan *harness.Runner, goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			results <- a.HarnessRunner()
		}()
	}

	var runners []*harness.Runner
	for i := 0; i < goroutines; i++ {
		runners = append(runners, <-results)
	}

	// All goroutines should get the same runner instance
	first := runners[0]
	for _, r := range runners[1:] {
		assert.Same(t, first, r, "all concurrent HarnessRunner() calls should return the same instance")
	}
}

func TestHarness_ErrorWithoutProvider(t *testing.T) {
	// With no provider configured, Harness() defaults to codeaf and reaches
	// provider execution. A missing binary may fail, but provider resolution does not.
	t.Setenv(harness.ProviderEnvVar, "")
	a := newTestAgentForHarness(t)

	missingCodeAF := filepath.Join(t.TempDir(), "missing-codeaf")
	t.Setenv("AFORGE_BIN", "")
	t.Setenv("CODEAF_BIN", missingCodeAF)
	result, err := a.Harness(context.Background(), "do something", nil, nil, harness.Options{})
	assert.Nil(t, result)
	require.ErrorIs(t, err, harness.ErrProviderUnavailable)
	assert.Contains(t, err.Error(), "missing-codeaf")
	assert.Contains(t, err.Error(), "af codeaf ensure")
}

func TestHarness_PassesOptsToRunner(t *testing.T) {
	// Verify that per-call Options are forwarded to the runner.
	// Using a non-existent provider triggers a provider-build error,
	// which confirms the Options reached Run().
	a := newTestAgentForHarness(t)

	_, err := a.Harness(context.Background(), "test", nil, nil, harness.Options{
		Provider: "nonexistent-provider",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown harness provider")
}

func TestHarnessConfig_PartialOverride(t *testing.T) {
	// Only set some HarnessConfig fields; others should be zero.
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{
			Provider: "codex",
			// Model, MaxTurns, etc. left as zero values
		},
	}

	a, err := New(cfg)
	require.NoError(t, err)

	runner := a.HarnessRunner()
	require.NotNil(t, runner)

	assert.Equal(t, "codex", runner.DefaultOptions.Provider)
	assert.Equal(t, "", runner.DefaultOptions.Model)
	assert.Equal(t, 0, runner.DefaultOptions.MaxTurns)
}

func TestHarness_NilHarnessConfig(t *testing.T) {
	// Agent with nil HarnessConfig should still create a runner with default (empty) options
	a := newTestAgentForHarness(t)
	assert.Nil(t, a.cfg.HarnessConfig)

	runner := a.HarnessRunner()
	assert.NotNil(t, runner)
	// Default options should be zero
	assert.Equal(t, harness.Options{}, runner.DefaultOptions)
}

func TestHarnessDoctor_ReportsRequestedProviders(t *testing.T) {
	a := newTestAgentForHarness(t)

	reports, err := a.HarnessDoctor(context.Background(), "codex", "gemini")
	require.NoError(t, err)
	require.Len(t, reports, 2)

	// Ordered by provider name, and each report carries actionable install
	// guidance regardless of what happens to be installed on this machine.
	assert.Equal(t, "codex", reports[0].Provider)
	assert.Equal(t, "gemini", reports[1].Provider)
	for _, report := range reports {
		assert.NotEmpty(t, report.InstallCommand)
	}
}

func TestHarnessDoctor_NoArgsChecksEveryProvider(t *testing.T) {
	a := newTestAgentForHarness(t)

	reports, err := a.HarnessDoctor(context.Background())
	require.NoError(t, err)
	assert.Len(t, reports, len(harness.SupportedProviderNames()))
}

func TestHarnessDoctor_UnknownProviderIsAnError(t *testing.T) {
	a := newTestAgentForHarness(t)

	_, err := a.HarnessDoctor(context.Background(), "not-a-provider")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not-a-provider")
}

func TestHarnessDefaultCodeAFRecordsUsage(t *testing.T) {
	t.Setenv(harness.ProviderEnvVar, "")
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	path := filepath.Join(t.TempDir(), "codeaf")
	require.NoError(t, os.WriteFile(path, []byte(`#!/bin/sh
printf '%s\n' '{"ok":true,"stop":"done","answer":"ok","spend_usd":0,"tokens":{"in":10,"out":5},"model":"reported/model","steps":2}'
`), 0o755))
	t.Setenv("CODEAF_BIN", path)
	a := newTestAgentForHarness(t)
	tracker := NewCostTracker()
	ctx := contextWithCostTracker(context.Background(), tracker)
	result, err := a.Harness(ctx, "task", nil, nil, harness.Options{})
	require.NoError(t, err)
	require.False(t, result.IsError, result.ErrorMessage)
	assert.Equal(t, harness.ProviderCodeAF, result.Provider)
	entries := tracker.Serialize()["entries"].([]map[string]any)
	require.Len(t, entries, 1)
	assert.Equal(t, "codeaf", entries[0]["harness"])
	assert.Equal(t, "reported/model", entries[0]["model"])
	assert.Equal(t, 10, entries[0]["input_tokens"])
	assert.Equal(t, 5, entries[0]["output_tokens"])
	assert.Equal(t, 0.0, entries[0]["cost_usd"])
}

// F1a/d: agent-level defaults and per-call BinPath pins keep the legacy adapter and usage.
func TestHarnessF1ImplicitLegacyPinRecordsUsage(t *testing.T) {
	for _, source := range []string{"env", "config", "call"} {
		t.Run(source, func(t *testing.T) {
			t.Setenv(harness.ProviderEnvVar, " ")
			t.Setenv("AFORGE_BIN", "")
			t.Setenv("AGENTFIELD_AFORGE_COMMAND", "exec")
			path := filepath.Join(t.TempDir(), "legacy")
			require.NoError(t, os.WriteFile(path, []byte(`#!/bin/sh
printf '%s\n' '{"text":"legacy","stop":"done","turns":2,"usage":{"prompt_tokens":10,"completion_tokens":5,"cost":0.1}}'
`), 0o755))
			a := newTestAgentForHarness(t)
			opts := harness.Options{}
			if source == "env" {
				t.Setenv("AFORGE_BIN", path)
			}
			if source == "config" {
				a.cfg.HarnessConfig = &HarnessConfig{BinPath: path}
			}
			if source == "call" {
				opts.BinPath = path
			}
			tracker := NewCostTracker()
			ctx := contextWithCostTracker(context.Background(), tracker)
			result, err := a.Harness(ctx, "p", nil, nil, opts)
			require.NoError(t, err)
			require.False(t, result.IsError, result.ErrorMessage)
			assert.Equal(t, "legacy", result.Result)
			assert.Equal(t, harness.ProviderAforge, result.Provider)
			entries := tracker.Serialize()["entries"].([]map[string]any)
			require.Len(t, entries, 1)
			assert.Equal(t, "aforge", entries[0]["harness"])
		})
	}
}
