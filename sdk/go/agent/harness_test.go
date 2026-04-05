package agent

import (
	"context"
	"io"
	"log"
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
	// Agent with no HarnessConfig — runner gets zero-value Options
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
	// Harness() should fail when no provider is configured.
	// The runner will return an error about a missing provider.
	a := newTestAgentForHarness(t)

	_, err := a.Harness(context.Background(), "do something", nil, nil, harness.Options{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "provider")
}

func TestHarness_PassesOptsToRunner(t *testing.T) {
	// Verify that per-call Options are forwarded to the runner.
	// Using a non-existent provider triggers a provider-build error,
	// which confirms the Options reached Run() (otherwise we'd get
	// the "no harness provider specified" error instead).
	a := newTestAgentForHarness(t)

	_, err := a.Harness(context.Background(), "test", nil, nil, harness.Options{
		Provider: "nonexistent-provider",
	})
	assert.Error(t, err)
	// Should be a provider-build error, NOT the "no harness provider specified" error
	assert.NotContains(t, err.Error(), "no harness provider specified")
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
