package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/require"
)

func TestMarkExecutionCleanupEnabledIfSet(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  bool
		want bool
	}{{"absent", false, false}, {"explicit false", true, true}} {
		t.Run(tc.name, func(t *testing.T) {
			v := viper.New()
			if tc.set {
				v.Set("agentfield.execution_cleanup.enabled", false)
			}
			var cfg Config
			MarkExecutionCleanupEnabledIfSet(v, &cfg)
			require.Equal(t, tc.want, cfg.AgentField.ExecutionCleanup.enabledSet)
		})
	}
}

func TestLoadConfigHonorsExplicitExecutionCleanupDisabled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentfield.yaml")
	require.NoError(t, os.WriteFile(path, []byte("agentfield:\n  execution_cleanup:\n    enabled: false\n"), 0o600))

	cfg, err := LoadConfig(path)
	require.NoError(t, err)
	require.False(t, cfg.AgentField.ExecutionCleanup.Enabled)
}

func TestExecutionCleanupDefaultsWithoutConfiguration(t *testing.T) {
	cfg := Config{}
	ApplyDefaults(&cfg)
	require.True(t, cfg.AgentField.ExecutionCleanup.Enabled)
	require.Zero(t, cfg.AgentField.ExecutionCleanup.RetentionPeriod)
	require.Equal(t, 5*time.Minute, cfg.AgentField.ExecutionCleanup.CleanupInterval)
	require.Equal(t, 30*time.Minute, cfg.AgentField.ExecutionCleanup.StaleExecutionTimeout)
	require.Equal(t, 200, cfg.AgentField.ExecutionCleanup.BatchSize)
	require.Equal(t, time.Hour, cfg.AgentField.ExecutionCleanup.PayloadOrphanGrace)
	require.Equal(t, 90*time.Second, cfg.AgentField.ExecutionQueue.AgentCallTimeout)
}

func TestLoadConfigPreservesExplicitDisabledAgentCallTimeout(t *testing.T) {
	for _, value := range []string{"0s", "-1s"} {
		t.Run(value, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "agentfield.yaml")
			contents := "agentfield:\n  execution_queue:\n    agent_call_timeout: " + value + "\n"
			require.NoError(t, os.WriteFile(path, []byte(contents), 0o600))

			cfg, err := LoadConfig(path)
			require.NoError(t, err)
			want, err := time.ParseDuration(value)
			require.NoError(t, err)
			require.Equal(t, want, cfg.AgentField.ExecutionQueue.AgentCallTimeout)
		})
	}
}

func TestExecutionCleanupEnvironmentOverridesAndMalformedValues(t *testing.T) {
	cfg := Config{}
	ApplyDefaults(&cfg)
	values := map[string]string{
		"AGENTFIELD_EXECUTION_CLEANUP_ENABLED": "false", "AGENTFIELD_EXECUTION_RETENTION_PERIOD": "72h",
		"AGENTFIELD_EXECUTION_CLEANUP_INTERVAL": "30s", "AGENTFIELD_EXECUTION_STALE_TIMEOUT": "45m",
		"AGENTFIELD_EXECUTION_CLEANUP_BATCH_SIZE": "321", "AGENTFIELD_EXECUTION_PRESERVE_RECENT": "2h",
		"AGENTFIELD_PAYLOAD_ORPHAN_GRACE": "3h", "AGENTFIELD_AGENT_CALL_TIMEOUT": "4m",
	}
	for name, value := range values {
		t.Setenv(name, value)
	}
	ApplyEnvOverrides(&cfg)
	require.False(t, cfg.AgentField.ExecutionCleanup.Enabled)
	require.Equal(t, 72*time.Hour, cfg.AgentField.ExecutionCleanup.RetentionPeriod)
	require.Equal(t, 30*time.Second, cfg.AgentField.ExecutionCleanup.CleanupInterval)
	require.Equal(t, 45*time.Minute, cfg.AgentField.ExecutionCleanup.StaleExecutionTimeout)
	require.Equal(t, 321, cfg.AgentField.ExecutionCleanup.BatchSize)
	require.Equal(t, 2*time.Hour, cfg.AgentField.ExecutionCleanup.PreserveRecentDuration)
	require.Equal(t, 3*time.Hour, cfg.AgentField.ExecutionCleanup.PayloadOrphanGrace)
	require.Equal(t, 4*time.Minute, cfg.AgentField.ExecutionQueue.AgentCallTimeout)
	t.Setenv("AGENTFIELD_EXECUTION_RETENTION_PERIOD", "invalid")
	t.Setenv("AGENTFIELD_EXECUTION_CLEANUP_BATCH_SIZE", "invalid")
	ApplyEnvOverrides(&cfg)
	require.Equal(t, 72*time.Hour, cfg.AgentField.ExecutionCleanup.RetentionPeriod)
	require.Equal(t, 321, cfg.AgentField.ExecutionCleanup.BatchSize)
}

func TestExecutionCleanupNonPositiveIntervalIsClamped(t *testing.T) {
	for _, value := range []string{"0s", "-1s"} {
		t.Run(value, func(t *testing.T) {
			cfg := Config{}
			ApplyDefaults(&cfg)
			t.Setenv("AGENTFIELD_EXECUTION_CLEANUP_INTERVAL", value)
			ApplyEnvOverrides(&cfg)
			require.Equal(t, DefaultExecutionCleanupInterval, cfg.AgentField.ExecutionCleanup.CleanupInterval)
		})
	}
}
