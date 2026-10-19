package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// C3-C4: YAML is a first-class policy surface; environment settings override
// both fields, including false and an explicitly empty allowlist.
func TestAgentEndpointConfiguration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentfield.yaml")
	require.NoError(t, os.WriteFile(path, []byte("agentfield:\n  registration:\n    agent_endpoint_strict: true\n    agent_endpoint_allowed_hosts: [localhost, '*.cluster.local', 10.0.0.0/8]\n"), 0600))
	cfg, err := LoadConfig(path)
	require.NoError(t, err)
	require.True(t, cfg.AgentField.Registration.AgentEndpointStrict)
	require.Equal(t, []string{"localhost", "*.cluster.local", "10.0.0.0/8"}, cfg.AgentField.Registration.AgentEndpointAllowedHosts)
	t.Setenv("AGENTFIELD_REGISTRATION_AGENT_ENDPOINT_STRICT", "false")
	t.Setenv("AGENTFIELD_REGISTRATION_AGENT_ENDPOINT_ALLOWED_HOSTS", " pod.internal, , *.trusted, 192.168.0.0/16 ")
	ApplyEnvOverrides(cfg)
	require.False(t, cfg.AgentField.Registration.AgentEndpointStrict)
	require.Equal(t, []string{"pod.internal", "*.trusted", "192.168.0.0/16"}, cfg.AgentField.Registration.AgentEndpointAllowedHosts)
	t.Setenv("AGENTFIELD_REGISTRATION_AGENT_ENDPOINT_STRICT", "true")
	t.Setenv("AGENTFIELD_REGISTRATION_AGENT_ENDPOINT_ALLOWED_HOSTS", "")
	ApplyEnvOverrides(cfg)
	require.True(t, cfg.AgentField.Registration.AgentEndpointStrict)
	require.Empty(t, cfg.AgentField.Registration.AgentEndpointAllowedHosts)
	var defaults Config
	ApplyDefaults(&defaults)
	require.False(t, defaults.AgentField.Registration.AgentEndpointStrict)
	require.Empty(t, defaults.AgentField.Registration.AgentEndpointAllowedHosts)
}
