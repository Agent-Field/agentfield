package server

import (
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

func TestConfigureAgentRestartSettingsLogsWhenOrphanReapDisabled(t *testing.T) {
	previous := handlers.AgentOrphanReapEnabled()
	t.Cleanup(func() { handlers.SetAgentOrphanReapEnabled(previous) })
	logs := captureServerLogger(t, zerolog.DebugLevel)

	disabled := false
	configureAgentRestartSettings(config.NodeHealthConfig{AgentOrphanReapEnabled: &disabled})

	require.False(t, handlers.AgentOrphanReapEnabled())
	require.Contains(t, logs.String(), "agent orphan reap on re-registration is disabled")
	require.Equal(t, 1, strings.Count(logs.String(), "agent orphan reap on re-registration is disabled"))
}

func TestConfigureAgentRestartSettingsZeroValueDefaultsOrphanReapEnabled(t *testing.T) {
	previous := handlers.AgentOrphanReapEnabled()
	t.Cleanup(func() { handlers.SetAgentOrphanReapEnabled(previous) })
	logs := captureServerLogger(t, zerolog.DebugLevel)

	configureAgentRestartSettings(config.NodeHealthConfig{})

	require.True(t, handlers.AgentOrphanReapEnabled())
	require.NotContains(t, logs.String(), "agent orphan reap on re-registration is disabled")
}

// TestConfigureAgentRestartSettingsAppliesGraceWindows pins that moving the
// grace wiring into configureAgentRestartSettings kept its behaviour: a
// non-zero configured window reaches the handlers package, and a zero value
// leaves the existing default untouched (the "0 = use default" contract
// documented on NodeHealthConfig).
func TestConfigureAgentRestartSettingsAppliesGraceWindows(t *testing.T) {
	previousRestart := handlers.AgentRestartGrace()
	previousDrain := handlers.AgentDrainGrace()
	previousEnabled := handlers.AgentOrphanReapEnabled()
	t.Cleanup(func() {
		handlers.SetAgentRestartGrace(previousRestart)
		handlers.SetAgentDrainGrace(previousDrain)
		handlers.SetAgentOrphanReapEnabled(previousEnabled)
	})

	configureAgentRestartSettings(config.NodeHealthConfig{
		AgentRestartGrace: 7 * time.Second,
		AgentDrainGrace:   11 * time.Second,
	})
	require.Equal(t, 7*time.Second, handlers.AgentRestartGrace())
	require.Equal(t, 11*time.Second, handlers.AgentDrainGrace())

	// Zero means "leave the configured value alone", not "reset to zero".
	configureAgentRestartSettings(config.NodeHealthConfig{})
	require.Equal(t, 7*time.Second, handlers.AgentRestartGrace())
	require.Equal(t, 11*time.Second, handlers.AgentDrainGrace())
}
