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

func TestConfigureAgentRestartSettingsAppliesInterruptedResumeBounds(t *testing.T) {
	previousEnabled := handlers.ResumeInterruptedRuns()
	previousAttempts := handlers.ResumeInterruptedMaxAttempts()
	previousWindow := handlers.ResumeInterruptedWindow()
	previousLimit := handlers.ResumeInterruptedLimit()
	previousDelay := handlers.ResumeInterruptedDelay()
	t.Cleanup(func() {
		handlers.SetResumeInterruptedRuns(previousEnabled)
		handlers.SetResumeInterruptedMaxAttempts(previousAttempts)
		handlers.SetResumeInterruptedWindow(previousWindow)
		handlers.SetResumeInterruptedLimit(previousLimit)
		handlers.SetResumeInterruptedDelay(previousDelay)
	})
	enabled := true
	configureAgentRestartSettings(config.NodeHealthConfig{
		ResumeInterruptedRuns:        &enabled,
		ResumeInterruptedMaxAttempts: 3,
		ResumeInterruptedWindow:      2 * time.Hour,
		ResumeInterruptedLimit:       12,
		ResumeInterruptedDelay:       4 * time.Second,
	})
	require.True(t, handlers.ResumeInterruptedRuns())
	require.Equal(t, 3, handlers.ResumeInterruptedMaxAttempts())
	require.Equal(t, 2*time.Hour, handlers.ResumeInterruptedWindow())
	require.Equal(t, 12, handlers.ResumeInterruptedLimit())
	require.Equal(t, 4*time.Second, handlers.ResumeInterruptedDelay())
}
