package types

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAgentStatusHealthHelpers(t *testing.T) {
	status := &AgentStatus{State: AgentStateActive, HealthScore: 85}
	require.True(t, status.IsHealthy())
	require.False(t, status.IsTransitioning())
	require.Equal(t, AgentStateActive, status.GetEffectiveState())
	require.Equal(t, HealthStatusActive, status.ToLegacyHealthStatus())

	status.HealthScore = 50
	status.State = AgentStateInactive
	require.False(t, status.IsHealthy())
	require.Equal(t, HealthStatusInactive, status.ToLegacyHealthStatus())

	status.State = AgentStateActive
	status.StateTransition = &StateTransition{To: AgentStateStopping}
	require.Equal(t, AgentStateStopping, status.GetEffectiveState())
}

func TestAgentStatusLifecycleMapping(t *testing.T) {
	status := &AgentStatus{State: AgentStateActive, HealthScore: 95}
	require.Equal(t, AgentStatusReady, status.ToLegacyLifecycleStatus())

	status.HealthScore = 60
	require.Equal(t, AgentStatusDegraded, status.ToLegacyLifecycleStatus())

	status.State = AgentStateStarting
	require.Equal(t, AgentStatusStarting, status.ToLegacyLifecycleStatus())

	status.State = AgentStateInactive
	require.Equal(t, AgentStatusOffline, status.ToLegacyLifecycleStatus())
}

func TestNewAgentStatusAndLegacyConversion(t *testing.T) {
	status := NewAgentStatus(AgentStateStarting, StatusSourceManual)
	require.Equal(t, AgentStateStarting, status.State)
	require.Equal(t, StatusSourceManual, status.Source)
	require.Equal(t, 100, status.HealthScore)
	require.Equal(t, HealthStatusUnknown, status.HealthStatus)
	require.Equal(t, AgentStatusStarting, status.LifecycleStatus)
	require.WithinDuration(t, time.Now(), status.LastSeen, 2*time.Second)

	lastBeat := time.Now().Add(-2 * time.Minute)
	fromLegacy := FromLegacyStatus(HealthStatusActive, AgentStatusReady, lastBeat)
	require.Equal(t, AgentStateActive, fromLegacy.State)
	require.Equal(t, lastBeat, fromLegacy.LastSeen)
	require.Equal(t, StatusSourceReconcile, fromLegacy.Source)
}

func TestUpdateFromHeartbeat(t *testing.T) {
	lastBeat := time.Now().Add(-5 * time.Minute)
	status := FromLegacyStatus(HealthStatusInactive, AgentStatusOffline, lastBeat)

	lifecycle := AgentStatusReady
	status.UpdateFromHeartbeat(&lifecycle)

	require.Equal(t, AgentStateActive, status.State)
	require.Equal(t, lifecycle, status.LifecycleStatus)
	require.Equal(t, HealthStatusActive, status.HealthStatus)
	require.Equal(t, StatusSourceHeartbeat, status.Source)
	require.GreaterOrEqual(t, status.HealthScore, 80)
}

func TestStateTransitions(t *testing.T) {
	status := NewAgentStatus(AgentStateActive, StatusSourceManual)

	status.StartTransition(AgentStateStopping, "maintenance")
	require.True(t, status.IsTransitioning())
	require.Equal(t, AgentStateStopping, status.StateTransition.To)

	status.CompleteTransition()
	require.False(t, status.IsTransitioning())
	require.Equal(t, AgentStateStopping, status.State)
	require.Equal(t, status.HealthStatus, status.ToLegacyHealthStatus())
	require.Equal(t, status.LifecycleStatus, status.ToLegacyLifecycleStatus())
}
