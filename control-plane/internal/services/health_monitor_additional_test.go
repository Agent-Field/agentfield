package services

import (
	"context"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestHealthMonitorLegacyStatusTransitions(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	nodeID := "legacy-agent"
	require.NoError(t, provider.RegisterAgent(context.Background(), &types.AgentNode{
		ID:              nodeID,
		BaseURL:         "http://legacy.example",
		HealthStatus:    types.HealthStatusUnknown,
		LifecycleStatus: types.AgentStatusStarting,
		LastHeartbeat:   time.Now(),
		RegisteredAt:    time.Now(),
	}))

	hm := NewHealthMonitor(provider, HealthMonitorConfig{}, nil, &mockAgentClient{}, nil, nil)

	hm.markAgentActive(nodeID)
	agent, err := provider.GetAgent(context.Background(), nodeID)
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusActive, agent.HealthStatus)
	require.Equal(t, types.AgentStatusReady, agent.LifecycleStatus)

	hm.markAgentInactive(nodeID, 3)
	agent, err = provider.GetAgent(context.Background(), nodeID)
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, agent.HealthStatus)
	require.Equal(t, types.AgentStatusOffline, agent.LifecycleStatus)
}

func TestHealthMonitorRecoverFromDatabaseSkipsMissingBaseURL(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:            "recoverable-agent",
		BaseURL:       "http://recoverable.example",
		LastHeartbeat: time.Now(),
		RegisteredAt:  time.Now(),
	}))
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:           "no-url-agent",
		LastHeartbeat: time.Now(),
		RegisteredAt: time.Now(),
	}))

	mockClient.setStatusResponse("recoverable-agent", "running")

	require.NoError(t, hm.RecoverFromDatabase(ctx))
	require.Eventually(t, func() bool {
		hm.agentsMutex.RLock()
		defer hm.agentsMutex.RUnlock()
		_, okRecoverable := hm.activeAgents["recoverable-agent"]
		_, okNoURL := hm.activeAgents["no-url-agent"]
		return okRecoverable && !okNoURL && mockClient.getStatusCallCountFor("recoverable-agent") > 0
	}, 5*time.Second, 100*time.Millisecond)
}

func TestHealthMonitorUnregisterAgentLegacyFallback(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	nodeID := "legacy-unregister-agent"
	require.NoError(t, provider.RegisterAgent(context.Background(), &types.AgentNode{
		ID:              nodeID,
		BaseURL:         "http://legacy-unregister.example",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		RegisteredAt:    time.Now(),
	}))

	hm := NewHealthMonitor(provider, HealthMonitorConfig{}, nil, &mockAgentClient{}, nil, nil)
	hm.RegisterAgent(nodeID, "http://legacy-unregister.example")
	hm.UnregisterAgent(nodeID)

	agent, err := provider.GetAgent(context.Background(), nodeID)
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, agent.HealthStatus)
	require.Equal(t, types.AgentStatusOffline, agent.LifecycleStatus)
}
