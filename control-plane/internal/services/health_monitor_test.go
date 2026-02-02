package services

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock AgentClient for testing
type mockAgentClient struct {
	mu                   sync.RWMutex
	statusResponses      map[string]*interfaces.AgentStatusResponse
	statusErrors         map[string]error
	mcpHealthResponses   map[string]*interfaces.MCPHealthResponse
	mcpHealthErrors      map[string]error
	getStatusCallCount   map[string]int
	getMCPHealthCallCount map[string]int
}

func newMockAgentClient() *mockAgentClient {
	return &mockAgentClient{
		statusResponses:      make(map[string]*interfaces.AgentStatusResponse),
		statusErrors:         make(map[string]error),
		mcpHealthResponses:   make(map[string]*interfaces.MCPHealthResponse),
		mcpHealthErrors:      make(map[string]error),
		getStatusCallCount:   make(map[string]int),
		getMCPHealthCallCount: make(map[string]int),
	}
}

func (m *mockAgentClient) GetAgentStatus(ctx context.Context, nodeID string) (*interfaces.AgentStatusResponse, error) {
	m.mu.Lock()
	m.getStatusCallCount[nodeID]++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	if err, ok := m.statusErrors[nodeID]; ok {
		return nil, err
	}
	if resp, ok := m.statusResponses[nodeID]; ok {
		return resp, nil
	}
	return nil, errors.New("agent not found")
}

func (m *mockAgentClient) GetMCPHealth(ctx context.Context, nodeID string) (*interfaces.MCPHealthResponse, error) {
	m.mu.Lock()
	m.getMCPHealthCallCount[nodeID]++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	if err, ok := m.mcpHealthErrors[nodeID]; ok {
		return nil, err
	}
	if resp, ok := m.mcpHealthResponses[nodeID]; ok {
		return resp, nil
	}
	return nil, errors.New("MCP not available")
}

func (m *mockAgentClient) RestartMCPServer(ctx context.Context, nodeID, alias string) error {
	return nil
}

func (m *mockAgentClient) GetMCPTools(ctx context.Context, nodeID, alias string) (*interfaces.MCPToolsResponse, error) {
	return nil, nil
}

func (m *mockAgentClient) ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
	return nil, nil
}

func (m *mockAgentClient) setStatusResponse(nodeID string, status string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.statusResponses[nodeID] = &interfaces.AgentStatusResponse{
		Status: status,
	}
}

func (m *mockAgentClient) setStatusError(nodeID string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.statusErrors[nodeID] = err
}

func (m *mockAgentClient) setMCPHealthResponse(nodeID string, response *interfaces.MCPHealthResponse) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.mcpHealthResponses[nodeID] = response
}


func (m *mockAgentClient) getStatusCallCountFor(nodeID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.getStatusCallCount[nodeID]
}

func (m *mockAgentClient) getMCPHealthCallCountFor(nodeID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.getMCPHealthCallCount[nodeID]
}

func setupHealthMonitorTest(t *testing.T) (*HealthMonitor, storage.StorageProvider, *mockAgentClient, *StatusManager, *PresenceManager) {
	t.Helper()

	provider, ctx := setupTestStorage(t)

	// Create status manager
	statusConfig := StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, nil)

	// Create presence manager
	presenceConfig := PresenceManagerConfig{
		HeartbeatTTL:  5 * time.Second,
		SweepInterval: 1 * time.Second,
		HardEvictTTL:  10 * time.Second,
	}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)

	// Create mock agent client
	mockClient := newMockAgentClient()

	// Create health monitor
	config := HealthMonitorConfig{
		CheckInterval: 100 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	return hm, provider, mockClient, statusManager, presenceManager
}

func TestHealthMonitor_NewHealthMonitor(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	defer provider.Close(ctx)

	statusManager := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	presenceManager := NewPresenceManager(statusManager, PresenceManagerConfig{})
	mockClient := newMockAgentClient()

	config := HealthMonitorConfig{
		CheckInterval: 10 * time.Second,
	}

	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	require.NotNil(t, hm)
	assert.Equal(t, 10*time.Second, hm.config.CheckInterval)
	assert.NotNil(t, hm.activeAgents)
	assert.NotNil(t, hm.mcpHealthCache)
	assert.NotNil(t, hm.stopCh)
}

func TestHealthMonitor_NewHealthMonitor_DefaultConfig(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	defer provider.Close(ctx)

	statusManager := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	presenceManager := NewPresenceManager(statusManager, PresenceManagerConfig{})
	mockClient := newMockAgentClient()

	// Pass zero config to test defaults
	hm := NewHealthMonitor(provider, HealthMonitorConfig{}, nil, mockClient, statusManager, presenceManager)

	require.NotNil(t, hm)
	assert.Equal(t, 10*time.Second, hm.config.CheckInterval, "Should use default check interval")
}

func TestHealthMonitor_RegisterAgent(t *testing.T) {
	hm, _, _, _, presenceManager := setupHealthMonitorTest(t)

	nodeID := "test-agent-1"
	baseURL := "http://localhost:8001"

	hm.RegisterAgent(nodeID, baseURL)

	// Verify agent is in active registry
	hm.agentsMutex.RLock()
	agent, exists := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	require.True(t, exists, "Agent should be registered")
	assert.Equal(t, nodeID, agent.NodeID)
	assert.Equal(t, baseURL, agent.BaseURL)
	assert.Equal(t, types.HealthStatusUnknown, agent.LastStatus)

	// Verify presence manager was notified
	assert.True(t, presenceManager.HasLease(nodeID), "Presence manager should track agent")
}

func TestHealthMonitor_RegisterAgent_MultipleAgents(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	agents := map[string]string{
		"agent-1": "http://localhost:8001",
		"agent-2": "http://localhost:8002",
		"agent-3": "http://localhost:8003",
	}

	for nodeID, baseURL := range agents {
		hm.RegisterAgent(nodeID, baseURL)
	}

	hm.agentsMutex.RLock()
	defer hm.agentsMutex.RUnlock()

	assert.Equal(t, 3, len(hm.activeAgents), "Should have 3 registered agents")
	for nodeID := range agents {
		_, exists := hm.activeAgents[nodeID]
		assert.True(t, exists, "Agent %s should be registered", nodeID)
	}
}

func TestHealthMonitor_UnregisterAgent(t *testing.T) {
	hm, provider, _, _, presenceManager := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-unregister"
	baseURL := "http://localhost:8001"

	// First register the agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Verify agent is registered
	require.True(t, presenceManager.HasLease(nodeID))

	// Unregister
	hm.UnregisterAgent(nodeID)

	// Verify agent is removed from active registry
	hm.agentsMutex.RLock()
	_, exists := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	assert.False(t, exists, "Agent should be unregistered")

	// Verify presence manager was notified
	assert.False(t, presenceManager.HasLease(nodeID), "Presence manager should not track agent")
}

func TestHealthMonitor_UnregisterAgent_NonExistent(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	// Unregister non-existent agent should not panic
	hm.UnregisterAgent("non-existent-agent")

	hm.agentsMutex.RLock()
	count := len(hm.activeAgents)
	hm.agentsMutex.RUnlock()

	assert.Equal(t, 0, count, "Should have no registered agents")
}

func TestHealthMonitor_CheckAgentHealth_Healthy(t *testing.T) {
	hm, provider, mockClient, _, presenceManager := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-healthy"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set mock to return healthy status
	mockClient.setStatusResponse(nodeID, "running")

	// Perform health check
	hm.checkAgentHealth(nodeID)

	// Wait a bit for async updates
	time.Sleep(100 * time.Millisecond)

	// Verify status was updated to active
	hm.agentsMutex.RLock()
	updatedAgent := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	assert.Equal(t, types.HealthStatusActive, updatedAgent.LastStatus, "Agent should be marked as active")
	assert.True(t, presenceManager.HasLease(nodeID), "Presence should be updated")
}

func TestHealthMonitor_CheckAgentHealth_Inactive(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-inactive"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set mock to return error (simulating agent offline)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))

	// Consecutive failures required (default: 3) before marking inactive
	for i := 0; i < hm.config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth(nodeID)
	}

	// Wait a bit for async updates
	time.Sleep(100 * time.Millisecond)

	// Verify status was updated to inactive after consecutive failures
	hm.agentsMutex.RLock()
	updatedAgent := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	assert.Equal(t, types.HealthStatusInactive, updatedAgent.LastStatus, "Agent should be marked as inactive after consecutive failures")
}

func TestHealthMonitor_CheckAgentHealth_NotRunning(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-not-running"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set mock to return non-running status
	mockClient.setStatusResponse(nodeID, "stopped")

	// Consecutive failures required before marking inactive

	for i := 0; i < hm.config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth(nodeID)
	}

	// Wait a bit for async updates
	time.Sleep(100 * time.Millisecond)

	// Verify status was updated to inactive after consecutive failures
	hm.agentsMutex.RLock()
	updatedAgent := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	assert.Equal(t, types.HealthStatusInactive, updatedAgent.LastStatus, "Agent should be marked as inactive when not running")
}

func TestHealthMonitor_CheckAgentHealth_StatusTransitions(t *testing.T) {
	// Use short debounce for test speed
	provider, ctx := setupTestStorage(t)
	statusManager := NewStatusManager(provider, StatusManagerConfig{ReconcileInterval: 30 * time.Second}, nil, nil)
	presenceConfig := PresenceManagerConfig{HeartbeatTTL: 5 * time.Second, SweepInterval: 1 * time.Second, HardEvictTTL: 10 * time.Second}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)
	mockClient := newMockAgentClient()

	config := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		ConsecutiveFailures: 2, // Use 2 for faster test
		RecoveryDebounce:    200 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	nodeID := "test-agent-transitions"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	hm.RegisterAgent(nodeID, baseURL)


	// Test transition: Unknown -> Active
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus)
	hm.agentsMutex.RUnlock()

	// Test transition: Active -> Inactive (requires consecutive failures)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	for i := 0; i < config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth(nodeID)
	}
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus)
	hm.agentsMutex.RUnlock()

	// Test transition: Inactive -> Active (after debounce period)
	time.Sleep(300 * time.Millisecond) // Wait past 200ms debounce

	mockClient.setStatusResponse(nodeID, "running")
	mockClient.mu.Lock()
	delete(mockClient.statusErrors, nodeID)
	mockClient.mu.Unlock()
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus)
	hm.agentsMutex.RUnlock()
}

func TestHealthMonitor_CheckAgentHealth_UnregisteredAgent(t *testing.T) {
	hm, _, mockClient, _, _ := setupHealthMonitorTest(t)

	nodeID := "test-agent-unregistered"

	mockClient.setStatusResponse(nodeID, "running")

	// Should skip check for unregistered agent (not in active registry)
	hm.checkAgentHealth(nodeID)

	// Verify no status calls were made (agent was not in registry)
	assert.Equal(t, 0, mockClient.getStatusCallCountFor(nodeID))
}

func TestHealthMonitor_MCP_CheckMCPHealth(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-mcp"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set mock MCP health response
	mcpResponse := &interfaces.MCPHealthResponse{
		Summary: interfaces.MCPSummary{
			TotalServers:   3,
			RunningServers: 3,
			TotalTools:     15,
			OverallHealth:  0.95,
		},
	}
	mockClient.setMCPHealthResponse(nodeID, mcpResponse)

	// Set agent as healthy first
	mockClient.setStatusResponse(nodeID, "running")

	// Perform health check (should trigger MCP check for active agents)

	hm.checkAgentHealth(nodeID)
	time.Sleep(200 * time.Millisecond)

	// Verify MCP health was checked
	assert.Greater(t, mockClient.getMCPHealthCallCountFor(nodeID), 0, "MCP health should be checked for active agent")

	// Verify MCP health is cached
	cache := hm.GetMCPHealthCache()
	mcpData, exists := cache[nodeID]
	require.True(t, exists, "MCP health should be cached")
	assert.Equal(t, 3, mcpData.TotalServers)
	assert.Equal(t, 3, mcpData.RunningServers)
	assert.Equal(t, 15, mcpData.TotalTools)
	assert.Equal(t, 0.95, mcpData.OverallHealth)
}

func TestHealthMonitor_MCP_HealthChange(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-mcp-change"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set agent as healthy
	mockClient.setStatusResponse(nodeID, "running")

	// First MCP health check
	mcpResponse1 := &interfaces.MCPHealthResponse{
		Summary: interfaces.MCPSummary{
			TotalServers:   3,
			RunningServers: 3,
			TotalTools:     15,
			OverallHealth:  0.95,
		},
	}
	mockClient.setMCPHealthResponse(nodeID, mcpResponse1)


	hm.checkAgentHealth(nodeID)
	time.Sleep(200 * time.Millisecond)

	// Change MCP health
	mcpResponse2 := &interfaces.MCPHealthResponse{
		Summary: interfaces.MCPSummary{
			TotalServers:   3,
			RunningServers: 2, // One server failed
			TotalTools:     10,
			OverallHealth:  0.67,
		},
	}
	mockClient.setMCPHealthResponse(nodeID, mcpResponse2)

	// Second health check
	hm.checkAgentHealth(nodeID)
	time.Sleep(200 * time.Millisecond)

	// Verify MCP health was updated
	cache := hm.GetMCPHealthCache()
	mcpData, exists := cache[nodeID]
	require.True(t, exists)
	assert.Equal(t, 2, mcpData.RunningServers, "MCP health should be updated")
	assert.Equal(t, 0.67, mcpData.OverallHealth)
}

func TestHealthMonitor_MCP_NoChange(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-mcp-no-change"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set agent as healthy
	mockClient.setStatusResponse(nodeID, "running")

	// Set MCP health response
	mcpResponse := &interfaces.MCPHealthResponse{
		Summary: interfaces.MCPSummary{
			TotalServers:   3,
			RunningServers: 3,
			TotalTools:     15,
			OverallHealth:  0.95,
		},
	}
	mockClient.setMCPHealthResponse(nodeID, mcpResponse)


	// First check
	hm.checkAgentHealth(nodeID)
	time.Sleep(200 * time.Millisecond)

	// Verify hasMCPHealthChanged returns false for same data
	newSummary := &domain.MCPSummaryData{
		TotalServers:   3,
		RunningServers: 3,
		TotalTools:     15,
		OverallHealth:  0.95,
	}
	hasChanged := hm.hasMCPHealthChanged(nodeID, newSummary)
	assert.False(t, hasChanged, "Should detect no change in MCP health")
}

func TestHealthMonitor_MCP_InactiveAgent(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-agent-mcp-inactive"
	baseURL := "http://localhost:8001"

	// Register agent in storage
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: baseURL,
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Register in health monitor
	hm.RegisterAgent(nodeID, baseURL)

	// Set agent as inactive
	mockClient.setStatusError(nodeID, errors.New("connection refused"))


	hm.checkAgentHealth(nodeID)
	time.Sleep(200 * time.Millisecond)

	// MCP health should NOT be checked for inactive agents
	assert.Equal(t, 0, mockClient.getMCPHealthCallCountFor(nodeID), "MCP health should not be checked for inactive agent")
}

func TestHealthMonitor_GetMCPHealthCache(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	// Add some test data to cache
	hm.mcpCacheMutex.Lock()
	hm.mcpHealthCache["agent-1"] = &domain.MCPSummaryData{
		TotalServers:   3,
		RunningServers: 3,
		TotalTools:     15,
		OverallHealth:  0.95,
	}
	hm.mcpHealthCache["agent-2"] = &domain.MCPSummaryData{
		TotalServers:   2,
		RunningServers: 1,
		TotalTools:     8,
		OverallHealth:  0.50,
	}
	hm.mcpCacheMutex.Unlock()

	// Get cache
	cache := hm.GetMCPHealthCache()

	// Verify cache contents
	assert.Equal(t, 2, len(cache))
	assert.Contains(t, cache, "agent-1")
	assert.Contains(t, cache, "agent-2")
	assert.Equal(t, 3, cache["agent-1"].TotalServers)
	assert.Equal(t, 1, cache["agent-2"].RunningServers)
}

func TestHealthMonitor_ConcurrentAccess(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	// Register multiple agents
	agents := []string{"agent-1", "agent-2", "agent-3", "agent-4", "agent-5"}
	for _, nodeID := range agents {
		agent := &types.AgentNode{
			ID:      nodeID,
			BaseURL: "http://localhost:800" + nodeID[len(nodeID)-1:],
		}
		err := provider.RegisterAgent(ctx, agent)
		require.NoError(t, err)

		hm.RegisterAgent(nodeID, agent.BaseURL)
		mockClient.setStatusResponse(nodeID, "running")
	}

	var wg sync.WaitGroup

	// Concurrent health checks
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			hm.checkActiveAgents()
		}()
	}

	// Concurrent register/unregister
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			nodeID := "temp-agent-" + string(rune('0'+idx))
			hm.RegisterAgent(nodeID, "http://localhost:9000")
			time.Sleep(10 * time.Millisecond)
			hm.UnregisterAgent(nodeID)
		}(i)
	}

	// Concurrent MCP cache access
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = hm.GetMCPHealthCache()
		}()
	}

	wg.Wait()

	// Verify no race conditions
	hm.agentsMutex.RLock()
	activeCount := len(hm.activeAgents)
	hm.agentsMutex.RUnlock()

	assert.Equal(t, 5, activeCount, "Should have 5 active agents after concurrent operations")
}

func TestHealthMonitor_StartStop(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	// Start in goroutine
	go hm.Start()

	// Let it run for a bit
	time.Sleep(300 * time.Millisecond)

	// Stop should not block
	hm.Stop()

	// Verify stop worked (stop channel closed)
	select {
	case <-hm.stopCh:
		// Expected: channel is closed
	default:
		t.Fatal("Stop channel should be closed")
	}
}

func TestHealthMonitor_PeriodicChecks(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	defer provider.Close(ctx)

	mockClient := newMockAgentClient()
	statusManager := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	presenceManager := NewPresenceManager(statusManager, PresenceManagerConfig{})

	// Use very short interval for testing
	config := HealthMonitorConfig{
		CheckInterval: 50 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	// Register agent
	nodeID := "test-periodic"
	agent := &types.AgentNode{
		ID:      nodeID,
		BaseURL: "http://localhost:8001",
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	hm.RegisterAgent(nodeID, agent.BaseURL)
	mockClient.setStatusResponse(nodeID, "running")

	// Start monitoring
	go hm.Start()

	// Let it run for multiple check intervals
	time.Sleep(250 * time.Millisecond)

	// Stop
	hm.Stop()

	// Verify multiple checks occurred
	callCount := mockClient.getStatusCallCountFor(nodeID)
	assert.GreaterOrEqual(t, callCount, 3, "Should have performed multiple periodic checks")
}

func TestHealthMonitor_CheckActiveAgents_NoAgents(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	// Should not panic with no agents
	hm.checkActiveAgents()

	hm.agentsMutex.RLock()
	count := len(hm.activeAgents)
	hm.agentsMutex.RUnlock()

	assert.Equal(t, 0, count)
}

func TestHealthMonitor_RecoverFromDatabase_NoNodes(t *testing.T) {
	hm, _, _, _, _ := setupHealthMonitorTest(t)

	ctx := context.Background()

	// Should succeed with empty database
	err := hm.RecoverFromDatabase(ctx)
	require.NoError(t, err)

	// Verify no agents registered
	hm.agentsMutex.RLock()
	count := len(hm.activeAgents)
	hm.agentsMutex.RUnlock()

	assert.Equal(t, 0, count)
}

func TestHealthMonitor_RecoverFromDatabase_WithNodes(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)

	ctx := context.Background()

	// Create some agents in the database
	agent1 := &types.AgentNode{
		ID:      "agent-1",
		BaseURL: "http://localhost:8001",
	}
	agent2 := &types.AgentNode{
		ID:      "agent-2",
		BaseURL: "http://localhost:8002",
	}
	agent3 := &types.AgentNode{
		ID:      "agent-3",
		BaseURL: "", // No BaseURL - should be skipped
	}

	err := provider.RegisterAgent(ctx, agent1)
	require.NoError(t, err)
	err = provider.RegisterAgent(ctx, agent2)
	require.NoError(t, err)
	err = provider.RegisterAgent(ctx, agent3)
	require.NoError(t, err)

	// Set up mock responses - agent-1 is running, agent-2 is not reachable
	mockClient.setStatusResponse("agent-1", "running")
	mockClient.setStatusError("agent-2", errors.New("connection refused"))

	// Recover from database
	err = hm.RecoverFromDatabase(ctx)
	require.NoError(t, err)

	// Verify agents were registered (except agent-3 with no BaseURL)
	hm.agentsMutex.RLock()
	count := len(hm.activeAgents)
	_, agent1Exists := hm.activeAgents["agent-1"]
	_, agent2Exists := hm.activeAgents["agent-2"]
	_, agent3Exists := hm.activeAgents["agent-3"]
	hm.agentsMutex.RUnlock()

	assert.Equal(t, 2, count, "Should have registered 2 agents (agent-3 has no BaseURL)")
	assert.True(t, agent1Exists, "agent-1 should be registered")
	assert.True(t, agent2Exists, "agent-2 should be registered")
	assert.False(t, agent3Exists, "agent-3 should not be registered (no BaseURL)")

	// Wait for async health checks to complete (RecoverFromDatabase runs them in a goroutine)
	time.Sleep(200 * time.Millisecond)

	// Verify health checks were performed
	assert.GreaterOrEqual(t, mockClient.getStatusCallCountFor("agent-1"), 1, "Should have checked agent-1 health")
	assert.GreaterOrEqual(t, mockClient.getStatusCallCountFor("agent-2"), 1, "Should have checked agent-2 health")
}

func TestHealthMonitor_RecoverFromDatabase_MarksUnreachableNodesInactive(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)

	ctx := context.Background()

	// Create an agent in the database
	agent := &types.AgentNode{
		ID:           "unreachable-agent",
		BaseURL:      "http://localhost:9999",
		HealthStatus: types.HealthStatusActive, // Was active before restart
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Mock: agent is not reachable
	mockClient.setStatusError("unreachable-agent", errors.New("connection refused"))

	// Recover from database
	err = hm.RecoverFromDatabase(ctx)
	require.NoError(t, err)

	// Wait for async health checks to complete
	time.Sleep(200 * time.Millisecond)

	// After recovery, agent is registered but a single check won't mark inactive
	// (consecutive failures required). Run additional checks to reach the threshold.
	hm.agentsMutex.RLock()
	activeAgent, exists := hm.activeAgents["unreachable-agent"]
	hm.agentsMutex.RUnlock()

	assert.True(t, exists, "Agent should be registered")

	// Run enough checks to trigger consecutive failure threshold
	for i := 0; i < hm.config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth("unreachable-agent")
	}
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	activeAgent = hm.activeAgents["unreachable-agent"]
	hm.agentsMutex.RUnlock()

	assert.Equal(t, types.HealthStatusInactive, activeAgent.LastStatus, "Agent should be marked inactive after consecutive failed health checks")
}

func TestHealthMonitor_RecoverFromDatabase_MarksReachableNodesActive(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)

	ctx := context.Background()

	// Create an agent in the database
	agent := &types.AgentNode{
		ID:           "reachable-agent",
		BaseURL:      "http://localhost:8001",
		HealthStatus: types.HealthStatusInactive, // Was inactive before restart
	}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Mock: agent is running
	mockClient.setStatusResponse("reachable-agent", "running")

	// Recover from database
	err = hm.RecoverFromDatabase(ctx)
	require.NoError(t, err)

	// Wait for async health checks to complete (RecoverFromDatabase runs them in a goroutine)
	time.Sleep(200 * time.Millisecond)

	// Verify agent was registered and marked active
	hm.agentsMutex.RLock()
	activeAgent, exists := hm.activeAgents["reachable-agent"]
	hm.agentsMutex.RUnlock()

	assert.True(t, exists, "Agent should be registered")
	assert.Equal(t, types.HealthStatusActive, activeAgent.LastStatus, "Agent should be marked active after successful health check")
}

// --- Consecutive failure tests ---

func TestHealthMonitor_ConsecutiveFailures_SingleFailureKeepsActive(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-single-failure"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	hm.RegisterAgent(nodeID, baseURL)

	// First make agent active
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus, "Agent should be active initially")
	hm.agentsMutex.RUnlock()

	// Now simulate one failure — should NOT mark inactive
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.NotEqual(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"Single failure should NOT mark agent inactive (consecutive failures required)")
	assert.Equal(t, 1, hm.activeAgents[nodeID].ConsecutiveFailures, "Should have 1 consecutive failure")
	hm.agentsMutex.RUnlock()
}

func TestHealthMonitor_ConsecutiveFailures_ThreeFailuresMarksInactive(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-three-failures"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	hm.RegisterAgent(nodeID, baseURL)

	// Make agent active first
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	// Now fail 3 times (default ConsecutiveFailures threshold)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))

	for i := 0; i < hm.config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth(nodeID)
	}
	time.Sleep(100 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"Agent should be inactive after %d consecutive failures", hm.config.ConsecutiveFailures)
	hm.agentsMutex.RUnlock()
}

func TestHealthMonitor_ConsecutiveFailures_SuccessResetsCounter(t *testing.T) {
	hm, provider, mockClient, _, _ := setupHealthMonitorTest(t)
	ctx := context.Background()

	nodeID := "test-success-resets"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	hm.RegisterAgent(nodeID, baseURL)

	// Make active first
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(100 * time.Millisecond)

	// Fail twice (2 out of 3)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	hm.checkAgentHealth(nodeID)
	hm.checkAgentHealth(nodeID)

	hm.agentsMutex.RLock()
	assert.Equal(t, 2, hm.activeAgents[nodeID].ConsecutiveFailures, "Should have 2 failures")
	hm.agentsMutex.RUnlock()

	// Success resets the counter
	mockClient.mu.Lock()
	delete(mockClient.statusErrors, nodeID)
	mockClient.mu.Unlock()
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)

	hm.agentsMutex.RLock()
	assert.Equal(t, 0, hm.activeAgents[nodeID].ConsecutiveFailures, "Success should reset failure counter")
	assert.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus, "Agent should still be active")
	hm.agentsMutex.RUnlock()

	// Fail twice more — should still NOT be inactive (counter was reset)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	hm.checkAgentHealth(nodeID)
	hm.checkAgentHealth(nodeID)

	hm.agentsMutex.RLock()
	assert.NotEqual(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"Agent should NOT be inactive — counter was reset by success")
	hm.agentsMutex.RUnlock()
}

func TestHealthMonitor_RecoveryDebounce_BlocksTooFastRecovery(t *testing.T) {
	// Use a short but measurable debounce for testing
	provider, ctx := setupTestStorage(t)
	statusManager := NewStatusManager(provider, StatusManagerConfig{ReconcileInterval: 30 * time.Second}, nil, nil)
	presenceConfig := PresenceManagerConfig{HeartbeatTTL: 5 * time.Second, SweepInterval: 1 * time.Second, HardEvictTTL: 10 * time.Second}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)
	mockClient := newMockAgentClient()

	config := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		ConsecutiveFailures: 2,
		RecoveryDebounce:    500 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	nodeID := "test-debounce"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)
	hm.RegisterAgent(nodeID, baseURL)

	// Make active, then inactive
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	for i := 0; i < config.ConsecutiveFailures; i++ {
		hm.checkAgentHealth(nodeID)
	}
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	require.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus)
	hm.agentsMutex.RUnlock()

	// Immediately try to recover — should be blocked by debounce
	mockClient.mu.Lock()
	delete(mockClient.statusErrors, nodeID)
	mockClient.mu.Unlock()
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"Recovery should be blocked by debounce")
	hm.agentsMutex.RUnlock()

	// Wait past debounce, then try again
	time.Sleep(600 * time.Millisecond)
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus,
		"Recovery should succeed after debounce period")
	hm.agentsMutex.RUnlock()
}

func TestHealthMonitor_Config_ConsecutiveFailuresConfigurable(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	statusManager := NewStatusManager(provider, StatusManagerConfig{ReconcileInterval: 30 * time.Second}, nil, nil)
	presenceConfig := PresenceManagerConfig{HeartbeatTTL: 5 * time.Second, SweepInterval: 1 * time.Second, HardEvictTTL: 10 * time.Second}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)
	mockClient := newMockAgentClient()

	// Configure to require 5 consecutive failures
	config := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		ConsecutiveFailures: 5,
		RecoveryDebounce:    100 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	nodeID := "test-config-failures"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)
	hm.RegisterAgent(nodeID, baseURL)

	// Make active first
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	// Fail 4 times — should NOT be inactive (threshold is 5)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	for i := 0; i < 4; i++ {
		hm.checkAgentHealth(nodeID)
	}

	hm.agentsMutex.RLock()
	assert.NotEqual(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"4 failures should not mark inactive when threshold is 5")
	assert.Equal(t, 4, hm.activeAgents[nodeID].ConsecutiveFailures)
	hm.agentsMutex.RUnlock()

	// 5th failure — NOW it should be inactive
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"5th failure should mark inactive when threshold is 5")
	hm.agentsMutex.RUnlock()
}

// =============================================================================
// INTEGRATION TESTS — All 3 services running concurrently with competing signals
// =============================================================================
//
// These tests start HealthMonitor, StatusManager, and PresenceManager running
// concurrently (as they do in production) and simulate the exact scenario that
// causes flapping: health check failures happening while heartbeats are flowing.

// TestIntegration_NoFlapping_HeartbeatsDuringTransientFailures is the critical
// integration test. It runs all 3 services concurrently and verifies that an
// agent NEVER flaps to inactive while heartbeats are being sent, even when
// HTTP health checks are intermittently failing.
func TestIntegration_NoFlapping_HeartbeatsDuringTransientFailures(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// --- Wire services exactly as production (server.go) ---
	mockClient := newMockAgentClient()

	// StatusManager with short reconciliation and the same mock client
	// (so UpdateFromHeartbeat -> GetAgentStatus uses mock HTTP too)
	statusConfig := StatusManagerConfig{
		ReconcileInterval:       200 * time.Millisecond,
		StatusCacheTTL:          50 * time.Millisecond, // Short cache so state changes propagate fast
		HeartbeatStaleThreshold: 2 * time.Second,
		MaxTransitionTime:       1 * time.Second,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, mockClient)

	// PresenceManager with short TTLs
	presenceConfig := PresenceManagerConfig{
		HeartbeatTTL:  2 * time.Second,
		SweepInterval: 200 * time.Millisecond,
		HardEvictTTL:  5 * time.Second,
	}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)

	// HealthMonitor with fast checks but requiring 3 consecutive failures
	hmConfig := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		CheckTimeout:        2 * time.Second,
		ConsecutiveFailures: 3,
		RecoveryDebounce:    200 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, hmConfig, nil, mockClient, statusManager, presenceManager)
	presenceManager.SetExpireCallback(hm.UnregisterAgent)

	// Cleanup
	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		statusManager.Stop()
		_ = provider.Close(ctx)
	})

	// --- Register agent in storage ---
	nodeID := "integration-agent-1"
	node := &types.AgentNode{
		ID:              nodeID,
		TeamID:          "team",
		BaseURL:         "http://localhost:9999",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Agent starts healthy
	mockClient.setStatusResponse(nodeID, "running")

	// Register with health monitor and presence
	hm.RegisterAgent(nodeID, "http://localhost:9999")
	presenceManager.Touch(nodeID, time.Now())

	// --- Start all 3 services concurrently (like production) ---
	go hm.Start()
	statusManager.Start()
	presenceManager.Start()

	// Let services stabilize — agent should be active
	time.Sleep(300 * time.Millisecond)

	// Verify agent is active before the test
	snapshot, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, snapshot.State,
		"Agent should be active before flapping test begins")

	// --- THE FLAPPING SCENARIO ---
	// Health checks start failing (simulating network blip), but agent keeps
	// sending heartbeats (proving it's alive). The agent should NEVER go inactive.

	// Track all status transitions to detect flapping
	var statusHistory []types.AgentState
	var historyMu sync.Mutex

	// Start heartbeat sender goroutine (simulates agent sending heartbeats every 100ms)
	heartbeatDone := make(chan struct{})
	go func() {
		defer close(heartbeatDone)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for i := 0; i < 30; i++ { // 30 heartbeats over ~3 seconds
			<-ticker.C
			readyStatus := types.AgentStatusReady
			_ = statusManager.UpdateFromHeartbeat(ctx, nodeID, &readyStatus, nil)
			presenceManager.Touch(nodeID, time.Now())

			// Record current state
			snap, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
			if err == nil {
				historyMu.Lock()
				statusHistory = append(statusHistory, snap.State)
				historyMu.Unlock()
			}
		}
	}()

	// Introduce intermittent health check failures after 200ms
	time.Sleep(200 * time.Millisecond)
	mockClient.setStatusError(nodeID, errors.New("connection timeout"))

	// Let it run with failing health checks + flowing heartbeats for 2 seconds
	time.Sleep(2 * time.Second)

	// Restore health checks
	mockClient.mu.Lock()
	delete(mockClient.statusErrors, nodeID)
	mockClient.mu.Unlock()
	mockClient.setStatusResponse(nodeID, "running")

	// Wait for heartbeat goroutine to finish
	<-heartbeatDone

	// Let services settle
	time.Sleep(300 * time.Millisecond)

	// --- ASSERTIONS ---

	// 1. Agent should be active now
	finalSnapshot, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	assert.Equal(t, types.AgentStateActive, finalSnapshot.State,
		"Agent must be active — heartbeats were flowing the entire time")

	// 2. Agent should NEVER have been inactive during the test
	historyMu.Lock()
	defer historyMu.Unlock()

	inactiveCount := 0
	for _, state := range statusHistory {
		if state == types.AgentStateInactive {
			inactiveCount++
		}
	}
	assert.Zero(t, inactiveCount,
		"Agent should NEVER have been inactive while heartbeats were flowing. "+
			"Got %d inactive readings out of %d samples. This is the flapping bug.",
		inactiveCount, len(statusHistory))

	// 3. Log the full history for debugging if test fails
	if t.Failed() {
		t.Logf("Status history (%d samples):", len(statusHistory))
		for i, state := range statusHistory {
			t.Logf("  [%d] %s", i, state)
		}
	}
}

// TestIntegration_ProperInactiveWhenHeartbeatsStop verifies that when an agent
// genuinely goes down (both heartbeats AND health checks stop), the system
// correctly marks it inactive after the consecutive failure threshold.
func TestIntegration_ProperInactiveWhenHeartbeatsStop(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	mockClient := newMockAgentClient()

	statusConfig := StatusManagerConfig{
		ReconcileInterval:       200 * time.Millisecond,
		StatusCacheTTL:          50 * time.Millisecond,
		HeartbeatStaleThreshold: 1 * time.Second,
		MaxTransitionTime:       500 * time.Millisecond,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, mockClient)

	presenceConfig := PresenceManagerConfig{
		HeartbeatTTL:  1 * time.Second,
		SweepInterval: 200 * time.Millisecond,
		HardEvictTTL:  3 * time.Second,
	}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)

	hmConfig := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		CheckTimeout:        2 * time.Second,
		ConsecutiveFailures: 3,
		RecoveryDebounce:    200 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, hmConfig, nil, mockClient, statusManager, presenceManager)
	presenceManager.SetExpireCallback(hm.UnregisterAgent)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		statusManager.Stop()
		_ = provider.Close(ctx)
	})

	// Register healthy agent
	nodeID := "integration-agent-down"
	node := &types.AgentNode{
		ID:              nodeID,
		TeamID:          "team",
		BaseURL:         "http://localhost:9998",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	mockClient.setStatusResponse(nodeID, "running")
	hm.RegisterAgent(nodeID, "http://localhost:9998")
	presenceManager.Touch(nodeID, time.Now())

	// Start all services
	go hm.Start()
	statusManager.Start()
	presenceManager.Start()

	// Let agent stabilize as active
	time.Sleep(300 * time.Millisecond)

	snapshot, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, snapshot.State)

	// --- Agent goes down: no more heartbeats, health checks fail ---
	mockClient.setStatusError(nodeID, errors.New("connection refused"))

	// Wait for consecutive failures to trigger inactive marking
	// 3 failures at 100ms interval = ~300ms + some processing time
	time.Sleep(800 * time.Millisecond)

	// Agent should now be inactive
	finalSnapshot, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	assert.Equal(t, types.AgentStateInactive, finalSnapshot.State,
		"Agent should be inactive when both heartbeats and health checks have stopped")
}

// TestIntegration_RecoveryAfterGenuineOutage verifies the full lifecycle:
// healthy -> outage (goes inactive) -> comes back (sends heartbeat) -> recovers to active.
func TestIntegration_RecoveryAfterGenuineOutage(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	mockClient := newMockAgentClient()

	statusConfig := StatusManagerConfig{
		ReconcileInterval:       200 * time.Millisecond,
		StatusCacheTTL:          50 * time.Millisecond,
		HeartbeatStaleThreshold: 1 * time.Second,
		MaxTransitionTime:       500 * time.Millisecond,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, mockClient)

	presenceConfig := PresenceManagerConfig{
		HeartbeatTTL:  1 * time.Second,
		SweepInterval: 200 * time.Millisecond,
		HardEvictTTL:  3 * time.Second,
	}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)

	hmConfig := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		CheckTimeout:        2 * time.Second,
		ConsecutiveFailures: 3,
		RecoveryDebounce:    200 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, hmConfig, nil, mockClient, statusManager, presenceManager)
	presenceManager.SetExpireCallback(hm.UnregisterAgent)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		statusManager.Stop()
		_ = provider.Close(ctx)
	})

	// Register healthy agent
	nodeID := "integration-agent-recovery"
	node := &types.AgentNode{
		ID:              nodeID,
		TeamID:          "team",
		BaseURL:         "http://localhost:9997",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	mockClient.setStatusResponse(nodeID, "running")
	hm.RegisterAgent(nodeID, "http://localhost:9997")
	presenceManager.Touch(nodeID, time.Now())

	// Start all services
	go hm.Start()
	statusManager.Start()
	presenceManager.Start()

	// Phase 1: Agent is healthy
	time.Sleep(300 * time.Millisecond)
	snapshot, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, snapshot.State, "Phase 1: should be active")

	// Phase 2: Agent goes down (genuine outage)
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	time.Sleep(800 * time.Millisecond)

	snapshot, err = statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	require.Equal(t, types.AgentStateInactive, snapshot.State, "Phase 2: should be inactive after outage")

	// Phase 3: Agent comes back — health checks pass and heartbeat sent
	mockClient.mu.Lock()
	delete(mockClient.statusErrors, nodeID)
	mockClient.mu.Unlock()
	mockClient.setStatusResponse(nodeID, "running")

	// Re-register with health monitor (agent would re-register on reconnect)
	hm.RegisterAgent(nodeID, "http://localhost:9997")
	presenceManager.Touch(nodeID, time.Now())

	// Send a heartbeat to signal recovery
	readyStatus := types.AgentStatusReady
	err = statusManager.UpdateFromHeartbeat(ctx, nodeID, &readyStatus, nil)
	require.NoError(t, err)

	// Wait for health check cycle + debounce
	time.Sleep(600 * time.Millisecond)

	snapshot, err = statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
	require.NoError(t, err)
	assert.Equal(t, types.AgentStateActive, snapshot.State,
		"Phase 3: agent should recover to active after coming back")
}

// TestHealthMonitor_Config_ConsecutiveFailuresOne verifies that setting
// ConsecutiveFailures=1 restores the old "single failure = instant inactive"
// behavior for operators who want aggressive failure detection.
func TestHealthMonitor_Config_ConsecutiveFailuresOne(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	statusManager := NewStatusManager(provider, StatusManagerConfig{ReconcileInterval: 30 * time.Second}, nil, nil)
	presenceConfig := PresenceManagerConfig{HeartbeatTTL: 5 * time.Second, SweepInterval: 1 * time.Second, HardEvictTTL: 10 * time.Second}
	presenceManager := NewPresenceManager(statusManager, presenceConfig)
	mockClient := newMockAgentClient()

	config := HealthMonitorConfig{
		CheckInterval:       100 * time.Millisecond,
		ConsecutiveFailures: 1, // Single failure = instant inactive
		RecoveryDebounce:    100 * time.Millisecond,
	}
	hm := NewHealthMonitor(provider, config, nil, mockClient, statusManager, presenceManager)

	t.Cleanup(func() {
		hm.Stop()
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	nodeID := "test-instant-fail"
	baseURL := "http://localhost:8001"

	agent := &types.AgentNode{ID: nodeID, BaseURL: baseURL}
	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)
	hm.RegisterAgent(nodeID, baseURL)

	// Make active first
	mockClient.setStatusResponse(nodeID, "running")
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	require.Equal(t, types.HealthStatusActive, hm.activeAgents[nodeID].LastStatus)
	hm.agentsMutex.RUnlock()

	// Single failure should immediately mark inactive
	mockClient.setStatusError(nodeID, errors.New("connection refused"))
	hm.checkAgentHealth(nodeID)
	time.Sleep(50 * time.Millisecond)

	hm.agentsMutex.RLock()
	assert.Equal(t, types.HealthStatusInactive, hm.activeAgents[nodeID].LastStatus,
		"With ConsecutiveFailures=1, a single failure should mark agent inactive immediately")
	assert.Equal(t, 1, hm.activeAgents[nodeID].ConsecutiveFailures)
	hm.agentsMutex.RUnlock()
}
