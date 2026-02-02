package services

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeAgentClient struct {
	statusResponse *interfaces.AgentStatusResponse
	err            error
	calls          int
}

func (f *fakeAgentClient) setError(err error) {
	f.err = err
}

func (f *fakeAgentClient) GetAgentStatus(ctx context.Context, nodeID string) (*interfaces.AgentStatusResponse, error) {
	f.calls++
	if f.err != nil {
		err := f.err
		f.err = nil
		return nil, err
	}
	return f.statusResponse, nil
}

func (f *fakeAgentClient) GetMCPHealth(ctx context.Context, nodeID string) (*interfaces.MCPHealthResponse, error) {
	return nil, nil
}

func (f *fakeAgentClient) RestartMCPServer(ctx context.Context, nodeID, alias string) error {
	return nil
}

func (f *fakeAgentClient) GetMCPTools(ctx context.Context, nodeID, alias string) (*interfaces.MCPToolsResponse, error) {
	return nil, nil
}

func (f *fakeAgentClient) ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
	return nil, nil
}

func setupStatusManagerStorage(t *testing.T) (storage.StorageProvider, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := provider.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping status manager test")
		}
		require.NoError(t, err)
	}
	t.Cleanup(func() { _ = provider.Close(ctx) })

	return provider, ctx
}

func registerTestAgent(t *testing.T, provider storage.StorageProvider, ctx context.Context, nodeID string) {
	t.Helper()

	node := &types.AgentNode{
		ID:              nodeID,
		TeamID:          "team",
		BaseURL:         "http://localhost",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		LastHeartbeat:   time.Now().Add(-1 * time.Minute),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}

	require.NoError(t, provider.RegisterAgent(ctx, node))
}

func ptrAgentState(state types.AgentState) *types.AgentState {
	return &state
}

func TestStatusManagerCachingAndFallback(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-1")

	fakeClient := &fakeAgentClient{statusResponse: &interfaces.AgentStatusResponse{Status: "running"}}
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 100 * time.Millisecond,
		StatusCacheTTL:    30 * time.Second,
		MaxTransitionTime: time.Second,
	}, nil, fakeClient)

	status, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, status.State)
	require.Equal(t, 1, fakeClient.calls)

	// Subsequent call within cache window should not re-hit client even if error is configured.
	fakeClient.setError(errors.New("boom"))
	statusCached, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, statusCached.State)
	require.Equal(t, 1, fakeClient.calls)

	// After cache expiry, a new health check should occur and fall back to inactive state on failure.
	time.Sleep(1100 * time.Millisecond)
	fakeClient.setError(errors.New("still failing"))
	statusAfterError, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateInactive, statusAfterError.State)
	require.Equal(t, 2, fakeClient.calls)

	storedAgent, err := provider.GetAgent(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, storedAgent.HealthStatus)
}

func TestStatusManagerAllowsInactiveToActiveTransition(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-transition")

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

	update := &types.AgentStatusUpdate{
		State:  ptrAgentState(types.AgentStateActive),
		Source: types.StatusSourceHeartbeat,
		Reason: "heartbeat indicates agent active",
	}

	require.NoError(t, sm.UpdateAgentStatus(ctx, "node-transition", update))

	status, err := sm.GetAgentStatus(ctx, "node-transition")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, status.State)
}

func TestStatusManagerSnapshotUsesStorage(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-snapshot")

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

	snapshot, err := sm.GetAgentStatusSnapshot(ctx, "node-snapshot", nil)
	require.NoError(t, err)
	require.Equal(t, types.StatusSourceReconcile, snapshot.Source)
	require.Equal(t, types.AgentStatusOffline, snapshot.LifecycleStatus)

	// Ensure snapshot is cached and returned without additional storage lookups when provided with cached node data.
	smNoCache := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	node := &types.AgentNode{ID: "node-snapshot", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: time.Now()}
	snapshot2, err := smNoCache.GetAgentStatusSnapshot(ctx, "node-snapshot", node)
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusReady, snapshot2.LifecycleStatus)
}

// TestStatusManagerBroadcastsNodeOfflineEvent verifies that when a node transitions
// from active to inactive, the proper events are broadcast. This tests the fix for
// the race condition where UpdateAgentStatus was calling GetAgentStatus (with live
// health check) instead of GetAgentStatusSnapshot, causing oldStatus == newStatus
// and preventing events from being broadcast.
func TestStatusManagerBroadcastsNodeOfflineEvent(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)

	// Register an agent that starts as ACTIVE
	node := &types.AgentNode{
		ID:              "node-offline-test",
		TeamID:          "team",
		BaseURL:         "http://localhost",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Subscribe to node events to capture broadcasts
	var mu sync.Mutex
	var receivedEvents []events.NodeEvent

	subscriberID := "test-offline-subscriber"
	eventCh := events.GlobalNodeEventBus.Subscribe(subscriberID)
	defer events.GlobalNodeEventBus.Unsubscribe(subscriberID)

	// Collect events in background
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case event, ok := <-eventCh:
				if !ok {
					return
				}
				mu.Lock()
				receivedEvents = append(receivedEvents, event)
				mu.Unlock()
			case <-time.After(2 * time.Second):
				return
			}
		}
	}()

	// Create status manager WITHOUT agent client (no live health checks)
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 10 * time.Second, // Long interval to avoid interference
		StatusCacheTTL:    30 * time.Second,
		MaxTransitionTime: time.Second,
	}, nil, nil)

	// Prime the cache with active status
	sm.cacheMutex.Lock()
	sm.statusCache["node-offline-test"] = &cachedAgentStatus{
		Status: &types.AgentStatus{
			State:           types.AgentStateActive,
			HealthScore:     85,
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastSeen:        time.Now(),
			LastUpdated:     time.Now(),
			Source:          types.StatusSourceHeartbeat,
		},
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	// Now simulate node going offline - this is what the health monitor does
	inactiveState := types.AgentStateInactive
	healthScore := 0
	update := &types.AgentStatusUpdate{
		State:       &inactiveState,
		HealthScore: &healthScore,
		Source:      types.StatusSourceHealthCheck,
		Reason:      "HTTP health check failed",
	}

	err := sm.UpdateAgentStatus(ctx, "node-offline-test", update)
	require.NoError(t, err)

	// Give events time to propagate
	time.Sleep(200 * time.Millisecond)

	// Stop event collection
	events.GlobalNodeEventBus.Unsubscribe(subscriberID)
	<-done

	// Verify we received the expected events
	mu.Lock()
	defer mu.Unlock()

	// Should have received at least NodeOffline or NodeUnifiedStatusChanged
	var foundOfflineEvent bool
	var foundUnifiedStatusEvent bool
	for _, event := range receivedEvents {
		if event.Type == events.NodeOffline && event.NodeID == "node-offline-test" {
			foundOfflineEvent = true
		}
		if event.Type == events.NodeUnifiedStatusChanged && event.NodeID == "node-offline-test" {
			foundUnifiedStatusEvent = true
		}
	}

	require.True(t, foundOfflineEvent || foundUnifiedStatusEvent,
		"Expected NodeOffline or NodeUnifiedStatusChanged event, got events: %+v", receivedEvents)
}

// TestStatusManagerBroadcastsNodeOnlineEvent verifies that when a node transitions
// from inactive to active, the proper events are broadcast.
func TestStatusManagerBroadcastsNodeOnlineEvent(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)

	// Register an agent that starts as INACTIVE
	registerTestAgent(t, provider, ctx, "node-online-test")

	// Subscribe to node events to capture broadcasts
	var mu sync.Mutex
	var receivedEvents []events.NodeEvent

	subscriberID := "test-online-subscriber"
	eventCh := events.GlobalNodeEventBus.Subscribe(subscriberID)
	defer events.GlobalNodeEventBus.Unsubscribe(subscriberID)

	// Collect events in background
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case event, ok := <-eventCh:
				if !ok {
					return
				}
				mu.Lock()
				receivedEvents = append(receivedEvents, event)
				mu.Unlock()
			case <-time.After(2 * time.Second):
				return
			}
		}
	}()

	// Create status manager WITHOUT agent client
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 10 * time.Second,
		StatusCacheTTL:    30 * time.Second,
		MaxTransitionTime: time.Second,
	}, nil, nil)

	// Prime the cache with inactive status (as it would be from storage)
	sm.cacheMutex.Lock()
	sm.statusCache["node-online-test"] = &cachedAgentStatus{
		Status: &types.AgentStatus{
			State:           types.AgentStateInactive,
			HealthScore:     0,
			HealthStatus:    types.HealthStatusInactive,
			LifecycleStatus: types.AgentStatusOffline,
			LastSeen:        time.Now().Add(-1 * time.Minute),
			LastUpdated:     time.Now().Add(-1 * time.Minute),
			Source:          types.StatusSourceReconcile,
		},
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	// Simulate node coming online - this is what heartbeat processing does
	activeState := types.AgentStateActive
	healthScore := 85
	lifecycleStatus := types.AgentStatusReady
	update := &types.AgentStatusUpdate{
		State:           &activeState,
		HealthScore:     &healthScore,
		LifecycleStatus: &lifecycleStatus,
		Source:          types.StatusSourceHeartbeat,
		Reason:          "agent heartbeat received",
	}

	err := sm.UpdateAgentStatus(ctx, "node-online-test", update)
	require.NoError(t, err)

	// Give events time to propagate
	time.Sleep(200 * time.Millisecond)

	// Stop event collection
	events.GlobalNodeEventBus.Unsubscribe(subscriberID)
	<-done

	// Verify we received the expected events
	mu.Lock()
	defer mu.Unlock()

	var foundOnlineEvent bool
	var foundUnifiedStatusEvent bool
	for _, event := range receivedEvents {
		if event.Type == events.NodeOnline && event.NodeID == "node-online-test" {
			foundOnlineEvent = true
		}
		if event.Type == events.NodeUnifiedStatusChanged && event.NodeID == "node-online-test" {
			foundUnifiedStatusEvent = true
		}
	}

	require.True(t, foundOnlineEvent || foundUnifiedStatusEvent,
		"Expected NodeOnline or NodeUnifiedStatusChanged event, got events: %+v", receivedEvents)
}

// TestStatusManagerPreservesOldStatusForEventBroadcast verifies that UpdateAgentStatus
// correctly captures the old status before applying updates, ensuring that status change
// events are broadcast with accurate old/new state information.
func TestStatusManagerPreservesOldStatusForEventBroadcast(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)

	// Register an agent that starts as ACTIVE
	node := &types.AgentNode{
		ID:              "node-preserve-test",
		TeamID:          "team",
		BaseURL:         "http://localhost",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Track event handler calls to verify old/new status
	var mu sync.Mutex
	var statusChanges []struct {
		OldState types.AgentState
		NewState types.AgentState
	}

	handler := &testStatusEventHandler{
		onStatusChanged: func(nodeID string, oldStatus, newStatus *types.AgentStatus) {
			if nodeID == "node-preserve-test" {
				mu.Lock()
				statusChanges = append(statusChanges, struct {
					OldState types.AgentState
					NewState types.AgentState
				}{
					OldState: oldStatus.State,
					NewState: newStatus.State,
				})
				mu.Unlock()
			}
		},
	}

	// Create status manager with event handler
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 10 * time.Second,
		StatusCacheTTL:    30 * time.Second,
		MaxTransitionTime: time.Second,
	}, nil, nil)
	sm.AddEventHandler(handler)

	// Prime the cache with active status
	sm.cacheMutex.Lock()
	sm.statusCache["node-preserve-test"] = &cachedAgentStatus{
		Status: &types.AgentStatus{
			State:           types.AgentStateActive,
			HealthScore:     85,
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastSeen:        time.Now(),
			LastUpdated:     time.Now(),
			Source:          types.StatusSourceHeartbeat,
		},
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	// Update to inactive
	inactiveState := types.AgentStateInactive
	healthScore := 0
	update := &types.AgentStatusUpdate{
		State:       &inactiveState,
		HealthScore: &healthScore,
		Source:      types.StatusSourceHealthCheck,
		Reason:      "HTTP health check failed",
	}

	err := sm.UpdateAgentStatus(ctx, "node-preserve-test", update)
	require.NoError(t, err)

	// Give event handler time to be called
	time.Sleep(100 * time.Millisecond)

	// Verify the status change captured correct old and new states
	mu.Lock()
	defer mu.Unlock()

	require.Len(t, statusChanges, 1, "Expected exactly one status change event")
	require.Equal(t, types.AgentStateActive, statusChanges[0].OldState, "Old state should be Active")
	require.Equal(t, types.AgentStateInactive, statusChanges[0].NewState, "New state should be Inactive")
}

// testStatusEventHandler is a test implementation of StatusEventHandler
type testStatusEventHandler struct {
	onStatusChanged func(nodeID string, oldStatus, newStatus *types.AgentStatus)
}

func (h *testStatusEventHandler) OnStatusChanged(nodeID string, oldStatus, newStatus *types.AgentStatus) {
	if h.onStatusChanged != nil {
		h.onStatusChanged(nodeID, oldStatus, newStatus)
	}
}

// --- Heartbeat priority and reconciliation threshold tests ---

func TestStatusManager_UpdateFromHeartbeat_NeverDropped(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-heartbeat-priority")

	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
		StatusCacheTTL:    5 * time.Minute,
	}, nil, nil)

	// First, mark the agent as inactive via a health check source
	// (simulating what HealthMonitor would do)
	inactiveState := types.AgentStateInactive
	healthScore := 0
	update := &types.AgentStatusUpdate{
		State:       &inactiveState,
		HealthScore: &healthScore,
		Source:      types.StatusSourceHealthCheck,
		Reason:      "HTTP health check failed",
	}
	err := sm.UpdateAgentStatus(ctx, "node-heartbeat-priority", update)
	require.NoError(t, err)

	// Verify agent is inactive
	status, err := sm.GetAgentStatusSnapshot(ctx, "node-heartbeat-priority", nil)
	require.NoError(t, err)
	require.Equal(t, types.AgentStateInactive, status.State)
	require.Equal(t, types.StatusSourceHealthCheck, status.Source)

	// Now send a heartbeat IMMEDIATELY (within what used to be the 10s drop window).
	// Previously this heartbeat would be silently ignored. Now it MUST be processed.
	readyStatus := types.AgentStatusReady
	err = sm.UpdateFromHeartbeat(ctx, "node-heartbeat-priority", &readyStatus, nil)
	require.NoError(t, err, "Heartbeat should never be dropped")

	// Verify the heartbeat was processed — agent should no longer be inactive
	status, err = sm.GetAgentStatusSnapshot(ctx, "node-heartbeat-priority", nil)
	require.NoError(t, err)
	require.Equal(t, types.StatusSourceHeartbeat, status.Source,
		"Source should be heartbeat, proving it was processed")
	require.NotEqual(t, types.AgentStateInactive, status.State,
		"Agent should not be inactive after receiving a heartbeat")
}

func TestStatusManager_Reconciliation_UsesConfiguredThreshold(t *testing.T) {
	provider, _ := setupStatusManagerStorage(t)

	// Create StatusManager with default 60s threshold
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval:       30 * time.Second,
		HeartbeatStaleThreshold: 60 * time.Second,
	}, nil, nil)

	// Agent with heartbeat 45 seconds ago — should NOT need reconciliation
	recentAgent := &types.AgentNode{
		ID:            "node-recent",
		HealthStatus:  types.HealthStatusActive,
		LastHeartbeat: time.Now().Add(-45 * time.Second),
	}
	assert.False(t, sm.needsReconciliation(recentAgent),
		"Agent with 45s-old heartbeat should NOT need reconciliation (threshold is 60s)")

	// Agent with heartbeat 65 seconds ago — SHOULD need reconciliation
	staleAgent := &types.AgentNode{
		ID:            "node-stale",
		HealthStatus:  types.HealthStatusActive,
		LastHeartbeat: time.Now().Add(-65 * time.Second),
	}
	assert.True(t, sm.needsReconciliation(staleAgent),
		"Agent with 65s-old heartbeat should need reconciliation (threshold is 60s)")

	// Agent already inactive — should NOT need reconciliation even if stale
	inactiveAgent := &types.AgentNode{
		ID:            "node-inactive",
		HealthStatus:  types.HealthStatusInactive,
		LastHeartbeat: time.Now().Add(-120 * time.Second),
	}
	assert.False(t, sm.needsReconciliation(inactiveAgent),
		"Already inactive agent should not need reconciliation")
}
