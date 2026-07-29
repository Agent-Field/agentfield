package services

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupPresenceManagerTest(t *testing.T) (*PresenceManager, storage.StorageProvider) {
	t.Helper()

	provider, ctx := setupTestStorage(t)

	// Create a minimal status manager for testing
	statusConfig := StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, nil)

	config := PresenceManagerConfig{
		HeartbeatTTL:  5 * time.Second,
		SweepInterval: 1 * time.Second,
		HardEvictTTL:  10 * time.Second,
	}

	presenceManager := NewPresenceManager(statusManager, config)

	t.Cleanup(func() {
		presenceManager.Stop()
		_ = provider.Close(ctx)
	})

	return presenceManager, provider
}

func TestPresenceManager_NewPresenceManager(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	statusConfig := StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, nil)

	config := PresenceManagerConfig{
		HeartbeatTTL:  10 * time.Second,
		SweepInterval: 2 * time.Second,
		HardEvictTTL:  30 * time.Second,
	}

	pm := NewPresenceManager(statusManager, config)
	require.NotNil(t, pm)
	require.Equal(t, 10*time.Second, pm.config.HeartbeatTTL)
	require.Equal(t, 2*time.Second, pm.config.SweepInterval)
	require.Equal(t, 30*time.Second, pm.config.HardEvictTTL)

	_ = ctx
}

func TestPresenceManager_NewPresenceManager_Defaults(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	statusConfig := StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
	}
	statusManager := NewStatusManager(provider, statusConfig, nil, nil)

	// Test with zero values (should use defaults)
	config := PresenceManagerConfig{}
	pm := NewPresenceManager(statusManager, config)
	require.NotNil(t, pm)
	require.Equal(t, 15*time.Second, pm.config.HeartbeatTTL)
	require.Greater(t, pm.config.SweepInterval, time.Duration(0))
	require.Equal(t, 5*time.Minute, pm.config.HardEvictTTL)

	_ = ctx
}

func TestPresenceManager_Touch(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	nodeID := "node-touch-1"
	now := time.Now()

	pm.Touch(nodeID, "", now)

	// Verify lease exists
	require.True(t, pm.HasLease(nodeID))
}

func TestPresenceManager_Touch_UpdateExisting(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	nodeID := "node-touch-update"
	now1 := time.Now()
	pm.Touch(nodeID, "", now1)

	time.Sleep(10 * time.Millisecond)
	now2 := time.Now()
	pm.Touch(nodeID, "", now2)

	// Verify lease still exists
	require.True(t, pm.HasLease(nodeID))
}

func TestPresenceManager_Forget(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	nodeID := "node-forget-1"
	pm.Touch(nodeID, "", time.Now())
	require.True(t, pm.HasLease(nodeID))

	pm.Forget(nodeID)
	require.False(t, pm.HasLease(nodeID))
}

func TestPresenceManager_HasLease(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	nodeID := "node-lease-1"
	require.False(t, pm.HasLease(nodeID))

	pm.Touch(nodeID, "", time.Now())
	require.True(t, pm.HasLease(nodeID))

	pm.Forget(nodeID)
	require.False(t, pm.HasLease(nodeID))
}

func TestPresenceManager_SetExpireCallback(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	// Register the agent in storage so UpdateAgentStatus can look up its status.
	// Without this, markInactive → UpdateAgentStatus → GetAgentStatusSnapshot → GetAgent
	// fails and returns early before invoking the callback.
	ctx := context.Background()
	nodeID := "node-callback-1"
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:            nodeID,
		BaseURL:       "http://localhost:9999",
		LastHeartbeat: time.Now(),
	}))

	var mu sync.Mutex
	var callbackInvoked bool
	var callbackNodeID string

	callback := func(id string) {
		mu.Lock()
		callbackInvoked = true
		callbackNodeID = id
		mu.Unlock()
	}

	pm.SetExpireCallback(callback)
	require.NotNil(t, pm.expireCallback)

	// Use shorter intervals for faster test execution
	pm.config.HeartbeatTTL = 500 * time.Millisecond
	pm.config.SweepInterval = 200 * time.Millisecond

	// Start the presence manager to trigger expiration
	pm.Start()

	// Touch a node in the past so it's already expired
	pm.Touch(nodeID, "", time.Now().Add(-10*time.Second))

	// Wait for sweep to detect the expired node (generous margin for CI)
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return callbackInvoked
	}, 5*time.Second, 100*time.Millisecond, "expire callback should have been invoked")

	pm.Stop()

	mu.Lock()
	require.Equal(t, nodeID, callbackNodeID)
	mu.Unlock()
}

func TestPresenceManager_ExpirationDetection(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	// Set shorter TTL for testing
	pm.config.HeartbeatTTL = 500 * time.Millisecond
	pm.config.SweepInterval = 200 * time.Millisecond
	// Set hard evict TTL short so the lease gets deleted after expiration.
	// The sweep first marks offline (MarkedOffline=true) keeping the lease,
	// then on the next sweep removes it if HardEvictTTL has elapsed.
	pm.config.HardEvictTTL = 1 * time.Second

	pm.Start()

	nodeID := "node-expire-1"
	pm.Touch(nodeID, "", time.Now())
	require.True(t, pm.HasLease(nodeID))

	// Wait for expiration: TTL expires → marked offline → hard evict removes lease
	require.Eventually(t, func() bool {
		return !pm.HasLease(nodeID)
	}, 5*time.Second, 100*time.Millisecond, "node should be removed after TTL + hard evict expiration")

	pm.Stop()
}

func TestPresenceManager_ConcurrentAccess(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	var wg sync.WaitGroup
	numGoroutines := 10
	numNodes := 5

	// Concurrent touches
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numNodes; j++ {
				nodeID := "node-concurrent-" + string(rune('0'+j))
				pm.Touch(nodeID, "", time.Now())
				_ = pm.HasLease(nodeID)
			}
		}(i)
	}

	wg.Wait()

	// Verify all nodes have leases
	for j := 0; j < numNodes; j++ {
		nodeID := "node-concurrent-" + string(rune('0'+j))
		require.True(t, pm.HasLease(nodeID))
	}
}

func TestPresenceManager_StartStop(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	pm.Start()

	// Verify it's running
	nodeID := "node-start-stop"
	pm.Touch(nodeID, "", time.Now())
	require.True(t, pm.HasLease(nodeID))

	pm.Stop()

	// Stop should be idempotent
	pm.Stop()
}

func TestPresenceManager_HardEviction(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	// Set shorter hard evict TTL
	pm.config.HardEvictTTL = 1 * time.Second
	pm.config.HeartbeatTTL = 500 * time.Millisecond
	pm.config.SweepInterval = 100 * time.Millisecond

	pm.Start()

	nodeID := "node-hard-evict"
	pm.Touch(nodeID, "", time.Now().Add(-2*time.Second)) // Touch in the past beyond hard evict TTL

	// Wait for hard eviction
	time.Sleep(1 * time.Second)

	// Node should be removed
	require.False(t, pm.HasLease(nodeID))

	pm.Stop()
}

func TestPresenceManager_MultipleNodes(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	nodeIDs := []string{"node-1", "node-2", "node-3"}

	for _, nodeID := range nodeIDs {
		pm.Touch(nodeID, "", time.Now())
		require.True(t, pm.HasLease(nodeID))
	}

	// Forget one node
	pm.Forget("node-2")
	require.False(t, pm.HasLease("node-2"))
	require.True(t, pm.HasLease("node-1"))
	require.True(t, pm.HasLease("node-3"))
}

func TestPresenceManager_RecoverFromDatabase_NoNodes(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	ctx := context.Background()

	// Should succeed with empty database
	err := pm.RecoverFromDatabase(ctx, provider)
	require.NoError(t, err)

	// Verify no leases created
	pm.mu.RLock()
	count := len(pm.leases)
	pm.mu.RUnlock()

	assert.Equal(t, 0, count)
}

func TestPresenceManager_RecoverFromDatabase_WithNodes(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	ctx := context.Background()

	// Create some agents in the database with different heartbeat times
	recentHeartbeat := time.Now().Add(-1 * time.Second)
	staleHeartbeat := time.Now().Add(-1 * time.Hour)

	agent1 := &types.AgentNode{
		ID:            "agent-recent",
		BaseURL:       "http://localhost:8001",
		LastHeartbeat: recentHeartbeat,
	}
	agent2 := &types.AgentNode{
		ID:            "agent-stale",
		BaseURL:       "http://localhost:8002",
		LastHeartbeat: staleHeartbeat,
	}

	err := provider.RegisterAgent(ctx, agent1)
	require.NoError(t, err)
	err = provider.RegisterAgent(ctx, agent2)
	require.NoError(t, err)

	// Recover from database
	err = pm.RecoverFromDatabase(ctx, provider)
	require.NoError(t, err)

	// Verify leases were created
	pm.mu.RLock()
	count := len(pm.leases)
	lease1, exists1 := pm.leases["agent-recent"]
	lease2, exists2 := pm.leases["agent-stale"]
	pm.mu.RUnlock()

	assert.Equal(t, 2, count, "Should have created 2 leases")
	assert.True(t, exists1, "agent-recent lease should exist")
	assert.True(t, exists2, "agent-stale lease should exist")

	// Check that recent heartbeat is not marked offline
	assert.False(t, lease1.MarkedOffline, "agent-recent should not be marked offline")

	// Check that stale heartbeat IS marked offline
	assert.True(t, lease2.MarkedOffline, "agent-stale should be marked offline")
}

func TestPresenceManager_RecoverFromDatabase_PreservesHeartbeatTimestamps(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	ctx := context.Background()

	// Create an agent with a specific heartbeat time
	heartbeatTime := time.Now().Add(-30 * time.Second)
	agent := &types.AgentNode{
		ID:            "agent-with-timestamp",
		BaseURL:       "http://localhost:8001",
		LastHeartbeat: heartbeatTime,
	}

	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Recover from database
	err = pm.RecoverFromDatabase(ctx, provider)
	require.NoError(t, err)

	// Verify the lease has the correct LastSeen time
	pm.mu.RLock()
	lease, exists := pm.leases["agent-with-timestamp"]
	pm.mu.RUnlock()

	assert.True(t, exists, "Lease should exist")
	assert.Equal(t, heartbeatTime.Unix(), lease.LastSeen.Unix(), "LastSeen should match LastHeartbeat from database")
}

func TestPresenceManager_RecoverFromDatabase_SkipsNilNodes(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	ctx := context.Background()

	// Create a valid agent
	agent := &types.AgentNode{
		ID:            "valid-agent",
		BaseURL:       "http://localhost:8001",
		LastHeartbeat: time.Now(),
	}

	err := provider.RegisterAgent(ctx, agent)
	require.NoError(t, err)

	// Recover from database - should not panic on nil nodes
	err = pm.RecoverFromDatabase(ctx, provider)
	require.NoError(t, err)

	// Verify the valid agent has a lease
	assert.True(t, pm.HasLease("valid-agent"))
}

// TestPresenceManager_Start_Idempotent verifies that calling Start() multiple
// times does not spawn additional sweep goroutines. Regression test for the
// duplicate sweep goroutine guard.
func TestPresenceManager_Start_Idempotent(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)

	// Use long intervals so no sweeps fire during the test. This isolates the
	// goroutine count to the sweep loop itself (no callback goroutines).
	pm.config.HeartbeatTTL = 1 * time.Hour
	pm.config.SweepInterval = 1 * time.Hour
	pm.config.HardEvictTTL = 1 * time.Hour

	// Allow the runtime goroutine count to settle before measuring.
	time.Sleep(50 * time.Millisecond)
	before := runtime.NumGoroutine()

	pm.Start()

	// The first Start() must spawn exactly one sweep goroutine.
	require.Eventually(t, func() bool {
		return runtime.NumGoroutine() > before
	}, 500*time.Millisecond, 5*time.Millisecond,
		"first Start() should spawn a sweep goroutine")

	afterFirst := runtime.NumGoroutine()

	// Calling Start() multiple times must NOT spawn additional goroutines.
	pm.Start()
	pm.Start()
	pm.Start()

	// Give any (incorrectly) spawned goroutines time to start.
	time.Sleep(100 * time.Millisecond)

	afterMultiple := runtime.NumGoroutine()

	// Tolerate ±1 for runtime fluctuations (GC, finalizer goroutines, etc.).
	// With the bug present, afterMultiple would exceed afterFirst by ~3.
	delta := afterMultiple - afterFirst
	assert.LessOrEqual(t, delta, 1,
		"multiple Start() calls should not spawn additional sweep goroutines (delta=%d)", delta)
}

// TestPresenceManager_ExpireCallback_OncePerExpiration verifies that the
// expireCallback fires at most once per expired node lease per sweep cycle.
// Even across multiple sweep cycles, a single expired lease (without a fresh
// Touch in between) must only trigger the callback once.
func TestPresenceManager_ExpireCallback_OncePerExpiration(t *testing.T) {
	pm, provider := setupPresenceManagerTest(t)

	// Register the agent in storage so UpdateAgentStatus can look it up.
	// Without this, markInactive returns early before invoking the callback.
	ctx := context.Background()
	nodeID := "node-once-per-expire"
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:            nodeID,
		BaseURL:       "http://localhost:9999",
		LastHeartbeat: time.Now(),
	}))

	var callbackCount int32
	pm.SetExpireCallback(func(id string) {
		atomic.AddInt32(&callbackCount, 1)
	})

	// Short intervals for fast, deterministic sweeps across multiple cycles.
	pm.config.HeartbeatTTL = 500 * time.Millisecond
	pm.config.SweepInterval = 100 * time.Millisecond
	// Long hard-evict TTL so the lease stays around (we want to verify no
	// duplicate callbacks fire while the lease lingers in MarkedOffline state).
	pm.config.HardEvictTTL = 1 * time.Hour

	pm.Start()

	// Touch with an already-expired timestamp so the first sweep marks it
	// offline and invokes the callback.
	pm.Touch(nodeID, "", time.Now().Add(-10*time.Second))

	// Wait for the callback to fire at least once.
	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&callbackCount) >= 1
	}, 3*time.Second, 50*time.Millisecond,
		"expire callback should fire at least once")

	// Wait through multiple additional sweep cycles to ensure no duplicate
	// callbacks fire for the same expired lease.
	time.Sleep(500 * time.Millisecond)

	assert.Equal(t, int32(1), atomic.LoadInt32(&callbackCount),
		"expire callback should fire exactly once per expired node lease")
}
