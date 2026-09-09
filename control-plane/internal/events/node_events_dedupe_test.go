package events

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func resetNodeEventTestState(t *testing.T) {
	t.Helper()

	lastEventCacheMutex.Lock()
	originalCache := lastEventCache
	lastEventCache = make(map[string]NodeEvent)
	lastEventCacheMutex.Unlock()

	GlobalNodeEventBus.mutex.Lock()
	GlobalNodeEventBus.subscribers = make(map[string]chan NodeEvent)
	GlobalNodeEventBus.mutex.Unlock()

	t.Cleanup(func() {
		lastEventCacheMutex.Lock()
		lastEventCache = originalCache
		lastEventCacheMutex.Unlock()
	})
}

func receiveNodeEvent(t *testing.T, ch <-chan NodeEvent) NodeEvent {
	t.Helper()

	select {
	case event := <-ch:
		return event
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for node event")
		return NodeEvent{}
	}
}

func TestNodeEventBusShouldFilterEvent(t *testing.T) {
	resetNodeEventTestState(t)

	bus := GlobalNodeEventBus

	t.Run("filters heartbeat without subscribers", func(t *testing.T) {
		event := NodeEvent{
			Type:      NodeHeartbeat,
			Timestamp: time.Now(),
		}

		require.True(t, bus.shouldFilterEvent(event))
	})

	t.Run("does not filter ordinary event", func(t *testing.T) {
		event := NodeEvent{
			Type:      NodesRefresh,
			Timestamp: time.Now(),
		}

		require.False(t, bus.shouldFilterEvent(event))
	})
	t.Run("heartbeat is not filtered with subscriber", func(t *testing.T) {
		ch := bus.Subscribe("heartbeat-filter-test")
		defer bus.Unsubscribe("heartbeat-filter-test")
		require.False(t, bus.shouldFilterEvent(NodeEvent{Type: NodeHeartbeat, Timestamp: time.Now()}))
		_ = ch
	})
}

func TestNodeEventBusDuplicateStatusEvent(t *testing.T) {
	t.Run("filters identical status event within one second", func(t *testing.T) {
		resetNodeEventTestState(t)

		bus := GlobalNodeEventBus
		first := NodeEvent{
			Type:      NodeStatusUpdated,
			NodeID:    "node-1",
			Status:    "online",
			Timestamp: time.Now(),
		}

		require.False(t, bus.isDuplicateStatusEvent(first))

		second := first
		second.Timestamp = time.Now()

		require.True(t, bus.isDuplicateStatusEvent(second))
	})

	t.Run("allows changed status within one second", func(t *testing.T) {
		resetNodeEventTestState(t)

		bus := GlobalNodeEventBus
		first := NodeEvent{
			Type:      NodeStatusUpdated,
			NodeID:    "node-1",
			Status:    "online",
			Timestamp: time.Now(),
		}

		require.False(t, bus.isDuplicateStatusEvent(first))

		second := first
		second.Status = "offline"
		second.Timestamp = time.Now()

		require.False(t, bus.isDuplicateStatusEvent(second))
	})

	t.Run("allows changed unified new status", func(t *testing.T) {
		resetNodeEventTestState(t)

		bus := GlobalNodeEventBus
		first := NodeEvent{
			Type:      NodeUnifiedStatusChanged,
			NodeID:    "node-1",
			OldStatus: "offline",
			NewStatus: "online",
			Timestamp: time.Now(),
		}

		require.False(t, bus.isDuplicateStatusEvent(first))

		second := first
		second.NewStatus = "degraded"
		second.Timestamp = time.Now()

		require.False(t, bus.isDuplicateStatusEvent(second))
	})

	t.Run("filters identical unified status", func(t *testing.T) {
		resetNodeEventTestState(t)

		bus := GlobalNodeEventBus
		first := NodeEvent{
			Type:      NodeUnifiedStatusChanged,
			NodeID:    "node-1",
			OldStatus: "offline",
			NewStatus: "online",
			Timestamp: time.Now(),
		}

		require.False(t, bus.isDuplicateStatusEvent(first))

		second := first
		second.Timestamp = time.Now()

		require.True(t, bus.isDuplicateStatusEvent(second))
	})
}

func TestNodeEventBusCleanupEventCache(t *testing.T) {
	resetNodeEventTestState(t)

	staleKey := "node_status_changed:stale"
	freshKey := "node_status_changed:fresh"

	lastEventCacheMutex.Lock()
	lastEventCache[staleKey] = NodeEvent{
		Type:      NodeStatusUpdated,
		NodeID:    "stale",
		Timestamp: time.Now().Add(-10 * time.Minute),
	}
	lastEventCache[freshKey] = NodeEvent{
		Type:      NodeStatusUpdated,
		NodeID:    "fresh",
		Timestamp: time.Now(),
	}
	lastEventCacheMutex.Unlock()

	GlobalNodeEventBus.cleanupEventCache()

	lastEventCacheMutex.RLock()
	_, staleExists := lastEventCache[staleKey]
	_, freshExists := lastEventCache[freshKey]
	lastEventCacheMutex.RUnlock()

	require.False(t, staleExists)
	require.True(t, freshExists)
}

func TestNodeEventBusPartialDeduplicationBranches(t *testing.T) {
	resetNodeEventTestState(t)
	bus := GlobalNodeEventBus
	require.False(t, bus.isDuplicateStatusEvent(NodeEvent{Type: NodesRefresh, NodeID: "node-partial-refresh", Timestamp: time.Now()}))
	require.False(t, bus.isDuplicateStatusEvent(NodeEvent{Type: NodeOnline, NodeID: "node-partial-online", Timestamp: time.Now()}))
	require.True(t, bus.isDuplicateStatusEvent(NodeEvent{Type: NodeOnline, NodeID: "node-partial-online", Timestamp: time.Now()}))

	health := NodeEvent{Type: NodeHealthChanged, NodeID: "node-partial-health", Status: "healthy", Timestamp: time.Now()}
	require.False(t, bus.isDuplicateStatusEvent(health))
	health.Timestamp = time.Now()
	health.Status = "degraded"
	require.False(t, bus.isDuplicateStatusEvent(health))

	base := NodeEvent{Type: NodeUnifiedStatusChanged, NodeID: "node-partial-unified", Status: "same", OldStatus: "a", NewStatus: "b", Timestamp: time.Now()}
	require.False(t, bus.isDuplicateStatusEvent(base))
	changedOld := base
	changedOld.Timestamp = time.Now()
	changedOld.OldStatus = "c"
	require.False(t, bus.isDuplicateStatusEvent(changedOld))
}

func TestPublishNodeStatusUpdatedEnhanced(t *testing.T) {
	resetNodeEventTestState(t)

	ch := GlobalNodeEventBus.Subscribe("enhanced-status-test")
	defer GlobalNodeEventBus.Unsubscribe("enhanced-status-test")

	oldStatus := map[string]interface{}{
		"state": "idle",
	}
	newStatus := map[string]interface{}{
		"state":  "running",
		"detail": "healthy",
	}

	PublishNodeStatusUpdatedEnhanced(
		"node-1",
		oldStatus,
		newStatus,
		"health-check",
		"state changed",
	)

	unified := receiveNodeEvent(t, ch)
	require.Equal(t, NodeUnifiedStatusChanged, unified.Type)
	require.Equal(t, "node-1", unified.NodeID)
	require.Equal(t, oldStatus, unified.OldStatus)
	require.Equal(t, newStatus, unified.NewStatus)
	require.Equal(t, "health-check", unified.Source)
	require.Equal(t, "state changed", unified.Reason)

	legacy := receiveNodeEvent(t, ch)
	require.Equal(t, NodeStatusUpdated, legacy.Type)
	require.Equal(t, "node-1", legacy.NodeID)
	require.Equal(t, "running", legacy.Status)
	require.Equal(t, newStatus, legacy.Data)
}

func TestPublishNodeStatusUpdatedEnhancedFallbacks(t *testing.T) {
	for _, tc := range []struct {
		name      string
		newStatus interface{}
		want      string
	}{
		{"nil", nil, "unknown"},
		{"non-map", "running", "unknown"},
		{"map-without-state", map[string]interface{}{"detail": "ok"}, "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetNodeEventTestState(t)
			ch := GlobalNodeEventBus.Subscribe("enhanced-fallback-" + tc.name)
			defer GlobalNodeEventBus.Unsubscribe("enhanced-fallback-" + tc.name)
			PublishNodeStatusUpdatedEnhanced("node-fallback-"+tc.name, nil, tc.newStatus, "source", "reason")
			_ = receiveNodeEvent(t, ch)
			event := receiveNodeEvent(t, ch)
			require.Equal(t, tc.want, event.Status)
			require.Equal(t, tc.newStatus, event.Data)
		})
	}
}

func receiveReasonerEvent(t *testing.T, ch <-chan ReasonerEvent) ReasonerEvent {
	t.Helper()
	select {
	case event := <-ch:
		return event
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for reasoner event")
		return ReasonerEvent{}
	}
}

func TestNodePublishHelperPayloads(t *testing.T) {
	t.Run("state transition populates fields and payload", func(t *testing.T) {
		resetNodeEventTestState(t)

		ch := GlobalNodeEventBus.Subscribe("state-transition-test")
		defer GlobalNodeEventBus.Unsubscribe("state-transition-test")

		PublishNodeStateTransition(
			"node-1",
			"idle",
			"running",
			"execution started",
		)

		event := receiveNodeEvent(t, ch)

		require.Equal(t, NodeStateTransition, event.Type)
		require.Equal(t, "node-1", event.NodeID)
		require.Equal(t, "running", event.Status)
		require.Equal(t, "state_transition", event.Source)
		require.Equal(t, "execution started", event.Reason)

		payload, ok := event.Data.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, "idle", payload["from_state"])
		require.Equal(t, "running", payload["to_state"])
		require.Equal(t, "execution started", payload["reason"])
	})

	t.Run("bulk status update populates counts", func(t *testing.T) {
		resetNodeEventTestState(t)

		ch := GlobalNodeEventBus.Subscribe("bulk-status-test")
		defer GlobalNodeEventBus.Unsubscribe("bulk-status-test")

		errors := []string{"node-3 failed"}
		PublishBulkStatusUpdate(3, 2, 1, errors)

		event := receiveNodeEvent(t, ch)

		require.Equal(t, BulkStatusUpdate, event.Type)

		payload, ok := event.Data.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, 3, payload["total_nodes"])
		require.Equal(t, 2, payload["successful"])
		require.Equal(t, 1, payload["failed"])
		require.Equal(t, errors, payload["errors"])
	})

	t.Run("system state snapshot preserves payload", func(t *testing.T) {
		resetNodeEventTestState(t)

		ch := GlobalNodeEventBus.Subscribe("snapshot-test")
		defer GlobalNodeEventBus.Unsubscribe("snapshot-test")

		payload := map[string]interface{}{
			"nodes":     4,
			"reasoners": 7,
		}

		PublishSystemStateSnapshot(payload)

		event := receiveNodeEvent(t, ch)

		require.Equal(t, SystemStateSnapshot, event.Type)
		require.Equal(t, payload, event.Data)
	})
}

func TestNodeEventBusCompareStatusEventData(t *testing.T) {
	bus := NewNodeEventBus()
	require.False(t, bus.compareStatusEventData(NodeEvent{Status: "a"}, NodeEvent{Status: "b"}))
	require.True(t, bus.compareStatusEventData(
		NodeEvent{Type: NodeStatusUpdated, Status: "a"},
		NodeEvent{Type: NodeStatusUpdated, Status: "a"},
	))
	last := NodeEvent{Type: NodeUnifiedStatusChanged, Status: "a", OldStatus: "x", NewStatus: "y"}
	require.True(t, bus.compareStatusEventData(last, NodeEvent{Type: NodeUnifiedStatusChanged, Status: "a", OldStatus: "x", NewStatus: "y"}))
	require.False(t, bus.compareStatusEventData(last, NodeEvent{Type: NodeUnifiedStatusChanged, Status: "a", OldStatus: "x", NewStatus: "z"}))
}

func TestIsDuplicateStatusEventConcurrent(t *testing.T) {
	resetNodeEventTestState(t)
	bus := GlobalNodeEventBus
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = bus.isDuplicateStatusEvent(NodeEvent{Type: NodeStatusUpdated, NodeID: "race", Status: "active", Timestamp: time.Now()})
		}()
	}
	wg.Wait()
}
