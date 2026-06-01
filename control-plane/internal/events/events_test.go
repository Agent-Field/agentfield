package events

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestExecutionEventBus_Subscribe tests subscription functionality
func TestExecutionEventBus_Subscribe(t *testing.T) {
	bus := NewExecutionEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	require.NotNil(t, ch)
	require.Equal(t, 1, bus.GetSubscriberCount())

	// Verify channel is buffered
	select {
	case ch <- ExecutionEvent{}:
		// Success - channel accepts events
	default:
		t.Fatal("channel should be buffered")
	}
}

// TestExecutionEventBus_Unsubscribe tests unsubscription
func TestExecutionEventBus_Unsubscribe(t *testing.T) {
	bus := NewExecutionEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	require.Equal(t, 1, bus.GetSubscriberCount())

	bus.Unsubscribe(subscriberID)
	require.Equal(t, 0, bus.GetSubscriberCount())

	// Verify channel is closed
	_, ok := <-ch
	require.False(t, ok, "channel should be closed after unsubscribe")
}

// TestExecutionEventBus_Publish tests event publishing
func TestExecutionEventBus_Publish(t *testing.T) {
	bus := NewExecutionEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	defer bus.Unsubscribe(subscriberID)

	event := ExecutionEvent{
		Type:        ExecutionCreated,
		ExecutionID: "exec-1",
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		Status:      "created",
		Timestamp:   time.Now(),
	}

	bus.Publish(event)

	// Verify event is received
	select {
	case received := <-ch:
		require.Equal(t, event.Type, received.Type)
		require.Equal(t, event.ExecutionID, received.ExecutionID)
	case <-time.After(1 * time.Second):
		t.Fatal("event not received within timeout")
	}
}

// TestExecutionEventBus_MultipleSubscribers tests multiple subscribers
func TestExecutionEventBus_MultipleSubscribers(t *testing.T) {
	bus := NewExecutionEventBus()
	sub1 := bus.Subscribe("sub-1")
	sub2 := bus.Subscribe("sub-2")
	sub3 := bus.Subscribe("sub-3")
	defer bus.Unsubscribe("sub-1")
	defer bus.Unsubscribe("sub-2")
	defer bus.Unsubscribe("sub-3")

	require.Equal(t, 3, bus.GetSubscriberCount())

	event := ExecutionEvent{
		Type:        ExecutionCreated,
		ExecutionID: "exec-multi",
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		Status:      "created",
		Timestamp:   time.Now(),
	}

	bus.Publish(event)

	// All subscribers should receive the event
	for i, ch := range []chan ExecutionEvent{sub1, sub2, sub3} {
		select {
		case received := <-ch:
			require.Equal(t, event.ExecutionID, received.ExecutionID, "subscriber %d", i)
		case <-time.After(1 * time.Second):
			t.Fatalf("subscriber %d did not receive event", i)
		}
	}
}

// TestExecutionEventBus_ConcurrentPublish tests concurrent publishing
func TestExecutionEventBus_ConcurrentPublish(t *testing.T) {
	bus := NewExecutionEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	defer bus.Unsubscribe(subscriberID)

	const numEvents = 100
	var wg sync.WaitGroup
	wg.Add(numEvents)

	// Publish events concurrently
	for i := 0; i < numEvents; i++ {
		go func(id int) {
			defer wg.Done()
			event := ExecutionEvent{
				Type:        ExecutionCreated,
				ExecutionID: "exec-" + string(rune(id)),
				WorkflowID:  "workflow-1",
				AgentNodeID: "agent-1",
				Status:      "created",
				Timestamp:   time.Now(),
			}
			bus.Publish(event)
		}(i)
	}

	wg.Wait()

	// Verify all events are received (may not be in order)
	received := 0
	timeout := time.After(2 * time.Second)
	for received < numEvents {
		select {
		case <-ch:
			received++
		case <-timeout:
			t.Fatalf("only received %d/%d events", received, numEvents)
		}
	}
}

// TestNodeEventBus_Subscribe tests node event bus subscription
func TestNodeEventBus_Subscribe(t *testing.T) {
	bus := NewNodeEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	require.NotNil(t, ch)
	require.Equal(t, 1, bus.GetSubscriberCount())

	bus.Unsubscribe(subscriberID)
	require.Equal(t, 0, bus.GetSubscriberCount())
}

// TestNodeEventBus_Publish tests node event publishing
func TestNodeEventBus_Publish(t *testing.T) {
	bus := NewNodeEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	defer bus.Unsubscribe(subscriberID)

	event := NodeEvent{
		Type:      NodeOnline,
		NodeID:    "node-1",
		Status:    "online",
		Timestamp: time.Now(),
	}

	bus.Publish(event)

	select {
	case received := <-ch:
		require.Equal(t, event.Type, received.Type)
		require.Equal(t, event.NodeID, received.NodeID)
	case <-time.After(1 * time.Second):
		t.Fatal("event not received within timeout")
	}
}

// TestReasonerEventBus_Subscribe tests reasoner event bus subscription
func TestReasonerEventBus_Subscribe(t *testing.T) {
	bus := NewReasonerEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	require.NotNil(t, ch)
	require.Equal(t, 1, bus.GetSubscriberCount())

	bus.Unsubscribe(subscriberID)
	require.Equal(t, 0, bus.GetSubscriberCount())
}

// TestReasonerEventBus_Publish tests reasoner event publishing
func TestReasonerEventBus_Publish(t *testing.T) {
	bus := NewReasonerEventBus()
	subscriberID := "test-subscriber"

	ch := bus.Subscribe(subscriberID)
	defer bus.Unsubscribe(subscriberID)

	event := ReasonerEvent{
		Type:       ReasonerOnline,
		ReasonerID: "reasoner-1",
		NodeID:     "node-1",
		Status:     "online",
		Timestamp:  time.Now(),
	}

	bus.Publish(event)

	select {
	case received := <-ch:
		require.Equal(t, event.Type, received.Type)
		require.Equal(t, event.ReasonerID, received.ReasonerID)
	case <-time.After(1 * time.Second):
		t.Fatal("event not received within timeout")
	}
}

// TestExecutionEvent_ToJSON tests JSON serialization
func TestExecutionEvent_ToJSON(t *testing.T) {
	event := ExecutionEvent{
		Type:        ExecutionCreated,
		ExecutionID: "exec-json",
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		Status:      "created",
		Timestamp:   time.Now(),
		Data:        map[string]interface{}{"key": "value"},
	}

	jsonStr, err := event.ToJSON()
	require.NoError(t, err)
	require.Contains(t, jsonStr, "exec-json")
	require.Contains(t, jsonStr, "execution_created")
}

// TestNodeEvent_ToJSON tests node event JSON serialization
func TestNodeEvent_ToJSON(t *testing.T) {
	event := NodeEvent{
		Type:      NodeOnline,
		NodeID:    "node-json",
		Status:    "online",
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"key": "value"},
	}

	jsonStr, err := event.ToJSON()
	require.NoError(t, err)
	require.Contains(t, jsonStr, "node-json")
	require.Contains(t, jsonStr, "node_online")
}

// TestReasonerEvent_ToJSON tests reasoner event JSON serialization
func TestReasonerEvent_ToJSON(t *testing.T) {
	event := ReasonerEvent{
		Type:       ReasonerOnline,
		ReasonerID: "reasoner-json",
		NodeID:     "node-1",
		Status:     "online",
		Timestamp:  time.Now(),
		Data:       map[string]interface{}{"key": "value"},
	}

	jsonStr, err := event.ToJSON()
	require.NoError(t, err)
	require.Contains(t, jsonStr, "reasoner-json")
	require.Contains(t, jsonStr, "reasoner_online")
}

// TestGlobalEventBuses tests global event bus instances
func TestGlobalEventBuses(t *testing.T) {
	// Test global execution event bus
	require.NotNil(t, GlobalExecutionEventBus)
	ch1 := GlobalExecutionEventBus.Subscribe("global-test-1")
	defer GlobalExecutionEventBus.Unsubscribe("global-test-1")

	// Test global node event bus
	require.NotNil(t, GlobalNodeEventBus)
	ch2 := GlobalNodeEventBus.Subscribe("global-test-2")
	defer GlobalNodeEventBus.Unsubscribe("global-test-2")

	// Test global reasoner event bus
	require.NotNil(t, GlobalReasonerEventBus)
	ch3 := GlobalReasonerEventBus.Subscribe("global-test-3")
	defer GlobalReasonerEventBus.Unsubscribe("global-test-3")

	// Verify all channels are functional
	require.NotNil(t, ch1)
	require.NotNil(t, ch2)
	require.NotNil(t, ch3)
}
