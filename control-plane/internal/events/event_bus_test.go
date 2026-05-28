package events

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestEventBus_Subscribe(t *testing.T) {
	bus := NewEventBus[string]()

	subscriberID := "subscriber-1"
	ch := bus.Subscribe(subscriberID)

	require.NotNil(t, ch)
	require.Equal(t, 1, bus.SubscriberCount())
}

func TestEventBus_Subscribe_Multiple(t *testing.T) {
	bus := NewEventBus[string]()

	ch1 := bus.Subscribe("subscriber-1")
	ch2 := bus.Subscribe("subscriber-2")
	ch3 := bus.Subscribe("subscriber-3")

	require.NotNil(t, ch1)
	require.NotNil(t, ch2)
	require.NotNil(t, ch3)
	require.Equal(t, 3, bus.SubscriberCount())
}

func TestEventBus_Unsubscribe(t *testing.T) {
	bus := NewEventBus[string]()

	subscriberID := "subscriber-1"
	ch := bus.Subscribe(subscriberID)
	require.Equal(t, 1, bus.SubscriberCount())

	bus.Unsubscribe(subscriberID)
	require.Equal(t, 0, bus.SubscriberCount())

	// Channel should be closed
	_, ok := <-ch
	require.False(t, ok)
}

func TestEventBus_Unsubscribe_NotSubscribed(t *testing.T) {
	bus := NewEventBus[string]()

	// Unsubscribe non-existent subscriber should not panic
	bus.Unsubscribe("nonexistent")
	require.Equal(t, 0, bus.SubscriberCount())
}

func TestEventBus_Publish_SingleSubscriber(t *testing.T) {
	bus := NewEventBus[string]()

	subscriberID := "subscriber-1"
	ch := bus.Subscribe(subscriberID)

	event := "test-event"
	bus.Publish(event)

	// Receive event
	received := <-ch
	require.Equal(t, event, received)
}

func TestEventBus_Publish_MultipleSubscribers(t *testing.T) {
	bus := NewEventBus[string]()

	ch1 := bus.Subscribe("subscriber-1")
	ch2 := bus.Subscribe("subscriber-2")
	ch3 := bus.Subscribe("subscriber-3")

	event := "test-event"
	bus.Publish(event)

	// All subscribers should receive the event
	received1 := <-ch1
	received2 := <-ch2
	received3 := <-ch3

	require.Equal(t, event, received1)
	require.Equal(t, event, received2)
	require.Equal(t, event, received3)
}

func TestEventBus_Publish_NoSubscribers(t *testing.T) {
	bus := NewEventBus[string]()

	// Publishing with no subscribers should not panic
	bus.Publish("test-event")
	require.Equal(t, 0, bus.SubscriberCount())
}

func TestEventBus_Publish_AfterUnsubscribe(t *testing.T) {
	bus := NewEventBus[string]()

	ch1 := bus.Subscribe("subscriber-1")
	ch2 := bus.Subscribe("subscriber-2")

	bus.Unsubscribe("subscriber-1")

	event := "test-event"
	bus.Publish(event)

	// Only subscriber-2 should receive
	received2 := <-ch2
	require.Equal(t, event, received2)

	// subscriber-1's channel should be closed
	_, ok := <-ch1
	require.False(t, ok)
}

func TestEventBus_SubscriberCount(t *testing.T) {
	bus := NewEventBus[string]()

	require.Equal(t, 0, bus.SubscriberCount())

	bus.Subscribe("subscriber-1")
	require.Equal(t, 1, bus.SubscriberCount())

	bus.Subscribe("subscriber-2")
	require.Equal(t, 2, bus.SubscriberCount())

	bus.Unsubscribe("subscriber-1")
	require.Equal(t, 1, bus.SubscriberCount())

	bus.Unsubscribe("subscriber-2")
	require.Equal(t, 0, bus.SubscriberCount())
}

func TestEventBus_ConcurrentSubscribeUnsubscribePublish(t *testing.T) {
	bus := NewEventBus[int]()

	var wg sync.WaitGroup
	numGoroutines := 10

	// Concurrent subscribe
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			bus.Subscribe("subscriber-" + string(rune('0'+id)))
		}(i)
	}

	wg.Wait()
	require.Equal(t, numGoroutines, bus.SubscriberCount())

	// Concurrent unsubscribe
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			bus.Unsubscribe("subscriber-" + string(rune('0'+id)))
		}(i)
	}

	wg.Wait()
	require.Equal(t, 0, bus.SubscriberCount())
}

func TestEventBus_ConcurrentPublish(t *testing.T) {
	bus := NewEventBus[int]()

	ch := bus.Subscribe("subscriber-1")

	var wg sync.WaitGroup
	numEvents := 100

	// Concurrent publish
	for i := 0; i < numEvents; i++ {
		wg.Add(1)
		go func(event int) {
			defer wg.Done()
			bus.Publish(event)
		}(i)
	}

	wg.Wait()

	// Receive all events
	received := make(map[int]bool)
	for i := 0; i < numEvents; i++ {
		select {
		case event := <-ch:
			received[event] = true
		case <-time.After(1 * time.Second):
			t.Fatal("Timeout waiting for event")
		}
	}

	require.Equal(t, numEvents, len(received))
}

func TestEventBus_SlowSubscriber(t *testing.T) {
	bus := NewEventBus[string]()

	// Create subscriber with small buffer
	ch := bus.Subscribe("slow-subscriber")

	// Publish many events rapidly
	for i := 0; i < 200; i++ {
		bus.Publish("event-" + string(rune('0'+(i%10))))
	}

	// Slow subscriber should not block publisher
	// Some events may be dropped if buffer is full
	received := 0
	timeout := time.After(100 * time.Millisecond)
	for {
		select {
		case <-ch:
			received++
		case <-timeout:
			goto done
		}
	}
done:

	// Should have received at least some events (up to buffer size)
	require.Greater(t, received, 0)
	require.LessOrEqual(t, received, 200)
}

func TestEventBus_CustomType(t *testing.T) {
	type CustomEvent struct {
		ID      string
		Message string
		Value   int
	}

	bus := NewEventBus[CustomEvent]()

	ch := bus.Subscribe("subscriber-1")

	event := CustomEvent{
		ID:      "event-1",
		Message: "test message",
		Value:   42,
	}

	bus.Publish(event)

	received := <-ch
	require.Equal(t, event.ID, received.ID)
	require.Equal(t, event.Message, received.Message)
	require.Equal(t, event.Value, received.Value)
}

func TestEventBus_EventOrdering(t *testing.T) {
	bus := NewEventBus[int]()

	ch := bus.Subscribe("subscriber-1")

	// Publish events in sequence
	for i := 0; i < 10; i++ {
		bus.Publish(i)
	}

	// Receive events and verify order
	for i := 0; i < 10; i++ {
		select {
		case event := <-ch:
			require.Equal(t, i, event)
		case <-time.After(1 * time.Second):
			t.Fatal("Timeout waiting for event")
		}
	}
}

func TestEventBus_Resubscribe(t *testing.T) {
	bus := NewEventBus[string]()

	subscriberID := "subscriber-1"

	// Subscribe
	ch1 := bus.Subscribe(subscriberID)
	require.Equal(t, 1, bus.SubscriberCount())

	// Unsubscribe
	bus.Unsubscribe(subscriberID)
	require.Equal(t, 0, bus.SubscriberCount())

	// Resubscribe
	ch2 := bus.Subscribe(subscriberID)
	require.Equal(t, 1, bus.SubscriberCount())
	require.NotEqual(t, ch1, ch2) // Should be a new channel

	// Publish event
	bus.Publish("test-event")
	received := <-ch2
	require.Equal(t, "test-event", received)
}
