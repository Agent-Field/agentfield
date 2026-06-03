package events

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInvariant_EventBus_OrderingPreserved verifies that events published in
// order 1, 2, 3 are received in the same order by a subscriber whose channel
// never fills up.
func TestInvariant_EventBus_OrderingPreserved(t *testing.T) {
	bus := NewEventBus[int]()
	ch := bus.Subscribe("ordered-sub")

	const n = 50
	for i := 0; i < n; i++ {
		bus.Publish(i)
	}

	for i := 0; i < n; i++ {
		select {
		case got := <-ch:
			assert.Equal(t, i, got, "event at position %d must equal %d, got %d", i, i, got)
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for event %d", i)
		}
	}
}

// TestInvariant_EventBus_SlowSubscriberDropsNotBlocks verifies that when a
// subscriber's buffer (100) is full, the 101st Publish call returns immediately
// (does not block) and the 101st event is dropped.
func TestInvariant_EventBus_SlowSubscriberDropsNotBlocks(t *testing.T) {
	bus := NewEventBus[int]()
	ch := bus.Subscribe("slow-sub")

	// Fill the buffer completely.
	for i := 0; i < 100; i++ {
		bus.Publish(i)
	}

	// The 101st publish must not block; use a goroutine with timeout to detect blocking.
	done := make(chan struct{})
	go func() {
		bus.Publish(9999) // This should be dropped silently.
		close(done)
	}()

	select {
	case <-done:
		// Good — the publish returned without blocking.
	case <-time.After(time.Second):
		t.Fatal("Publish blocked when subscriber buffer was full — violates drop invariant")
	}

	// Drain the channel and verify the dropped event is absent.
	received := make([]int, 0, 101)
	draining := true
	for draining {
		select {
		case v := <-ch:
			received = append(received, v)
		default:
			draining = false
		}
	}

	assert.Len(t, received, 100, "should have exactly 100 events (the 101st must be dropped)")
	for _, v := range received {
		assert.NotEqual(t, 9999, v, "dropped event 9999 must not appear in received slice")
	}
}

// TestInvariant_EventBus_SlowSubscriberDoesNotAffectFastSubscriber verifies
// that a slow subscriber whose channel is full does not cause a fast subscriber
// to lose events.
func TestInvariant_EventBus_SlowSubscriberDoesNotAffectFastSubscriber(t *testing.T) {
	bus := NewEventBus[int]()
	slow := bus.Subscribe("slow")
	fast := bus.Subscribe("fast")

	// Fill the slow subscriber's buffer.
	for i := 0; i < 100; i++ {
		bus.Publish(i)
		// Drain fast immediately so it never fills.
		select {
		case <-fast:
		default:
		}
	}

	// Drain fast one more time (might have up to 100 buffered).
	fastReceived := 0
	draining := true
	for draining {
		select {
		case <-fast:
			fastReceived++
		default:
			draining = false
		}
	}

	// Publish 10 more events with the slow buffer already full.
	const extra = 10
	for i := 100; i < 100+extra; i++ {
		bus.Publish(i)
	}

	// The fast subscriber must receive all extra events.
	for i := 0; i < extra; i++ {
		select {
		case <-fast:
			fastReceived++
		case <-time.After(time.Second):
			t.Fatalf("fast subscriber missed event %d — slow subscriber isolation violated", i)
		}
	}

	// Slow subscriber's count may be less than 110 due to drops — that's fine.
	_ = slow
	_ = fastReceived
}

// TestInvariant_EventBus_ConcurrentPublishNoPanic verifies that concurrent
// Publish from many goroutines neither panics nor causes data races.
func TestInvariant_EventBus_ConcurrentPublishNoPanic(t *testing.T) {
	bus := NewEventBus[int]()
	_ = bus.Subscribe("concurrent-sub")

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			bus.Publish(g)
		}()
	}

	wg.Wait() // Must complete without panic; -race flag will catch data races.
}

// TestInvariant_EventBus_UnsubscribeClosesChannel verifies that after
// Unsubscribe the channel is closed so reads return the zero value immediately.
func TestInvariant_EventBus_UnsubscribeClosesChannel(t *testing.T) {
	bus := NewEventBus[int]()
	ch := bus.Subscribe("unsub-test")

	bus.Unsubscribe("unsub-test")

	// A closed channel must be readable immediately and return zero value.
	select {
	case v, ok := <-ch:
		assert.False(t, ok, "channel should be closed after Unsubscribe")
		assert.Equal(t, 0, v, "zero value expected from closed channel")
	case <-time.After(time.Second):
		t.Fatal("reading from closed channel timed out — channel may not be closed")
	}
}

// TestInvariant_EventBus_SubscriberCountReflectsState verifies the subscriber
// count is accurate across Subscribe/Unsubscribe operations.
func TestInvariant_EventBus_SubscriberCountReflectsState(t *testing.T) {
	bus := NewEventBus[string]()

	require.Equal(t, 0, bus.SubscriberCount())

	bus.Subscribe("a")
	require.Equal(t, 1, bus.SubscriberCount())

	bus.Subscribe("b")
	require.Equal(t, 2, bus.SubscriberCount())

	bus.Unsubscribe("a")
	require.Equal(t, 1, bus.SubscriberCount())

	bus.Unsubscribe("b")
	require.Equal(t, 0, bus.SubscriberCount())
}

// TestInvariant_EventBus_LateSubscriberReceivesNoHistoricalEvents verifies
// that events published before a Subscribe call are NOT delivered to the new
// subscriber (no replay / retroactive delivery).
func TestInvariant_EventBus_LateSubscriberReceivesNoHistoricalEvents(t *testing.T) {
	bus := NewEventBus[int]()

	// Publish before any subscriber exists.
	bus.Publish(1)
	bus.Publish(2)
	bus.Publish(3)

	// Now subscribe.
	ch := bus.Subscribe("late")

	// The channel must be empty — no historical events.
	select {
	case v := <-ch:
		t.Fatalf("late subscriber received historical event %d — retroactive delivery must not happen", v)
	default:
		// Correct: channel is empty.
	}
}

// TestInvariant_EventBus_UnsubscribeNonExistentIsSafe verifies that calling
// Unsubscribe for an unknown subscriber ID does not panic.
func TestInvariant_EventBus_UnsubscribeNonExistentIsSafe(t *testing.T) {
	bus := NewEventBus[int]()

	assert.NotPanics(t, func() {
		bus.Unsubscribe("does-not-exist")
	})
}
