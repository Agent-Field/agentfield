package storage

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func memoryChangeEvent(key string) types.MemoryChangeEvent {
	return types.MemoryChangeEvent{Scope: "session", ScopeID: "s1", Key: key, Action: "set"}
}

func memorySubscriberCount(ls *LocalStorage) int {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	n := 0
	for _, subs := range ls.subscribers {
		n += len(subs)
	}
	return n
}

func TestSubscribeToMemoryChanges_CancelUnsubscribes(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	subCtx, cancel := context.WithCancel(ctx)
	ch, err := ls.SubscribeToMemoryChanges(subCtx, "session", "s1")
	require.NoError(t, err)
	require.Equal(t, 1, memorySubscriberCount(ls))

	cancel()
	select {
	case _, ok := <-ch:
		assert.False(t, ok, "channel should be closed, not deliver an event")
	case <-time.After(2 * time.Second):
		t.Fatal("channel not closed after cancel")
	}
	assert.Equal(t, 0, memorySubscriberCount(ls))

	// Publishing after the subscriber left must not panic on a closed channel.
	require.NoError(t, ls.PublishMemoryChange(ctx, memoryChangeEvent("k")))
}

func TestPublishMemoryChange_DeliversInOrderBeforeReturning(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	ch, err := ls.SubscribeToMemoryChanges(ctx, "", "")
	require.NoError(t, err)

	for i := 0; i < 50; i++ {
		require.NoError(t, ls.PublishMemoryChange(ctx, memoryChangeEvent(fmt.Sprintf("k%d", i))))
	}
	for i := 0; i < 50; i++ {
		select {
		case ev := <-ch:
			assert.Equal(t, fmt.Sprintf("k%d", i), ev.Key)
		default:
			t.Fatalf("event %d not delivered by the time Publish returned", i)
		}
	}
}

func TestPublishMemoryChange_SlowSubscriberIsDisconnected(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	slow, err := ls.SubscribeToMemoryChanges(ctx, "session", "s1")
	require.NoError(t, err)
	fast, err := ls.SubscribeToMemoryChanges(ctx, "session", "")
	require.NoError(t, err)

	for i := 0; i <= memorySubscriberBuffer; i++ {
		require.NoError(t, ls.PublishMemoryChange(ctx, memoryChangeEvent("k")))
		if i < memorySubscriberBuffer {
			<-fast
		}
	}
	<-fast

	// The slow subscriber gets every buffered event, then a closed channel.
	for i := 0; i < memorySubscriberBuffer; i++ {
		_, ok := <-slow
		require.True(t, ok)
	}
	_, ok := <-slow
	assert.False(t, ok, "overflowing subscriber should be closed")
	assert.Equal(t, 1, memorySubscriberCount(ls), "only the slow subscriber is removed")
}

func TestPublishMemoryChange_CacheSubscriberIsNotClosedOnOverflow(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	_, err := ls.Subscribe(subscriberKey("session", "s1"))
	require.NoError(t, err)
	// The cache bridge goroutine drains into a 100-slot channel nobody reads,
	// so the subscriber eventually overflows.
	for i := 0; i < 3*memorySubscriberBuffer; i++ {
		require.NoError(t, ls.PublishMemoryChange(ctx, memoryChangeEvent("k")))
	}
	assert.Equal(t, 1, memorySubscriberCount(ls))
}

func TestGetEventHistory_SinceIsExclusive(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	var stored []*types.MemoryChangeEvent
	for i := 0; i < 3; i++ {
		ev := memoryChangeEvent(fmt.Sprintf("k%d", i))
		require.NoError(t, ls.StoreEvent(ctx, &ev))
		stored = append(stored, &ev)
		time.Sleep(time.Millisecond)
	}

	since := stored[0].Timestamp
	events, err := ls.GetEventHistory(ctx, types.EventFilter{Since: &since})
	require.NoError(t, err)
	require.Len(t, events, 2)
	assert.Equal(t, "k1", events[0].Key)
	assert.Equal(t, "k2", events[1].Key)
}

func TestGetEventHistory_LocalModeReturnsPublicationOrder(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	// More than ten events, so decimal keys no longer sort lexicographically.
	for i := 1; i <= 12; i++ {
		ev := memoryChangeEvent(fmt.Sprintf("k%d", i))
		require.NoError(t, ls.StoreEvent(ctx, &ev))
	}

	events, err := ls.GetEventHistory(ctx, types.EventFilter{})
	require.NoError(t, err)
	require.Len(t, events, 12)
	for i, ev := range events {
		assert.Equal(t, fmt.Sprintf("k%d", i+1), ev.Key)
	}

	events, err = ls.GetEventHistory(ctx, types.EventFilter{Limit: 1})
	require.NoError(t, err)
	require.Len(t, events, 1)
	assert.Equal(t, "k12", events[0].Key, "Limit keeps the most recent event")
}
