package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/boltdb/bolt"
	"github.com/stretchr/testify/require"
)

func TestLocalStorageStoreEventAndGetEventHistoryApplyFilters(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	events := []*types.MemoryChangeEvent{
		{
			Type:    "memory_changed",
			Scope:   "agent",
			ScopeID: "agent-alpha",
			Key:     "memory/profile/name",
			Action:  "set",
			Metadata: types.EventMetadata{
				AgentID: "agent-alpha",
			},
		},
		{
			Type:    "memory_changed",
			Scope:   "agent",
			ScopeID: "agent-alpha",
			Key:     "memory/profile/email",
			Action:  "set",
			Metadata: types.EventMetadata{
				AgentID: "agent-alpha",
			},
		},
		{
			Type:    "memory_changed",
			Scope:   "workflow",
			ScopeID: "wf-123",
			Key:     "workflow/context/summary",
			Action:  "delete",
			Metadata: types.EventMetadata{
				WorkflowID: "wf-123",
			},
		},
	}

	for _, event := range events {
		require.NoError(t, ls.StoreEvent(ctx, event))
		require.NotEmpty(t, event.ID)
	}

	now := time.Now().UTC()
	rewriteStoredEvent(t, ls, events[0].ID, func(event *types.MemoryChangeEvent) {
		event.Timestamp = now.Add(-2 * time.Hour)
	})
	rewriteStoredEvent(t, ls, events[1].ID, func(event *types.MemoryChangeEvent) {
		event.Timestamp = now.Add(-20 * time.Minute)
	})
	rewriteStoredEvent(t, ls, events[2].ID, func(event *types.MemoryChangeEvent) {
		event.Timestamp = now.Add(-10 * time.Minute)
	})
	putRawEventPayload(t, ls, "corrupted", []byte("not-json"))

	scope := "agent"
	scopeID := "agent-alpha"
	since := now.Add(-30 * time.Minute)

	history, err := ls.GetEventHistory(ctx, types.EventFilter{
		Scope:    &scope,
		ScopeID:  &scopeID,
		Patterns: []string{"memory/profile/*"},
		Since:    &since,
		Limit:    1,
	})
	require.NoError(t, err)
	require.Len(t, history, 1)
	require.Equal(t, events[1].ID, history[0].ID)
	require.Equal(t, "memory/profile/email", history[0].Key)

	allHistory, err := ls.GetEventHistory(ctx, types.EventFilter{})
	require.NoError(t, err)
	require.Len(t, allHistory, 3)
	ids := []string{allHistory[0].ID, allHistory[1].ID, allHistory[2].ID}
	require.ElementsMatch(t, []string{events[0].ID, events[1].ID, events[2].ID}, ids)
}

func TestLocalStorageEventOperationsHonorContextCancellation(t *testing.T) {
	ls, _ := setupLocalStorage(t)

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	err := ls.StoreEvent(cancelledCtx, &types.MemoryChangeEvent{
		Type:    "memory_changed",
		Scope:   "agent",
		ScopeID: "agent-alpha",
		Key:     "memory/profile/name",
		Action:  "set",
	})
	require.ErrorContains(t, err, "context cancelled during store event")

	_, err = ls.GetEventHistory(cancelledCtx, types.EventFilter{})
	require.ErrorContains(t, err, "context cancelled during get event history")
}

func TestLocalStorageCleanupExpiredEventsRemovesExpiredAndCorruptedEntries(t *testing.T) {
	ls, _ := setupLocalStorage(t)

	now := time.Now().UTC()
	putStoredEvent(t, ls, &types.MemoryChangeEvent{
		ID:        "expired",
		Type:      "memory_changed",
		Timestamp: now.Add(-72 * time.Hour),
		Scope:     "agent",
		ScopeID:   "agent-alpha",
		Key:       "memory/profile/name",
		Action:    "set",
	})
	putStoredEvent(t, ls, &types.MemoryChangeEvent{
		ID:        "fresh",
		Type:      "memory_changed",
		Timestamp: now.Add(-1 * time.Hour),
		Scope:     "agent",
		ScopeID:   "agent-alpha",
		Key:       "memory/profile/email",
		Action:    "set",
	})
	putRawEventPayload(t, ls, "corrupted", []byte("not-json"))

	ls.cleanupExpiredEvents()

	history, err := ls.GetEventHistory(context.Background(), types.EventFilter{})
	require.NoError(t, err)
	require.Len(t, history, 1)
	require.Equal(t, "fresh", history[0].ID)
	require.Equal(t, "memory/profile/email", history[0].Key)
}

func rewriteStoredEvent(t *testing.T, ls *LocalStorage, id string, mutate func(*types.MemoryChangeEvent)) {
	t.Helper()

	require.NoError(t, ls.kvStore.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(eventsBucket))
		require.NotNil(t, bucket)

		raw := bucket.Get([]byte(id))
		require.NotNil(t, raw)

		var event types.MemoryChangeEvent
		require.NoError(t, json.Unmarshal(raw, &event))
		mutate(&event)

		encoded, err := json.Marshal(event)
		require.NoError(t, err)

		return bucket.Put([]byte(id), encoded)
	}))
}

func putStoredEvent(t *testing.T, ls *LocalStorage, event *types.MemoryChangeEvent) {
	t.Helper()

	payload, err := json.Marshal(event)
	require.NoError(t, err)
	putRawEventPayload(t, ls, event.ID, payload)
}

func putRawEventPayload(t *testing.T, ls *LocalStorage, id string, payload []byte) {
	t.Helper()

	require.NoError(t, ls.kvStore.Update(func(tx *bolt.Tx) error {
		bucket, err := tx.CreateBucketIfNotExists([]byte(eventsBucket))
		if err != nil {
			return err
		}

		return bucket.Put([]byte(id), payload)
	}))
}
