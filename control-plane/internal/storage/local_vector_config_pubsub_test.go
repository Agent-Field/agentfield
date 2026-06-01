package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalStorageVectorLifecycleAndSearch(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	recordA := &types.VectorRecord{
		Scope:     "session",
		ScopeID:   "scope-1",
		Key:       "doc-a",
		Embedding: []float32{1, 0, 0},
		Metadata: map[string]interface{}{
			"kind": "doc",
		},
	}
	recordB := &types.VectorRecord{
		Scope:     "session",
		ScopeID:   "scope-1",
		Key:       "doc-b",
		Embedding: []float32{0, 1, 0},
		Metadata: map[string]interface{}{
			"kind": "doc",
		},
	}

	require.NoError(t, ls.SetVector(ctx, recordA))
	require.NoError(t, ls.SetVector(ctx, recordB))

	loaded, err := ls.GetVector(ctx, "session", "scope-1", "doc-a")
	require.NoError(t, err)
	assert.Equal(t, recordA.Key, loaded.Key)
	assert.Equal(t, recordA.Scope, loaded.Scope)
	assert.Equal(t, recordA.ScopeID, loaded.ScopeID)
	assert.InDeltaSlice(t, []float32{1, 0, 0}, loaded.Embedding, 0.0001)
	assert.Equal(t, "doc", loaded.Metadata["kind"])

	results, err := ls.SimilaritySearch(ctx, "session", "scope-1", []float32{1, 0, 0}, 2, map[string]interface{}{"kind": "doc"})
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "doc-a", results[0].Key)
	assert.GreaterOrEqual(t, results[0].Score, results[1].Score)

	deleted, err := ls.DeleteVectorsByPrefix(ctx, "session", "scope-1", "doc-")
	require.NoError(t, err)
	assert.Equal(t, 2, deleted)

	missing, err := ls.GetVector(ctx, "session", "scope-1", "doc-a")
	require.NoError(t, err)
	assert.Nil(t, missing)

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	err = ls.SetVector(canceled, recordA)
	require.Error(t, err)

	enabled := false
	ls.vectorConfig.Enabled = &enabled
	ls.vectorStore = nil
	err = ls.DeleteVector(context.Background(), "session", "scope-1", "doc-a")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector store is disabled")
}

func TestLocalStorageConfigLifecycle(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.SetConfig(ctx, "ui.theme", "light", "alice"))
	entry, err := ls.GetConfig(ctx, "ui.theme")
	require.NoError(t, err)
	require.NotNil(t, entry)
	assert.Equal(t, "light", entry.Value)
	assert.Equal(t, 1, entry.Version)
	assert.Equal(t, "alice", entry.CreatedBy)
	assert.Equal(t, "alice", entry.UpdatedBy)

	require.NoError(t, ls.SetConfig(ctx, "ui.theme", "dark", "bob"))
	entry, err = ls.GetConfig(ctx, "ui.theme")
	require.NoError(t, err)
	require.NotNil(t, entry)
	assert.Equal(t, "dark", entry.Value)
	assert.Equal(t, 2, entry.Version)
	assert.Equal(t, "alice", entry.CreatedBy)
	assert.Equal(t, "bob", entry.UpdatedBy)

	require.NoError(t, ls.SetConfig(ctx, "ui.locale", "en-US", "bob"))
	entries, err := ls.ListConfigs(ctx)
	require.NoError(t, err)
	require.Len(t, entries, 2)
	assert.Equal(t, "ui.locale", entries[0].Key)
	assert.Equal(t, "ui.theme", entries[1].Key)

	require.NoError(t, ls.DeleteConfig(ctx, "ui.locale"))
	missing, err := ls.GetConfig(ctx, "ui.locale")
	require.NoError(t, err)
	assert.Nil(t, missing)

	err = ls.DeleteConfig(ctx, "ui.locale")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	err = ls.SetConfig(canceled, "ui.theme", "solarized", "bob")
	require.Error(t, err)

	_, err = ls.ListConfigs(canceled)
	require.Error(t, err)
}

func TestLocalStorageCacheAndPubSub(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	type cachedValue struct {
		Name string `json:"name"`
	}

	require.NoError(t, ls.Set("agent", cachedValue{Name: "field"}, time.Minute))
	assert.True(t, ls.Exists("agent"))

	var structDest cachedValue
	require.NoError(t, ls.Get("agent", &structDest))
	assert.Equal(t, "field", structDest.Name)

	require.NoError(t, ls.Set("count", 7, time.Minute))
	var count int
	require.NoError(t, ls.Get("count", &count))
	assert.Equal(t, 7, count)

	var badType string
	err := ls.Get("count", &badType)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cached value type mismatch")

	err = ls.Get("missing", &badType)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in cache")

	require.NoError(t, ls.Delete("count"))
	assert.False(t, ls.Exists("count"))

	event := types.MemoryChangeEvent{
		ID:        "evt-1",
		Type:      "memory.changed",
		Timestamp: time.Now().UTC(),
		Scope:     "session",
		ScopeID:   "scope-1",
		Key:       "doc-a",
		Action:    "set",
		Data:      json.RawMessage(`{"value":1}`),
	}

	cacheChannel, err := ls.Subscribe("memory_changes:session:scope-1")
	require.NoError(t, err)
	require.NoError(t, ls.Publish("memory_changes:session:scope-1", event))

	select {
	case msg := <-cacheChannel:
		assert.Equal(t, "memory_changes:session:scope-1", msg.Channel)
		var got types.MemoryChangeEvent
		require.NoError(t, json.Unmarshal(msg.Payload, &got))
		assert.Equal(t, event.ID, got.ID)
		assert.Equal(t, event.Key, got.Key)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for cache pub/sub message")
	}

	memChannel, err := ls.SubscribeToMemoryChanges(ctx, "session", "scope-2")
	require.NoError(t, err)
	require.NoError(t, ls.PublishMemoryChange(ctx, types.MemoryChangeEvent{
		ID:        "evt-2",
		Type:      "memory.changed",
		Timestamp: time.Now().UTC(),
		Scope:     "session",
		ScopeID:   "scope-2",
		Key:       "doc-b",
		Action:    "delete",
	}))

	select {
	case got := <-memChannel:
		assert.Equal(t, "evt-2", got.ID)
		assert.Equal(t, "delete", got.Action)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for memory change event")
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.SubscribeToMemoryChanges(canceled, "session", "scope-3")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context cancelled")

	err = ls.PublishMemoryChange(canceled, event)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context cancelled")

	assert.Equal(t, "memory_changes:*:*", subscriberKey("", ""))
	assert.Equal(t, "memory_changes:session:*", subscriberKey("session", ""))
}
