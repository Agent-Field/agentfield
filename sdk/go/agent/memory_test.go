package agent

import (
	"context"
	"io"
	"log"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInMemoryBackend(t *testing.T) {
	backend := NewInMemoryBackend()

	t.Run("Set and Get", func(t *testing.T) {
		err := backend.Set(ScopeSession, "session-1", "key1", "value1")
		require.NoError(t, err)

		val, found, err := backend.Get(ScopeSession, "session-1", "key1")
		require.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, "value1", val)
	})

	t.Run("Get non-existent key", func(t *testing.T) {
		val, found, err := backend.Get(ScopeSession, "session-1", "nonexistent")
		require.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, val)
	})

	t.Run("Get from non-existent scope", func(t *testing.T) {
		val, found, err := backend.Get(ScopeSession, "nonexistent-session", "key1")
		require.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, val)
	})

	t.Run("Delete key", func(t *testing.T) {
		err := backend.Set(ScopeSession, "session-1", "to-delete", "value")
		require.NoError(t, err)

		err = backend.Delete(ScopeSession, "session-1", "to-delete")
		require.NoError(t, err)

		val, found, err := backend.Get(ScopeSession, "session-1", "to-delete")
		require.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, val)
	})

	t.Run("Delete non-existent key (no error)", func(t *testing.T) {
		err := backend.Delete(ScopeSession, "session-1", "nonexistent")
		require.NoError(t, err)
	})

	t.Run("List keys", func(t *testing.T) {
		// Clear and set up fresh data
		backend.ClearScope(ScopeWorkflow, "workflow-1")
		err := backend.Set(ScopeWorkflow, "workflow-1", "key-a", "value-a")
		require.NoError(t, err)
		err = backend.Set(ScopeWorkflow, "workflow-1", "key-b", "value-b")
		require.NoError(t, err)

		keys, err := backend.List(ScopeWorkflow, "workflow-1")
		require.NoError(t, err)
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, "key-a")
		assert.Contains(t, keys, "key-b")
	})

	t.Run("List empty scope", func(t *testing.T) {
		keys, err := backend.List(ScopeGlobal, "nonexistent")
		require.NoError(t, err)
		assert.Nil(t, keys)
	})

	t.Run("Scope isolation", func(t *testing.T) {
		// Set same key in different scopes
		err := backend.Set(ScopeSession, "id-1", "shared-key", "session-value")
		require.NoError(t, err)
		err = backend.Set(ScopeWorkflow, "id-1", "shared-key", "workflow-value")
		require.NoError(t, err)

		sessionVal, _, _ := backend.Get(ScopeSession, "id-1", "shared-key")
		workflowVal, _, _ := backend.Get(ScopeWorkflow, "id-1", "shared-key")

		assert.Equal(t, "session-value", sessionVal)
		assert.Equal(t, "workflow-value", workflowVal)
	})

	t.Run("ScopeID isolation", func(t *testing.T) {
		// Same scope, different IDs
		err := backend.Set(ScopeSession, "session-a", "key", "value-a")
		require.NoError(t, err)
		err = backend.Set(ScopeSession, "session-b", "key", "value-b")
		require.NoError(t, err)

		valA, _, _ := backend.Get(ScopeSession, "session-a", "key")
		valB, _, _ := backend.Get(ScopeSession, "session-b", "key")

		assert.Equal(t, "value-a", valA)
		assert.Equal(t, "value-b", valB)
	})

	t.Run("Clear all data", func(t *testing.T) {
		err := backend.Set(ScopeGlobal, "global", "test", "value")
		require.NoError(t, err)

		backend.Clear()

		val, found, _ := backend.Get(ScopeGlobal, "global", "test")
		assert.False(t, found)
		assert.Nil(t, val)
	})

	t.Run("Store complex types", func(t *testing.T) {
		complexData := map[string]any{
			"name":   "test",
			"count":  42,
			"nested": map[string]any{"key": "value"},
		}
		err := backend.Set(ScopeSession, "session-1", "complex", complexData)
		require.NoError(t, err)

		val, found, err := backend.Get(ScopeSession, "session-1", "complex")
		require.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, complexData, val)
	})
}

func TestMemory_DefaultScope(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	// Create a context with execution context
	execCtx := ExecutionContext{
		SessionID:  "test-session",
		WorkflowID: "test-workflow",
		RunID:      "test-run",
	}
	ctx := contextWithExecution(context.Background(), execCtx)

	t.Run("Set and Get", func(t *testing.T) {
		err := memory.Set(ctx, "key1", "value1")
		require.NoError(t, err)

		val, err := memory.Get(ctx, "key1")
		require.NoError(t, err)
		assert.Equal(t, "value1", val)
	})

	t.Run("GetWithDefault - key exists", func(t *testing.T) {
		err := memory.Set(ctx, "existing", "real-value")
		require.NoError(t, err)

		val, err := memory.GetWithDefault(ctx, "existing", "default")
		require.NoError(t, err)
		assert.Equal(t, "real-value", val)
	})

	t.Run("GetWithDefault - key missing", func(t *testing.T) {
		val, err := memory.GetWithDefault(ctx, "missing", "default-value")
		require.NoError(t, err)
		assert.Equal(t, "default-value", val)
	})

	t.Run("Delete", func(t *testing.T) {
		err := memory.Set(ctx, "to-delete", "value")
		require.NoError(t, err)

		err = memory.Delete(ctx, "to-delete")
		require.NoError(t, err)

		val, err := memory.Get(ctx, "to-delete")
		require.NoError(t, err)
		assert.Nil(t, val)
	})

	t.Run("List", func(t *testing.T) {
		// Clear the session scope
		backend.ClearScope(ScopeSession, "test-session")

		err := memory.Set(ctx, "list-key-1", "v1")
		require.NoError(t, err)
		err = memory.Set(ctx, "list-key-2", "v2")
		require.NoError(t, err)

		keys, err := memory.List(ctx)
		require.NoError(t, err)
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, "list-key-1")
		assert.Contains(t, keys, "list-key-2")
	})
}

func TestMemory_WorkflowScope(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	execCtx := ExecutionContext{
		SessionID:  "test-session",
		WorkflowID: "test-workflow",
		RunID:      "test-run",
	}
	ctx := contextWithExecution(context.Background(), execCtx)

	t.Run("Set and Get", func(t *testing.T) {
		err := memory.WorkflowScope().Set(ctx, "wf-key", "wf-value")
		require.NoError(t, err)

		val, err := memory.WorkflowScope().Get(ctx, "wf-key")
		require.NoError(t, err)
		assert.Equal(t, "wf-value", val)
	})

	t.Run("Isolation from session scope", func(t *testing.T) {
		err := memory.Set(ctx, "shared-key", "session-val")
		require.NoError(t, err)
		err = memory.WorkflowScope().Set(ctx, "shared-key", "workflow-val")
		require.NoError(t, err)

		sessionVal, _ := memory.Get(ctx, "shared-key")
		workflowVal, _ := memory.WorkflowScope().Get(ctx, "shared-key")

		assert.Equal(t, "session-val", sessionVal)
		assert.Equal(t, "workflow-val", workflowVal)
	})
}

func TestMemory_GlobalScope(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	// Two different execution contexts
	ctx1 := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "session-1",
	})
	ctx2 := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "session-2",
	})

	t.Run("Global data shared across sessions", func(t *testing.T) {
		err := memory.GlobalScope().Set(ctx1, "global-key", "global-value")
		require.NoError(t, err)

		// Access from different session
		val, err := memory.GlobalScope().Get(ctx2, "global-key")
		require.NoError(t, err)
		assert.Equal(t, "global-value", val)
	})

	t.Run("Session data isolated", func(t *testing.T) {
		err := memory.Set(ctx1, "session-key", "session-1-value")
		require.NoError(t, err)

		// Should not see session-1's data from session-2
		val, err := memory.Get(ctx2, "session-key")
		require.NoError(t, err)
		assert.Nil(t, val) // Not found
	})
}

func TestMemory_UserScope(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	// Same actor, different sessions
	ctx1 := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "session-1",
		ActorID:   "user-123",
	})
	ctx2 := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "session-2",
		ActorID:   "user-123",
	})

	t.Run("User data persists across sessions", func(t *testing.T) {
		err := memory.UserScope().Set(ctx1, "user-pref", "dark-mode")
		require.NoError(t, err)

		// Access from different session, same user
		val, err := memory.UserScope().Get(ctx2, "user-pref")
		require.NoError(t, err)
		assert.Equal(t, "dark-mode", val)
	})
}

func TestMemory_ScopedGetTyped(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "test-session",
	})

	t.Run("GetTyped with struct", func(t *testing.T) {
		type TestData struct {
			Name  string `json:"name"`
			Count int    `json:"count"`
		}

		original := TestData{Name: "test", Count: 42}
		err := memory.SessionScope().Set(ctx, "typed-data", original)
		require.NoError(t, err)

		var retrieved TestData
		err = memory.SessionScope().GetTyped(ctx, "typed-data", &retrieved)
		require.NoError(t, err)
		assert.Equal(t, original.Name, retrieved.Name)
		assert.Equal(t, original.Count, retrieved.Count)
	})

	t.Run("GetTyped with slice", func(t *testing.T) {
		original := []string{"a", "b", "c"}
		err := memory.SessionScope().Set(ctx, "slice-data", original)
		require.NoError(t, err)

		var retrieved []string
		err = memory.SessionScope().GetTyped(ctx, "slice-data", &retrieved)
		require.NoError(t, err)
		assert.Equal(t, original, retrieved)
	})

	t.Run("GetTyped with non-existent key", func(t *testing.T) {
		var retrieved string
		err := memory.SessionScope().GetTyped(ctx, "nonexistent", &retrieved)
		require.NoError(t, err)
		assert.Equal(t, "", retrieved) // zero value
	})
}

func TestMemory_FallbackToRunID(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	// Context with only RunID (no SessionID, WorkflowID, ActorID)
	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	t.Run("Session scope falls back to RunID", func(t *testing.T) {
		err := memory.Set(ctx, "key", "value")
		require.NoError(t, err)

		// Verify it was stored under RunID
		val, found, _ := backend.Get(ScopeSession, "run-123", "key")
		assert.True(t, found)
		assert.Equal(t, "value", val)
	})

	t.Run("Workflow scope falls back to RunID", func(t *testing.T) {
		err := memory.WorkflowScope().Set(ctx, "wf-key", "wf-value")
		require.NoError(t, err)

		val, found, _ := backend.Get(ScopeWorkflow, "run-123", "wf-key")
		assert.True(t, found)
		assert.Equal(t, "wf-value", val)
	})
}

func TestMemory_NilBackend(t *testing.T) {
	// NewMemory should create InMemoryBackend if nil is passed
	memory := NewMemory(nil)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "test",
	})

	err := memory.Set(ctx, "key", "value")
	require.NoError(t, err)

	val, err := memory.Get(ctx, "key")
	require.NoError(t, err)
	assert.Equal(t, "value", val)
}

func TestMemory_VectorAndScopedHelpers(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)
	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:      "run-1",
		SessionID:  "session-1",
		WorkflowID: "workflow-1",
		ActorID:    "actor-1",
	})

	require.NoError(t, memory.SetVector(ctx, "session-vec", []float64{0.1, 0.2}, map[string]any{"kind": "session"}))
	embedding, metadata, err := memory.GetVector(ctx, "session-vec")
	require.NoError(t, err)
	assert.Equal(t, []float64{0.1, 0.2}, embedding)
	assert.Equal(t, map[string]any{"kind": "session"}, metadata)

	results, err := memory.SearchVector(ctx, []float64{0.1, 0.2}, SearchOptions{Limit: 3})
	require.NoError(t, err)
	assert.Empty(t, results)
	require.NoError(t, memory.DeleteVector(ctx, "session-vec"))
	embedding, metadata, err = memory.GetVector(ctx, "session-vec")
	require.NoError(t, err)
	assert.Nil(t, embedding)
	assert.Nil(t, metadata)

	custom := memory.Scoped(ScopeGlobal, "tenant-1")
	require.NoError(t, custom.Set(ctx, "answer", 42))
	value, err := custom.Get(ctx, "answer")
	require.NoError(t, err)
	assert.Equal(t, 42, value)
	value, err = custom.GetWithDefault(ctx, "missing", "fallback")
	require.NoError(t, err)
	assert.Equal(t, "fallback", value)
	keys, err := custom.List(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"answer"}, keys)
	require.NoError(t, custom.Delete(ctx, "answer"))

	require.NoError(t, custom.SetVector(ctx, "global-vec", []float64{1, 2}, map[string]any{"kind": "global"}))
	embedding, metadata, err = custom.GetVector(ctx, "global-vec")
	require.NoError(t, err)
	assert.Equal(t, []float64{1, 2}, embedding)
	assert.Equal(t, map[string]any{"kind": "global"}, metadata)
	results, err = custom.SearchVector(ctx, []float64{1, 2}, SearchOptions{Limit: 1})
	require.NoError(t, err)
	assert.Empty(t, results)
	require.NoError(t, custom.DeleteVector(ctx, "global-vec"))

	userScoped := memory.UserScope()
	require.NoError(t, userScoped.Set(ctx, "pref", "dark"))
	value, err = userScoped.GetWithDefault(ctx, "pref", "light")
	require.NoError(t, err)
	assert.Equal(t, "dark", value)
	keys, err = userScoped.List(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"pref"}, keys)
	require.NoError(t, userScoped.Delete(ctx, "pref"))
	keys, err = userScoped.List(ctx)
	require.NoError(t, err)
	assert.Empty(t, keys)
}

func TestAgentMemory(t *testing.T) {
	cfg := Config{
		NodeID:  "test-node",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "test-session",
	})

	t.Run("Agent has memory", func(t *testing.T) {
		assert.NotNil(t, agent.Memory())
	})

	t.Run("Agent memory works", func(t *testing.T) {
		err := agent.Memory().Set(ctx, "agent-key", "agent-value")
		require.NoError(t, err)

		val, err := agent.Memory().Get(ctx, "agent-key")
		require.NoError(t, err)
		assert.Equal(t, "agent-value", val)
	})
}

func TestAgentWithCustomMemoryBackend(t *testing.T) {
	// Create a custom backend
	customBackend := NewInMemoryBackend()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		Logger:        log.New(io.Discard, "", 0),
		MemoryBackend: customBackend,
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		SessionID: "test-session",
	})

	// Set via agent
	err = agent.Memory().Set(ctx, "custom-key", "custom-value")
	require.NoError(t, err)

	// Verify directly on backend
	val, found, _ := customBackend.Get(ScopeSession, "test-session", "custom-key")
	assert.True(t, found)
	assert.Equal(t, "custom-value", val)
}

func TestInMemoryBackend_DeepCopy(t *testing.T) {
	backend := NewInMemoryBackend()

	t.Run("Set deep-copies map - mutation does not affect stored value", func(t *testing.T) {
		input := map[string]any{"key": "original", "nested": map[string]any{"a": 1}}
		err := backend.Set(ScopeSession, "s1", "map-key", input)
		require.NoError(t, err)

		input["key"] = "mutated"
		input["new-field"] = "added"
		if nested, ok := input["nested"].(map[string]any); ok {
			nested["a"] = 999
		}

		val, found, err := backend.Get(ScopeSession, "s1", "map-key")
		require.NoError(t, err)
		require.True(t, found)

		retrieved, ok := val.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "original", retrieved["key"])
		assert.NotContains(t, retrieved, "new-field")
		if nested, ok := retrieved["nested"].(map[string]any); ok {
			assert.Equal(t, 1, nested["a"])
		} else {
			t.Fatal("expected nested map")
		}
	})

	t.Run("Get deep-copies map - mutation does not affect stored value", func(t *testing.T) {
		err := backend.Set(ScopeSession, "s2", "map2", map[string]any{"x": "y", "inner": map[string]any{"z": "w"}})
		require.NoError(t, err)

		val1, found, err := backend.Get(ScopeSession, "s2", "map2")
		require.NoError(t, err)
		require.True(t, found)

		m1, ok := val1.(map[string]any)
		require.True(t, ok)
		m1["x"] = "mutated"
		if inner, ok := m1["inner"].(map[string]any); ok {
			inner["z"] = "also-mutated"
		}
		m1["new-key"] = "new-value"

		val2, found, err := backend.Get(ScopeSession, "s2", "map2")
		require.NoError(t, err)
		require.True(t, found)

		m2, ok := val2.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "y", m2["x"])
		assert.NotContains(t, m2, "new-key")
		if inner, ok := m2["inner"].(map[string]any); ok {
			assert.Equal(t, "w", inner["z"])
		} else {
			t.Fatal("expected nested map")
		}
	})

	t.Run("SetVector deep-copies embedding and metadata", func(t *testing.T) {
		embedding := []float64{1.0, 2.0, 3.0}
		metadata := map[string]any{"kind": "original", "tags": map[string]any{"env": "dev"}}

		err := backend.SetVector(ScopeSession, "s3", "vec1", embedding, metadata)
		require.NoError(t, err)

		embedding[0] = 999.0
		metadata["kind"] = "mutated"
		if tags, ok := metadata["tags"].(map[string]any); ok {
			tags["env"] = "prod"
		}
		metadata["added"] = "extra"
		embedding = append(embedding, 4.0)

		retrievedEmb, retrievedMeta, found, err := backend.GetVector(ScopeSession, "s3", "vec1")
		require.NoError(t, err)
		require.True(t, found)

		assert.Equal(t, []float64{1.0, 2.0, 3.0}, retrievedEmb)
		assert.Equal(t, "original", retrievedMeta["kind"])
		assert.NotContains(t, retrievedMeta, "added")
		if tags, ok := retrievedMeta["tags"].(map[string]any); ok {
			assert.Equal(t, "dev", tags["env"])
		} else {
			t.Fatal("expected nested metadata map")
		}
	})

	t.Run("GetVector deep-copies embedding and metadata", func(t *testing.T) {
		err := backend.SetVector(ScopeSession, "s4", "vec2", []float64{0.1, 0.2}, map[string]any{"meta": "orig"})
		require.NoError(t, err)

		emb1, meta1, found, err := backend.GetVector(ScopeSession, "s4", "vec2")
		require.NoError(t, err)
		require.True(t, found)

		emb1[0] = 555.0
		meta1["meta"] = "changed"

		emb2, meta2, found, err := backend.GetVector(ScopeSession, "s4", "vec2")
		require.NoError(t, err)
		require.True(t, found)

		assert.Equal(t, []float64{0.1, 0.2}, emb2)
		assert.Equal(t, "orig", meta2["meta"])
	})
}

func TestInMemoryBackend_DeepCopyConcurrent(t *testing.T) {
	backend := NewInMemoryBackend()

	err := backend.Set(ScopeSession, "s1", "shared", map[string]any{"counter": 0})
	require.NoError(t, err)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			val, found, _ := backend.Get(ScopeSession, "s1", "shared")
			if found {
				if m, ok := val.(map[string]any); ok {
					m["counter"] = i
				}
			}
		}
	}()

	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			val, found, _ := backend.Get(ScopeSession, "s1", "shared")
			if found {
				if m, ok := val.(map[string]any); ok {
					_ = m["counter"]
				}
			}
		}
	}()

	wg.Wait()
}

func TestDeepCopyAny_SliceAny(t *testing.T) {
	t.Run("shallow []any copy", func(t *testing.T) {
		orig := []any{"a", "b", 42}
		copied, err := deepCopyAny(orig)
		require.NoError(t, err)

		assert.Equal(t, orig, copied)
		orig[0] = "mutated"
		assert.NotEqual(t, "mutated", copied.([]any)[0])
	})

	t.Run("nested []any recursion", func(t *testing.T) {
		orig := []any{
			"outer",
			[]any{"inner1", "inner2"},
			map[string]any{"key": "val"},
		}
		copiedAny, err := deepCopyAny(orig)
		require.NoError(t, err)
		copied := copiedAny.([]any)

		orig[0] = "changed"
		orig[1].([]any)[0] = "changed-inner"
		orig[2].(map[string]any)["key"] = "changed-map"

		assert.Equal(t, "outer", copied[0])
		assert.Equal(t, "inner1", copied[1].([]any)[0])
		assert.Equal(t, "val", copied[2].(map[string]any)["key"])
	})

	t.Run("empty []any", func(t *testing.T) {
		orig := []any{}
		copied, err := deepCopyAny(orig)
		require.NoError(t, err)
		assert.Equal(t, []any{}, copied)
	})
}

func TestDeepCopyAny_SliceFloat64(t *testing.T) {
	t.Run("float64 slice is copied not shared", func(t *testing.T) {
		orig := []float64{1.0, 2.0, 3.0}
		copiedAny, err := deepCopyAny(orig)
		require.NoError(t, err)
		copied := copiedAny.([]float64)

		orig[0] = 999.0
		assert.Equal(t, []float64{1.0, 2.0, 3.0}, copied)
	})

	t.Run("empty []float64", func(t *testing.T) {
		orig := []float64{}
		copiedAny, err := deepCopyAny(orig)
		require.NoError(t, err)
		copied := copiedAny.([]float64)
		assert.Equal(t, []float64{}, copied)
	})
}

func TestDeepCopyAny_DefaultPassthrough(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"string", "hello"},
		{"integer", 42},
		{"float", 3.14},
		{"boolean", true},
		{"nil", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			copied, err := deepCopyAny(tc.value)
			require.NoError(t, err)
			assert.Equal(t, tc.value, copied)
		})
	}
}

func TestInMemoryBackendRejectsCyclicValues(t *testing.T) {
	backend := NewInMemoryBackend()
	cycle := map[string]any{}
	cycle["self"] = cycle

	err := backend.Set(ScopeSession, "session", "key", cycle)
	require.ErrorIs(t, err, ErrCyclicMemoryValue)

	_, found, err := backend.Get(ScopeSession, "session", "key")
	require.NoError(t, err)
	assert.False(t, found)
}

func TestInMemoryBackendRejectsCyclicVectorMetadata(t *testing.T) {
	backend := NewInMemoryBackend()
	cycle := []any{nil}
	cycle[0] = cycle

	err := backend.SetVector(ScopeSession, "session", "key", []float64{1}, map[string]any{"cycle": cycle})
	require.ErrorIs(t, err, ErrCyclicMemoryValue)

	_, _, found, err := backend.GetVector(ScopeSession, "session", "key")
	require.NoError(t, err)
	assert.False(t, found)
}

func TestDeepCopyFloat64Slice_Nil(t *testing.T) {
	assert.Nil(t, deepCopyFloat64Slice(nil))
}

func TestDeepCopyFloat64Slice_Empty(t *testing.T) {
	result := deepCopyFloat64Slice([]float64{})
	assert.NotNil(t, result)
	assert.Empty(t, result)
}
