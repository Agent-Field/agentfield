package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type memoryErrorBackend struct {
	lastScope   MemoryScope
	lastScopeID string
	getErr      error
	listErr     error
	vectorErr   error
	deleteErr   error
	searchErr   error
}

func (b *memoryErrorBackend) Set(scope MemoryScope, scopeID, key string, value any) error {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil
}

func (b *memoryErrorBackend) Get(scope MemoryScope, scopeID, key string) (any, bool, error) {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil, false, b.getErr
}

func (b *memoryErrorBackend) Delete(scope MemoryScope, scopeID, key string) error {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return b.deleteErr
}

func (b *memoryErrorBackend) List(scope MemoryScope, scopeID string) ([]string, error) {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil, b.listErr
}

func (b *memoryErrorBackend) SetVector(scope MemoryScope, scopeID, key string, embedding []float64, metadata map[string]any) error {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil
}

func (b *memoryErrorBackend) GetVector(scope MemoryScope, scopeID, key string) ([]float64, map[string]any, bool, error) {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil, nil, false, b.vectorErr
}

func (b *memoryErrorBackend) SearchVector(scope MemoryScope, scopeID string, embedding []float64, opts SearchOptions) ([]VectorSearchResult, error) {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return nil, b.searchErr
}

func (b *memoryErrorBackend) DeleteVector(scope MemoryScope, scopeID, key string) error {
	b.lastScope = scope
	b.lastScopeID = scopeID
	return b.deleteErr
}

func TestMemory_AdditionalFallbackAndErrorBranches(t *testing.T) {
	ctx := contextWithExecution(context.Background(), ExecutionContext{RunID: "run-only"})

	t.Run("session-scoped methods fall back to run id", func(t *testing.T) {
		backend := &memoryErrorBackend{}
		memory := NewMemory(backend)

		require.NoError(t, memory.Set(ctx, "key", "value"))
		assert.Equal(t, ScopeSession, backend.lastScope)
		assert.Equal(t, "run-only", backend.lastScopeID)

		require.NoError(t, memory.SetVector(ctx, "vec", []float64{1}, nil))
		assert.Equal(t, ScopeSession, backend.lastScope)
		assert.Equal(t, "run-only", backend.lastScopeID)

		require.NoError(t, memory.Delete(ctx, "key"))
		assert.Equal(t, "run-only", backend.lastScopeID)

		require.NoError(t, memory.DeleteVector(ctx, "vec"))
		assert.Equal(t, "run-only", backend.lastScopeID)

		_, _ = memory.List(ctx)
		assert.Equal(t, "run-only", backend.lastScopeID)

		_, _ = memory.SearchVector(ctx, []float64{1}, SearchOptions{})
		assert.Equal(t, "run-only", backend.lastScopeID)
	})

	t.Run("propagates backend errors", func(t *testing.T) {
		backend := &memoryErrorBackend{
			getErr:    errors.New("get failed"),
			listErr:   errors.New("list failed"),
			vectorErr: errors.New("vector failed"),
			deleteErr: errors.New("delete failed"),
			searchErr: errors.New("search failed"),
		}
		memory := NewMemory(backend)

		_, err := memory.GetWithDefault(ctx, "missing", "fallback")
		require.EqualError(t, err, "get failed")

		_, err = memory.List(ctx)
		require.EqualError(t, err, "list failed")

		_, _, err = memory.GetVector(ctx, "vec")
		require.EqualError(t, err, "vector failed")

		err = memory.Delete(ctx, "key")
		require.EqualError(t, err, "delete failed")

		_, err = memory.SearchVector(ctx, []float64{1}, SearchOptions{})
		require.EqualError(t, err, "search failed")

		err = memory.DeleteVector(ctx, "vec")
		require.EqualError(t, err, "delete failed")
	})
}
