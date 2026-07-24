package knowledge

import (
	"context"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// erroringEmbedder fails on Embed, or returns a deliberately wrong number of
// vectors, to exercise the embed error / count-mismatch paths.
type erroringEmbedder struct {
	err       error
	wrongLen  bool
	dimsValue int
}

func (e erroringEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	if e.wrongLen {
		// Return one fewer vector than requested (or one for a single query).
		n := len(texts) - 1
		if n < 0 {
			n = 0
		}
		out := make([][]float32, n)
		for i := range out {
			out[i] = make([]float32, e.Dimensions())
		}
		return out, nil
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = make([]float32, e.Dimensions())
	}
	return out, nil
}

func (e erroringEmbedder) Dimensions() int {
	if e.dimsValue > 0 {
		return e.dimsValue
	}
	return embedding.Dimensions
}

// failingStore fails the named operation; everything else delegates to memStore.
type failingStore struct {
	*memStore
	failSet    bool
	failSearch bool
	failDelete bool
}

func (f failingStore) SetVector(ctx context.Context, r *types.VectorRecord) error {
	if f.failSet {
		return errors.New("boom set")
	}
	return f.memStore.SetVector(ctx, r)
}

func (f failingStore) SimilaritySearch(ctx context.Context, scope, scopeID string, q []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	if f.failSearch {
		return nil, errors.New("boom search")
	}
	return f.memStore.SimilaritySearch(ctx, scope, scopeID, q, topK, filters)
}

func (f failingStore) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	if f.failDelete {
		return 0, errors.New("boom delete")
	}
	return f.memStore.DeleteVectorsByPrefix(ctx, scope, scopeID, prefix)
}

var wsScopeErr = Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}

// TestUpsert_RejectsEmptySourceID covers the source_id validation branch.
func TestUpsert_RejectsEmptySourceID(t *testing.T) {
	svc := newTestService()
	if _, err := svc.Upsert(context.Background(), wsScopeErr, "  ", chunksOf("x")); err == nil {
		t.Fatal("expected error for empty source_id")
	}
}

// TestUpsert_RejectsEmptyChunks covers the no-chunks validation branch.
func TestUpsert_RejectsEmptyChunks(t *testing.T) {
	svc := newTestService()
	if _, err := svc.Upsert(context.Background(), wsScopeErr, "src", nil); err == nil {
		t.Fatal("expected error for empty chunks")
	}
}

// TestUpsert_RejectsEmptyChunkText covers the per-chunk empty-text branch.
func TestUpsert_RejectsEmptyChunkText(t *testing.T) {
	svc := newTestService()
	if _, err := svc.Upsert(context.Background(), wsScopeErr, "src", []Chunk{{Text: "ok"}, {Text: "   "}}); err == nil {
		t.Fatal("expected error for an empty chunk text")
	}
}

// TestUpsert_EmbedFailure surfaces an embedder error.
func TestUpsert_EmbedFailure(t *testing.T) {
	svc := NewService(newMemStore(), erroringEmbedder{err: errors.New("no embed")})
	if _, err := svc.Upsert(context.Background(), wsScopeErr, "src", chunksOf("x")); err == nil {
		t.Fatal("expected embed failure to propagate")
	}
}

// TestUpsert_EmbedCountMismatch surfaces a vector/chunk count mismatch.
func TestUpsert_EmbedCountMismatch(t *testing.T) {
	svc := NewService(newMemStore(), erroringEmbedder{wrongLen: true})
	if _, err := svc.Upsert(context.Background(), wsScopeErr, "src", chunksOf("a", "b")); err == nil {
		t.Fatal("expected count-mismatch error")
	}
}

// TestUpsert_StoreFailure surfaces a store SetVector error and reports the
// partial index.
func TestUpsert_StoreFailure(t *testing.T) {
	svc := NewService(failingStore{memStore: newMemStore(), failSet: true}, embedding.NewFakeEmbedder())
	n, err := svc.Upsert(context.Background(), wsScopeErr, "src", chunksOf("a"))
	if err == nil {
		t.Fatal("expected store failure to propagate")
	}
	if n != 0 {
		t.Fatalf("partial index on first-chunk failure = %d, want 0", n)
	}
}

// TestSearch_RejectsEmptyQuery covers the empty-query validation branch.
func TestSearch_RejectsEmptyQuery(t *testing.T) {
	svc := newTestService()
	if _, err := svc.Search(context.Background(), wsScopeErr, "   ", 5); err == nil {
		t.Fatal("expected error for empty query")
	}
}

// TestSearch_EmbedFailure surfaces an embedder error during search.
func TestSearch_EmbedFailure(t *testing.T) {
	svc := NewService(newMemStore(), erroringEmbedder{err: errors.New("no embed")})
	if _, err := svc.Search(context.Background(), wsScopeErr, "q", 5); err == nil {
		t.Fatal("expected embed failure to propagate")
	}
}

// TestSearch_EmbedWrongVectorCount surfaces a non-single query vector result.
func TestSearch_EmbedWrongVectorCount(t *testing.T) {
	svc := NewService(newMemStore(), erroringEmbedder{wrongLen: true})
	if _, err := svc.Search(context.Background(), wsScopeErr, "q", 5); err == nil {
		t.Fatal("expected wrong-vector-count error")
	}
}

// TestSearch_StoreFailure surfaces a store SimilaritySearch error.
func TestSearch_StoreFailure(t *testing.T) {
	svc := NewService(failingStore{memStore: newMemStore(), failSearch: true}, embedding.NewFakeEmbedder())
	if _, err := svc.Search(context.Background(), wsScopeErr, "q", 5); err == nil {
		t.Fatal("expected search failure to propagate")
	}
}

// TestSearch_DefaultTopK exercises the topK<=0 default branch (returns without error).
func TestSearch_DefaultTopK(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()
	if _, err := svc.Upsert(ctx, wsScopeErr, "src", chunksOf("hello world")); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if _, err := svc.Search(ctx, wsScopeErr, "hello world", 0); err != nil {
		t.Fatalf("search with topK=0 must default and succeed: %v", err)
	}
}

// TestDeleteSource_RejectsEmptySourceID covers the source_id validation branch.
func TestDeleteSource_RejectsEmptySourceID(t *testing.T) {
	svc := newTestService()
	if _, err := svc.DeleteSource(context.Background(), wsScopeErr, "  "); err == nil {
		t.Fatal("expected error for empty source_id on delete")
	}
}

// TestDeleteSource_StoreFailure surfaces a store delete error.
func TestDeleteSource_StoreFailure(t *testing.T) {
	svc := NewService(failingStore{memStore: newMemStore(), failDelete: true}, embedding.NewFakeEmbedder())
	if _, err := svc.DeleteSource(context.Background(), wsScopeErr, "src"); err == nil {
		t.Fatal("expected delete failure to propagate")
	}
}
