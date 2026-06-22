package knowledge

import (
	"context"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// memStore is an in-memory VectorStore that faithfully mirrors the production
// vector store contract (see internal/storage/vector_store_sqlite.go):
//   - records keyed by (scope, scopeID, key); Set upserts on that triple
//   - Search matches a single (scope, scopeID), applies an exact-match metadata
//     filter, ranks by cosine similarity, and limits to topK
//   - DeleteVectorsByPrefix removes keys under (scope, scopeID) with a prefix
//
// Tests are written against the documented knowledge-scope behavior, not the
// service implementation; this store stands in for the real pgvector/sqlite-vec
// backend with equivalent semantics.
type memStore struct {
	records map[string]*types.VectorRecord // composite key -> record
}

func newMemStore() *memStore { return &memStore{records: map[string]*types.VectorRecord{}} }

func compositeKey(scope, scopeID, key string) string {
	return scope + "\x00" + scopeID + "\x00" + key
}

func (m *memStore) SetVector(_ context.Context, record *types.VectorRecord) error {
	cp := *record
	m.records[compositeKey(record.Scope, record.ScopeID, record.Key)] = &cp
	return nil
}

func (m *memStore) DeleteVectorsByPrefix(_ context.Context, scope, scopeID, prefix string) (int, error) {
	n := 0
	for k, r := range m.records {
		if r.Scope == scope && r.ScopeID == scopeID && strings.HasPrefix(r.Key, prefix) {
			delete(m.records, k)
			n++
		}
	}
	return n, nil
}

func (m *memStore) SimilaritySearch(_ context.Context, scope, scopeID string, query []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	var results []*types.VectorSearchResult
	for _, r := range m.records {
		if r.Scope != scope || r.ScopeID != scopeID {
			continue
		}
		if !matchesFilters(r.Metadata, filters) {
			continue
		}
		score := cosine(query, r.Embedding)
		results = append(results, &types.VectorSearchResult{
			Scope:    r.Scope,
			ScopeID:  r.ScopeID,
			Key:      r.Key,
			Score:    score,
			Distance: 1 - score,
			Metadata: r.Metadata,
		})
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func matchesFilters(meta, filters map[string]interface{}) bool {
	for k, want := range filters {
		got, ok := meta[k]
		if !ok || got != want {
			return false
		}
	}
	return true
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func newTestService() *Service {
	return NewService(newMemStore(), embedding.NewFakeEmbedder())
}

func chunksOf(texts ...string) []Chunk {
	out := make([]Chunk, len(texts))
	for i, t := range texts {
		out[i] = Chunk{Text: t}
	}
	return out
}

func sourceIDs(hits []SearchHit) map[string]bool {
	m := map[string]bool{}
	for _, h := range hits {
		m[h.SourceID] = true
	}
	return m
}

// Contract: upsert then search returns the stored chunk (exact text match wins).
func TestUpsertThenSearchReturnsChunk(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()
	scope := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}

	if _, err := svc.Upsert(ctx, scope, "src1", chunksOf("the quick brown fox", "lorem ipsum dolor")); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	hits, err := svc.Search(ctx, scope, "the quick brown fox", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one hit")
	}
	if hits[0].Text != "the quick brown fox" {
		t.Fatalf("top hit text = %q, want exact-match chunk", hits[0].Text)
	}
	if hits[0].SourceID != "src1" {
		t.Fatalf("top hit source = %q, want src1", hits[0].SourceID)
	}
}

// Contract: a workspace-tier search never returns another workspace's chunks.
func TestCrossTenantIsolation(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	wsA := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}
	wsB := Scope{Tier: TierWorkspace, WorkspaceID: "wsB"}

	if _, err := svc.Upsert(ctx, wsA, "a1", chunksOf("shared secret text")); err != nil {
		t.Fatalf("upsert A: %v", err)
	}
	if _, err := svc.Upsert(ctx, wsB, "b1", chunksOf("shared secret text")); err != nil {
		t.Fatalf("upsert B: %v", err)
	}

	hits, err := svc.Search(ctx, wsA, "shared secret text", 10)
	if err != nil {
		t.Fatalf("search A: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected a hit in workspace A")
	}
	for _, h := range hits {
		if h.SourceID == "b1" {
			t.Fatalf("workspace A search leaked workspace B chunk %q", h.SourceID)
		}
		if ws, _ := h.Metadata["workspace_id"].(string); ws != "wsA" {
			t.Fatalf("hit has workspace_id %q, want wsA", ws)
		}
	}
}

// Contract: a project-tier search returns BOTH the project's own chunks and the
// parent workspace's chunks (inheritance).
func TestProjectInheritsWorkspace(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	wsScope := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}
	projScope := Scope{Tier: TierProject, WorkspaceID: "wsA", ProjectID: "projX"}

	if _, err := svc.Upsert(ctx, wsScope, "ws-src", chunksOf("workspace level policy")); err != nil {
		t.Fatalf("upsert ws: %v", err)
	}
	if _, err := svc.Upsert(ctx, projScope, "proj-src", chunksOf("project specific note")); err != nil {
		t.Fatalf("upsert proj: %v", err)
	}

	// Project search for the workspace chunk should find it (inheritance).
	hits, err := svc.Search(ctx, projScope, "workspace level policy", 10)
	if err != nil {
		t.Fatalf("search proj for ws chunk: %v", err)
	}
	if !sourceIDs(hits)["ws-src"] {
		t.Fatal("project search did not inherit parent workspace chunk")
	}

	// Project search for the project chunk should also find it.
	hits, err = svc.Search(ctx, projScope, "project specific note", 10)
	if err != nil {
		t.Fatalf("search proj for proj chunk: %v", err)
	}
	if !sourceIDs(hits)["proj-src"] {
		t.Fatal("project search did not return its own chunk")
	}

	// A workspace-tier search must NOT see the project's chunk (no downward leak).
	hits, err = svc.Search(ctx, wsScope, "project specific note", 10)
	if err != nil {
		t.Fatalf("search ws for proj chunk: %v", err)
	}
	if sourceIDs(hits)["proj-src"] {
		t.Fatal("workspace search leaked a project-only chunk")
	}
}

// Contract: a sender-scoped upsert then a sender-scoped search returns the chunk.
func TestSenderUpsertThenSearchReturnsChunk(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()
	scope := Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX", SenderID: "sndA"}

	if _, err := svc.Upsert(ctx, scope, "snd-src", chunksOf("sender private note")); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	hits, err := svc.Search(ctx, scope, "sender private note", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !sourceIDs(hits)["snd-src"] {
		t.Fatal("sender search did not return its own chunk")
	}
}

// Contract: a search carrying workspace_id + project_id + sender_id sees
// workspace, project, and sender chunks additively.
func TestSenderSearchIsAdditive(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	wsScope := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}
	projScope := Scope{Tier: TierProject, WorkspaceID: "wsA", ProjectID: "projX"}
	sndScope := Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX", SenderID: "sndA"}

	if _, err := svc.Upsert(ctx, wsScope, "ws-src", chunksOf("workspace level policy")); err != nil {
		t.Fatalf("upsert ws: %v", err)
	}
	if _, err := svc.Upsert(ctx, projScope, "proj-src", chunksOf("project specific note")); err != nil {
		t.Fatalf("upsert proj: %v", err)
	}
	if _, err := svc.Upsert(ctx, sndScope, "snd-src", chunksOf("sender private note")); err != nil {
		t.Fatalf("upsert snd: %v", err)
	}

	// A query in the sender's scope (ws + proj + sender ids present) must see
	// all three tiers' chunks.
	for _, want := range []struct {
		query  string
		source string
	}{
		{"workspace level policy", "ws-src"},
		{"project specific note", "proj-src"},
		{"sender private note", "snd-src"},
	} {
		hits, err := svc.Search(ctx, sndScope, want.query, 10)
		if err != nil {
			t.Fatalf("search %q: %v", want.query, err)
		}
		if !sourceIDs(hits)[want.source] {
			t.Fatalf("sender-scoped search for %q did not return %q (additive set failed)", want.query, want.source)
		}
	}
}

// Contract: cross-sender isolation. A query for sender A never returns sender
// B's chunks; and a workspace-only query does NOT see sender-scoped chunks.
func TestCrossSenderIsolation(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	sndA := Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX", SenderID: "sndA"}
	sndB := Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX", SenderID: "sndB"}
	wsOnly := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}

	if _, err := svc.Upsert(ctx, sndA, "a1", chunksOf("shared sender text")); err != nil {
		t.Fatalf("upsert A: %v", err)
	}
	if _, err := svc.Upsert(ctx, sndB, "b1", chunksOf("shared sender text")); err != nil {
		t.Fatalf("upsert B: %v", err)
	}

	// Sender A search must not leak sender B's chunk.
	hits, err := svc.Search(ctx, sndA, "shared sender text", 10)
	if err != nil {
		t.Fatalf("search A: %v", err)
	}
	if !sourceIDs(hits)["a1"] {
		t.Fatal("sender A search did not return its own chunk")
	}
	if sourceIDs(hits)["b1"] {
		t.Fatal("sender A search leaked sender B's chunk")
	}

	// A workspace-only query (no sender_id) must not see any sender-scoped chunk.
	hits, err = svc.Search(ctx, wsOnly, "shared sender text", 10)
	if err != nil {
		t.Fatalf("search ws-only: %v", err)
	}
	if sourceIDs(hits)["a1"] || sourceIDs(hits)["b1"] {
		t.Fatal("workspace-only search leaked a sender-scoped chunk")
	}
}

// Contract: deleting a sender source removes its chunks.
func TestDeleteSenderSourceRemovesChunks(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()
	scope := Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX", SenderID: "sndA"}

	if _, err := svc.Upsert(ctx, scope, "snd-src", chunksOf("alpha", "beta", "gamma")); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	deleted, err := svc.DeleteSource(ctx, scope, "snd-src")
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if deleted != 3 {
		t.Fatalf("deleted = %d, want 3", deleted)
	}

	hits, err := svc.Search(ctx, scope, "alpha", 10)
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	if sourceIDs(hits)["snd-src"] {
		t.Fatal("expected no sender chunks after delete")
	}
}

// Contract: sender tier without a sender_id is rejected.
func TestScopeValidationRejectsSenderWithoutSenderID(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	if _, err := svc.Search(ctx, Scope{Tier: TierSender, WorkspaceID: "wsA", ProjectID: "projX"}, "q", 5); err == nil {
		t.Fatal("expected error for sender tier without sender_id")
	}
	if _, err := svc.Upsert(ctx, Scope{Tier: TierSender, WorkspaceID: "wsA"}, "s", chunksOf("x")); err == nil {
		t.Fatal("expected error for sender tier without sender_id on upsert")
	}
}

// Contract: project of a DIFFERENT workspace cannot see workspace A's chunks.
func TestProjectCrossWorkspaceIsolation(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	if _, err := svc.Upsert(ctx, Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}, "ws-src", chunksOf("alpha knowledge")); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	otherProj := Scope{Tier: TierProject, WorkspaceID: "wsB", ProjectID: "projY"}
	hits, err := svc.Search(ctx, otherProj, "alpha knowledge", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if sourceIDs(hits)["ws-src"] {
		t.Fatal("project in workspace B saw workspace A's chunk")
	}
}

// Contract: deleting a source removes all its chunks.
func TestDeleteSourceRemovesChunks(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()
	scope := Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}

	if _, err := svc.Upsert(ctx, scope, "src1", chunksOf("chunk one", "chunk two", "chunk three")); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	deleted, err := svc.DeleteSource(ctx, scope, "src1")
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if deleted != 3 {
		t.Fatalf("deleted = %d, want 3", deleted)
	}

	hits, err := svc.Search(ctx, scope, "chunk one", 10)
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("expected no hits after delete, got %d", len(hits))
	}
}

// Contract: an empty workspace_id is rejected (no unscoped operations).
func TestScopeValidationRejectsEmptyWorkspace(t *testing.T) {
	ctx := context.Background()
	svc := newTestService()

	if _, err := svc.Upsert(ctx, Scope{Tier: TierWorkspace}, "s", chunksOf("x")); err == nil {
		t.Fatal("expected error for empty workspace_id on upsert")
	}
	if _, err := svc.Search(ctx, Scope{Tier: TierWorkspace}, "q", 5); err == nil {
		t.Fatal("expected error for empty workspace_id on search")
	}
	if _, err := svc.Search(ctx, Scope{Tier: TierProject, WorkspaceID: "wsA"}, "q", 5); err == nil {
		t.Fatal("expected error for project tier without project_id")
	}
	if _, err := svc.Search(ctx, Scope{Tier: "bogus", WorkspaceID: "wsA"}, "q", 5); err == nil {
		t.Fatal("expected error for invalid tier")
	}
}

// leakyStore ignores metadata filters entirely, returning every record under
// the queried (scope, scopeID). It models a backend whose in-query scope filter
// is absent/broken, so the test can prove the service's in-Go re-verification
// (defense in depth) still drops out-of-scope chunks.
type leakyStore struct{ *memStore }

func (l leakyStore) SimilaritySearch(ctx context.Context, scope, scopeID string, query []float32, topK int, _ map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return l.memStore.SimilaritySearch(ctx, scope, scopeID, query, topK, nil)
}

// Contract: a chunk with mismatched workspace/namespace metadata is dropped by
// the in-Go re-verification even when the underlying store fails to filter it.
func TestSearchReverifiesScope(t *testing.T) {
	ctx := context.Background()
	base := newMemStore()
	svc := NewService(leakyStore{base}, embedding.NewFakeEmbedder())

	// Plant a poisoned record under wsA's namespace whose metadata lies about
	// its workspace. The leaky store returns it (no filtering), so only the
	// Go-side re-verification can keep it out of results.
	vecs, _ := embedding.NewFakeEmbedder().Embed(ctx, []string{"poison"})
	base.records[compositeKey(vectorScope, "ws:wsA", "evil:0")] = &types.VectorRecord{
		Scope:     vectorScope,
		ScopeID:   "ws:wsA",
		Key:       "evil:0",
		Embedding: vecs[0],
		Metadata: map[string]interface{}{
			"namespace":    "ws:wsB", // mismatched namespace
			"workspace_id": "wsB",    // mismatched workspace
			"source_id":    "evil",
			"text":         "poison",
		},
	}

	hits, err := svc.Search(ctx, Scope{Tier: TierWorkspace, WorkspaceID: "wsA"}, "poison", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, h := range hits {
		if h.SourceID == "evil" {
			t.Fatal("re-verification did not drop a scope-mismatched chunk")
		}
	}
}
