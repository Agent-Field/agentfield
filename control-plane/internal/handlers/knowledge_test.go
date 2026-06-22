package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/internal/knowledge"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// knMemStore is an in-memory VectorStore mirroring the production vector store
// contract closely enough to drive the knowledge handlers end-to-end: records
// keyed by (scope, scopeID, key); search matches a single (scope, scopeID),
// applies an exact-match metadata filter, ranks by cosine, and limits to topK.
type knMemStore struct {
	records map[string]*types.VectorRecord
}

func newKnMemStore() *knMemStore { return &knMemStore{records: map[string]*types.VectorRecord{}} }

func knKey(scope, scopeID, key string) string { return scope + "\x00" + scopeID + "\x00" + key }

func (m *knMemStore) SetVector(_ context.Context, record *types.VectorRecord) error {
	cp := *record
	m.records[knKey(record.Scope, record.ScopeID, record.Key)] = &cp
	return nil
}

func (m *knMemStore) DeleteVectorsByPrefix(_ context.Context, scope, scopeID, prefix string) (int, error) {
	n := 0
	for k, r := range m.records {
		if r.Scope == scope && r.ScopeID == scopeID && strings.HasPrefix(r.Key, prefix) {
			delete(m.records, k)
			n++
		}
	}
	return n, nil
}

func (m *knMemStore) SimilaritySearch(_ context.Context, scope, scopeID string, query []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	var results []*types.VectorSearchResult
	for _, r := range m.records {
		if r.Scope != scope || r.ScopeID != scopeID {
			continue
		}
		match := true
		for k, want := range filters {
			if got, ok := r.Metadata[k]; !ok || got != want {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		score := knCosine(query, r.Embedding)
		results = append(results, &types.VectorSearchResult{
			Scope: r.Scope, ScopeID: r.ScopeID, Key: r.Key,
			Score: score, Distance: 1 - score, Metadata: r.Metadata,
		})
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func knCosine(a, b []float32) float64 {
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

func newKnowledgeHandlerService() *knowledge.Service {
	return knowledge.NewService(newKnMemStore(), embedding.NewFakeEmbedder())
}

func doJSON(t *testing.T, h gin.HandlerFunc, method, path string, params gin.Params, body interface{}) (*httptest.ResponseRecorder, map[string]interface{}) {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, json.NewEncoder(&buf).Encode(body))
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(method, path, &buf)
	c.Request.Header.Set("Content-Type", "application/json")
	c.Params = params
	h(c)
	var out map[string]interface{}
	if w.Body.Len() > 0 {
		_ = json.Unmarshal(w.Body.Bytes(), &out)
	}
	return w, out
}

func senderScope() gin.H {
	return gin.H{"tier": "sender", "workspace_id": "wsA", "project_id": "projX", "sender_id": "sndA"}
}

// Contract (handler level): a sender-scoped upsert then a search carrying
// workspace_id + project_id + sender_id returns the sender chunk, and the
// additive set surfaces a workspace chunk too. Cross-sender isolation holds.
func TestKnowledgeHandlers_SenderScopeAdditiveAndIsolation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newKnowledgeHandlerService()
	upsert := KnowledgeUpsertHandler(svc)
	search := KnowledgeSearchHandler(svc)
	del := KnowledgeDeleteSourceHandler(svc)

	// Workspace-level chunk.
	w, _ := doJSON(t, upsert, http.MethodPost, "/knowledge/upsert", nil, gin.H{
		"scope":     gin.H{"tier": "workspace", "workspace_id": "wsA"},
		"source_id": "ws-src",
		"chunks":    []gin.H{{"text": "workspace level policy"}},
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	// Sender A chunk.
	w, _ = doJSON(t, upsert, http.MethodPost, "/knowledge/upsert", nil, gin.H{
		"scope":     senderScope(),
		"source_id": "snd-src",
		"chunks":    []gin.H{{"text": "sender private note"}},
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	// Sender B chunk (different sender, same workspace/project).
	w, _ = doJSON(t, upsert, http.MethodPost, "/knowledge/upsert", nil, gin.H{
		"scope":     gin.H{"tier": "sender", "workspace_id": "wsA", "project_id": "projX", "sender_id": "sndB"},
		"source_id": "snd-b",
		"chunks":    []gin.H{{"text": "sender private note"}},
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	// Sender A search sees its own sender chunk + inherited workspace chunk,
	// but NOT sender B's chunk.
	w, out := doJSON(t, search, http.MethodPost, "/knowledge/search", nil, gin.H{
		"scope": senderScope(),
		"query": "sender private note",
		"top_k": 10,
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	got := resultSources(out)
	require.True(t, got["snd-src"], "sender A search should return its own chunk")
	require.False(t, got["snd-b"], "sender A search must not leak sender B's chunk")

	w, out = doJSON(t, search, http.MethodPost, "/knowledge/search", nil, gin.H{
		"scope": senderScope(),
		"query": "workspace level policy",
		"top_k": 10,
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.True(t, resultSources(out)["ws-src"], "sender search should inherit workspace chunk (additive)")

	// A workspace-only search must not see sender-scoped chunks.
	w, out = doJSON(t, search, http.MethodPost, "/knowledge/search", nil, gin.H{
		"scope": gin.H{"tier": "workspace", "workspace_id": "wsA"},
		"query": "sender private note",
		"top_k": 10,
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.False(t, resultSources(out)["snd-src"], "workspace search must not see sender chunk")

	// Delete the sender source, then confirm it is gone.
	w, _ = doJSON(t, del, http.MethodDelete, "/knowledge/source/snd-src",
		gin.Params{{Key: "id", Value: "snd-src"}}, gin.H{"scope": senderScope()})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	w, out = doJSON(t, search, http.MethodPost, "/knowledge/search", nil, gin.H{
		"scope": senderScope(),
		"query": "sender private note",
		"top_k": 10,
	})
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.False(t, resultSources(out)["snd-src"], "deleted sender chunk should be gone")
}

// Contract: an empty workspace_id is rejected at the handler layer. Here the
// binding requires workspace_id, so an empty value is a 400.
func TestKnowledgeHandlers_RejectsEmptyWorkspace(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newKnowledgeHandlerService()
	upsert := KnowledgeUpsertHandler(svc)

	w, _ := doJSON(t, upsert, http.MethodPost, "/knowledge/upsert", nil, gin.H{
		"scope":     gin.H{"tier": "workspace", "workspace_id": ""},
		"source_id": "s",
		"chunks":    []gin.H{{"text": "x"}},
	})
	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
}

// Contract: sender tier without a sender_id is rejected as a 400 (validation
// error surfaced by the service and mapped by the handler).
func TestKnowledgeHandlers_RejectsSenderWithoutSenderID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newKnowledgeHandlerService()
	search := KnowledgeSearchHandler(svc)

	w, _ := doJSON(t, search, http.MethodPost, "/knowledge/search", nil, gin.H{
		"scope": gin.H{"tier": "sender", "workspace_id": "wsA", "project_id": "projX"},
		"query": "q",
		"top_k": 5,
	})
	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
}

func resultSources(out map[string]interface{}) map[string]bool {
	m := map[string]bool{}
	results, _ := out["results"].([]interface{})
	for _, r := range results {
		hit, _ := r.(map[string]interface{})
		if sid, ok := hit["source_id"].(string); ok {
			m[sid] = true
		}
	}
	return m
}
