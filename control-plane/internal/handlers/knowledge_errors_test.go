package handlers

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/internal/knowledge"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// knFailingStore fails SetVector/SimilaritySearch/DeleteVectorsByPrefix so the
// handler's internal-error (500) mapping can be exercised. These are NOT
// validation errors, so writeKnowledgeError must classify them as 500.
type knFailingStore struct{ *knMemStore }

func (f knFailingStore) SetVector(context.Context, *types.VectorRecord) error {
	return errors.New("backend down")
}

func (f knFailingStore) SimilaritySearch(context.Context, string, string, []float32, int, map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, errors.New("backend down")
}

func (f knFailingStore) DeleteVectorsByPrefix(context.Context, string, string, string) (int, error) {
	return 0, errors.New("backend down")
}

func newFailingKnowledgeService() *knowledge.Service {
	return knowledge.NewService(knFailingStore{newKnMemStore()}, embedding.NewFakeEmbedder())
}

// doRaw posts a raw (possibly malformed) body without JSON-encoding it.
func doRaw(h gin.HandlerFunc, method, path, body string, params gin.Params) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(method, path, bytes.NewReader([]byte(body)))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Params = params
	h(c)
	return w
}

// TestKnowledgeUpsertHandler_InvalidBody returns 400 on malformed JSON.
func TestKnowledgeUpsertHandler_InvalidBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := doRaw(KnowledgeUpsertHandler(newKnowledgeHandlerService()), http.MethodPost, "/knowledge/upsert", "{bad", nil)
	require.Equal(t, http.StatusBadRequest, w.Code)
}

// TestKnowledgeSearchHandler_InvalidBody returns 400 on malformed JSON.
func TestKnowledgeSearchHandler_InvalidBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := doRaw(KnowledgeSearchHandler(newKnowledgeHandlerService()), http.MethodPost, "/knowledge/search", "{bad", nil)
	require.Equal(t, http.StatusBadRequest, w.Code)
}

// TestKnowledgeDeleteHandler_InvalidBody returns 400 on malformed JSON.
func TestKnowledgeDeleteHandler_InvalidBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := doRaw(KnowledgeDeleteSourceHandler(newKnowledgeHandlerService()), http.MethodDelete,
		"/knowledge/source/s", "{bad", gin.Params{{Key: "id", Value: "s"}})
	require.Equal(t, http.StatusBadRequest, w.Code)
}

// TestKnowledgeDeleteHandler_MissingID returns 400 when the path id is empty.
func TestKnowledgeDeleteHandler_MissingID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w, _ := doJSON(t, KnowledgeDeleteSourceHandler(newKnowledgeHandlerService()), http.MethodDelete,
		"/knowledge/source/", gin.Params{{Key: "id", Value: ""}}, gin.H{"scope": gin.H{"tier": "workspace", "workspace_id": "wsA"}})
	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
}

// TestKnowledgeUpsertHandler_InternalError maps a store failure to a 500.
func TestKnowledgeUpsertHandler_InternalError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w, _ := doJSON(t, KnowledgeUpsertHandler(newFailingKnowledgeService()), http.MethodPost,
		"/knowledge/upsert", nil, gin.H{
			"scope":     gin.H{"tier": "workspace", "workspace_id": "wsA"},
			"source_id": "s",
			"chunks":    []gin.H{{"text": "x"}},
		})
	require.Equal(t, http.StatusInternalServerError, w.Code, w.Body.String())
}

// TestKnowledgeSearchHandler_InternalError maps a store failure to a 500.
func TestKnowledgeSearchHandler_InternalError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w, _ := doJSON(t, KnowledgeSearchHandler(newFailingKnowledgeService()), http.MethodPost,
		"/knowledge/search", nil, gin.H{
			"scope": gin.H{"tier": "workspace", "workspace_id": "wsA"},
			"query": "q",
			"top_k": 5,
		})
	require.Equal(t, http.StatusInternalServerError, w.Code, w.Body.String())
}

// TestKnowledgeDeleteHandler_InternalError maps a store failure to a 500.
func TestKnowledgeDeleteHandler_InternalError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w, _ := doJSON(t, KnowledgeDeleteSourceHandler(newFailingKnowledgeService()), http.MethodDelete,
		"/knowledge/source/s", gin.Params{{Key: "id", Value: "s"}},
		gin.H{"scope": gin.H{"tier": "workspace", "workspace_id": "wsA"}})
	require.Equal(t, http.StatusInternalServerError, w.Code, w.Body.String())
}

// TestKnowledgeUpsertHandler_ValidationError400 maps a service validation error
// (empty chunk text) to a 400, not a 500.
func TestKnowledgeUpsertHandler_ValidationError400(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w, _ := doJSON(t, KnowledgeUpsertHandler(newKnowledgeHandlerService()), http.MethodPost,
		"/knowledge/upsert", nil, gin.H{
			"scope":     gin.H{"tier": "workspace", "workspace_id": "wsA"},
			"source_id": "s",
			"chunks":    []gin.H{{"text": "   "}}, // empty text -> service validation error
		})
	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
}

// TestIsKnowledgeValidationError covers the classifier directly across its
// validation fragments and the internal-error fall-through.
func TestIsKnowledgeValidationError(t *testing.T) {
	for _, msg := range []string{
		"workspace_id is required",
		"project_id is required",
		"sender_id is required",
		"invalid tier: bogus",
		"source_id is required",
		"query is required",
		"at least one chunk is required",
		"chunk 2 has empty text",
	} {
		if !isKnowledgeValidationError(errors.New(msg)) {
			t.Errorf("isKnowledgeValidationError(%q) = false, want true", msg)
		}
	}
	if isKnowledgeValidationError(nil) {
		t.Error("isKnowledgeValidationError(nil) must be false")
	}
	if isKnowledgeValidationError(errors.New("store chunk 0: backend down")) {
		t.Error("internal store error must NOT be classified as a validation error")
	}
}
