package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type vectorStorageStub struct {
	mu            sync.Mutex
	setRecord     *types.VectorRecord
	setErr        error
	searchResults []*types.VectorSearchResult
	searchErr     error
	searchScope   string
	searchScopeID string
	searchQuery   []float32
	searchTopK    int
	searchFilters map[string]interface{}
	// GetVector fields
	getResult *types.VectorRecord
	getErr    error
	getScope  string
	getScopeID string
	getKey    string
	// DeleteVector fields
	deleteErr     error
	deleteScope   string
	deleteScopeID string
	deleteKey     string
}

func (v *vectorStorageStub) SetMemory(ctx context.Context, memory *types.Memory) error {
	return nil
}

func (v *vectorStorageStub) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, errors.New("not implemented")
}

func (v *vectorStorageStub) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	return nil
}

func (v *vectorStorageStub) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}

func (v *vectorStorageStub) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}

func (v *vectorStorageStub) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}

func (v *vectorStorageStub) SetVector(ctx context.Context, record *types.VectorRecord) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.setErr != nil {
		return v.setErr
	}
	v.setRecord = &types.VectorRecord{
		Scope:     record.Scope,
		ScopeID:   record.ScopeID,
		Key:       record.Key,
		Embedding: append([]float32(nil), record.Embedding...),
		Metadata:  cloneMetadata(record.Metadata),
	}
	return nil
}

func (v *vectorStorageStub) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.getScope = scope
	v.getScopeID = scopeID
	v.getKey = key
	if v.getErr != nil {
		return nil, v.getErr
	}
	return v.getResult, nil
}

func (v *vectorStorageStub) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.deleteScope = scope
	v.deleteScopeID = scopeID
	v.deleteKey = key
	if v.deleteErr != nil {
		return v.deleteErr
	}
	return nil
}

func (v *vectorStorageStub) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}

func (v *vectorStorageStub) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.searchScope = scope
	v.searchScopeID = scopeID
	v.searchQuery = append([]float32(nil), queryEmbedding...)
	v.searchTopK = topK
	v.searchFilters = cloneMetadata(filters)
	if v.searchErr != nil {
		return nil, v.searchErr
	}
	return v.searchResults, nil
}

func cloneMetadata(input map[string]interface{}) map[string]interface{} {
	if input == nil {
		return nil
	}
	copied := make(map[string]interface{}, len(input))
	for key, value := range input {
		copied[key] = value
	}
	return copied
}

func TestSetVectorHandler_StoresVectorWithMetadata(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.POST("/vectors/set", SetVectorHandler(storage))

	body := `{"key":"vec-1","embedding":[0.1,0.2],"metadata":{"source":"doc"},"scope":"session"}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Session-ID", "session-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.NotNil(t, storage.setRecord)
	require.Equal(t, "session", storage.setRecord.Scope)
	require.Equal(t, "session-1", storage.setRecord.ScopeID)
	require.Equal(t, "vec-1", storage.setRecord.Key)
	require.Equal(t, []float32{0.1, 0.2}, storage.setRecord.Embedding)
	require.Equal(t, map[string]interface{}{"source": "doc"}, storage.setRecord.Metadata)

	var payload struct {
		Key      string                 `json:"key"`
		Scope    string                 `json:"scope"`
		ScopeID  string                 `json:"scope_id"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "vec-1", payload.Key)
	require.Equal(t, "session", payload.Scope)
	require.Equal(t, "session-1", payload.ScopeID)
	require.Equal(t, map[string]interface{}{"source": "doc"}, payload.Metadata)
}

func TestSetVectorHandler_EmptyEmbedding(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.POST("/vectors/set", SetVectorHandler(storage))

	body := `{"key":"vec-empty","embedding":[]}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "embedding cannot be empty")
}

func TestSetVectorHandler_InvalidRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.POST("/vectors/set", SetVectorHandler(storage))

	body := `{}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "invalid_request")
}

func TestSetVectorHandler_StorageError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{setErr: errors.New("store failed")}
	router := gin.New()
	router.POST("/vectors/set", SetVectorHandler(storage))

	body := `{"key":"vec-err","embedding":[0.4]}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	require.Contains(t, resp.Body.String(), "storage_error")
}

func TestSimilaritySearchHandler_DefaultTopKAndFilters(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		searchResults: []*types.VectorSearchResult{
			{
				Scope:    "actor",
				ScopeID:  "actor-1",
				Key:      "vec-1",
				Score:    0.9,
				Metadata: map[string]interface{}{"source": "doc"},
			},
		},
	}
	router := gin.New()
	router.POST("/vectors/search", SimilaritySearchHandler(storage))

	body := `{"query_embedding":[0.1,0.2],"top_k":0,"filters":{"source":"doc"}}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Actor-ID", "actor-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "actor", storage.searchScope)
	require.Equal(t, "actor-1", storage.searchScopeID)
	require.Equal(t, []float32{0.1, 0.2}, storage.searchQuery)
	require.Equal(t, 10, storage.searchTopK)
	require.Equal(t, map[string]interface{}{"source": "doc"}, storage.searchFilters)

	var results []types.VectorSearchResult
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &results))
	require.Len(t, results, 1)
	require.Equal(t, "vec-1", results[0].Key)
	require.Equal(t, map[string]interface{}{"source": "doc"}, results[0].Metadata)
}

func TestSimilaritySearchHandler_CustomTopKNoFilters(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.POST("/vectors/search", SimilaritySearchHandler(storage))

	body := `{"query_embedding":[0.3,0.4],"top_k":3}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-9")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "workflow", storage.searchScope)
	require.Equal(t, "wf-9", storage.searchScopeID)
	require.Equal(t, 3, storage.searchTopK)
	require.Nil(t, storage.searchFilters)
}

func TestSimilaritySearchHandler_EmptyEmbedding(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.POST("/vectors/search", SimilaritySearchHandler(storage))

	body := `{"query_embedding":[]}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "query_embedding cannot be empty")
}

func TestSimilaritySearchHandler_StorageError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{searchErr: errors.New("search failed")}
	router := gin.New()
	router.POST("/vectors/search", SimilaritySearchHandler(storage))

	body := `{"query_embedding":[0.1]}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	require.Contains(t, resp.Body.String(), "storage_error")
}

// GetVectorHandler tests

func TestGetVectorHandler_ReturnsVectorWithMetadata(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		getResult: &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "session-123",
			Key:       "vec-key",
			Embedding: []float32{0.1, 0.2, 0.3},
			Metadata:  map[string]interface{}{"source": "test"},
		},
	}
	router := gin.New()
	router.GET("/vectors/:key", GetVectorHandler(storage))

	req := httptest.NewRequest(http.MethodGet, "/vectors/vec-key?scope=session", nil)
	req.Header.Set("X-Session-ID", "session-123")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "session", storage.getScope)
	require.Equal(t, "session-123", storage.getScopeID)
	require.Equal(t, "vec-key", storage.getKey)

	var result types.VectorRecord
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	require.Equal(t, "vec-key", result.Key)
	require.Equal(t, "session", result.Scope)
	require.Equal(t, []float32{0.1, 0.2, 0.3}, result.Embedding)
	require.Equal(t, map[string]interface{}{"source": "test"}, result.Metadata)
}

func TestGetVectorHandler_NotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		getResult: nil, // No vector found
	}
	router := gin.New()
	router.GET("/vectors/:key", GetVectorHandler(storage))

	req := httptest.NewRequest(http.MethodGet, "/vectors/nonexistent-key", nil)
	req.Header.Set("X-Actor-ID", "actor-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusNotFound, resp.Code)
	require.Contains(t, resp.Body.String(), "not_found")
	require.Contains(t, resp.Body.String(), "vector not found")
}

func TestGetVectorHandler_StorageError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		getErr: errors.New("database connection lost"),
	}
	router := gin.New()
	router.GET("/vectors/:key", GetVectorHandler(storage))

	req := httptest.NewRequest(http.MethodGet, "/vectors/vec-key", nil)

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	require.Contains(t, resp.Body.String(), "storage_error")
}

func TestGetVectorHandler_DefaultScope(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		getResult: &types.VectorRecord{
			Scope:     "actor",
			ScopeID:   "actor-xyz",
			Key:       "my-vec",
			Embedding: []float32{0.5},
		},
	}
	router := gin.New()
	router.GET("/vectors/:key", GetVectorHandler(storage))

	// No scope query param - should use default scope resolution from headers
	req := httptest.NewRequest(http.MethodGet, "/vectors/my-vec", nil)
	req.Header.Set("X-Actor-ID", "actor-xyz")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "actor", storage.getScope)
	require.Equal(t, "actor-xyz", storage.getScopeID)
	require.Equal(t, "my-vec", storage.getKey)
}

// DeleteVectorHandler tests

func TestDeleteVectorHandler_RESTfulDelete(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	router.DELETE("/vectors/:key", DeleteVectorHandler(storage))

	req := httptest.NewRequest(http.MethodDelete, "/vectors/vec-to-delete?scope=workflow", nil)
	req.Header.Set("X-Workflow-ID", "wf-42")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusNoContent, resp.Code)
	require.Equal(t, "workflow", storage.deleteScope)
	require.Equal(t, "wf-42", storage.deleteScopeID)
	require.Equal(t, "vec-to-delete", storage.deleteKey)
}

func TestDeleteVectorHandler_BackwardCompatibilityWithBody(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	// Register on POST path for backward compatibility test
	router.POST("/vectors/delete", DeleteVectorHandler(storage))

	body := `{"key":"legacy-vec","scope":"session"}`
	req := httptest.NewRequest(http.MethodPost, "/vectors/delete", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Session-ID", "session-old")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusNoContent, resp.Code)
	require.Equal(t, "session", storage.deleteScope)
	require.Equal(t, "session-old", storage.deleteScopeID)
	require.Equal(t, "legacy-vec", storage.deleteKey)
}

func TestDeleteVectorHandler_StorageError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{
		deleteErr: errors.New("delete failed"),
	}
	router := gin.New()
	router.DELETE("/vectors/:key", DeleteVectorHandler(storage))

	req := httptest.NewRequest(http.MethodDelete, "/vectors/vec-fail", nil)
	req.Header.Set("X-Actor-ID", "actor-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	require.Contains(t, resp.Body.String(), "storage_error")
}

func TestDeleteVectorHandler_MissingKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorStorageStub{}
	router := gin.New()
	// Register handler without path parameter to test missing key validation
	router.DELETE("/vectors/", DeleteVectorHandler(storage))

	req := httptest.NewRequest(http.MethodDelete, "/vectors/", nil)

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "key is required")
}
