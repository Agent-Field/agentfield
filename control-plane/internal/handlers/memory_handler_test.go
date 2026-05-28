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

type memoryStorageStub struct {
	mu        sync.Mutex
	store     map[string]*types.Memory
	events    []*types.MemoryChangeEvent
	published []types.MemoryChangeEvent
	setErr    error
	eventErr  error
}

func newMemoryStorageStub() *memoryStorageStub {
	return &memoryStorageStub{store: make(map[string]*types.Memory)}
}

func (m *memoryStorageStub) composite(scope, scopeID, key string) string {
	return scope + "|" + scopeID + "|" + key
}

func (m *memoryStorageStub) SetMemory(ctx context.Context, memory *types.Memory) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.setErr != nil {
		return m.setErr
	}
	k := m.composite(memory.Scope, memory.ScopeID, memory.Key)
	m.store[k] = memory
	return nil
}

func (m *memoryStorageStub) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	k := m.composite(scope, scopeID, key)
	mem, ok := m.store[k]
	if !ok {
		return nil, errors.New("not found")
	}
	return mem, nil
}

func (m *memoryStorageStub) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	k := m.composite(scope, scopeID, key)
	if _, ok := m.store[k]; !ok {
		return errors.New("not found")
	}
	delete(m.store, k)
	return nil
}

func (m *memoryStorageStub) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var result []*types.Memory
	prefix := scope + "|" + scopeID + "|"
	for k, mem := range m.store {
		if strings.HasPrefix(k, prefix) {
			result = append(result, mem)
		}
	}
	return result, nil
}

func (m *memoryStorageStub) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.eventErr != nil {
		return m.eventErr
	}
	m.events = append(m.events, event)
	return nil
}

func (m *memoryStorageStub) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.published = append(m.published, event)
	return nil
}

func (m *memoryStorageStub) SetVector(ctx context.Context, record *types.VectorRecord) error {
	return nil
}

func (m *memoryStorageStub) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	return nil, nil
}

func (m *memoryStorageStub) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	return nil
}

func (m *memoryStorageStub) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}

func (m *memoryStorageStub) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return []*types.VectorSearchResult{}, nil
}

func TestSetMemoryHandler_StoresMemoryAndEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newMemoryStorageStub()
	router := gin.New()
	router.POST("/memory/set", SetMemoryHandler(storage))

	body := `{"key":"alpha","data":{"value":1}}`
	req := httptest.NewRequest(http.MethodPost, "/memory/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-123")
	req.Header.Set("X-Agent-Node-ID", "agent-1")
	req.Header.Set("X-Actor-ID", "user-42")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var memory types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &memory))
	require.Equal(t, "alpha", memory.Key)
	require.Equal(t, "workflow", memory.Scope)
	require.Equal(t, "wf-123", memory.ScopeID)

	stored, err := storage.GetMemory(context.Background(), "workflow", "wf-123", "alpha")
	require.NoError(t, err)
	require.Equal(t, "alpha", stored.Key)

	require.Len(t, storage.events, 1)
	require.Equal(t, "set", storage.events[0].Action)
	require.Equal(t, "agent-1", storage.events[0].Metadata.AgentID)
	require.Equal(t, "user-42", storage.events[0].Metadata.ActorID)
	require.Equal(t, "wf-123", storage.events[0].Metadata.WorkflowID)
	require.Len(t, storage.published, 1)
	require.Equal(t, "set", storage.published[0].Action)
}

func TestGetMemoryHandler_HierarchicalLookup(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newMemoryStorageStub()
	memData, _ := json.Marshal(map[string]any{"v": 99})
	_ = storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "session",
		ScopeID: "session-1",
		Key:     "beta",
		Data:    memData,
	})

	router := gin.New()
	router.POST("/memory/get", GetMemoryHandler(storage))

	body := `{"key":"beta"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Session-ID", "session-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var memory types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &memory))
	require.Equal(t, "session", memory.Scope)
	require.Equal(t, "session-1", memory.ScopeID)
}

func TestDeleteMemoryHandler_RemovesEntry(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newMemoryStorageStub()
	_ = storage.SetMemory(context.Background(), &types.Memory{Scope: "global", ScopeID: "global", Key: "gamma", Data: json.RawMessage(`"value"`)})

	router := gin.New()
	router.POST("/memory/delete", DeleteMemoryHandler(storage))

	body := `{"key":"gamma","scope":"global"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/delete", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-2")
	req.Header.Set("X-Actor-ID", "user-99")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusNoContent, resp.Code)

	_, err := storage.GetMemory(context.Background(), "global", "global", "gamma")
	require.Error(t, err)
	require.Len(t, storage.events, 1)
	require.Equal(t, "delete", storage.events[0].Action)
	require.Equal(t, "agent-2", storage.events[0].Metadata.AgentID)
	require.Equal(t, "user-99", storage.events[0].Metadata.ActorID)
	require.Len(t, storage.published, 1)
	require.Equal(t, "delete", storage.published[0].Action)
}

func TestListMemoryHandler_ReturnsMemories(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newMemoryStorageStub()
	_ = storage.SetMemory(context.Background(), &types.Memory{Scope: "workflow", ScopeID: "wf-1", Key: "k1", Data: json.RawMessage(`1`)})
	_ = storage.SetMemory(context.Background(), &types.Memory{Scope: "workflow", ScopeID: "wf-1", Key: "k2", Data: json.RawMessage(`2`)})

	router := gin.New()
	router.GET("/memory/list", ListMemoryHandler(storage))

	req := httptest.NewRequest(http.MethodGet, "/memory/list?scope=workflow", nil)
	req.Header.Set("X-Workflow-ID", "wf-1")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var memories []types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &memories))
	require.Len(t, memories, 2)
}
