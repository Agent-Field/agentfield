package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ---------------------------------------------------------------------------
// Set-Get roundtrip
// ---------------------------------------------------------------------------

// TestInvariant_Memory_SetGetRoundtrip verifies that data stored via
// SetMemoryHandler is exactly retrievable via GetMemoryHandler for the same
// scope and key.
func TestInvariant_Memory_SetGetRoundtrip(t *testing.T) {
	storage := newMemoryStorageStub()
	router := gin.New()
	router.POST("/memory/set", SetMemoryHandler(storage))
	router.POST("/memory/get", GetMemoryHandler(storage))

	payload := `{"key":"roundtrip-key","data":{"answer":42}}`
	setReq := httptest.NewRequest(http.MethodPost, "/memory/set", strings.NewReader(payload))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Workflow-ID", "wf-rt-1")
	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code, "Set must succeed")

	getBody := `{"key":"roundtrip-key","scope":"workflow"}`
	getReq := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getReq.Header.Set("X-Workflow-ID", "wf-rt-1")
	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)
	require.Equal(t, http.StatusOK, getResp.Code, "Get must succeed after Set")

	var got types.Memory
	require.NoError(t, json.Unmarshal(getResp.Body.Bytes(), &got))
	assert.Equal(t, "roundtrip-key", got.Key)
	assert.Equal(t, "workflow", got.Scope)
	assert.Equal(t, "wf-rt-1", got.ScopeID)

	var data map[string]interface{}
	require.NoError(t, json.Unmarshal(got.Data, &data))
	assert.Equal(t, float64(42), data["answer"])
}

// ---------------------------------------------------------------------------
// Scope isolation
// ---------------------------------------------------------------------------

// TestInvariant_Memory_ScopeIsolation verifies that setting a key in scope_a
// does NOT make it visible when querying the same key in scope_b.
func TestInvariant_Memory_ScopeIsolation(t *testing.T) {
	storage := newMemoryStorageStub()

	// Insert directly into the stub for scope "workflow" / ID "wf-scope-a".
	dataA, _ := json.Marshal(map[string]any{"origin": "scope-a"})
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "workflow",
		ScopeID: "wf-scope-a",
		Key:     "shared-key",
		Data:    dataA,
	}))

	// Query the same key but from a different scope ID.
	router := gin.New()
	router.POST("/memory/get", GetMemoryHandler(storage))

	getBody := `{"key":"shared-key","scope":"workflow"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(getBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-scope-b") // different scope ID
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNotFound, resp.Code,
		"Key set in scope-a must not be visible when querying scope-b")
}

// ---------------------------------------------------------------------------
// Delete removes
// ---------------------------------------------------------------------------

// TestInvariant_Memory_DeleteMakesKeyNotFound verifies that after
// DeleteMemoryHandler completes, a subsequent GetMemoryHandler returns 404.
func TestInvariant_Memory_DeleteMakesKeyNotFound(t *testing.T) {
	storage := newMemoryStorageStub()

	// Seed a value directly.
	dataVal, _ := json.Marshal("to-be-deleted")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "global",
		ScopeID: "global",
		Key:     "del-key",
		Data:    dataVal,
	}))

	router := gin.New()
	router.POST("/memory/delete", DeleteMemoryHandler(storage))
	router.POST("/memory/get", GetMemoryHandler(storage))

	delBody := `{"key":"del-key","scope":"global"}`
	delReq := httptest.NewRequest(http.MethodPost, "/memory/delete", strings.NewReader(delBody))
	delReq.Header.Set("Content-Type", "application/json")
	delResp := httptest.NewRecorder()
	router.ServeHTTP(delResp, delReq)
	require.Equal(t, http.StatusNoContent, delResp.Code, "Delete must succeed")

	getBody := `{"key":"del-key","scope":"global"}`
	getReq := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)
	assert.Equal(t, http.StatusNotFound, getResp.Code,
		"Key must be absent after Delete")
}

// ---------------------------------------------------------------------------
// Event action invariant
// ---------------------------------------------------------------------------

// TestInvariant_Memory_SetProducesSetEvent verifies that a Set operation
// publishes exactly one event with Action="set".
func TestInvariant_Memory_SetProducesSetEvent(t *testing.T) {
	storage := newMemoryStorageStub()
	router := gin.New()
	router.POST("/memory/set", SetMemoryHandler(storage))

	body := `{"key":"evt-key","data":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/set", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-event-1")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	require.Len(t, storage.events, 1, "Set must produce exactly one stored event")
	assert.Equal(t, "set", storage.events[0].Action,
		"event Action must be 'set' for a Set operation")
	require.Len(t, storage.published, 1)
	assert.Equal(t, "set", storage.published[0].Action)
}

// TestInvariant_Memory_DeleteProducesDeleteEvent verifies that a Delete
// operation publishes exactly one event with Action="delete".
func TestInvariant_Memory_DeleteProducesDeleteEvent(t *testing.T) {
	storage := newMemoryStorageStub()

	dataVal, _ := json.Marshal("value")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "global",
		ScopeID: "global",
		Key:     "del-evt-key",
		Data:    dataVal,
	}))

	router := gin.New()
	router.POST("/memory/delete", DeleteMemoryHandler(storage))

	body := `{"key":"del-evt-key","scope":"global"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/delete", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusNoContent, resp.Code)

	require.Len(t, storage.events, 1, "Delete must produce exactly one stored event")
	assert.Equal(t, "delete", storage.events[0].Action,
		"event Action must be 'delete' for a Delete operation")
	require.Len(t, storage.published, 1)
	assert.Equal(t, "delete", storage.published[0].Action)
}

// ---------------------------------------------------------------------------
// Hierarchical search order
// ---------------------------------------------------------------------------

// TestInvariant_Memory_HierarchicalSearchOrder verifies that GetMemoryHandler
// (no explicit scope) searches scopes in the order:
//   workflow → session → actor → global
//
// Concretely: if a key exists in both session and global, the session result
// must be returned (not global), because session comes first.
func TestInvariant_Memory_HierarchicalSearchOrder(t *testing.T) {
	storage := newMemoryStorageStub()

	// Seed the same key at session and global level with distinct values.
	sessionData, _ := json.Marshal("from-session")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "session",
		ScopeID: "sess-1",
		Key:     "hier-key",
		Data:    sessionData,
	}))

	globalData, _ := json.Marshal("from-global")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "global",
		ScopeID: "global",
		Key:     "hier-key",
		Data:    globalData,
	}))

	router := gin.New()
	router.POST("/memory/get", GetMemoryHandler(storage))

	// No explicit scope — hierarchical lookup should fire.
	// Provide session header so scope "session" has an ID to look up.
	body := `{"key":"hier-key"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Session-ID", "sess-1")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	var mem types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &mem))
	assert.Equal(t, "session", mem.Scope,
		"hierarchical search must return session-scope result when both session and global exist")
	assert.Equal(t, "sess-1", mem.ScopeID)
}

// TestInvariant_Memory_HierarchicalSearchFallsBackToGlobal verifies that when
// a key exists only in global, the hierarchical lookup returns the global entry.
func TestInvariant_Memory_HierarchicalSearchFallsBackToGlobal(t *testing.T) {
	storage := newMemoryStorageStub()

	globalData, _ := json.Marshal("only-global")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "global",
		ScopeID: "global",
		Key:     "fallback-key",
		Data:    globalData,
	}))

	router := gin.New()
	router.POST("/memory/get", GetMemoryHandler(storage))

	// No workflow/session/actor headers → only global will match.
	body := `{"key":"fallback-key"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	var mem types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &mem))
	assert.Equal(t, "global", mem.Scope)
}

// TestInvariant_Memory_HierarchicalSearchWorkflowBeatsSession verifies that
// when both workflow and session have the key, workflow (higher priority) wins.
func TestInvariant_Memory_HierarchicalSearchWorkflowBeatsSession(t *testing.T) {
	storage := newMemoryStorageStub()

	wfData, _ := json.Marshal("from-workflow")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "workflow",
		ScopeID: "wf-prio",
		Key:     "prio-key",
		Data:    wfData,
	}))

	sessData, _ := json.Marshal("from-session")
	require.NoError(t, storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "session",
		ScopeID: "sess-prio",
		Key:     "prio-key",
		Data:    sessData,
	}))

	router := gin.New()
	router.POST("/memory/get", GetMemoryHandler(storage))

	body := `{"key":"prio-key"}`
	req := httptest.NewRequest(http.MethodPost, "/memory/get", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-prio")
	req.Header.Set("X-Session-ID", "sess-prio")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	var mem types.Memory
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &mem))
	assert.Equal(t, "workflow", mem.Scope,
		"workflow scope must take priority over session in hierarchical search")
}
