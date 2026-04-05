package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockMemoryStorage implements the MemoryStorage interface for testing.
type MockMemoryStorage struct {
	mock.Mock
}

func (m *MockMemoryStorage) SetMemory(ctx context.Context, memory *types.Memory) error {
	args := m.Called(ctx, memory)
	return args.Error(0)
}

func (m *MockMemoryStorage) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	args := m.Called(ctx, scope, scopeID, key)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.Memory), args.Error(1)
}

func (m *MockMemoryStorage) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	args := m.Called(ctx, scope, scopeID, key)
	return args.Error(0)
}

func (m *MockMemoryStorage) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	args := m.Called(ctx, scope, scopeID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.Memory), args.Error(1)
}

func (m *MockMemoryStorage) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockMemoryStorage) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockMemoryStorage) SetVector(ctx context.Context, record *types.VectorRecord) error {
	return nil
}

func (m *MockMemoryStorage) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	return nil, nil
}

func (m *MockMemoryStorage) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	return nil
}

func (m *MockMemoryStorage) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}

func (m *MockMemoryStorage) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, nil
}

func setupGinRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	return gin.New()
}

func TestSetMemoryHandler(t *testing.T) {
	tests := []struct {
		name           string
		body           interface{}
		headers        map[string]string
		setupMock      func(m *MockMemoryStorage)
		expectedStatus int
		checkBody      func(t *testing.T, body []byte)
	}{
		{
			name: "successful set with workflow scope",
			body: SetMemoryRequest{Key: "test-key", Data: "test-value"},
			headers: map[string]string{
				"X-Workflow-ID": "wf-123",
			},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, "workflow", "wf-123", "test-key").
					Return(nil, errors.New("not found"))
				m.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
					Return(nil)
				m.On("StoreEvent", mock.Anything, mock.AnythingOfType("*types.MemoryChangeEvent")).
					Return(nil)
				m.On("PublishMemoryChange", mock.Anything, mock.AnythingOfType("types.MemoryChangeEvent")).
					Return(nil)
			},
			expectedStatus: http.StatusOK,
			checkBody: func(t *testing.T, body []byte) {
				var mem types.Memory
				err := json.Unmarshal(body, &mem)
				assert.NoError(t, err)
				assert.Equal(t, "test-key", mem.Key)
				assert.Equal(t, "workflow", mem.Scope)
				assert.Equal(t, "wf-123", mem.ScopeID)
			},
		},
		{
			name: "successful set with global scope (no headers)",
			body: SetMemoryRequest{Key: "global-key", Data: map[string]string{"hello": "world"}},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, "global", "global", "global-key").
					Return(nil, errors.New("not found"))
				m.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
					Return(nil)
				m.On("StoreEvent", mock.Anything, mock.AnythingOfType("*types.MemoryChangeEvent")).
					Return(nil)
				m.On("PublishMemoryChange", mock.Anything, mock.AnythingOfType("types.MemoryChangeEvent")).
					Return(nil)
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:           "invalid JSON body",
			body:           "not-json{{{",
			expectedStatus: http.StatusBadRequest,
			checkBody: func(t *testing.T, body []byte) {
				var errResp ErrorResponse
				err := json.Unmarshal(body, &errResp)
				assert.NoError(t, err)
				assert.Equal(t, "invalid_request", errResp.Error)
			},
		},
		{
			name: "storage error on set",
			body: SetMemoryRequest{Key: "fail-key", Data: "value"},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
					Return(nil, errors.New("not found"))
				m.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
					Return(errors.New("database unavailable"))
			},
			expectedStatus: http.StatusInternalServerError,
			checkBody: func(t *testing.T, body []byte) {
				var errResp ErrorResponse
				err := json.Unmarshal(body, &errResp)
				assert.NoError(t, err)
				assert.Equal(t, "storage_error", errResp.Error)
			},
		},
		{
			name: "event storage failure does not fail request and skips publish",
			body: SetMemoryRequest{Key: "event-fail-key", Data: "value"},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
					Return(nil, errors.New("not found"))
				m.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
					Return(nil)
				m.On("StoreEvent", mock.Anything, mock.AnythingOfType("*types.MemoryChangeEvent")).
					Return(errors.New("event store down"))
				// PublishMemoryChange should NOT be called when StoreEvent fails
				// (the source uses else-if: only publishes if StoreEvent succeeds)
			},
			expectedStatus: http.StatusOK,
			checkBody: func(t *testing.T, body []byte) {
				// Verify response is still valid memory object
				var mem map[string]interface{}
				err := json.Unmarshal(body, &mem)
				assert.NoError(t, err)
				assert.Equal(t, "event-fail-key", mem["key"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStorage := new(MockMemoryStorage)
			if tt.setupMock != nil {
				tt.setupMock(mockStorage)
			}

			router := setupGinRouter()
			router.POST("/memory/set", SetMemoryHandler(mockStorage))

			var bodyBytes []byte
			switch v := tt.body.(type) {
			case string:
				bodyBytes = []byte(v)
			default:
				var err error
				bodyBytes, err = json.Marshal(v)
				assert.NoError(t, err)
			}

			req, _ := http.NewRequest("POST", "/memory/set", bytes.NewBuffer(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
			if tt.checkBody != nil {
				tt.checkBody(t, w.Body.Bytes())
			}
		})
	}
}

func TestGetMemoryHandler(t *testing.T) {
	tests := []struct {
		name           string
		body           interface{}
		headers        map[string]string
		setupMock      func(m *MockMemoryStorage)
		expectedStatus int
	}{
		{
			name: "get with explicit scope",
			body: GetMemoryRequest{Key: "my-key", Scope: strPtr("workflow")},
			headers: map[string]string{
				"X-Workflow-ID": "wf-456",
			},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, "workflow", "wf-456", "my-key").
					Return(&types.Memory{
						Key:     "my-key",
						Scope:   "workflow",
						ScopeID: "wf-456",
						Data:    json.RawMessage(`"hello"`),
					}, nil)
			},
			expectedStatus: http.StatusOK,
		},
		{
			name: "get with hierarchical search finds in session",
			body: GetMemoryRequest{Key: "my-key"},
			headers: map[string]string{
				"X-Session-ID": "sess-789",
			},
			setupMock: func(m *MockMemoryStorage) {
				// Workflow scope — no workflow ID header, so scopeID is empty → skip
				// Session scope — has header, found
				m.On("GetMemory", mock.Anything, "session", "sess-789", "my-key").
					Return(&types.Memory{
						Key:     "my-key",
						Scope:   "session",
						ScopeID: "sess-789",
						Data:    json.RawMessage(`"found"`),
					}, nil)
			},
			expectedStatus: http.StatusOK,
		},
		{
			name: "get not found in any scope",
			body: GetMemoryRequest{Key: "missing-key"},
			setupMock: func(m *MockMemoryStorage) {
				// Only global scope will be checked (no headers)
				m.On("GetMemory", mock.Anything, "global", "global", "missing-key").
					Return(nil, errors.New("not found"))
			},
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "invalid body",
			body:           "bad",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStorage := new(MockMemoryStorage)
			if tt.setupMock != nil {
				tt.setupMock(mockStorage)
			}

			router := setupGinRouter()
			router.POST("/memory/get", GetMemoryHandler(mockStorage))

			var bodyBytes []byte
			switch v := tt.body.(type) {
			case string:
				bodyBytes = []byte(v)
			default:
				var err error
				bodyBytes, err = json.Marshal(v)
				assert.NoError(t, err)
			}

			req, _ := http.NewRequest("POST", "/memory/get", bytes.NewBuffer(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
		})
	}
}

func TestDeleteMemoryHandler(t *testing.T) {
	tests := []struct {
		name           string
		body           interface{}
		headers        map[string]string
		setupMock      func(m *MockMemoryStorage)
		expectedStatus int
	}{
		{
			name: "successful delete",
			body: GetMemoryRequest{Key: "del-key"},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, "global", "global", "del-key").
					Return(&types.Memory{Data: json.RawMessage(`"old"`)}, nil)
				m.On("DeleteMemory", mock.Anything, "global", "global", "del-key").
					Return(nil)
				m.On("StoreEvent", mock.Anything, mock.AnythingOfType("*types.MemoryChangeEvent")).
					Return(nil)
				m.On("PublishMemoryChange", mock.Anything, mock.AnythingOfType("types.MemoryChangeEvent")).
					Return(nil)
			},
			expectedStatus: http.StatusNoContent,
		},
		{
			name: "delete not found",
			body: GetMemoryRequest{Key: "no-key"},
			setupMock: func(m *MockMemoryStorage) {
				m.On("GetMemory", mock.Anything, "global", "global", "no-key").
					Return(nil, errors.New("not found"))
				m.On("DeleteMemory", mock.Anything, "global", "global", "no-key").
					Return(errors.New("not found"))
			},
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "invalid body",
			body:           "bad",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStorage := new(MockMemoryStorage)
			if tt.setupMock != nil {
				tt.setupMock(mockStorage)
			}

			router := setupGinRouter()
			router.POST("/memory/delete", DeleteMemoryHandler(mockStorage))

			var bodyBytes []byte
			switch v := tt.body.(type) {
			case string:
				bodyBytes = []byte(v)
			default:
				var err error
				bodyBytes, err = json.Marshal(v)
				assert.NoError(t, err)
			}

			req, _ := http.NewRequest("POST", "/memory/delete", bytes.NewBuffer(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
		})
	}
}

func TestListMemoryHandler(t *testing.T) {
	tests := []struct {
		name           string
		query          string
		headers        map[string]string
		setupMock      func(m *MockMemoryStorage)
		expectedStatus int
	}{
		{
			name:  "successful list",
			query: "scope=global",
			setupMock: func(m *MockMemoryStorage) {
				m.On("ListMemory", mock.Anything, "global", "global").
					Return([]*types.Memory{
						{Key: "key1", Scope: "global", ScopeID: "global"},
						{Key: "key2", Scope: "global", ScopeID: "global"},
					}, nil)
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:  "list with workflow scope",
			query: "scope=workflow",
			headers: map[string]string{
				"X-Workflow-ID": "wf-list",
			},
			setupMock: func(m *MockMemoryStorage) {
				m.On("ListMemory", mock.Anything, "workflow", "wf-list").
					Return([]*types.Memory{}, nil)
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:           "missing scope parameter",
			query:          "",
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:  "storage error",
			query: "scope=global",
			setupMock: func(m *MockMemoryStorage) {
				m.On("ListMemory", mock.Anything, "global", "global").
					Return(nil, errors.New("db error"))
			},
			expectedStatus: http.StatusInternalServerError,
		},
		{
			name:  "nil result returns empty array",
			query: "scope=global",
			setupMock: func(m *MockMemoryStorage) {
				m.On("ListMemory", mock.Anything, "global", "global").
					Return(nil, nil)
			},
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStorage := new(MockMemoryStorage)
			if tt.setupMock != nil {
				tt.setupMock(mockStorage)
			}

			router := setupGinRouter()
			router.GET("/memory/list", ListMemoryHandler(mockStorage))

			url := "/memory/list"
			if tt.query != "" {
				url += "?" + tt.query
			}

			req, _ := http.NewRequest("GET", url, nil)
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
		})
	}
}

func TestResolveScope(t *testing.T) {
	tests := []struct {
		name          string
		explicitScope *string
		headers       map[string]string
		wantScope     string
		wantScopeID   string
	}{
		{
			name:          "explicit workflow scope",
			explicitScope: strPtr("workflow"),
			headers:       map[string]string{"X-Workflow-ID": "wf-abc"},
			wantScope:     "workflow",
			wantScopeID:   "wf-abc",
		},
		{
			name:        "implicit from workflow header",
			headers:     map[string]string{"X-Workflow-ID": "wf-def"},
			wantScope:   "workflow",
			wantScopeID: "wf-def",
		},
		{
			name:        "implicit from session header",
			headers:     map[string]string{"X-Session-ID": "sess-123"},
			wantScope:   "session",
			wantScopeID: "sess-123",
		},
		{
			name:        "implicit from actor header",
			headers:     map[string]string{"X-Actor-ID": "actor-1"},
			wantScope:   "actor",
			wantScopeID: "actor-1",
		},
		{
			name:        "no headers defaults to global",
			wantScope:   "global",
			wantScopeID: "global",
		},
		{
			name: "workflow takes precedence over session",
			headers: map[string]string{
				"X-Workflow-ID": "wf-pri",
				"X-Session-ID":  "sess-pri",
			},
			wantScope:   "workflow",
			wantScopeID: "wf-pri",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("GET", "/", nil)
			for k, v := range tt.headers {
				c.Request.Header.Set(k, v)
			}

			gotScope, gotScopeID := resolveScope(c, tt.explicitScope)
			assert.Equal(t, tt.wantScope, gotScope)
			assert.Equal(t, tt.wantScopeID, gotScopeID)
		})
	}
}

func TestGetScopeID(t *testing.T) {
	tests := []struct {
		name    string
		scope   string
		headers map[string]string
		want    string
	}{
		{"workflow scope", "workflow", map[string]string{"X-Workflow-ID": "wf-1"}, "wf-1"},
		{"session scope", "session", map[string]string{"X-Session-ID": "s-1"}, "s-1"},
		{"actor scope", "actor", map[string]string{"X-Actor-ID": "a-1"}, "a-1"},
		{"global scope", "global", nil, "global"},
		{"unknown scope", "unknown", nil, ""},
		{"missing header", "workflow", nil, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("GET", "/", nil)
			for k, v := range tt.headers {
				c.Request.Header.Set(k, v)
			}

			got := getScopeID(c, tt.scope)
			assert.Equal(t, tt.want, got)
		})
	}
}

// marshalDataWithLogging is defined in memory.go but unexported — we can't test it directly
// but we test it indirectly through SetMemoryHandler above.

// Helper function
func strPtr(s string) *string {
	return &s
}

// Verify mocks are used
func TestMockMemoryStorageImplementsInterface(t *testing.T) {
	var _ MemoryStorage = (*MockMemoryStorage)(nil)
}

// Test that SetMemoryHandler captures previous data for events
func TestSetMemoryHandler_CapturesPreviousData(t *testing.T) {
	mockStorage := new(MockMemoryStorage)

	existingMemory := &types.Memory{
		Key:     "update-key",
		Scope:   "global",
		ScopeID: "global",
		Data:    json.RawMessage(`"old-value"`),
	}

	mockStorage.On("GetMemory", mock.Anything, "global", "global", "update-key").
		Return(existingMemory, nil)
	mockStorage.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
		Return(nil)
	mockStorage.On("StoreEvent", mock.Anything, mock.MatchedBy(func(event *types.MemoryChangeEvent) bool {
		return event.Action == "set" && string(event.PreviousData) == `"old-value"`
	})).Return(nil)
	mockStorage.On("PublishMemoryChange", mock.Anything, mock.AnythingOfType("types.MemoryChangeEvent")).
		Return(nil)

	router := setupGinRouter()
	router.POST("/memory/set", SetMemoryHandler(mockStorage))

	body, _ := json.Marshal(SetMemoryRequest{Key: "update-key", Data: "new-value"})
	req, _ := http.NewRequest("POST", "/memory/set", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	mockStorage.AssertExpectations(t)
}

// Test memory events carry correct metadata from headers
func TestSetMemoryHandler_EventMetadata(t *testing.T) {
	mockStorage := new(MockMemoryStorage)

	mockStorage.On("GetMemory", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil, errors.New("not found"))
	mockStorage.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
		Return(nil)
	mockStorage.On("StoreEvent", mock.Anything, mock.MatchedBy(func(event *types.MemoryChangeEvent) bool {
		return event.Metadata.AgentID == "agent-node-1" &&
			event.Metadata.ActorID == "actor-42" &&
			event.Metadata.WorkflowID == "wf-meta"
	})).Return(nil)
	mockStorage.On("PublishMemoryChange", mock.Anything, mock.AnythingOfType("types.MemoryChangeEvent")).
		Return(nil)

	router := setupGinRouter()
	router.POST("/memory/set", SetMemoryHandler(mockStorage))

	body, _ := json.Marshal(SetMemoryRequest{Key: "meta-key", Data: "value"})
	req, _ := http.NewRequest("POST", "/memory/set", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-meta")
	req.Header.Set("X-Agent-Node-ID", "agent-node-1")
	req.Header.Set("X-Actor-ID", "actor-42")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	mockStorage.AssertExpectations(t)
}

// TestSetMemoryHandler_StoreEventFailure_SkipsPublish verifies that when
// StoreEvent fails, PublishMemoryChange is NOT called. This tests the else-if
// branching in the source (lines 147-149 of memory.go). A regression that
// changed "else if" to two separate "if" blocks would be caught here.
func TestSetMemoryHandler_StoreEventFailure_SkipsPublish(t *testing.T) {
	mockStorage := new(MockMemoryStorage)

	mockStorage.On("GetMemory", mock.Anything, "global", "global", "skip-publish-key").
		Return(nil, errors.New("not found"))
	mockStorage.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).
		Return(nil)
	mockStorage.On("StoreEvent", mock.Anything, mock.AnythingOfType("*types.MemoryChangeEvent")).
		Return(errors.New("event store unavailable"))
	// Do NOT register PublishMemoryChange — it should never be called

	router := setupGinRouter()
	router.POST("/memory/set", SetMemoryHandler(mockStorage))

	body, _ := json.Marshal(SetMemoryRequest{Key: "skip-publish-key", Data: "value"})
	req, _ := http.NewRequest("POST", "/memory/set", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	mockStorage.AssertExpectations(t)
	// This is the critical assertion: PublishMemoryChange must NOT have been called
	mockStorage.AssertNotCalled(t, "PublishMemoryChange", mock.Anything, mock.Anything)
}

// TestDeleteMemoryHandler_PublishesEvent verifies delete handler calls both
// StoreEvent and PublishMemoryChange, not just DeleteMemory.
func TestDeleteMemoryHandler_PublishesEvent(t *testing.T) {
	mockStorage := new(MockMemoryStorage)

	mockStorage.On("GetMemory", mock.Anything, "global", "global", "del-with-event").
		Return(&types.Memory{
			Key:  "del-with-event",
			Data: json.RawMessage(`"old-data"`),
		}, nil)
	mockStorage.On("DeleteMemory", mock.Anything, "global", "global", "del-with-event").
		Return(nil)
	mockStorage.On("StoreEvent", mock.Anything, mock.MatchedBy(func(event *types.MemoryChangeEvent) bool {
		return event.Action == "delete" &&
			event.Key == "del-with-event" &&
			event.Data == nil &&
			string(event.PreviousData) == `"old-data"`
	})).Return(nil)
	mockStorage.On("PublishMemoryChange", mock.Anything, mock.MatchedBy(func(event types.MemoryChangeEvent) bool {
		return event.Action == "delete" && event.Key == "del-with-event"
	})).Return(nil)

	router := setupGinRouter()
	router.POST("/memory/delete", DeleteMemoryHandler(mockStorage))

	body, _ := json.Marshal(GetMemoryRequest{Key: "del-with-event"})
	req, _ := http.NewRequest("POST", "/memory/delete", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNoContent, w.Code)
	mockStorage.AssertExpectations(t)
}

// marshalDataWithLogging helper needed by memory.go
func init() {
	// Ensure gin test mode for all tests in this file
	gin.SetMode(gin.TestMode)
}
