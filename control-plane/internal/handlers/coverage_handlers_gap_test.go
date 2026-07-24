package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type vectorNamespaceStorageStub struct {
	*vectorStorageStub
	deletePrefixScope   string
	deletePrefixScopeID string
	deletePrefix        string
	deletePrefixCount   int
	deletePrefixErr     error
}

func (v *vectorNamespaceStorageStub) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	v.deletePrefixScope = scope
	v.deletePrefixScopeID = scopeID
	v.deletePrefix = prefix
	if v.deletePrefixErr != nil {
		return 0, v.deletePrefixErr
	}
	return v.deletePrefixCount, nil
}

type lifecycleUpdateErrorStorageStub struct {
	*nodeRESTStorageStub
	updateErr error
}

func (s *lifecycleUpdateErrorStorageStub) UpdateAgentLifecycleStatus(ctx context.Context, nodeID string, status types.AgentLifecycleStatus) error {
	if s.updateErr != nil {
		return s.updateErr
	}
	return s.nodeRESTStorageStub.UpdateAgentLifecycleStatus(ctx, nodeID, status)
}

type registerErrorStorageStub struct {
	*nodeRESTStorageStub
	registerErr error
}

func (s *registerErrorStorageStub) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	if s.registerErr != nil {
		return s.registerErr
	}
	return s.nodeRESTStorageStub.RegisterAgent(ctx, agent)
}

type executionLogStoreStub struct {
	*testExecutionStorage
	storeErr     error
	pruneErr     error
	batchErr     error
	batchEntries []*types.ExecutionLogEntry
}

func (s *executionLogStoreStub) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	if s.storeErr != nil {
		return s.storeErr
	}
	return s.testExecutionStorage.StoreExecutionLogEntry(ctx, entry)
}

func (s *executionLogStoreStub) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	if s.pruneErr != nil {
		return s.pruneErr
	}
	return s.testExecutionStorage.PruneExecutionLogEntries(ctx, executionID, maxEntries, olderThan)
}

func (s *executionLogStoreStub) StoreExecutionLogEntries(ctx context.Context, executionID string, entries []*types.ExecutionLogEntry) error {
	if s.batchErr != nil {
		return s.batchErr
	}
	s.batchEntries = append(s.batchEntries, entries...)
	for _, entry := range entries {
		if err := s.testExecutionStorage.StoreExecutionLogEntry(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

type workflowPersistErrorStore struct {
	*testExecutionStorage
	storeErr error
}

func (s *workflowPersistErrorStore) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	if s.storeErr != nil {
		return s.storeErr
	}
	return s.testExecutionStorage.StoreWorkflowExecution(ctx, execution)
}

type executionRecordLookupErrorStore struct {
	*testExecutionStorage
	getErr error
}

func (s *executionRecordLookupErrorStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	return s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
}

func TestDeleteNamespaceVectorsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name               string
		body               string
		sessionID          string
		deleteCount        int
		deleteErr          error
		expectedStatusCode int
		expectedBody       string
		expectedScope      string
		expectedScopeID    string
		expectedPrefix     string
	}{
		{
			name:               "invalid json",
			body:               `{`,
			expectedStatusCode: http.StatusBadRequest,
			expectedBody:       "invalid_request",
		},
		{
			name:               "storage error",
			body:               `{"namespace":"team/docs","scope":"session"}`,
			sessionID:          "session-1",
			deleteErr:          errors.New("delete failed"),
			expectedStatusCode: http.StatusInternalServerError,
			expectedBody:       "storage_error",
			expectedScope:      "session",
			expectedScopeID:    "session-1",
			expectedPrefix:     "team/docs",
		},
		{
			name:               "success",
			body:               `{"namespace":"team/docs","scope":"session"}`,
			sessionID:          "session-1",
			deleteCount:        3,
			expectedStatusCode: http.StatusOK,
			expectedBody:       `"deleted":3`,
			expectedScope:      "session",
			expectedScopeID:    "session-1",
			expectedPrefix:     "team/docs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storage := &vectorNamespaceStorageStub{vectorStorageStub: &vectorStorageStub{}, deletePrefixCount: tt.deleteCount, deletePrefixErr: tt.deleteErr}
			router := gin.New()
			router.POST("/vectors/delete-namespace", DeleteNamespaceVectorsHandler(storage))

			req := httptest.NewRequest(http.MethodPost, "/vectors/delete-namespace", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			if tt.sessionID != "" {
				req.Header.Set("X-Session-ID", tt.sessionID)
			}

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.expectedStatusCode, resp.Code)
			assert.Contains(t, resp.Body.String(), tt.expectedBody)
			if tt.expectedPrefix != "" {
				assert.Equal(t, tt.expectedScope, storage.deletePrefixScope)
				assert.Equal(t, tt.expectedScopeID, storage.deletePrefixScopeID)
				assert.Equal(t, tt.expectedPrefix, storage.deletePrefix)
			}
		})
	}
}

func TestParsePauseResumeRequest(t *testing.T) {
	t.Run("nil request returns empty request", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		req, err := parsePauseResumeRequest(c)
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Empty(t, req.Reason)
	})

	t.Run("empty body is treated as empty request", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/", http.NoBody)
		req, err := parsePauseResumeRequest(c)
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Empty(t, req.Reason)
	})

	t.Run("invalid json returns error", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/", strings.NewReader("{"))
		c.Request.Header.Set("Content-Type", "application/json")
		req, err := parsePauseResumeRequest(c)
		require.Error(t, err)
		assert.Nil(t, req)
	})

	t.Run("valid body is parsed", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"reason":"resume work"}`))
		c.Request.Header.Set("Content-Type", "application/json")
		req, err := parsePauseResumeRequest(c)
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Equal(t, "resume work", req.Reason)
	})
}

func TestGetExecutionIDFromContext(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(*gin.Context)
		expected string
	}{
		{
			name: "context value wins",
			setup: func(c *gin.Context) {
				c.Set("execution_id", "ctx-exec")
				c.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-exec", nil)
				c.Request.Header.Set("X-Execution-ID", "header-exec")
			},
			expected: "ctx-exec",
		},
		{
			name: "header used when context value is wrong type",
			setup: func(c *gin.Context) {
				c.Set("execution_id", 42)
				c.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-exec", nil)
				c.Request.Header.Set("X-Execution-ID", "header-exec")
			},
			expected: "header-exec",
		},
		{
			name: "query used as fallback",
			setup: func(c *gin.Context) {
				c.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-exec", nil)
			},
			expected: "query-exec",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			tt.setup(c)
			assert.Equal(t, tt.expected, getExecutionIDFromContext(c))
		})
	}
}

func TestReadCloserClose(t *testing.T) {
	rc := &readCloser{Reader: bytes.NewBufferString("payload")}
	assert.NoError(t, rc.Close())
}

func TestUpdateLifecycleStatusHandler_ErrorPaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name               string
		body               string
		expectedStatusCode int
		expectedBody       string
	}{
		{
			name: "invalid json",
			body: `{`,
			expectedStatusCode: http.StatusBadRequest,
			expectedBody: "Invalid JSON format",
		},
		{
			name: "invalid lifecycle status",
			body: `{"lifecycle_status":"unknown"}`,
			expectedStatusCode: http.StatusBadRequest,
			expectedBody: "Invalid lifecycle status",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/nodes/:node_id/lifecycle", UpdateLifecycleStatusHandler(&nodeRESTStorageStub{agent: &types.AgentNode{ID: "node-1"}}, nil, nil))

			req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/lifecycle", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.expectedStatusCode, resp.Code)
			assert.Contains(t, resp.Body.String(), tt.expectedBody)
		})
	}

	t.Run("missing node returns not found", func(t *testing.T) {
		store := &nodeRESTStorageStub{getAgentErr: errors.New("missing")}
		router := gin.New()
		router.POST("/nodes/:node_id/lifecycle", UpdateLifecycleStatusHandler(store, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/lifecycle", strings.NewReader(`{"lifecycle_status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusNotFound, resp.Code)
		assert.Contains(t, resp.Body.String(), "node not found")
	})

	t.Run("storage update failure returns internal server error", func(t *testing.T) {
		store := &lifecycleUpdateErrorStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{agent: &types.AgentNode{ID: "node-1", LifecycleStatus: types.AgentStatusReady}},
			updateErr:           errors.New("update failed"),
		}
		router := gin.New()
		router.POST("/nodes/:node_id/lifecycle", UpdateLifecycleStatusHandler(store, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/lifecycle", strings.NewReader(`{"lifecycle_status":"offline"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to update lifecycle status")
	})
}

func TestRegisterServerlessAgentHandler_ErrorPaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("invalid request body", func(t *testing.T) {
		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader("{"))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "Invalid request")
	})

	t.Run("host must be allowlisted", func(t *testing.T) {
		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"example.com"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"https://disallowed.test"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "Invalid invocation_url")
	})

	t.Run("discovery status failure", func(t *testing.T) {
		discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "nope", http.StatusTeapot)
		}))
		defer discoveryServer.Close()

		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadGateway, resp.Code)
		assert.Contains(t, resp.Body.String(), "Discovery endpoint failed")
	})

	t.Run("discovery json failure", func(t *testing.T) {
		discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"node_id":`))
		}))
		defer discoveryServer.Close()

		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadGateway, resp.Code)
		assert.Contains(t, resp.Body.String(), "Invalid discovery response")
	})

	t.Run("registration failure bubbles up", func(t *testing.T) {
		discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"node_id":"serverless-err",
				"version":"v1",
				"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"}}],
				"skills":[{"id":"s1","input_schema":{"type":"object"}}]
			}`))
		}))
		defer discoveryServer.Close()

		store := &registerErrorStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{},
			registerErr:         errors.New("register failed"),
		}
		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "Failed to register serverless agent")
	})
}

func TestDeleteNamespaceVectorsHandler_ResponsePayload(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vectorNamespaceStorageStub{
		vectorStorageStub: &vectorStorageStub{},
		deletePrefixCount: 2,
	}
	router := gin.New()
	router.POST("/vectors/delete-namespace", DeleteNamespaceVectorsHandler(storage))

	req := httptest.NewRequest(http.MethodPost, "/vectors/delete-namespace", strings.NewReader(`{"namespace":"tenant/","scope":"actor"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Actor-ID", "actor-9")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	assert.Equal(t, "tenant/", payload["namespace"])
	assert.Equal(t, float64(2), payload["deleted"])
	assert.Equal(t, "actor", payload["scope"])
	assert.Equal(t, "actor-9", payload["scope_id"])
	assert.Equal(t, "deleted", payload["status"])
}

func TestStructuredExecutionLogsHandler_EdgeCases(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("validates body and required fields", func(t *testing.T) {
		tests := []struct {
			name               string
			body               string
			expectedStatusCode int
			expectedBody       string
		}{
			{name: "empty body", body: "", expectedStatusCode: http.StatusBadRequest, expectedBody: "request body is required"},
			{name: "invalid payload", body: `{"entries":`, expectedStatusCode: http.StatusBadRequest, expectedBody: "invalid payload"},
			{name: "missing message", body: `{"agent_node_id":"node-1"}`, expectedStatusCode: http.StatusBadRequest, expectedBody: "message is required"},
			{name: "missing agent node id", body: `{"message":"hello"}`, expectedStatusCode: http.StatusBadRequest, expectedBody: "agent_node_id is required"},
			{name: "path mismatch", body: `{"execution_id":"other","agent_node_id":"node-1","message":"hello"}`, expectedStatusCode: http.StatusBadRequest, expectedBody: "execution_id_mismatch"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				store := &executionLogStoreStub{testExecutionStorage: newTestExecutionStorage(nil)}
				router := gin.New()
				router.POST("/api/v1/executions/:execution_id/logs", StructuredExecutionLogsHandler(store, func() config.ExecutionLogsConfig {
					return config.ExecutionLogsConfig{MaxEntriesPerExecution: 100}
				}))

				req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", strings.NewReader(tt.body))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)

				require.Equal(t, tt.expectedStatusCode, resp.Code)
				assert.Contains(t, resp.Body.String(), tt.expectedBody)
			})
		}
	})

	t.Run("batch storage applies defaults", func(t *testing.T) {
		store := &executionLogStoreStub{testExecutionStorage: newTestExecutionStorage(nil)}
		router := gin.New()
		router.POST("/api/v1/executions/:execution_id/logs", StructuredExecutionLogsHandler(store, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{MaxEntriesPerExecution: 100}
		}))

		req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", strings.NewReader(`{"entries":[{"agent_node_id":"node-1","message":"hello"}]}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusAccepted, resp.Code)
		require.Len(t, store.batchEntries, 1)
		assert.Equal(t, "exec-1", store.batchEntries[0].ExecutionID)
		assert.Equal(t, "exec-1", store.batchEntries[0].WorkflowID)
		require.NotNil(t, store.batchEntries[0].RootWorkflowID)
		assert.Equal(t, "exec-1", *store.batchEntries[0].RootWorkflowID)
		assert.JSONEq(t, `{}`, string(store.batchEntries[0].Attributes))
	})

	t.Run("storage and prune failures surface", func(t *testing.T) {
		tests := []struct {
			name               string
			store              *executionLogStoreStub
			expectedStatusCode int
			expectedBody       string
		}{
			{
				name: "batch store error",
				store: &executionLogStoreStub{
					testExecutionStorage: newTestExecutionStorage(nil),
					batchErr:             errors.New("batch failed"),
				},
				expectedStatusCode: http.StatusInternalServerError,
				expectedBody:       "failed to store execution log entry batch",
			},
			{
				name: "prune error",
				store: &executionLogStoreStub{
					testExecutionStorage: newTestExecutionStorage(nil),
					pruneErr:             errors.New("prune failed"),
				},
				expectedStatusCode: http.StatusInternalServerError,
				expectedBody:       "failed to prune execution logs",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				router := gin.New()
				router.POST("/api/v1/executions/:execution_id/logs", StructuredExecutionLogsHandler(tt.store, func() config.ExecutionLogsConfig {
					return config.ExecutionLogsConfig{MaxEntriesPerExecution: 100}
				}))

				req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", strings.NewReader(`{"agent_node_id":"node-1","message":"hello"}`))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)

				require.Equal(t, tt.expectedStatusCode, resp.Code)
				assert.Contains(t, resp.Body.String(), tt.expectedBody)
			})
		}
	})
}

func TestConfigStorageHandlers_ErrorBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("list configs error", func(t *testing.T) {
		store := &configStorageMock{}
		store.On("ListConfigs", mock.Anything).Return(([]*storage.ConfigEntry)(nil), errors.New("list failed")).Once()

		router := gin.New()
		NewConfigStorageHandlers(store, nil).RegisterRoutes(router.Group("/api/v1"))

		req := httptest.NewRequest(http.MethodGet, "/api/v1/configs", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "list failed")
		store.AssertExpectations(t)
	})

	t.Run("get config error", func(t *testing.T) {
		store := &configStorageMock{}
		store.On("GetConfig", mock.Anything, "broken").Return((*storage.ConfigEntry)(nil), errors.New("read failed")).Once()

		router := gin.New()
		NewConfigStorageHandlers(store, nil).RegisterRoutes(router.Group("/api/v1"))

		req := httptest.NewRequest(http.MethodGet, "/api/v1/configs/broken", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "read failed")
		store.AssertExpectations(t)
	})

	t.Run("set config enforces size limit", func(t *testing.T) {
		store := &configStorageMock{}
		router := gin.New()
		NewConfigStorageHandlers(store, nil).RegisterRoutes(router.Group("/api/v1"))

		req := httptest.NewRequest(http.MethodPut, "/api/v1/configs/large", strings.NewReader(strings.Repeat("a", maxConfigBodySize+1)))
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
		assert.Contains(t, resp.Body.String(), "config body exceeds maximum size")
	})

	t.Run("set config store and readback failures", func(t *testing.T) {
		tests := []struct {
			name         string
			mockSetup    func(*configStorageMock)
			expectedBody string
		}{
			{
				name: "store failure",
				mockSetup: func(store *configStorageMock) {
					store.On("SetConfig", mock.Anything, "key", "value", "api").Return(errors.New("write failed")).Once()
				},
				expectedBody: "write failed",
			},
			{
				name: "readback failure",
				mockSetup: func(store *configStorageMock) {
					store.On("SetConfig", mock.Anything, "key", "value", "api").Return(nil).Once()
					store.On("GetConfig", mock.Anything, "key").Return((*storage.ConfigEntry)(nil), errors.New("reload failed")).Once()
				},
				expectedBody: "reload failed",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				store := &configStorageMock{}
				tt.mockSetup(store)

				router := gin.New()
				NewConfigStorageHandlers(store, nil).RegisterRoutes(router.Group("/api/v1"))

				req := httptest.NewRequest(http.MethodPut, "/api/v1/configs/key", strings.NewReader("value"))
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)

				require.Equal(t, http.StatusInternalServerError, resp.Code)
				assert.Contains(t, resp.Body.String(), tt.expectedBody)
				store.AssertExpectations(t)
			})
		}
	})
}

func TestMarshalDataWithLogging(t *testing.T) {
	jsonData, err := marshalDataWithLogging(nil, "payload")
	require.NoError(t, err)
	assert.Equal(t, "null", string(jsonData))

	jsonData, err = marshalDataWithLogging(map[string]string{"hello": "world"}, "payload")
	require.NoError(t, err)
	assert.JSONEq(t, `{"hello":"world"}`, string(jsonData))

	jsonData, err = marshalDataWithLogging(map[string]interface{}{"bad": make(chan int)}, "payload")
	require.Error(t, err)
	assert.Nil(t, jsonData)
	assert.Contains(t, err.Error(), "failed to marshal payload")
}

func newSkillAgent(baseURL string) *types.AgentNode {
	return &types.AgentNode{
		ID:              "node-1",
		BaseURL:         baseURL,
		Skills:          []types.SkillDefinition{{ID: "summarize"}},
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
}

func TestExecuteSkillHandler_BasicPaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("malformed skill id", func(t *testing.T) {
		store := newReasonerHandlerStorage(nil)
		router := gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/skills/not-valid", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "node_id.skill_name")
	})

	t.Run("missing skill on node", func(t *testing.T) {
		store := newReasonerHandlerStorage(newSkillAgent("http://agent.invalid"))
		router := gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/skills/node-1.unknown", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusNotFound, resp.Code)
		assert.Contains(t, resp.Body.String(), "skill 'unknown' not found")
	})

	t.Run("successful execution persists workflow execution", func(t *testing.T) {
		agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/skills/summarize", r.URL.Path)
			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			defer r.Body.Close()
			require.JSONEq(t, `{"topic":"status"}`, string(body))
			require.Equal(t, "wf-skill", r.Header.Get("X-Workflow-ID"))
			require.Equal(t, "session-1", r.Header.Get("X-Session-ID"))

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"summary":"done"}`))
		}))
		defer agentServer.Close()

		store := newReasonerHandlerStorage(newSkillAgent(agentServer.URL))
		router := gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/skills/node-1.summarize", strings.NewReader(`{"input":{"topic":"status"}}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Workflow-ID", "wf-skill")
		req.Header.Set("X-Session-ID", "session-1")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)

		var payload ExecuteReasonerResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
		assert.Equal(t, "node-1", payload.NodeID)

		records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
		require.NoError(t, err)
		require.Len(t, records, 1)
		assert.Equal(t, string(types.ExecutionStatusSucceeded), records[0].Status)
		assert.Equal(t, "summarize", records[0].ReasonerID)
		assert.JSONEq(t, `{"summary":"done"}`, string(records[0].OutputData))
	})
}

func TestExecuteReasonerHandler_AdditionalBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("body validation", func(t *testing.T) {
		store := newReasonerHandlerStorage(newReasonerAgent("http://agent.invalid"))
		router := gin.New()
		router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
	})

	t.Run("missing reasoner on node", func(t *testing.T) {
		store := newReasonerHandlerStorage(newReasonerAgent("http://agent.invalid"))
		router := gin.New()
		router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.other", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusNotFound, resp.Code)
		assert.Contains(t, resp.Body.String(), "reasoner 'other' not found")
	})

	t.Run("invalid agent response persists failure", func(t *testing.T) {
		agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`not-json`))
		}))
		defer agentServer.Close()

		store := newReasonerHandlerStorage(newReasonerAgent(agentServer.URL))
		router := gin.New()
		router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{"ok":true}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to parse agent response")

		records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
		require.NoError(t, err)
		require.Len(t, records, 1)
		assert.Equal(t, string(types.ExecutionStatusFailed), records[0].Status)
		require.NotNil(t, records[0].ErrorMessage)
		assert.Contains(t, *records[0].ErrorMessage, "failed to parse agent response")
	})
}

func TestExecuteSkillHandler_AdditionalBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("node availability errors", func(t *testing.T) {
		tests := []struct {
			name       string
			agent      *types.AgentNode
			getAgentErr error
			wantCode   int
			wantBody   string
		}{
			{
				name:       "node not found",
				getAgentErr: errors.New("missing"),
				wantCode:   http.StatusNotFound,
				wantBody:   "node 'node-1' not found",
			},
			{
				name: "inactive node",
				agent: &types.AgentNode{
					ID:              "node-1",
					BaseURL:         "http://agent.invalid",
					Skills:          []types.SkillDefinition{{ID: "summarize"}},
					HealthStatus:    types.HealthStatusInactive,
					LifecycleStatus: types.AgentStatusReady,
				},
				wantCode: http.StatusServiceUnavailable,
				wantBody: "is not healthy",
			},
			{
				name: "offline node",
				agent: &types.AgentNode{
					ID:              "node-1",
					BaseURL:         "http://agent.invalid",
					Skills:          []types.SkillDefinition{{ID: "summarize"}},
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusOffline,
				},
				wantCode: http.StatusServiceUnavailable,
				wantBody: "is offline",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				store := newReasonerHandlerStorage(tt.agent)
				store.getAgentErr = tt.getAgentErr

				router := gin.New()
				router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

				req := httptest.NewRequest(http.MethodPost, "/skills/node-1.summarize", strings.NewReader(`{"input":{}}`))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)

				require.Equal(t, tt.wantCode, resp.Code)
				assert.Contains(t, resp.Body.String(), tt.wantBody)
			})
		}
	})

	t.Run("request validation and upstream parse failure", func(t *testing.T) {
		store := newReasonerHandlerStorage(newSkillAgent("http://agent.invalid"))
		router := gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/skills/node-1.summarize", strings.NewReader(`{"input":`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)

		agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`not-json`))
		}))
		defer agentServer.Close()

		store = newReasonerHandlerStorage(newSkillAgent(agentServer.URL))
		router = gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req = httptest.NewRequest(http.MethodPost, "/skills/node-1.summarize", strings.NewReader(`{"input":{"topic":"status"}}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to parse agent response")

		records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
		require.NoError(t, err)
		require.Len(t, records, 1)
		assert.Equal(t, string(types.ExecutionStatusFailed), records[0].Status)
	})
}

func TestExecutionHelpersAndStatusHandlers(t *testing.T) {
	t.Run("extract requested llm endpoint", func(t *testing.T) {
		tests := []struct {
			name     string
			req      ExecuteRequest
			expected string
		}{
			{name: "primary key", req: ExecuteRequest{Context: map[string]interface{}{"llm_endpoint": " openai "}}, expected: "openai"},
			{name: "fallback key", req: ExecuteRequest{Context: map[string]interface{}{"provider": "anthropic"}}, expected: "anthropic"},
			{name: "missing", req: ExecuteRequest{Context: map[string]interface{}{}}, expected: ""},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				assert.Equal(t, tt.expected, extractRequestedLLMEndpoint(tt.req))
			})
		}
	})

	t.Run("parse target and classify raw error", func(t *testing.T) {
		target, err := parseTarget("node-1.ping")
		require.NoError(t, err)
		require.NotNil(t, target)
		assert.Equal(t, "node-1", target.NodeID)
		assert.Equal(t, "ping", target.TargetName)

		_, err = parseTarget("")
		require.Error(t, err)
		_, err = parseTarget("node-only")
		require.Error(t, err)

		assert.Equal(t, ErrorCategoryAgentTimeout, classifyRawError(context.DeadlineExceeded))
		assert.Equal(t, ErrorCategoryAgentUnreachable, classifyRawError(errors.New("connection refused")))
		assert.Equal(t, ErrorCategoryInternal, classifyRawError(context.Canceled))
		assert.Equal(t, ErrorCategoryInternal, classifyRawError(nil))
	})

	t.Run("build and ensure workflow execution record", func(t *testing.T) {
		parentID := "parent-exec"
		rootWorkflowID := "root-workflow"
		store := &workflowPersistErrorStore{testExecutionStorage: newTestExecutionStorage(nil)}
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID:    parentID,
			WorkflowID:     "parent-workflow",
			RootWorkflowID: &rootWorkflowID,
			WorkflowDepth:  2,
			AgentNodeID:    "node-1",
			Status:         types.ExecutionStatusRunning,
			StartedAt:      time.Now().UTC(),
		}))

		controller := newExecutionController(store, nil, nil, 0, "")
		exec := &types.Execution{
			ExecutionID:       "exec-1",
			RunID:             "run-1",
			ParentExecutionID: &parentID,
			SessionID:         pointerString("session-1"),
			ActorID:           pointerString("actor-1"),
			AgentNodeID:       "node-1",
			NodeID:            "node-1",
			ReasonerID:        "ping",
			Status:            types.ExecutionStatusRunning,
		}
		target := &parsedTarget{NodeID: "node-1", TargetName: "ping", TargetType: "reasoner"}

		record := controller.buildWorkflowExecutionRecord(context.Background(), exec, target, []byte(`{"hello":"world"}`))
		require.NotNil(t, record)
		assert.Equal(t, "run-1", record.WorkflowID)
		require.NotNil(t, record.RootWorkflowID)
		assert.Equal(t, "root-workflow", *record.RootWorkflowID)
		require.NotNil(t, record.ParentWorkflowID)
		assert.Equal(t, "parent-workflow", *record.ParentWorkflowID)
		assert.Equal(t, 3, record.WorkflowDepth)
		assert.Equal(t, 17, record.InputSize)
		assert.Equal(t, []string{"reasoner"}, record.WorkflowTags)

		controller.ensureWorkflowExecutionRecord(context.Background(), exec, target, []byte(`{"hello":"world"}`))
		stored, err := store.GetWorkflowExecution(context.Background(), "exec-1")
		require.NoError(t, err)
		require.NotNil(t, stored)

		store.storeErr = errors.New("persist failed")
		controller.ensureWorkflowExecutionRecord(context.Background(), exec, target, []byte(`{"hello":"world"}`))
		controller.ensureWorkflowExecutionRecord(context.Background(), nil, target, nil)
		controller.ensureWorkflowExecutionRecord(context.Background(), exec, nil, nil)
	})

	t.Run("render status with approval and handler branches", func(t *testing.T) {
		baseStore := newTestExecutionStorage(nil)
		require.NoError(t, baseStore.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-1",
			RunID:       "run-1",
			Status:      types.ExecutionStatusFailed,
			ResultPayload: json.RawMessage(`{"kind":"denied"}`),
			ErrorMessage: pointerString("failed"),
			StartedAt:    time.Now().UTC(),
		}))
		approvalID := "approval-1"
		approvalStatus := "pending"
		approvalURL := "https://example.test/approval/1"
		require.NoError(t, baseStore.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID:        "exec-1",
			WorkflowID:         "run-1",
			ApprovalRequestID:  &approvalID,
			ApprovalStatus:     &approvalStatus,
			ApprovalRequestURL: &approvalURL,
			AgentNodeID:        "node-1",
			Status:             types.ExecutionStatusFailed,
			StartedAt:          time.Now().UTC(),
		}))

		controller := newExecutionController(baseStore, nil, nil, 0, "")
		resp := controller.renderStatusWithApproval(context.Background(), &types.Execution{
			ExecutionID:   "exec-1",
			RunID:         "run-1",
			Status:        types.ExecutionStatusFailed,
			ResultPayload: json.RawMessage(`{"kind":"denied"}`),
			ErrorMessage:  pointerString("failed"),
			StartedAt:     time.Now().UTC(),
		})
		require.NotNil(t, resp.ErrorDetails)
		assert.Equal(t, approvalID, *resp.ApprovalRequestID)
		assert.Equal(t, approvalStatus, *resp.ApprovalStatus)
		assert.Equal(t, approvalURL, *resp.ApprovalRequestURL)

		errStore := &executionRecordLookupErrorStore{testExecutionStorage: newTestExecutionStorage(nil), getErr: errors.New("boom")}
		controller = newExecutionController(errStore, nil, nil, 0, "")
		router := gin.New()
		router.GET("/executions/:execution_id/status", controller.handleStatus)

		req := httptest.NewRequest(http.MethodGet, "/executions/exec-1/status", nil)
		respRec := httptest.NewRecorder()
		router.ServeHTTP(respRec, req)
		require.Equal(t, http.StatusInternalServerError, respRec.Code)

		router = gin.New()
		router.GET("/executions/:execution_id/status", newExecutionController(newTestExecutionStorage(nil), nil, nil, 0, "").handleStatus)
		req = httptest.NewRequest(http.MethodGet, "/executions/missing/status", nil)
		respRec = httptest.NewRecorder()
		router.ServeHTTP(respRec, req)
		require.Equal(t, http.StatusNotFound, respRec.Code)

		router = gin.New()
		router.GET("/executions/:execution_id/status", newExecutionController(baseStore, nil, nil, 0, "").handleStatus)
		req = httptest.NewRequest(http.MethodGet, "/executions/exec-1/status", nil)
		respRec = httptest.NewRecorder()
		router.ServeHTTP(respRec, req)
		require.Equal(t, http.StatusOK, respRec.Code)
		assert.Contains(t, respRec.Body.String(), `"approval_request_id":"approval-1"`)
	})
}
