//go:build integration
// +build integration

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/internal/services"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockStorageProvider is defined in config_test.go to avoid duplication
// in the same package.

func setupExecutionTestRouter() (*gin.Engine, *MockStorageProvider) {
	gin.SetMode(gin.TestMode)

	mockStorage := &MockStorageProvider{}
	executionHandler := NewExecutionHandler(mockStorage, nil, nil)

	router := gin.New()
	v1 := router.Group("/api/ui/v1")
	{
		agents := v1.Group("/agents")
		{
			agents.GET("/:agentId/executions", executionHandler.ListExecutionsHandler)
			agents.GET("/:agentId/executions/:executionId", executionHandler.GetExecutionDetailsHandler)
		}
	}

	return router, mockStorage
}

type stubPayloadStore struct {
	payloads map[string][]byte
}

func newStubPayloadStore() *stubPayloadStore {
	return &stubPayloadStore{payloads: make(map[string][]byte)}
}

func (s *stubPayloadStore) SaveFromReader(ctx context.Context, r io.Reader) (*services.PayloadRecord, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *stubPayloadStore) SaveBytes(ctx context.Context, data []byte) (*services.PayloadRecord, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *stubPayloadStore) Open(ctx context.Context, uri string) (io.ReadCloser, error) {
	data, ok := s.payloads[uri]
	if !ok {
		return nil, fmt.Errorf("payload not found: %s", uri)
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

func (s *stubPayloadStore) Remove(ctx context.Context, uri string) error {
	delete(s.payloads, uri)
	return nil
}

func TestListExecutionsHandler(t *testing.T) {
	t.Run("successful_list_executions", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		// Mock data
		now := time.Now()
		executions := []*types.AgentExecution{
			{
				ID:          1,
				WorkflowID:  "workflow-1",
				SessionID:   stringPtrForExecutions("session-1"),
				AgentNodeID: "test-agent",
				ReasonerID:  "test-reasoner",
				Status:      string(types.ExecutionStatusSucceeded),
				DurationMS:  1000,
				InputSize:   100,
				OutputSize:  200,
				CreatedAt:   now,
			},
			{
				ID:           2,
				WorkflowID:   "workflow-2",
				AgentNodeID:  "test-agent",
				ReasonerID:   "test-reasoner-2",
				Status:       "failed",
				DurationMS:   500,
				InputSize:    50,
				OutputSize:   0,
				ErrorMessage: stringPtrForExecutions("test error"),
				CreatedAt:    now.Add(-time.Hour),
			},
		}

		expectedFilters := types.ExecutionFilters{
			AgentNodeID: stringPtrForExecutions("test-agent"),
			Limit:       10,
			Offset:      0,
		}

		mockStorage.On("QueryExecutions", mock.AnythingOfType("context.Context"), expectedFilters).Return(executions, nil)

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assert response
		assert.Equal(t, http.StatusOK, w.Code)

		var response ExecutionListResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Len(t, response.Executions, 2)
		assert.Equal(t, 2, response.Total)
		assert.Equal(t, 1, response.Page)
		assert.Equal(t, 10, response.PageSize)

		// Check first execution
		exec1 := response.Executions[0]
		assert.Equal(t, int64(1), exec1.ID)
		assert.Equal(t, "workflow-1", exec1.WorkflowID)
		assert.Equal(t, "session-1", *exec1.SessionID)
		assert.Equal(t, "test-agent", exec1.AgentNodeID)
		assert.Equal(t, string(types.ExecutionStatusSucceeded), exec1.Status)

		mockStorage.AssertExpectations(t)
	})

	t.Run("with_pagination_and_filters", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		executions := []*types.AgentExecution{}

		expectedFilters := types.ExecutionFilters{
			AgentNodeID: stringPtrForExecutions("test-agent"),
			Status:      stringPtrForExecutions(string(types.ExecutionStatusSucceeded)),
			WorkflowID:  stringPtrForExecutions("workflow-1"),
			Limit:       5,
			Offset:      10,
		}

		mockStorage.On("QueryExecutions", mock.AnythingOfType("context.Context"), expectedFilters).Return(executions, nil)

		// Make request with query parameters
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions?page=3&pageSize=5&status=succeeded&workflowId=workflow-1", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assert response
		assert.Equal(t, http.StatusOK, w.Code)

		var response ExecutionListResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Equal(t, 3, response.Page)
		assert.Equal(t, 5, response.PageSize)

		mockStorage.AssertExpectations(t)
	})

	t.Run("missing_agent_id", func(t *testing.T) {
		router, _ := setupExecutionTestRouter()

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents//executions", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "agentId is required", response.Error)
	})

	t.Run("storage_error", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		expectedFilters := types.ExecutionFilters{
			AgentNodeID: stringPtrForExecutions("test-agent"),
			Limit:       10,
			Offset:      0,
		}

		mockStorage.On("QueryExecutions", mock.AnythingOfType("context.Context"), expectedFilters).Return([]*types.AgentExecution(nil), assert.AnError)

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusInternalServerError, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Contains(t, response.Error, "failed to query executions")

		mockStorage.AssertExpectations(t)
	})
}

func TestGetExecutionDetailsHandler(t *testing.T) {
	t.Run("successful_get_execution_details", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		// Mock data
		now := time.Now()
		inputData := json.RawMessage(`{"input": "test"}`)
		outputData := json.RawMessage(`{"output": "result"}`)

		execution := &types.AgentExecution{
			ID:          123,
			WorkflowID:  "workflow-1",
			SessionID:   stringPtrForExecutions("session-1"),
			AgentNodeID: "test-agent",
			ReasonerID:  "test-reasoner",
			InputData:   inputData,
			OutputData:  outputData,
			InputSize:   100,
			OutputSize:  200,
			DurationMS:  1000,
			Status:      string(types.ExecutionStatusSucceeded),
			UserID:      stringPtrForExecutions("user-1"),
			NodeID:      stringPtrForExecutions("node-1"),
			Metadata:    types.ExecutionMetadata{},
			CreatedAt:   now,
		}

		mockStorage.On("GetExecution", mock.AnythingOfType("context.Context"), int64(123)).Return(execution, nil)

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions/123", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assert response
		assert.Equal(t, http.StatusOK, w.Code)

		var response ExecutionDetailsResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Equal(t, int64(123), response.ID)
		assert.Equal(t, "workflow-1", response.WorkflowID)
		assert.Equal(t, "session-1", *response.SessionID)
		assert.Equal(t, "test-agent", response.AgentNodeID)
		assert.Equal(t, "test-reasoner", response.ReasonerID)
		assert.Equal(t, string(types.ExecutionStatusSucceeded), response.Status)
		assert.Equal(t, 1000, response.DurationMS)
		// UserID and NodeID are not directly part of ExecutionDetailsResponse in pkg/types,
		// they are part of AgentExecution. Remove these assertions if not applicable to the response struct.
		// assert.Equal(t, "user-1", *response.UserID)
		// assert.Equal(t, "node-1", *response.NodeID)

		// Check parsed input/output data
		inputMap, ok := response.InputData.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "test", inputMap["input"])

		outputMap, ok := response.OutputData.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "result", outputMap["output"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("missing_agent_id", func(t *testing.T) {
		router, _ := setupExecutionTestRouter()

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents//executions/123", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "agentId is required", response.Error)
	})

	t.Run("missing_execution_id", func(t *testing.T) {
		router, _ := setupExecutionTestRouter()

		// Test with a URL that has trailing slash - Gin will redirect
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions/", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Gin redirects trailing slash URLs, so we expect 301
		assert.Equal(t, http.StatusMovedPermanently, w.Code)
	})

	t.Run("invalid_execution_id_format", func(t *testing.T) {
		router, _ := setupExecutionTestRouter()

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions/invalid", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "invalid executionId format", response.Error)
	})

	t.Run("execution_not_found", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		mockStorage.On("GetExecution", mock.AnythingOfType("context.Context"), int64(123)).Return((*types.AgentExecution)(nil), assert.AnError)

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions/123", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "execution not found", response.Error)

		mockStorage.AssertExpectations(t)
	})

	t.Run("execution_belongs_to_different_agent", func(t *testing.T) {
		router, mockStorage := setupExecutionTestRouter()

		execution := &types.AgentExecution{
			ID:          123,
			AgentNodeID: "different-agent",
		}

		mockStorage.On("GetExecution", mock.AnythingOfType("context.Context"), int64(123)).Return(execution, nil)

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/executions/123", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "execution not found for this agent", response.Error)

		mockStorage.AssertExpectations(t)
	})
}

func TestGetExecutionDetailsHandler_FallbacksToPayloadStore(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockStorage := &MockStorageProvider{}
	payloadStore := newStubPayloadStore()

	uri := "payload://stub-input"
	payloadBytes := []byte("{\"foo\":\"bar\"}")
	payloadStore.payloads[uri] = payloadBytes

	execution := &types.AgentExecution{
		ID:          123,
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		InputData:   json.RawMessage("{}"),
		OutputData:  json.RawMessage("{}"),
		InputSize:   0,
		OutputSize:  0,
		DurationMS:  50,
		Status:      string(types.ExecutionStatusSucceeded),
		CreatedAt:   time.Now(),
	}

	mockStorage.On("GetExecution", mock.AnythingOfType("context.Context"), int64(123)).Return(execution, nil)
	mockStorage.On("GetWorkflowStep", mock.AnythingOfType("context.Context"), "exec_123").Return(&types.WorkflowStep{InputURI: &uri}, nil)

	handler := NewExecutionHandler(mockStorage, payloadStore, nil)
	router := gin.New()
	router.GET("/api/ui/v1/agents/:agentId/executions/:executionId", handler.GetExecutionDetailsHandler)

	req, _ := http.NewRequest("GET", "/api/ui/v1/agents/agent-1/executions/123", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	var response ExecutionDetailsResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

	inputMap, ok := response.InputData.(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "bar", inputMap["foo"])

	assert.Equal(t, len(payloadBytes), response.InputSize)

	mockStorage.AssertExpectations(t)
}

// Helper function to create string pointers for executions tests
func stringPtrForExecutions(s string) *string {
	return &s
}
