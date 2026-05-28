//go:build integration
// +build integration

package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// Add CleanupWorkflow method to existing MockStorageProvider
func (m *MockStorageProvider) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	args := m.Called(ctx, workflowID, dryRun)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.WorkflowCleanupResult), args.Error(1)
}

func TestCleanupWorkflowHandler_RequiresConfirmation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockStorage := new(MockStorageProvider)
	router := gin.New()
	router.DELETE("/workflows/:workflow_id/cleanup", CleanupWorkflowHandler(mockStorage))

	req, _ := http.NewRequest("DELETE", "/workflows/test-workflow-123/cleanup", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "confirmation_required")
}

func TestCleanupWorkflowHandler_DryRun(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockStorage := new(MockStorageProvider)
	expectedResult := &types.WorkflowCleanupResult{
		WorkflowID: "test-workflow-123",
		DryRun:     true,
		DeletedRecords: map[string]int{
			"execution_vcs":       5,
			"workflow_vcs":        2,
			"workflow_executions": 10,
			"workflows":           1,
			"total":               18,
		},
		FreedSpaceBytes: 0,
		Success:         true,
		DurationMS:      150,
	}

	mockStorage.On("CleanupWorkflow", mock.Anything, "test-workflow-123", true).Return(expectedResult, nil)

	router := gin.New()
	router.DELETE("/workflows/:workflow_id/cleanup", CleanupWorkflowHandler(mockStorage))

	req, _ := http.NewRequest("DELETE", "/workflows/test-workflow-123/cleanup?dry_run=true", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "test-workflow-123")
	assert.Contains(t, w.Body.String(), "\"dry_run\":true")

	mockStorage.AssertExpectations(t)
}

func TestCleanupWorkflowHandler_UIRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockStorage := new(MockStorageProvider)
	expectedResult := &types.WorkflowCleanupResult{
		WorkflowID: "ui-workflow-456",
		DryRun:     false,
		DeletedRecords: map[string]int{
			"workflow_executions": 3,
			"workflows":           1,
			"total":               4,
		},
		FreedSpaceBytes: 2048,
		Success:         true,
		DurationMS:      275,
	}

	mockStorage.On("CleanupWorkflow", mock.Anything, "ui-workflow-456", false).Return(expectedResult, nil)

	router := gin.New()
	router.DELETE("/workflows/:workflowId/cleanup", CleanupWorkflowHandler(mockStorage))

	req, _ := http.NewRequest("DELETE", "/workflows/ui-workflow-456/cleanup?confirm=true", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "ui-workflow-456")
	assert.Contains(t, w.Body.String(), "\"dry_run\":false")

	mockStorage.AssertExpectations(t)
}
