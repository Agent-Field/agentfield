package handlers

import (
	"net/http"
	"strings"

	"github.com/your-org/agentfield/control-plane/internal/storage"

	"github.com/gin-gonic/gin"
)

// WorkflowCleanupRequest represents the request body for workflow cleanup
type WorkflowCleanupRequest struct {
	Confirm bool `json:"confirm" form:"confirm"`
	DryRun  bool `json:"dry_run" form:"dry_run"`
}

// WorkflowCleanupResponse represents the response from workflow cleanup
type WorkflowCleanupResponse struct {
	Success         bool           `json:"success"`
	WorkflowID      string         `json:"workflow_id"`
	DeletedRecords  map[string]int `json:"deleted_records"`
	FreedSpaceBytes int64          `json:"freed_space_bytes"`
	DurationMS      int64          `json:"duration_ms"`
	DryRun          bool           `json:"dry_run,omitempty"`
	ErrorMessage    *string        `json:"error_message,omitempty"`
}

// CleanupWorkflowHandler handles DELETE /api/v1/workflows/{workflow_id}/cleanup
func CleanupWorkflowHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()

		// Get workflow ID from URL parameter (support both API and UI route patterns)
		workflowID := c.Param("workflow_id")
		if workflowID == "" {
			workflowID = c.Param("workflowId")
		}
		if workflowID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "workflow_id_required",
				"message": "Workflow ID is required",
			})
			return
		}

		// Parse query parameters
		dryRun := c.Query("dry_run") == "true"
		confirm := c.Query("confirm") == "true"

		// Parse request body if present (for POST-style requests)
		var req WorkflowCleanupRequest
		if c.Request.Method == "POST" || c.Request.ContentLength > 0 {
			if err := c.ShouldBindJSON(&req); err == nil {
				dryRun = req.DryRun
				confirm = req.Confirm
			}
		}

		// Safety check: require confirmation for non-dry-run operations
		if !dryRun && !confirm {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "confirmation_required",
				"message": "Confirmation required for workflow cleanup. Use ?confirm=true or set confirm: true in request body",
			})
			return
		}

		// Perform the cleanup
		result, err := storageProvider.CleanupWorkflow(ctx, workflowID, dryRun)
		if err != nil {
			// Determine appropriate HTTP status code based on error
			statusCode := http.StatusInternalServerError
			errorType := "internal_error"

			if result != nil && result.ErrorMessage != nil {
				errorMsg := *result.ErrorMessage
				if strings.Contains(errorMsg, "not found") {
					statusCode = http.StatusNotFound
					errorType = "workflow_not_found"
				} else if strings.Contains(errorMsg, "active") || strings.Contains(errorMsg, "running") {
					statusCode = http.StatusConflict
					errorType = "workflow_active"
				} else if strings.Contains(errorMsg, "empty") {
					statusCode = http.StatusBadRequest
					errorType = "invalid_workflow_id"
				}
			}

			c.JSON(statusCode, gin.H{
				"success": false,
				"error":   errorType,
				"message": err.Error(),
			})
			return
		}

		// Convert result to response format
		response := WorkflowCleanupResponse{
			Success:         result.Success,
			WorkflowID:      result.WorkflowID,
			DeletedRecords:  result.DeletedRecords,
			FreedSpaceBytes: result.FreedSpaceBytes,
			DurationMS:      result.DurationMS,
			DryRun:          result.DryRun,
			ErrorMessage:    result.ErrorMessage,
		}

		// Return appropriate status code
		statusCode := http.StatusOK
		if !result.Success {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, response)
	}
}
