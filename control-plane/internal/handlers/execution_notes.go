package handlers

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ExecutionNoteStorage captures the storage operations required for execution note handlers.
type ExecutionNoteStorage interface {
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	UpdateExecutionRecord(ctx context.Context, executionID string, updateFunc func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	GetExecutionEventBus() *events.ExecutionEventBus
}

// AddNoteRequest represents the request body for adding a note to an execution
type AddNoteRequest struct {
	Message string   `json:"message" binding:"required"`
	Tags    []string `json:"tags"`
}

// AddNoteResponse represents the response for adding a note
type AddNoteResponse struct {
	Success bool                `json:"success"`
	Note    types.ExecutionNote `json:"note"`
	Message string              `json:"message"`
}

// GetNotesResponse represents the response for getting execution notes
type GetNotesResponse struct {
	ExecutionID string                `json:"execution_id"`
	Notes       []types.ExecutionNote `json:"notes"`
	Total       int                   `json:"total"`
}

// AddExecutionNoteHandler handles POST /api/v1/executions/note
// Adds a note to the current execution context
func AddExecutionNoteHandler(storageProvider ExecutionNoteStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req AddNoteRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid request body: %v", err)})
			return
		}

		// Get execution ID from context or header
		executionID := getExecutionIDFromContext(c)
		if executionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required in context or X-Execution-ID header"})
			return
		}

		// Validate message
		if strings.TrimSpace(req.Message) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "message cannot be empty"})
			return
		}

		// Create the note
		note := types.ExecutionNote{
			Message:   strings.TrimSpace(req.Message),
			Tags:      req.Tags,
			Timestamp: time.Now(),
		}

		// Ensure tags is not nil
		if note.Tags == nil {
			note.Tags = []string{}
		}

		// Update the execution with the new note
		ctx := context.Background()
		var runID string
		updated, err := storageProvider.UpdateExecutionRecord(ctx, executionID, func(execution *types.Execution) (*types.Execution, error) {
			if execution == nil {
				return nil, fmt.Errorf("execution with ID %s not found", executionID)
			}

			// Store run ID for SSE event (run_id is the workflow ID equivalent)
			runID = execution.RunID

			// Initialize notes if nil
			if execution.Notes == nil {
				execution.Notes = []types.ExecutionNote{}
			}

			// Add the new note
			execution.Notes = append(execution.Notes, note)
			execution.UpdatedAt = time.Now()

			return execution, nil
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to add note: %v", err)})
			return
		}

		// Broadcast SSE event for workflow node notes if update was successful
		if updated != nil && runID != "" {
			event := events.ExecutionEvent{
				Type:        "workflow_note_added",
				ExecutionID: executionID,
				WorkflowID:  runID, // Use run_id as workflow_id for SSE events
				AgentNodeID: updated.AgentNodeID,
				Status:      "note_added",
				Timestamp:   time.Now(),
				Data: map[string]interface{}{
					"workflow_id":  runID,
					"execution_id": executionID,
					"note":         note,
					"timestamp":    time.Now().Format(time.RFC3339),
				},
			}
			storageProvider.GetExecutionEventBus().Publish(event)
		}

		c.JSON(http.StatusOK, AddNoteResponse{
			Success: true,
			Note:    note,
			Message: "Note added successfully",
		})
	}
}

// GetExecutionNotesHandler handles GET /api/v1/executions/:execution_id/notes
// Retrieves notes for a specific execution with optional tag filtering
func GetExecutionNotesHandler(storageProvider ExecutionNoteStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		executionID := c.Param("execution_id")
		if executionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
			return
		}

		// Get optional tag filter from query parameters
		tagFilter := c.Query("tags")
		var filterTags []string
		if tagFilter != "" {
			filterTags = strings.Split(tagFilter, ",")
			// Trim whitespace from tags
			for i, tag := range filterTags {
				filterTags[i] = strings.TrimSpace(tag)
			}
		}

		// Get the execution
		ctx := context.Background()
		execution, err := storageProvider.GetExecutionRecord(ctx, executionID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get execution: %v", err)})
			return
		}

		if execution == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "execution not found"})
			return
		}

		// Filter notes by tags if specified
		var filteredNotes []types.ExecutionNote
		if len(filterTags) > 0 {
			for _, note := range execution.Notes {
				if noteHasTags(note, filterTags) {
					filteredNotes = append(filteredNotes, note)
				}
			}
		} else {
			filteredNotes = execution.Notes
		}

		// Ensure notes is not nil for JSON response
		if filteredNotes == nil {
			filteredNotes = []types.ExecutionNote{}
		}

		c.JSON(http.StatusOK, GetNotesResponse{
			ExecutionID: executionID,
			Notes:       filteredNotes,
			Total:       len(filteredNotes),
		})
	}
}

// getExecutionIDFromContext extracts execution ID from gin context or headers
func getExecutionIDFromContext(c *gin.Context) string {
	// First try to get from context (set by middleware or previous handlers)
	if executionID, exists := c.Get("execution_id"); exists {
		if id, ok := executionID.(string); ok {
			return id
		}
	}

	// Then try to get from X-Execution-ID header
	if executionID := c.GetHeader("X-Execution-ID"); executionID != "" {
		return executionID
	}

	// Finally try to get from query parameter (fallback)
	return c.Query("execution_id")
}

// noteHasTags checks if a note contains any of the specified tags
func noteHasTags(note types.ExecutionNote, filterTags []string) bool {
	if len(filterTags) == 0 {
		return true
	}

	noteTagsMap := make(map[string]bool)
	for _, tag := range note.Tags {
		noteTagsMap[strings.ToLower(strings.TrimSpace(tag))] = true
	}

	for _, filterTag := range filterTags {
		if noteTagsMap[strings.ToLower(strings.TrimSpace(filterTag))] {
			return true
		}
	}

	return false
}
