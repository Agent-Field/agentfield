package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ExecutionNoteStorage captures the storage operations required for execution note handlers.
type ExecutionNoteStorage interface {
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	UpdateExecutionRecord(ctx context.Context, executionID string, updateFunc func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	GetExecutionEventBus() *events.ExecutionEventBus
}

type executionNoteDIDDocumentLookup interface {
	GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error)
}

type executionNoteAgentDIDLister interface {
	ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error)
}

type executionNoteAuthorizationError struct {
	message string
}

func (e *executionNoteAuthorizationError) Error() string {
	return e.message
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
// Adds a note to the current execution context.
//
// ownershipEnforced reports whether the server runs with an authentication
// method (API key or DID auth) that yields a trusted caller identity. When true,
// the caller must own the target execution or the write is rejected with 403.
// When false the server is fully unauthenticated, there is no trustworthy
// identity to check against, and the ownership guard is skipped (the route wiring
// logs a startup warning in that mode).
func AddExecutionNoteHandler(storageProvider ExecutionNoteStorage, ownershipEnforced bool) gin.HandlerFunc {
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

		// Update the execution with the new note. Resolve the caller identity and
		// enforce execution ownership only when an auth method is active; otherwise
		// there is no trusted identity to compare against (see ownershipEnforced).
		ctx := c.Request.Context()
		var callerAgentID string
		if ownershipEnforced {
			var resolveErr error
			callerAgentID, resolveErr = executionNoteCallerAgentID(ctx, c, storageProvider)
			if resolveErr != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to resolve caller identity: %v", resolveErr)})
				return
			}
		}

		var runID string
		updated, err := storageProvider.UpdateExecutionRecord(ctx, executionID, func(execution *types.Execution) (*types.Execution, error) {
			if execution == nil {
				return nil, fmt.Errorf("execution with ID %s not found", executionID)
			}
			if ownershipEnforced {
				if err := ensureExecutionNoteOwnership(callerAgentID, execution); err != nil {
					return nil, err
				}
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
			var authzErr *executionNoteAuthorizationError
			if errors.As(err, &authzErr) {
				c.JSON(http.StatusForbidden, gin.H{
					"error":   "execution_ownership_mismatch",
					"message": authzErr.message,
				})
				return
			}
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

func ensureExecutionNoteOwnership(callerAgentID string, execution *types.Execution) error {
	ownerAgentID := strings.TrimSpace(execution.AgentNodeID)
	if ownerAgentID == "" {
		return &executionNoteAuthorizationError{message: "execution owner is required to add notes"}
	}

	if callerAgentID == "" {
		return &executionNoteAuthorizationError{message: "caller agent identity is required to add notes to this execution"}
	}
	if callerAgentID != ownerAgentID {
		return &executionNoteAuthorizationError{message: "this execution does not belong to the requesting agent"}
	}

	return nil
}

// executionNoteCallerAgentID resolves the caller's owning agent node ID from a
// trusted source only: a cryptographically verified DID (set by DIDAuthMiddleware)
// or the authenticated middleware context populated by APIKeyAuth after a
// successful key check. Raw X-Caller-Agent-ID / X-Agent-Node-ID request headers
// are deliberately NOT consulted here — without an auth layer validating the
// request they are attacker-controlled, so trusting them would let any caller
// spoof execution ownership. Returns "" when no trusted identity is present,
// which fails the ownership check closed.
func executionNoteCallerAgentID(ctx context.Context, c *gin.Context, storageProvider ExecutionNoteStorage) (string, error) {
	if callerDID := strings.TrimSpace(middleware.GetVerifiedCallerDID(c)); callerDID != "" {
		return resolveExecutionNoteAgentIDByDID(ctx, storageProvider, callerDID)
	}

	if callerID, exists := c.Get(string(middleware.CallerAgentIDKey)); exists {
		if id, ok := callerID.(string); ok {
			if id = strings.TrimSpace(id); id != "" {
				return id, nil
			}
		}
	}

	return "", nil
}

// resolveExecutionNoteAgentIDByDID maps a verified caller DID to its owning agent
// node ID, consulting the did:web document table first and falling back to the
// agent_dids registry.
//
// The two sources expose the same value under differently-named fields —
// DIDDocumentRecord.AgentID (did_documents.agent_id) and AgentDIDInfo.AgentNodeID
// (agent_dids.agent_node_id). They are kept equivalent at registration time
// (services/nodes_register.go populates both from the same agent node ID), so
// either is a valid owner identifier for the ownership comparison.
//
// Revoked entries are treated as unresolved (fail closed): a revoked DID must not
// resolve to an agent identity even if the surrounding auth layer admitted the
// request (e.g. a self-verifying did:key whose registry entry was later revoked).
func resolveExecutionNoteAgentIDByDID(ctx context.Context, storageProvider ExecutionNoteStorage, callerDID string) (string, error) {
	if lookup, ok := storageProvider.(executionNoteDIDDocumentLookup); ok {
		if record, err := lookup.GetDIDDocument(ctx, callerDID); err == nil && record != nil && !record.IsRevoked() {
			return strings.TrimSpace(record.AgentID), nil
		}
	}

	lister, ok := storageProvider.(executionNoteAgentDIDLister)
	if !ok {
		return "", nil
	}
	agentDIDs, err := lister.ListAgentDIDs(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to resolve caller DID: %w", err)
	}
	for _, info := range agentDIDs {
		if info == nil {
			continue
		}
		if info.Status == types.AgentDIDStatusRevoked {
			continue
		}
		if strings.TrimSpace(info.DID) == callerDID {
			return strings.TrimSpace(info.AgentNodeID), nil
		}
	}

	return "", nil
}

func ensureExecutionNoteReadOwnership(callerAgentID string, execution *types.Execution) error {
	ownerAgentID := strings.TrimSpace(execution.AgentNodeID)
	if ownerAgentID == "" {
		return &executionNoteAuthorizationError{message: "execution owner is required to read notes"}
	}

	if callerAgentID == "" {
		return &executionNoteAuthorizationError{message: "caller agent identity is required to read notes for this execution"}
	}
	if callerAgentID != ownerAgentID {
		return &executionNoteAuthorizationError{message: "this execution does not belong to the requesting agent"}
	}

	return nil
}

// GetExecutionNotesHandler handles GET /api/v1/executions/:execution_id/notes.
// Retrieves notes for a specific execution with optional tag filtering.
//
// ownershipEnforced reports whether the server runs with an authentication
// method that yields a trusted caller identity. When true, the caller must own
// the target execution or the read is rejected with 403. When false the server
// is fully unauthenticated, matching the write endpoint's S2 behavior.
func GetExecutionNotesHandler(storageProvider ExecutionNoteStorage, ownershipEnforced bool) gin.HandlerFunc {
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

		ctx := c.Request.Context()
		var callerAgentID string
		if ownershipEnforced {
			var resolveErr error
			callerAgentID, resolveErr = executionNoteCallerAgentID(ctx, c, storageProvider)
			if resolveErr != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to resolve caller identity: %v", resolveErr)})
				return
			}
			if callerAgentID == "" {
				c.JSON(http.StatusForbidden, gin.H{
					"error":   "execution_ownership_mismatch",
					"message": "caller agent identity is required to read notes for this execution",
				})
				return
			}
		}

		// Get the execution
		execution, err := storageProvider.GetExecutionRecord(ctx, executionID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get execution: %v", err)})
			return
		}

		if execution == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "execution not found"})
			return
		}

		if ownershipEnforced {
			if err := ensureExecutionNoteReadOwnership(callerAgentID, execution); err != nil {
				var authzErr *executionNoteAuthorizationError
				if errors.As(err, &authzErr) {
					c.JSON(http.StatusForbidden, gin.H{
						"error":   "execution_ownership_mismatch",
						"message": authzErr.message,
					})
					return
				}
			}
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
