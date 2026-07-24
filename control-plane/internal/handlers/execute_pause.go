package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type executionPauseResumeRequest struct {
	Reason string `json:"reason"`
}

type executionPauseResponse struct {
	ExecutionID    string  `json:"execution_id"`
	PreviousStatus string  `json:"previous_status"`
	Status         string  `json:"status"`
	Reason         *string `json:"reason,omitempty"`
	PausedAt       string  `json:"paused_at"`
}

type executionResumeResponse struct {
	ExecutionID    string `json:"execution_id"`
	PreviousStatus string `json:"previous_status"`
	Status         string `json:"status"`
	ResumedAt      string `json:"resumed_at"`
}

func PauseExecutionHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		handlePauseResume(c, store, types.ExecutionStatusRunning, types.ExecutionStatusPaused)
	}
}

func ResumeExecutionHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		handlePauseResume(c, store, types.ExecutionStatusPaused, types.ExecutionStatusRunning)
	}
}

func handlePauseResume(c *gin.Context, store ExecutionStore, expectedFromStatus, nextStatus string) {
	executionID := c.Param("execution_id")
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	req, err := parsePauseResumeRequest(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}
	reason := strings.TrimSpace(req.Reason)

	reqCtx := c.Request.Context()
	exec, err := store.GetExecutionRecord(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get execution record")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if exec == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return
	}

	// workflow_executions may not exist for simple async executions; look it
	// up but treat a nil result as non-fatal — we only need it for the
	// secondary UpdateWorkflowExecution and for event metadata.
	wfExec, err := store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("workflow execution lookup failed (non-fatal)")
	}

	now := time.Now().UTC()
	var statusReason *string
	if reason != "" {
		statusReason = &reason
	}

	var previousStatus string
	updatedExec, err := store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		currentStatus := types.NormalizeExecutionStatus(current.Status)
		if currentStatus != expectedFromStatus {
			return nil, fmt.Errorf("execution is in '%s' state; must be '%s'", currentStatus, expectedFromStatus)
		}
		previousStatus = currentStatus
		current.Status = nextStatus
		current.StatusReason = statusReason
		current.UpdatedAt = now
		return current, nil
	})
	if err != nil {
		errMsg := err.Error()
		if strings.Contains(strings.ToLower(errMsg), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
			return
		}
		if strings.Contains(errMsg, "must be '") || strings.Contains(errMsg, "invalid execution state transition") {
			c.JSON(http.StatusConflict, gin.H{
				"error":   "invalid_state",
				"message": errMsg,
			})
			return
		}
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to update execution record")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution status"})
		return
	}

	// Keep workflow_executions in sync when the row exists.
	if wfExec != nil {
		err = store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", executionID)
			}
			current.Status = nextStatus
			current.StatusReason = statusReason
			current.UpdatedAt = now
			return current, nil
		})
		if err != nil {
			logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to update workflow execution (non-fatal)")
		}
	}

	eventData := map[string]interface{}{"reason": reason}
	if nextStatus == types.ExecutionStatusPaused {
		events.PublishExecutionPaused(executionID, updatedExec.RunID, updatedExec.AgentNodeID, eventData)
	} else {
		events.PublishExecutionResumed(executionID, updatedExec.RunID, updatedExec.AgentNodeID, eventData)
	}

	// Derive event metadata from workflow_executions when available,
	// otherwise fall back to the execution record.
	workflowID := updatedExec.RunID
	runID := &updatedExec.RunID
	if wfExec != nil {
		workflowID = wfExec.WorkflowID
		runID = wfExec.RunID
	}
	eventType := "execution.resumed"
	if nextStatus == types.ExecutionStatusPaused {
		eventType = "execution.paused"
	}
	payload, _ := json.Marshal(map[string]interface{}{"reason": reason})
	statusCopy := nextStatus
	event := &types.WorkflowExecutionEvent{
		ExecutionID:  executionID,
		WorkflowID:   workflowID,
		RunID:        runID,
		EventType:    eventType,
		Status:       &statusCopy,
		StatusReason: statusReason,
		Payload:      payload,
		EmittedAt:    now,
	}
	if storeErr := store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store execution pause/resume event")
	}

	if nextStatus == types.ExecutionStatusPaused {
		response := executionPauseResponse{
			ExecutionID:    executionID,
			PreviousStatus: previousStatus,
			Status:         nextStatus,
			Reason:         statusReason,
			PausedAt:       now.Format(time.RFC3339),
		}
		c.JSON(http.StatusOK, response)
		return
	}

	response := executionResumeResponse{
		ExecutionID:    executionID,
		PreviousStatus: previousStatus,
		Status:         nextStatus,
		ResumedAt:      now.Format(time.RFC3339),
	}
	c.JSON(http.StatusOK, response)
}

func parsePauseResumeRequest(c *gin.Context) (*executionPauseResumeRequest, error) {
	if c.Request == nil || c.Request.Body == nil {
		return &executionPauseResumeRequest{}, nil
	}

	var req executionPauseResumeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		if errors.Is(err, io.EOF) {
			return &executionPauseResumeRequest{}, nil
		}
		return nil, err
	}

	return &req, nil
}
