package handlers

import (
	"context"
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

type cancelExecutionRequest struct {
	Reason string `json:"reason"`
}

type cancelExecutionResponse struct {
	ExecutionID          string  `json:"execution_id"`
	PreviousStatus       string  `json:"previous_status"`
	Status               string  `json:"status"`
	Reason               *string `json:"reason,omitempty"`
	CancelledAt          string  `json:"cancelled_at"`
	Scope                string  `json:"scope"`
	RunID                string  `json:"run_id,omitempty"`
	RemainingActiveInRun int     `json:"remaining_active_in_run,omitempty"`
	SuggestedEndpoint    string  `json:"suggested_endpoint,omitempty"`
	Warning              string  `json:"warning,omitempty"`
}

func CancelExecutionHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		executionID := strings.TrimSpace(c.Param("execution_id"))
		if executionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
			return
		}

		var req cancelExecutionRequest
		if err := c.ShouldBindJSON(&req); err != nil && !errors.Is(err, io.EOF) {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
			return
		}

		reason := strings.TrimSpace(req.Reason)
		var reasonPtr *string
		if reason != "" {
			reasonPtr = &reason
		}

		reqCtx := c.Request.Context()
		exec, err := store.GetExecutionRecord(reqCtx, executionID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get execution record for cancellation")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
			return
		}
		if exec == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
			return
		}

		// workflow_executions may not exist for simple async executions;
		// treat a nil result as non-fatal.
		wfExec, err := store.GetWorkflowExecution(reqCtx, executionID)
		if err != nil {
			logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("workflow execution lookup failed (non-fatal)")
		}

		now := time.Now().UTC()

		var previousStatus string
		exec, err = store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", executionID)
			}
			if types.IsTerminalExecutionStatus(current.Status) {
				return nil, fmt.Errorf("execution is in terminal state '%s'", current.Status)
			}
			previousStatus = current.Status
			current.Status = types.ExecutionStatusCancelled
			if reasonPtr != nil {
				current.StatusReason = reasonPtr
			}
			return current, nil
		})
		if err != nil {
			errMsg := err.Error()
			if strings.Contains(errMsg, "terminal state") || strings.Contains(errMsg, "invalid execution state transition") {
				c.JSON(http.StatusConflict, gin.H{
					"error":   "invalid_state",
					"message": errMsg,
				})
				return
			}
			logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to update execution record for cancellation")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to cancel execution"})
			return
		}

		// Keep workflow_executions in sync when the row exists.
		if wfExec != nil {
			err = store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
				if current == nil {
					return nil, fmt.Errorf("execution %s not found", executionID)
				}
				current.Status = types.ExecutionStatusCancelled
				if reasonPtr != nil {
					current.StatusReason = reasonPtr
				}
				return current, nil
			})
			if err != nil {
				logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to update workflow execution for cancellation (non-fatal)")
			}
		}

		events.PublishExecutionCancelled(executionID, exec.RunID, exec.AgentNodeID, map[string]interface{}{"reason": reason})

		payload, marshalErr := json.Marshal(map[string]interface{}{
			"reason": reason,
		})
		if marshalErr != nil {
			payload = json.RawMessage(`{"reason":""}`)
		}

		// Derive event metadata from workflow_executions when available,
		// otherwise fall back to the execution record.
		workflowID := exec.RunID
		runID := &exec.RunID
		if wfExec != nil {
			workflowID = wfExec.WorkflowID
			runID = wfExec.RunID
		}

		cancelledStatus := types.ExecutionStatusCancelled
		workflowEvent := &types.WorkflowExecutionEvent{
			ExecutionID:  executionID,
			WorkflowID:   workflowID,
			RunID:        runID,
			EventType:    "execution.cancelled",
			Status:       &cancelledStatus,
			StatusReason: reasonPtr,
			Payload:      payload,
			EmittedAt:    now,
		}
		if eventErr := store.StoreWorkflowExecutionEvent(reqCtx, workflowEvent); eventErr != nil {
			logger.Logger.Warn().Err(eventErr).Str("execution_id", executionID).Msg("failed to store execution cancelled event (non-fatal)")
		}

		response := cancelExecutionResponse{
			ExecutionID:    executionID,
			PreviousStatus: previousStatus,
			Status:         types.ExecutionStatusCancelled,
			Reason:         reasonPtr,
			CancelledAt:    now.Format(time.RFC3339),
			Scope:          "execution",
			RunID:          exec.RunID,
		}
		remaining, queryErr := countRemainingActiveInRun(reqCtx, store, exec.RunID, executionID)
		if queryErr != nil {
			logger.Logger.Warn().Err(queryErr).Str("run_id", exec.RunID).Str("execution_id", executionID).Msg("failed to inspect run after single-execution cancel")
		} else if remaining > 0 {
			response.RemainingActiveInRun = remaining
			response.SuggestedEndpoint = fmt.Sprintf("/api/v1/workflows/%s/cancel-tree", exec.RunID)
			response.Warning = fmt.Sprintf(
				"Cancelled only execution %s; %d other non-terminal execution(s) remain in run %s. Use cancel-tree to stop the whole run.",
				executionID,
				remaining,
				exec.RunID,
			)
		}
		c.JSON(http.StatusOK, response)
	}
}

func countRemainingActiveInRun(ctx context.Context, store ExecutionStore, runID string, cancelledExecutionID string) (int, error) {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return 0, nil
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			logger.Logger.Warn().Interface("panic", recovered).Str("run_id", runID).Str("execution_id", cancelledExecutionID).Msg("single-execution cancel advisory query panicked")
		}
	}()
	executions, err := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		RunID:          &runID,
		SortBy:         "started_at",
		SortDescending: false,
	})
	if err != nil {
		return 0, err
	}
	remaining := 0
	for _, exec := range executions {
		if exec == nil || exec.ExecutionID == cancelledExecutionID {
			continue
		}
		if !types.IsTerminalExecutionStatus(exec.Status) {
			remaining++
		}
	}
	return remaining, nil
}
