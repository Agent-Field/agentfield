package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// cancel-tree exists because per-execution cancel cannot reach a run whose
// root execution has already terminated but whose children are still flagged
// non-terminal (zombied workers, fan-out reasoners that finished before
// their dispatched siblings, etc.). The user-facing model is "cancel the
// run" — this endpoint walks the DAG bottom-up and cancels every
// non-terminal execution under the run, leaving terminal ones alone.

type cancelTreeRequest struct {
	Reason string `json:"reason"`
}

type cancelTreeNodeResult struct {
	ExecutionID    string `json:"execution_id"`
	AgentNodeID    string `json:"agent_node_id,omitempty"`
	ReasonerID     string `json:"reasoner_id,omitempty"`
	WorkflowDepth  int    `json:"workflow_depth"`
	PreviousStatus string `json:"previous_status"`
	Status         string `json:"status"`
	SkipReason     string `json:"skip_reason,omitempty"`
}

type cancelTreeResponse struct {
	RunID          string                 `json:"run_id"`
	TotalNodes     int                    `json:"total_nodes"`
	CancelledCount int                    `json:"cancelled_count"`
	SkippedCount   int                    `json:"skipped_count"`
	ErrorCount     int                    `json:"error_count"`
	Nodes          []cancelTreeNodeResult `json:"nodes"`
	CancelledAt    string                 `json:"cancelled_at"`
}

// errAlreadyTerminal signals the storage update saw a terminal status under
// the lock — distinguished from a generic update failure so we can report
// it as a clean skip rather than an error.
var errAlreadyTerminal = errors.New("execution already terminal")

// CancelWorkflowTreeHandler cancels every non-terminal execution belonging
// to a run. Order is bottom-up by computed depth so leaves transition first
// and parents observe a consistent post-state before they themselves get
// cancelled. Already-terminal executions are left untouched.
func CancelWorkflowTreeHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		runID := strings.TrimSpace(c.Param("workflowId"))
		if runID == "" {
			runID = strings.TrimSpace(c.Param("workflow_id"))
		}
		if runID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "workflowId is required"})
			return
		}

		var req cancelTreeRequest
		if err := c.ShouldBindJSON(&req); err != nil && !errors.Is(err, io.EOF) {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
			return
		}
		reason := strings.TrimSpace(req.Reason)
		var reasonPtr *string
		if reason != "" {
			reasonPtr = &reason
		}

		ctx := c.Request.Context()

		executions, err := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
			RunID:          &runID,
			SortBy:         "started_at",
			SortDescending: false,
		})
		if err != nil {
			logger.Logger.Error().Err(err).Str("run_id", runID).Msg("cancel-tree: failed to load run executions")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load run"})
			return
		}
		if len(executions) == 0 {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("run %s not found", runID)})
			return
		}

		ordered, depthByID := orderExecutionsLeafFirst(executions)

		now := time.Now().UTC()
		results := make([]cancelTreeNodeResult, 0, len(ordered))
		var cancelledCount, skippedCount, errorCount int

		for _, exec := range ordered {
			if exec == nil {
				continue
			}
			depth := depthByID[exec.ExecutionID]
			result := cancelTreeNodeResult{
				ExecutionID:    exec.ExecutionID,
				AgentNodeID:    exec.AgentNodeID,
				ReasonerID:     exec.ReasonerID,
				WorkflowDepth:  depth,
				PreviousStatus: exec.Status,
			}

			if types.IsTerminalExecutionStatus(exec.Status) {
				result.Status = exec.Status
				result.SkipReason = "terminal"
				results = append(results, result)
				skippedCount++
				continue
			}

			updated, previousStatus, cancelErr := cancelOneExecution(ctx, store, exec.ExecutionID, reasonPtr, now, reason)
			switch {
			case errors.Is(cancelErr, errAlreadyTerminal):
				result.Status = "skipped"
				result.SkipReason = "raced_to_terminal"
				results = append(results, result)
				skippedCount++
			case cancelErr != nil:
				logger.Logger.Error().Err(cancelErr).Str("execution_id", exec.ExecutionID).Msg("cancel-tree: failed to cancel execution")
				result.Status = "error"
				result.SkipReason = cancelErr.Error()
				results = append(results, result)
				errorCount++
			default:
				if updated != nil {
					result.PreviousStatus = previousStatus
					result.Status = types.ExecutionStatusCancelled
				}
				results = append(results, result)
				cancelledCount++
			}
		}

		c.JSON(http.StatusOK, cancelTreeResponse{
			RunID:          runID,
			TotalNodes:     len(executions),
			CancelledCount: cancelledCount,
			SkippedCount:   skippedCount,
			ErrorCount:     errorCount,
			Nodes:          results,
			CancelledAt:    now.Format(time.RFC3339),
		})
	}
}

// cancelOneExecution applies the same status transition + sibling-table
// sync + event publishing as the per-execution cancel handler. Kept as a
// local helper rather than a refactor of execute_cancel.go to keep this
// PR's blast radius small. Returns errAlreadyTerminal if the execution
// raced to a terminal state under the storage lock.
func cancelOneExecution(
	ctx context.Context,
	store storage.StorageProvider,
	executionID string,
	reasonPtr *string,
	now time.Time,
	reasonRaw string,
) (*types.Execution, string, error) {
	wfExec, err := store.GetWorkflowExecution(ctx, executionID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("cancel-tree: workflow execution lookup failed (non-fatal)")
	}

	var previousStatus string
	updated, err := store.UpdateExecutionRecord(ctx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		if types.IsTerminalExecutionStatus(current.Status) {
			return nil, errAlreadyTerminal
		}
		previousStatus = current.Status
		current.Status = types.ExecutionStatusCancelled
		if reasonPtr != nil {
			current.StatusReason = reasonPtr
		}
		return current, nil
	})
	if err != nil {
		return nil, previousStatus, err
	}

	if wfExec != nil {
		if updateErr := store.UpdateWorkflowExecution(ctx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", executionID)
			}
			current.Status = types.ExecutionStatusCancelled
			if reasonPtr != nil {
				current.StatusReason = reasonPtr
			}
			return current, nil
		}); updateErr != nil {
			logger.Logger.Warn().Err(updateErr).Str("execution_id", executionID).Msg("cancel-tree: failed to sync workflow_executions (non-fatal)")
		}
	}

	events.PublishExecutionCancelled(updated.ExecutionID, updated.RunID, updated.AgentNodeID, map[string]interface{}{
		"reason": reasonRaw,
		"source": "cancel_tree",
	})

	payload, marshalErr := json.Marshal(map[string]interface{}{
		"reason": reasonRaw,
		"source": "cancel_tree",
	})
	if marshalErr != nil {
		payload = json.RawMessage(`{"reason":"","source":"cancel_tree"}`)
	}

	workflowID := updated.RunID
	runIDForEvent := &updated.RunID
	if wfExec != nil {
		workflowID = wfExec.WorkflowID
		runIDForEvent = wfExec.RunID
	}
	cancelledStatus := types.ExecutionStatusCancelled
	if eventErr := store.StoreWorkflowExecutionEvent(ctx, &types.WorkflowExecutionEvent{
		ExecutionID:  updated.ExecutionID,
		WorkflowID:   workflowID,
		RunID:        runIDForEvent,
		EventType:    "execution.cancelled",
		Status:       &cancelledStatus,
		StatusReason: reasonPtr,
		Payload:      payload,
		EmittedAt:    now,
	}); eventErr != nil {
		logger.Logger.Warn().Err(eventErr).Str("execution_id", updated.ExecutionID).Msg("cancel-tree: failed to store event (non-fatal)")
	}

	return updated, previousStatus, nil
}

// orderExecutionsLeafFirst returns executions sorted by depth descending so
// leaves are processed before their ancestors. Stable secondary order is
// most-recently-started first so older parents fall to the end.
func orderExecutionsLeafFirst(executions []*types.Execution) ([]*types.Execution, map[string]int) {
	depths := computeExecutionDepths(executions)

	ordered := make([]*types.Execution, 0, len(executions))
	for _, e := range executions {
		if e == nil {
			continue
		}
		ordered = append(ordered, e)
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		di, dj := depths[ordered[i].ExecutionID], depths[ordered[j].ExecutionID]
		if di != dj {
			return di > dj
		}
		return ordered[i].StartedAt.After(ordered[j].StartedAt)
	})
	return ordered, depths
}

// computeExecutionDepths walks parent_execution_id chains to assign each
// execution a depth from the run's root. Cycle-safe; orphan parents (parent
// id present but missing from the slice) are treated as depth 0.
func computeExecutionDepths(executions []*types.Execution) map[string]int {
	execMap := make(map[string]*types.Execution, len(executions))
	for _, e := range executions {
		if e == nil {
			continue
		}
		execMap[e.ExecutionID] = e
	}
	depths := make(map[string]int, len(executions))
	computing := make(map[string]bool)

	var compute func(e *types.Execution) int
	compute = func(e *types.Execution) int {
		if e == nil {
			return 0
		}
		if d, ok := depths[e.ExecutionID]; ok {
			return d
		}
		if computing[e.ExecutionID] {
			return 0
		}
		computing[e.ExecutionID] = true
		defer delete(computing, e.ExecutionID)

		d := 0
		if e.ParentExecutionID != nil && *e.ParentExecutionID != "" {
			if parent, ok := execMap[*e.ParentExecutionID]; ok {
				d = compute(parent) + 1
			}
		}
		depths[e.ExecutionID] = d
		return d
	}
	for _, e := range executions {
		compute(e)
	}
	return depths
}
