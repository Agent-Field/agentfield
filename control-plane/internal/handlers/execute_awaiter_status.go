package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// awaiterStatusReason marks WAITING transitions driven by the awaiter-status
// endpoint, so the matching RUNNING transition can be distinguished from one
// that would resume an approval (which has its own status_reason).
const awaiterStatusReason = "awaiting_child"

// AwaiterStatusRequest is the body for POST /executions/:execution_id/awaiter-status.
//
// An agent calls this on its OWN execution_id when its `app.call()` wait loop
// observes the awaited child enter or leave WAITING. The point is to make the
// awaiter visible as WAITING to anyone watching it — without this, pause-state
// only propagates one hop up the call tree, and a great-grandparent times out
// at wallclock even though a leaf descendant is paused on human approval.
type AwaiterStatusRequest struct {
	// Status is "waiting" (RUNNING -> WAITING) or "running" (WAITING -> RUNNING).
	Status string `json:"status" binding:"required"`
	// Reason is a free-text marker for observability, e.g. "awaiting child exec_abc".
	Reason string `json:"reason,omitempty"`
}

// AwaiterStatusResponse is returned to the agent. ``Applied`` indicates whether
// the transition actually changed state; the SDK treats no-ops as success.
type AwaiterStatusResponse struct {
	Status  string `json:"status"`
	Applied bool   `json:"applied"`
}

// awaiterStatusController handles POST /executions/:execution_id/awaiter-status.
type awaiterStatusController struct {
	store ExecutionStore
}

// UpdateAwaiterStatusHandler returns a gin handler for the agent-scoped
// awaiter-status route. Mirrors the agent-scoped approval handler.
func UpdateAwaiterStatusHandler(store ExecutionStore) gin.HandlerFunc {
	ctrl := &awaiterStatusController{store: store}
	approvalCtrl := &approvalController{store: store}
	return func(ctx *gin.Context) {
		nodeID := ctx.Param("node_id")
		executionID := ctx.Param("execution_id")
		if nodeID == "" || executionID == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "node_id and execution_id are required"})
			return
		}
		if !approvalCtrl.verifyExecutionOwnership(ctx, executionID, nodeID) {
			return
		}
		ctrl.handle(ctx)
	}
}

func (c *awaiterStatusController) handle(ctx *gin.Context) {
	executionID := ctx.Param("execution_id")

	var req AwaiterStatusRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}
	if req.Status != "waiting" && req.Status != "running" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "status must be 'waiting' or 'running'"})
		return
	}

	reqCtx := ctx.Request.Context()

	wfExec, err := c.store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to load execution for awaiter-status update")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if wfExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return
	}

	currentStatus := types.NormalizeExecutionStatus(wfExec.Status)
	if types.IsTerminalExecutionStatus(currentStatus) {
		// Race: child resolved after we fired the cascade. Benign — the
		// awaiter has already moved on. Report no-op so the SDK swallows.
		ctx.JSON(http.StatusOK, AwaiterStatusResponse{Status: currentStatus, Applied: false})
		return
	}

	var (
		targetStatus types.ExecutionStatus
		targetReason string
	)
	switch req.Status {
	case "waiting":
		// Only cascade from RUNNING. If already WAITING (regardless of cause)
		// we leave it alone — an approval-driven WAITING must not be
		// converted to an awaiter-driven one, and an awaiter-driven WAITING
		// is already what we'd be setting.
		if currentStatus != string(types.ExecutionStatusRunning) {
			ctx.JSON(http.StatusOK, AwaiterStatusResponse{Status: currentStatus, Applied: false})
			return
		}
		targetStatus = types.ExecutionStatusWaiting
		targetReason = awaiterStatusReason
	case "running":
		// Only resume from WAITING set BY US. If status_reason is something
		// else (e.g. waiting_for_approval), an approval is still pending and
		// must not be silently resumed.
		if currentStatus != string(types.ExecutionStatusWaiting) {
			ctx.JSON(http.StatusOK, AwaiterStatusResponse{Status: currentStatus, Applied: false})
			return
		}
		if wfExec.StatusReason == nil || *wfExec.StatusReason != awaiterStatusReason {
			// Not our WAITING to resume.
			ctx.JSON(http.StatusOK, AwaiterStatusResponse{Status: currentStatus, Applied: false})
			return
		}
		targetStatus = types.ExecutionStatusRunning
		targetReason = ""
	}

	now := time.Now().UTC()
	var reasonPtr *string
	if targetReason != "" {
		reasonPtr = &targetReason
	}

	// Update the lightweight execution record so downstream readers (UI,
	// polling-side overdue checks) see the new status.
	_, updateErr := c.store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = targetStatus
		current.StatusReason = reasonPtr
		return current, nil
	})
	if updateErr != nil {
		logger.Logger.Error().Err(updateErr).Str("execution_id", executionID).Msg("failed to update execution record for awaiter-status")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution status"})
		return
	}

	// Update the workflow execution row.
	err = c.store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = targetStatus
		current.StatusReason = reasonPtr
		return current, nil
	})
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to update workflow execution for awaiter-status")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution"})
		return
	}

	// Emit observability event so the timeline reflects the cascade.
	statusStr := string(targetStatus)
	eventStatus := targetStatus
	eventType := "execution.running"
	if targetStatus == types.ExecutionStatusWaiting {
		eventType = "execution.waiting"
	}
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"reason":    req.Reason,
		"wait_kind": "awaiter_cascade",
	})
	event := &types.WorkflowExecutionEvent{
		ExecutionID:  executionID,
		WorkflowID:   wfExec.WorkflowID,
		RunID:        wfExec.RunID,
		EventType:    eventType,
		Status:       &eventStatus,
		StatusReason: reasonPtr,
		Payload:      eventPayload,
		EmittedAt:    now,
	}
	if storeErr := c.store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store awaiter-status event (non-fatal)")
	}

	// Publish to the execution event bus.
	if bus := c.store.GetExecutionEventBus(); bus != nil {
		busType := events.ExecutionResumed
		if targetStatus == types.ExecutionStatusWaiting {
			busType = events.ExecutionWaiting
		}
		bus.Publish(events.ExecutionEvent{
			Type:        busType,
			ExecutionID: executionID,
			WorkflowID:  wfExec.WorkflowID,
			AgentNodeID: wfExec.AgentNodeID,
			Status:      targetStatus,
			Timestamp:   now,
			Data: map[string]interface{}{
				"reason":    req.Reason,
				"wait_kind": "awaiter_cascade",
			},
		})
	}

	logger.Logger.Debug().
		Str("execution_id", executionID).
		Str("status", statusStr).
		Str("reason", req.Reason).
		Msg("awaiter-status cascade applied")

	ctx.JSON(http.StatusOK, AwaiterStatusResponse{Status: statusStr, Applied: true})
}
