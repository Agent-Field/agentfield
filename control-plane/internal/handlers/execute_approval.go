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

// approvalController handles approval-related endpoints.
// The control plane manages execution state only — the agent is responsible
// for communicating with external approval services (e.g. hax-sdk).
type approvalController struct {
	store ExecutionStore
}

// RequestApprovalRequest is the body for POST /executions/:execution_id/request-approval.
// The agent creates the approval request externally and passes the metadata here
// so the CP can track it and transition the execution to "waiting".
type RequestApprovalRequest struct {
	ApprovalRequestID  string `json:"approval_request_id" binding:"required"`
	ApprovalRequestURL string `json:"approval_request_url,omitempty"`
	CallbackURL        string `json:"callback_url,omitempty"`
	ExpiresInHours     *int   `json:"expires_in_hours,omitempty"`
}

// RequestApprovalResponse is returned when the execution transitions to waiting.
type RequestApprovalResponse struct {
	ApprovalRequestID  string `json:"approval_request_id"`
	ApprovalRequestURL string `json:"approval_request_url"`
	Status             string `json:"status"`
}

// ApprovalStatusResponse is returned by GET /executions/:execution_id/approval-status.
type ApprovalStatusResponse struct {
	Status      string  `json:"status"`
	Response    *string `json:"response,omitempty"`
	RequestURL  string  `json:"request_url,omitempty"`
	RequestedAt string  `json:"requested_at,omitempty"`
	RespondedAt *string `json:"responded_at,omitempty"`
	ExpiresAt   *string `json:"expires_at,omitempty"`
}

// RequestApprovalHandler transitions an execution to "waiting" and stores
// the approval metadata provided by the agent.
func RequestApprovalHandler(store ExecutionStore) gin.HandlerFunc {
	ctrl := &approvalController{store: store}
	return ctrl.handleRequestApproval
}

// GetApprovalStatusHandler returns the approval status for an execution.
func GetApprovalStatusHandler(store ExecutionStore) gin.HandlerFunc {
	ctrl := &approvalController{store: store}
	return ctrl.handleGetApprovalStatus
}

func (c *approvalController) handleRequestApproval(ctx *gin.Context) {
	executionID := ctx.Param("execution_id")
	if executionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	var req RequestApprovalRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}

	reqCtx := ctx.Request.Context()

	// Look up the workflow execution and validate state
	wfExec, err := c.store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get workflow execution for approval request")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if wfExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return
	}

	// Execution must be in running state to request approval
	normalized := types.NormalizeExecutionStatus(wfExec.Status)
	if normalized != types.ExecutionStatusRunning {
		ctx.JSON(http.StatusConflict, gin.H{
			"error":   "invalid_state",
			"message": fmt.Sprintf("execution is in '%s' state; must be 'running' to request approval", normalized),
		})
		return
	}

	// Prevent duplicate approval requests.  Only block if there is an
	// active (pending) approval.  If a previous approval was resolved
	// (approved / request_changes), the agent is allowed to start a new
	// approval round within the same execution (multi-pause workflows).
	approvalPending := wfExec.ApprovalStatus == nil || *wfExec.ApprovalStatus == "pending"
	if wfExec.ApprovalRequestID != nil && *wfExec.ApprovalRequestID != "" && approvalPending {
		ctx.JSON(http.StatusConflict, gin.H{
			"error":                "approval_already_requested",
			"message":              "An approval request already exists for this execution",
			"approval_request_id": *wfExec.ApprovalRequestID,
		})
		return
	}

	now := time.Now().UTC()
	statusReason := "waiting_for_approval"
	approvalStatus := "pending"

	// Compute expiry timestamp from expires_in_hours (default 72h)
	expiryHours := 72
	if req.ExpiresInHours != nil && *req.ExpiresInHours > 0 {
		expiryHours = *req.ExpiresInHours
	}
	expiresAt := now.Add(time.Duration(expiryHours) * time.Hour)

	// Transition the lightweight execution record to waiting
	_, updateErr := c.store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = types.ExecutionStatusWaiting
		current.StatusReason = &statusReason
		return current, nil
	})
	if updateErr != nil {
		logger.Logger.Error().Err(updateErr).Str("execution_id", executionID).Msg("failed to update execution record to waiting")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution status"})
		return
	}

	// Update the workflow execution with approval metadata + waiting status
	err = c.store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = types.ExecutionStatusWaiting
		current.StatusReason = &statusReason
		current.ApprovalRequestID = &req.ApprovalRequestID
		if req.ApprovalRequestURL != "" {
			current.ApprovalRequestURL = &req.ApprovalRequestURL
		}
		current.ApprovalStatus = &approvalStatus
		current.ApprovalRequestedAt = &now
		if req.CallbackURL != "" {
			current.ApprovalCallbackURL = &req.CallbackURL
		}
		current.ApprovalExpiresAt = &expiresAt
		return current, nil
	})
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to update workflow execution with approval data")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution with approval data"})
		return
	}

	// Emit execution event for observability
	waitingStatus := types.ExecutionStatusWaiting
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"approval_request_id":  req.ApprovalRequestID,
		"approval_request_url": req.ApprovalRequestURL,
		"wait_kind":            "approval",
	})
	event := &types.WorkflowExecutionEvent{
		ExecutionID: executionID,
		WorkflowID:  wfExec.WorkflowID,
		RunID:       wfExec.RunID,
		EventType:   "execution.waiting",
		Status:      &waitingStatus,
		StatusReason: &statusReason,
		Payload:     eventPayload,
		EmittedAt:   now,
	}
	if storeErr := c.store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store approval event (non-fatal)")
	}

	// Publish waiting event to the execution event bus
	if bus := c.store.GetExecutionEventBus(); bus != nil {
		bus.Publish(events.ExecutionEvent{
			Type:        events.ExecutionWaiting,
			ExecutionID: executionID,
			WorkflowID:  wfExec.WorkflowID,
			AgentNodeID: wfExec.AgentNodeID,
			Status:      types.ExecutionStatusWaiting,
			Timestamp:   now,
			Data: map[string]interface{}{
				"status_reason":        statusReason,
				"approval_request_id":  req.ApprovalRequestID,
				"approval_request_url": req.ApprovalRequestURL,
				"wait_kind":            "approval",
			},
		})
	}

	logger.Logger.Info().
		Str("execution_id", executionID).
		Str("approval_request_id", req.ApprovalRequestID).
		Msg("execution transitioned to waiting for approval")

	ctx.JSON(http.StatusOK, RequestApprovalResponse{
		ApprovalRequestID:  req.ApprovalRequestID,
		ApprovalRequestURL: req.ApprovalRequestURL,
		Status:             "pending",
	})
}

func (c *approvalController) handleGetApprovalStatus(ctx *gin.Context) {
	executionID := ctx.Param("execution_id")
	if executionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	reqCtx := ctx.Request.Context()
	wfExec, err := c.store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get workflow execution for approval status")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if wfExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return
	}

	if wfExec.ApprovalRequestID == nil {
		ctx.JSON(http.StatusNotFound, gin.H{
			"error":   "no_approval_request",
			"message": "No approval request exists for this execution",
		})
		return
	}

	status := "unknown"
	if wfExec.ApprovalStatus != nil {
		status = *wfExec.ApprovalStatus
	}

	requestedAt := ""
	if wfExec.ApprovalRequestedAt != nil {
		requestedAt = wfExec.ApprovalRequestedAt.Format(time.RFC3339)
	}

	var respondedAt *string
	if wfExec.ApprovalRespondedAt != nil {
		formatted := wfExec.ApprovalRespondedAt.Format(time.RFC3339)
		respondedAt = &formatted
	}

	requestURL := ""
	if wfExec.ApprovalRequestURL != nil {
		requestURL = *wfExec.ApprovalRequestURL
	}

	var expiresAtStr *string
	if wfExec.ApprovalExpiresAt != nil {
		formatted := wfExec.ApprovalExpiresAt.Format(time.RFC3339)
		expiresAtStr = &formatted
	}

	ctx.JSON(http.StatusOK, ApprovalStatusResponse{
		Status:      status,
		Response:    wfExec.ApprovalResponse,
		RequestURL:  requestURL,
		RequestedAt: requestedAt,
		ExpiresAt:   expiresAtStr,
		RespondedAt: respondedAt,
	})
}

// AgentScopedRequestApprovalHandler is the agent-scoped version of RequestApprovalHandler.
// It enforces that the execution belongs to the agent identified by :node_id.
func AgentScopedRequestApprovalHandler(store ExecutionStore) gin.HandlerFunc {
	ctrl := &approvalController{store: store}
	return func(ctx *gin.Context) {
		nodeID := ctx.Param("node_id")
		executionID := ctx.Param("execution_id")
		if nodeID == "" || executionID == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "node_id and execution_id are required"})
			return
		}

		// Verify the execution belongs to this agent
		if !ctrl.verifyExecutionOwnership(ctx, executionID, nodeID) {
			return // verifyExecutionOwnership writes the response
		}

		// Delegate to the standard handler
		ctrl.handleRequestApproval(ctx)
	}
}

// AgentScopedGetApprovalStatusHandler is the agent-scoped version of GetApprovalStatusHandler.
// It enforces that the execution belongs to the agent identified by :node_id.
func AgentScopedGetApprovalStatusHandler(store ExecutionStore) gin.HandlerFunc {
	ctrl := &approvalController{store: store}
	return func(ctx *gin.Context) {
		nodeID := ctx.Param("node_id")
		executionID := ctx.Param("execution_id")
		if nodeID == "" || executionID == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "node_id and execution_id are required"})
			return
		}

		// Verify the execution belongs to this agent
		if !ctrl.verifyExecutionOwnership(ctx, executionID, nodeID) {
			return
		}

		// Delegate to the standard handler
		ctrl.handleGetApprovalStatus(ctx)
	}
}

// verifyExecutionOwnership checks that the execution's AgentNodeID matches the
// node_id from the URL path. Returns false and writes an error response if
// verification fails; returns true if the caller may proceed.
func (c *approvalController) verifyExecutionOwnership(ctx *gin.Context, executionID, nodeID string) bool {
	reqCtx := ctx.Request.Context()
	wfExec, err := c.store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get workflow execution for ownership check")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return false
	}
	if wfExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return false
	}
	if wfExec.AgentNodeID != nodeID {
		logger.Logger.Warn().
			Str("execution_id", executionID).
			Str("requested_node", nodeID).
			Str("actual_node", wfExec.AgentNodeID).
			Msg("agent-scoped approval request denied: execution belongs to a different agent")
		ctx.JSON(http.StatusForbidden, gin.H{
			"error":   "execution_ownership_mismatch",
			"message": "this execution does not belong to the requesting agent",
		})
		return false
	}
	return true
}
