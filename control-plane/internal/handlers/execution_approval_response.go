package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"

	"github.com/gin-gonic/gin"
)

// ResolveApprovalRequest is the body for POST /executions/:execution_id/approval-response.
type ResolveApprovalRequest struct {
	Decision string          `json:"decision" binding:"required"`
	Reason   string          `json:"reason,omitempty"`
	Response json.RawMessage `json:"response,omitempty"`
}

// ResolveApprovalHandler resolves a pending approval by execution ID under the
// standard API auth middleware. The HMAC-signed webhook
// (/api/v1/webhooks/approval-response) is keyed by approval_request_id and
// requires the shared webhook secret held by the external approval service;
// this endpoint lets an authenticated operator (e.g. the CLI) resolve the same
// approval directly. Both paths share applyApprovalDecision, so state
// transitions, idempotency, events, and callbacks are identical.
//
// A caller that names itself with X-AgentField-Run-Owner (WorkAF's node) is
// held to the runs it owns, exactly as steer and the relay are: only the run's
// owner may answer its question. A caller that names no node is the API-key
// holder speaking for itself, which the control plane already trusts. scope
// is nil on a store with no run-owner table, and then nothing is refused.
func ResolveApprovalHandler(store ExecutionStore, scope ApprovalScope) gin.HandlerFunc {
	ctrl := &webhookApprovalController{store: store, scope: scope}
	return ctrl.handleResolveApproval
}

// ApprovalScope refuses a caller that may not answer exec's approval; its
// error carries the HTTP status (nodelink.CodeOf) and the words.
type ApprovalScope interface {
	Scope(ctx context.Context, caller, exec string) error
}

func (c *webhookApprovalController) handleResolveApproval(ctx *gin.Context) {
	executionID := ctx.Param("execution_id")
	if executionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	var req ResolveApprovalRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}

	// "expired" is deliberately excluded: expiry is an approval-service
	// outcome, not an operator decision.
	decision := normalizeDecision(strings.TrimSpace(req.Decision))
	switch decision {
	case "approved", "rejected", "request_changes":
		// valid
	default:
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_decision",
			"message": fmt.Sprintf("invalid decision '%s'; must be approved, rejected, or request_changes", req.Decision),
		})
		return
	}

	reqCtx := ctx.Request.Context()
	if c.scope != nil {
		if err := c.scope.Scope(reqCtx, strings.TrimSpace(ctx.GetHeader(RunOwnerHeader)), executionID); err != nil {
			ctx.JSON(nodelink.CodeOf(err), gin.H{"error": err.Error()})
			return
		}
	}
	wfExec, err := c.store.GetWorkflowExecution(reqCtx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get workflow execution for approval resolution")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if wfExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", executionID)})
		return
	}
	if wfExec.ApprovalRequestID == nil || *wfExec.ApprovalRequestID == "" {
		ctx.JSON(http.StatusNotFound, gin.H{
			"error":   "no_approval_request",
			"message": "No approval request exists for this execution",
		})
		return
	}

	payload := &ApprovalWebhookPayload{
		RequestID: *wfExec.ApprovalRequestID,
		Decision:  decision,
		Feedback:  req.Reason,
		Response:  req.Response,
	}

	c.applyApprovalDecision(ctx, executionID, wfExec, payload, decision)
}
