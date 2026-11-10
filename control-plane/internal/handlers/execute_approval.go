package handlers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
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
//
// ApprovalRequestID is optional: a run with no external approval service (no
// hax) names none and the control plane mints one, so the request lives only
// here. Kind, Question, Body and Options carry the question itself, so any
// door can show a person the whole of what they are asked to judge.
type RequestApprovalRequest struct {
	ApprovalRequestID  string   `json:"approval_request_id,omitempty"`
	ApprovalRequestURL string   `json:"approval_request_url,omitempty"`
	CallbackURL        string   `json:"callback_url,omitempty"`
	ExpiresInHours     *int     `json:"expires_in_hours,omitempty"`
	Kind               string   `json:"kind,omitempty"`
	Question           string   `json:"question,omitempty"`
	Body               string   `json:"body,omitempty"`
	Options            []string `json:"options,omitempty"`
}

// The largest question a request may carry. THE BODY CAP KEEPS A RUN FRAME
// UNDER THE LINK'S 1 MiB CEILING: the body rides the frame that tells the
// run's owner the execution is waiting.
const (
	maxApprovalQuestion = 500
	maxApprovalBody     = 256 << 10
	maxApprovalOptions  = 12
	maxApprovalOption   = 200
)

// ApprovalQuestionStore keeps what an approval round asks. A store without
// it still pauses and resumes; it just cannot show the question.
type ApprovalQuestionStore interface {
	PutApprovalQuestion(ctx context.Context, q types.ApprovalQuestion) error
	GetApprovalQuestion(ctx context.Context, executionID, approvalRequestID string) (*types.ApprovalQuestion, error)
}

// normalizeApprovalQuestion checks and fills the question fields of req.
func normalizeApprovalQuestion(req *RequestApprovalRequest) error {
	req.Kind = strings.TrimSpace(req.Kind)
	switch req.Kind {
	case "":
		req.Kind = types.ApprovalKindApproval
	case types.ApprovalKindPlanReview, types.ApprovalKindQuestion, types.ApprovalKindApproval:
	default:
		return fmt.Errorf("kind must be %s, %s or %s", types.ApprovalKindPlanReview, types.ApprovalKindQuestion, types.ApprovalKindApproval)
	}
	req.Question = strings.TrimSpace(req.Question)
	if len(req.Question) > maxApprovalQuestion {
		return fmt.Errorf("question is at most %d bytes", maxApprovalQuestion)
	}
	if len(req.Body) > maxApprovalBody {
		return fmt.Errorf("body is at most %d bytes", maxApprovalBody)
	}
	opts := make([]string, 0, len(req.Options))
	for _, o := range req.Options {
		o = strings.TrimSpace(o)
		if o == "" {
			continue
		}
		if len(o) > maxApprovalOption {
			return fmt.Errorf("an option is at most %d bytes", maxApprovalOption)
		}
		opts = append(opts, o)
	}
	if len(opts) > maxApprovalOptions {
		return fmt.Errorf("at most %d options", maxApprovalOptions)
	}
	if len(opts) == 0 {
		opts = append(opts, types.DefaultApprovalOptions...)
	}
	req.Options = opts
	return nil
}

// mintApprovalRequestID names a request that no external service named.
func mintApprovalRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "apr_" + hex.EncodeToString(b[:])
}

// RequestApprovalResponse is returned when the execution transitions to waiting.
type RequestApprovalResponse struct {
	ApprovalRequestID  string `json:"approval_request_id"`
	ApprovalRequestURL string `json:"approval_request_url"`
	Status             string `json:"status"`
}

// ApprovalStatusResponse is returned by GET /executions/:execution_id/approval-status.
type ApprovalStatusResponse struct {
	Status            string   `json:"status"`
	ApprovalRequestID string   `json:"approval_request_id,omitempty"`
	Kind              string   `json:"kind,omitempty"`
	Question          string   `json:"question,omitempty"`
	Body              string   `json:"body,omitempty"`
	Options           []string `json:"options,omitempty"`
	// Feedback is the answerer's words (the reason given with the decision).
	Feedback    string  `json:"feedback,omitempty"`
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
	if err := normalizeApprovalQuestion(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid_question", "message": err.Error()})
		return
	}
	req.ApprovalRequestID = strings.TrimSpace(req.ApprovalRequestID)
	if req.ApprovalRequestID == "" {
		req.ApprovalRequestID = mintApprovalRequestID()
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
			"error":               "approval_already_requested",
			"message":             "An approval request already exists for this execution",
			"approval_request_id": *wfExec.ApprovalRequestID,
		})
		return
	}

	// SSRF protection: validate callback_url before storing it. Reject
	// URLs targeting private/internal addresses (cloud metadata, localhost,
	// RFC-1918) to prevent the control plane from being used as a proxy.
	if req.CallbackURL != "" {
		if err := services.ValidateWebhookURL(req.CallbackURL); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"error":   "invalid_callback_url",
				"message": fmt.Sprintf("callback_url rejected: %v", err),
			})
			return
		}
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

	// Keep the question beside the approval. A failure here is logged and
	// the pause goes on: the run still waits and can still be answered.
	if qs, ok := c.store.(ApprovalQuestionStore); ok {
		if qErr := qs.PutApprovalQuestion(reqCtx, types.ApprovalQuestion{
			ExecutionID:       executionID,
			ApprovalRequestID: req.ApprovalRequestID,
			Kind:              req.Kind,
			Question:          req.Question,
			Body:              req.Body,
			Options:           req.Options,
		}); qErr != nil {
			logger.Logger.Warn().Err(qErr).Str("execution_id", executionID).Msg("failed to keep the approval question (non-fatal)")
		}
	}

	// Emit execution event for observability
	waitingStatus := types.ExecutionStatusWaiting
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"approval_request_id":  req.ApprovalRequestID,
		"approval_request_url": req.ApprovalRequestURL,
		"wait_kind":            "approval",
		"kind":                 req.Kind,
		"question":             req.Question,
	})
	event := &types.WorkflowExecutionEvent{
		ExecutionID:  executionID,
		WorkflowID:   wfExec.WorkflowID,
		RunID:        wfExec.RunID,
		EventType:    "execution.waiting",
		Status:       &waitingStatus,
		StatusReason: &statusReason,
		Payload:      eventPayload,
		EmittedAt:    now,
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

	// THE WAIT GOES ON THE GLOBAL BUS TOO. The store's bus feeds the UI's
	// streams; the node link's pipe follows only the global bus, and it is
	// what tells the run's owner (WorkAF) within a second that a person is
	// needed, as a run frame carrying the question.
	events.PublishExecutionWaiting(executionID, wfExec.WorkflowID, wfExec.AgentNodeID, map[string]interface{}{
		"status_reason":       statusReason,
		"approval_request_id": req.ApprovalRequestID,
		"wait_kind":           "approval",
		"kind":                req.Kind,
	})

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

	out := ApprovalStatusResponse{
		Status:            status,
		ApprovalRequestID: *wfExec.ApprovalRequestID,
		Feedback:          approvalFeedback(wfExec.ApprovalResponse),
		Response:          wfExec.ApprovalResponse,
		RequestURL:        requestURL,
		RequestedAt:       requestedAt,
		ExpiresAt:         expiresAtStr,
		RespondedAt:       respondedAt,
	}
	if qs, ok := c.store.(ApprovalQuestionStore); ok {
		if q, qErr := qs.GetApprovalQuestion(reqCtx, executionID, *wfExec.ApprovalRequestID); qErr == nil && q != nil {
			out.Kind, out.Question, out.Body, out.Options = q.Kind, q.Question, q.Body, q.Options
		}
	}
	ctx.JSON(http.StatusOK, out)
}

// approvalFeedback reads the answerer's words out of a stored response: the
// "feedback" field the resolver writes, or a "reason"/"text" one an external
// service sent.
func approvalFeedback(resp *string) string {
	if resp == nil || *resp == "" {
		return ""
	}
	var m map[string]any
	if json.Unmarshal([]byte(*resp), &m) != nil {
		return ""
	}
	for _, k := range []string{"feedback", "reason", "text", "comment"} {
		if v, ok := m[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
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
