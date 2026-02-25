package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// approvalController handles approval-related endpoints.
type approvalController struct {
	store      ExecutionStore
	approvalCfg config.ApprovalConfig
	httpClient *http.Client
}

// RequestApprovalRequest is the body for POST /executions/:execution_id/request-approval.
type RequestApprovalRequest struct {
	Title        string                 `json:"title" binding:"required"`
	Description  string                 `json:"description"`
	TemplateType string                 `json:"template_type" binding:"required"`
	Payload      map[string]interface{} `json:"payload" binding:"required"`
	ProjectID    string                 `json:"project_id" binding:"required"`
	ExpiresInHours *int                 `json:"expires_in_hours,omitempty"`
}

// RequestApprovalResponse is returned when an approval request is created.
type RequestApprovalResponse struct {
	ApprovalRequestID  string `json:"approval_request_id"`
	ApprovalRequestURL string `json:"approval_request_url"`
	Status             string `json:"status"`
	ExpiresAt          string `json:"expires_at,omitempty"`
}

// ApprovalStatusResponse is returned by GET /executions/:execution_id/approval-status.
type ApprovalStatusResponse struct {
	Status      string  `json:"status"`
	Response    *string `json:"response,omitempty"`
	RequestURL  string  `json:"request_url,omitempty"`
	RequestedAt string  `json:"requested_at,omitempty"`
	RespondedAt *string `json:"responded_at,omitempty"`
}

// haxSDKCreateRequestBody is the payload sent to hax-sdk to create a request.
type haxSDKCreateRequestBody struct {
	Title            string                 `json:"title"`
	Description      string                 `json:"description"`
	Type             string                 `json:"type"`
	Payload          map[string]interface{} `json:"payload"`
	WebhookURL       string                 `json:"webhookUrl"`
	ExpiresInSeconds int                    `json:"expiresInSeconds,omitempty"`
}

// haxSDKCreateRequestResponse is the response from hax-sdk after creating a request.
type haxSDKCreateRequestResponse struct {
	ID         string `json:"id"`
	URL        string `json:"url"`
	Status     string `json:"status"`
	ExpiresAt  string `json:"expiresAt,omitempty"`
}

// RequestApprovalHandler creates a new approval request and transitions the execution to waiting.
func RequestApprovalHandler(store ExecutionStore, approvalCfg config.ApprovalConfig) gin.HandlerFunc {
	ctrl := &approvalController{
		store:       store,
		approvalCfg: approvalCfg,
		httpClient: &http.Client{
			Timeout: approvalCfg.RequestTimeout,
		},
	}
	if ctrl.httpClient.Timeout <= 0 {
		ctrl.httpClient.Timeout = 30 * time.Second
	}
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

	// Validate hax-sdk is configured
	if c.approvalCfg.HaxSDKURL == "" {
		ctx.JSON(http.StatusServiceUnavailable, gin.H{
			"error":   "approval_not_configured",
			"message": "Approval workflow is not configured. Set approval.hax_sdk_url in agentfield.yaml.",
		})
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

	// Prevent duplicate approval requests
	if wfExec.ApprovalRequestID != nil && *wfExec.ApprovalRequestID != "" {
		ctx.JSON(http.StatusConflict, gin.H{
			"error":   "approval_already_requested",
			"message": "An approval request already exists for this execution",
			"approval_request_id": *wfExec.ApprovalRequestID,
		})
		return
	}

	// Build the webhook callback URL for hax-sdk to call back
	// We use the scheme/host from the incoming request as a base
	scheme := "http"
	if ctx.Request.TLS != nil {
		scheme = "https"
	}
	if fwdProto := ctx.GetHeader("X-Forwarded-Proto"); fwdProto != "" {
		scheme = fwdProto
	}
	webhookURL := fmt.Sprintf("%s://%s/api/v1/webhooks/approval-response", scheme, ctx.Request.Host)

	// Determine expiry
	expiryHours := c.approvalCfg.DefaultExpiryHours
	if expiryHours <= 0 {
		expiryHours = 72
	}
	if req.ExpiresInHours != nil && *req.ExpiresInHours > 0 {
		expiryHours = *req.ExpiresInHours
	}

	// Call hax-sdk to create the request
	haxReq := haxSDKCreateRequestBody{
		Title:            req.Title,
		Description:      req.Description,
		Type:             req.TemplateType,
		Payload:          req.Payload,
		WebhookURL:       webhookURL,
		ExpiresInSeconds: expiryHours * 3600,
	}

	haxResp, err := c.callHaxSDKCreateRequest(reqCtx, &haxReq)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to create approval request in hax-sdk")
		ctx.JSON(http.StatusBadGateway, gin.H{
			"error":   "hax_sdk_error",
			"message": fmt.Sprintf("Failed to create approval request: %v", err),
		})
		return
	}

	now := time.Now().UTC()
	statusReason := "waiting_for_approval"
	approvalStatus := "pending"

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
	}

	// Update the workflow execution with approval fields + waiting status
	err = c.store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = types.ExecutionStatusWaiting
		current.StatusReason = &statusReason
		current.ApprovalRequestID = &haxResp.ID
		current.ApprovalRequestURL = &haxResp.URL
		current.ApprovalStatus = &approvalStatus
		current.ApprovalRequestedAt = &now
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
		"approval_request_id":  haxResp.ID,
		"approval_request_url": haxResp.URL,
		"wait_kind":            "approval",
		"expires_in_hours":     expiryHours,
	})
	event := &types.WorkflowExecutionEvent{
		ExecutionID:  executionID,
		WorkflowID:   wfExec.WorkflowID,
		RunID:         wfExec.RunID,
		EventType:    "execution.waiting",
		Status:       &waitingStatus,
		StatusReason: &statusReason,
		Payload:      eventPayload,
		EmittedAt:    now,
	}
	if storeErr := c.store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store approval event (non-fatal)")
	}

	// Publish dedicated waiting event to the execution event bus
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
				"approval_request_id":  haxResp.ID,
				"approval_request_url": haxResp.URL,
				"wait_kind":            "approval",
			},
		})
	}

	logger.Logger.Info().
		Str("execution_id", executionID).
		Str("approval_request_id", haxResp.ID).
		Str("approval_url", haxResp.URL).
		Msg("approval request created, execution transitioned to waiting")

	ctx.JSON(http.StatusOK, RequestApprovalResponse{
		ApprovalRequestID:  haxResp.ID,
		ApprovalRequestURL: haxResp.URL,
		Status:             "pending",
		ExpiresAt:          haxResp.ExpiresAt,
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

	ctx.JSON(http.StatusOK, ApprovalStatusResponse{
		Status:      status,
		Response:    wfExec.ApprovalResponse,
		RequestURL:  requestURL,
		RequestedAt: requestedAt,
		RespondedAt: respondedAt,
	})
}

// callHaxSDKCreateRequest sends a POST to hax-sdk to create an approval request.
func (c *approvalController) callHaxSDKCreateRequest(ctx interface{ Done() <-chan struct{} }, body *haxSDKCreateRequestBody) (*haxSDKCreateRequestResponse, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	url := strings.TrimRight(c.approvalCfg.HaxSDKURL, "/") + "/api/v1/requests"

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.approvalCfg.HaxSDKAPIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.approvalCfg.HaxSDKAPIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to hax-sdk failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 1MB limit
	if err != nil {
		return nil, fmt.Errorf("failed to read hax-sdk response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("hax-sdk returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result haxSDKCreateRequestResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to decode hax-sdk response: %w", err)
	}

	return &result, nil
}
