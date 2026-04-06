package handlers

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ApprovalWebhookPayload is the normalized payload for approval processing.
type ApprovalWebhookPayload struct {
	RequestID string          `json:"requestId"`
	Decision  string          `json:"decision"` // "approved", "rejected", "request_changes", "expired"
	Response  json.RawMessage `json:"response,omitempty"`
	Feedback  string          `json:"feedback,omitempty"`
	Timestamp string          `json:"timestamp,omitempty"`
}

// haxSDKWebhookEnvelope is the actual envelope format hax-sdk sends.
type haxSDKWebhookEnvelope struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"` // "completed", "expired", etc.
	CreatedAt string                 `json:"createdAt"`
	Data      map[string]interface{} `json:"data"`
}

// parseWebhookPayload attempts to extract an ApprovalWebhookPayload from the raw body.
// It supports two formats:
// 1. hax-sdk envelope: {"id":"evt_...","type":"completed","data":{"requestId":"...","response":{"decision":"approved"}}}
// 2. Direct flat format: {"requestId":"...","decision":"approved","feedback":"..."}
func parseWebhookPayload(bodyBytes []byte) (*ApprovalWebhookPayload, error) {
	// First try the hax-sdk envelope format
	var envelope haxSDKWebhookEnvelope
	if err := json.Unmarshal(bodyBytes, &envelope); err == nil && envelope.Data != nil && envelope.Type != "" {
		payload := &ApprovalWebhookPayload{}

		// Extract requestId from data
		if rid, ok := envelope.Data["requestId"].(string); ok {
			payload.RequestID = rid
		}
		payload.Timestamp = envelope.CreatedAt

		// Extract decision from data.response (supports multiple field names across templates)
		if respRaw, ok := envelope.Data["response"]; ok {
			if respMap, ok := respRaw.(map[string]interface{}); ok {
				// Check "decision" first (text-approval, data-table-review, side-by-side, code-changes, terminal-output)
				if dec, ok := respMap["decision"].(string); ok {
					payload.Decision = dec
				}
				// Fall back to "action" (rich-text-editor, confirm-action)
				if payload.Decision == "" {
					if act, ok := respMap["action"].(string); ok {
						payload.Decision = act
					}
				}
				if fb, ok := respMap["feedback"].(string); ok {
					payload.Feedback = fb
				}
				// Preserve full response
				if respJSON, err := json.Marshal(respMap); err == nil {
					payload.Response = respJSON
				}
			}
		}

		// Handle "expired" event type from hax-sdk
		if envelope.Type == "expired" {
			payload.Decision = "expired"
		}

		// If envelope type is "completed" but no decision/action field was found
		// (e.g. signature-capture, rich-text-editor submit), default to "approved"
		if payload.Decision == "" && envelope.Type == "completed" {
			payload.Decision = "approved"
		}

		if payload.RequestID != "" && payload.Decision != "" {
			return payload, nil
		}
		// If we couldn't extract enough from envelope, fall through to flat format
	}

	// Fall back to flat format
	var flat ApprovalWebhookPayload
	if err := json.Unmarshal(bodyBytes, &flat); err != nil {
		return nil, fmt.Errorf("could not parse webhook payload: %w", err)
	}
	return &flat, nil
}

// normalizeDecision maps the various decision values used by hax-sdk
// templates into the canonical set the control plane understands.
func normalizeDecision(raw string) string {
	switch raw {
	case "approve", "continue", "confirm":
		return "approved"
	case "reject", "deny", "abort", "cancel":
		return "rejected"
	case "approved", "rejected", "request_changes", "expired":
		return raw
	default:
		return raw
	}
}

// webhookApprovalController handles the approval webhook callback.
type webhookApprovalController struct {
	store         ExecutionStore
	webhookSecret string // optional HMAC-SHA256 secret for signature verification
}

// ApprovalWebhookHandler receives approval responses via webhook callback.
// Can be called by external services (e.g. hax-sdk) or by agents directly.
// The optional webhookSecret enables HMAC-SHA256 signature verification.
func ApprovalWebhookHandler(store ExecutionStore, webhookSecret string) gin.HandlerFunc {
	ctrl := &webhookApprovalController{
		store:         store,
		webhookSecret: webhookSecret,
	}
	return ctrl.handleApprovalWebhook
}

func (c *webhookApprovalController) handleApprovalWebhook(ctx *gin.Context) {
	// Read the raw body for signature verification
	bodyBytes, err := io.ReadAll(io.LimitReader(ctx.Request.Body, 1<<20)) // 1MB limit
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	// Verify HMAC-SHA256 signature if webhook secret is configured
	if c.webhookSecret != "" {
		signature := ctx.GetHeader("X-Hax-Signature")
		if signature == "" {
			signature = ctx.GetHeader("X-Webhook-Signature")
		}
		if signature == "" {
			signature = ctx.GetHeader("X-Hub-Signature-256")
		}
		if !c.verifySignature(bodyBytes, signature) {
			logger.Logger.Warn().Msg("approval webhook signature verification failed")
			ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid webhook signature"})
			return
		}
	}

	payload, parseErr := parseWebhookPayload(bodyBytes)
	if parseErr != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid payload: %v", parseErr)})
		return
	}

	if payload.RequestID == "" {
		logger.Logger.Warn().Str("raw_body", string(bodyBytes)).Msg("webhook payload missing requestId")
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "requestId is required"})
		return
	}

	// Normalize decision values from various hax-sdk template formats
	// into the canonical set the control plane understands.
	decision := normalizeDecision(payload.Decision)
	switch decision {
	case "approved", "rejected", "request_changes", "expired":
		// valid
	default:
		logger.Logger.Warn().Str("decision", payload.Decision).Str("raw_body", string(bodyBytes)).Msg("webhook payload has invalid decision")
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid decision '%s'; must be approved, rejected, request_changes, or expired", payload.Decision)})
		return
	}

	reqCtx := ctx.Request.Context()

	// Find the workflow execution by approval_request_id
	executionID, wfExec, err := c.findExecutionByApprovalRequestID(ctx, payload.RequestID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("request_id", payload.RequestID).Msg("failed to find execution for approval webhook")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up execution"})
		return
	}
	if wfExec == nil {
		logger.Logger.Warn().Str("request_id", payload.RequestID).Msg("no execution found for approval request ID")
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("no execution found for approval request %s", payload.RequestID)})
		return
	}

	// Idempotency: if execution is no longer in waiting state, it was already
	// processed by a previous webhook delivery.  Return 200 so the sender
	// (hax-sdk retry queue) considers it delivered and stops retrying.
	// Exception: if the stale reaper marked it as "timeout" but the approval
	// is still pending, allow the approval to proceed (the user is explicitly
	// resolving the approval via the UI or webhook).
	normalized := types.NormalizeExecutionStatus(wfExec.Status)
	approvalStillPending := wfExec.ApprovalStatus != nil && *wfExec.ApprovalStatus == "pending"
	reaperTimeout := normalized == string(types.ExecutionStatusTimeout) && approvalStillPending

	if normalized != types.ExecutionStatusWaiting && !reaperTimeout {
		logger.Logger.Info().
			Str("execution_id", executionID).
			Str("current_status", normalized).
			Str("request_id", payload.RequestID).
			Msg("approval webhook is a duplicate — execution already resolved")
		approvalStatus := ""
		if wfExec.ApprovalStatus != nil {
			approvalStatus = *wfExec.ApprovalStatus
		}
		ctx.JSON(http.StatusOK, gin.H{
			"status":          "already_processed",
			"execution_id":    executionID,
			"current_status":  normalized,
			"approval_status": approvalStatus,
		})
		return
	}

	now := time.Now().UTC()
	var responseStr *string
	if len(payload.Response) > 0 {
		s := string(payload.Response)
		responseStr = &s
	} else if payload.Feedback != "" {
		s := fmt.Sprintf(`{"feedback":%q}`, payload.Feedback)
		responseStr = &s
	}

	// Determine the new execution status based on decision
	var newStatus string
	var newStatusReason *string
	switch decision {
	case "approved":
		newStatus = types.ExecutionStatusRunning
		reason := "approval_granted"
		newStatusReason = &reason
	case "rejected":
		newStatus = types.ExecutionStatusCancelled
		reason := "approval_rejected"
		if payload.Feedback != "" {
			reason = fmt.Sprintf("approval_rejected: %s", payload.Feedback)
		}
		newStatusReason = &reason
	case "request_changes":
		newStatus = types.ExecutionStatusRunning
		reason := "approval_changes_requested"
		if payload.Feedback != "" {
			reason = fmt.Sprintf("approval_changes_requested: %s", payload.Feedback)
		}
		newStatusReason = &reason
	case "expired":
		newStatus = types.ExecutionStatusCancelled
		reason := "approval_expired"
		newStatusReason = &reason
	}

	// Update the lightweight execution record
	var recordSyncFailed bool
	_, updateErr := c.store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = newStatus
		current.StatusReason = newStatusReason
		if decision != "approved" && decision != "request_changes" {
			current.CompletedAt = &now
			dur := now.Sub(current.StartedAt).Milliseconds()
			current.DurationMS = &dur
		}
		return current, nil
	})
	if updateErr != nil {
		logger.Logger.Error().Err(updateErr).Str("execution_id", executionID).Msg("failed to update execution record from approval webhook — proceeding with workflow update")
		recordSyncFailed = true
	}

	// Capture callback URL before the update closure may clear it.  The
	// wfExec pointer may alias the same object the store mutates in-place
	// (e.g. in-memory stores), so read it now for the post-response callback.
	var savedCallbackURL string
	if wfExec.ApprovalCallbackURL != nil {
		savedCallbackURL = *wfExec.ApprovalCallbackURL
	}

	// Update the workflow execution with approval resolution (authoritative — must not lose the decision)
	err = c.store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		current.Status = newStatus
		current.StatusReason = newStatusReason
		current.ApprovalStatus = &decision
		current.ApprovalResponse = responseStr
		current.ApprovalRespondedAt = &now
		if decision != "approved" && decision != "request_changes" {
			current.CompletedAt = &now
			dur := now.Sub(current.StartedAt).Milliseconds()
			current.DurationMS = &dur
		}
		// Clear approval request fields so the agent can issue subsequent
		// approval requests within the same execution (multi-pause workflows).
		// The decision/response/respondedAt are preserved for audit.
		//
		// For "request_changes": clear everything so the agent can start fresh.
		// For "approved": keep ApprovalRequestID (needed for idempotent webhook
		// retries and approval-status lookups) but clear the URL fields.  The
		// request-approval handler allows new requests when ApprovalStatus is
		// no longer "pending", enabling multi-pause workflows.
		if decision == "request_changes" {
			current.ApprovalRequestID = nil
			current.ApprovalRequestURL = nil
			current.ApprovalCallbackURL = nil
		} else if decision == "approved" {
			current.ApprovalRequestURL = nil
			current.ApprovalCallbackURL = nil
		}
		return current, nil
	})
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to update workflow execution from approval webhook")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update execution"})
		return
	}

	// Emit observability event
	eventType := "execution.approval_resolved"
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"approval_request_id": payload.RequestID,
		"decision":            decision,
		"feedback":            payload.Feedback,
		"new_status":          newStatus,
	})
	event := &types.WorkflowExecutionEvent{
		ExecutionID:  executionID,
		WorkflowID:   wfExec.WorkflowID,
		RunID:         wfExec.RunID,
		EventType:    eventType,
		Status:       &newStatus,
		StatusReason: newStatusReason,
		Payload:      eventPayload,
		EmittedAt:    now,
	}
	if storeErr := c.store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store approval resolved event (non-fatal)")
	}

	// Publish dedicated approval resolved event to the execution event bus
	if bus := c.store.GetExecutionEventBus(); bus != nil {
		bus.Publish(events.ExecutionEvent{
			Type:        events.ExecutionApprovalResolved,
			ExecutionID: executionID,
			WorkflowID:  wfExec.WorkflowID,
			AgentNodeID: wfExec.AgentNodeID,
			Status:      newStatus,
			Timestamp:   now,
			Data: map[string]interface{}{
				"approval_decision":   decision,
				"approval_request_id": payload.RequestID,
				"feedback":            payload.Feedback,
			},
		})
	}

	logger.Logger.Info().
		Str("execution_id", executionID).
		Str("request_id", payload.RequestID).
		Str("decision", decision).
		Str("new_status", newStatus).
		Msg("approval webhook processed, execution state updated")

	response := gin.H{
		"status":       "processed",
		"execution_id": executionID,
		"decision":     decision,
		"new_status":   newStatus,
	}
	if recordSyncFailed {
		response["warning"] = "lightweight execution record update failed — workflow execution is authoritative"
	}
	ctx.JSON(http.StatusOK, response)

	// Notify the agent's callback URL if one was registered
	if savedCallbackURL != "" {
		go c.notifyApprovalCallback(savedCallbackURL, executionID, decision, newStatus, payload.Feedback, responseStr, payload.RequestID)
	}
}

// findExecutionByApprovalRequestID looks up a workflow execution by its approval_request_id.
// Uses the indexed approval_request_id column via QueryWorkflowExecutions.
func (c *webhookApprovalController) findExecutionByApprovalRequestID(ctx *gin.Context, requestID string) (string, *types.WorkflowExecution, error) {
	reqCtx := ctx.Request.Context()

	results, err := c.store.QueryWorkflowExecutions(reqCtx, types.WorkflowExecutionFilters{
		ApprovalRequestID: &requestID,
		Limit:             1,
	})
	if err != nil {
		return "", nil, fmt.Errorf("failed to query workflow executions by approval_request_id: %w", err)
	}
	if len(results) == 0 {
		return "", nil, nil
	}

	wfExec := results[0]
	return wfExec.ExecutionID, wfExec, nil
}

// verifySignature verifies the HMAC-SHA256 signature of the webhook payload.
// Supports multiple signature formats:
// - hax-sdk format: "t=1234567890,v1=<hex_signature>" (signs "timestamp.payload")
// - Raw hex: "<hex_signature>" (signs payload directly)
// - Prefixed: "sha256=<hex_signature>" (signs payload directly)
func (c *webhookApprovalController) verifySignature(body []byte, signature string) bool {
	if c.webhookSecret == "" {
		return true // No secret configured, skip verification
	}
	if signature == "" {
		return false
	}

	// Try hax-sdk format: "t=timestamp,v1=signature"
	if ts, sig, ok := parseHaxSignature(signature); ok {
		signedPayload := fmt.Sprintf("%s.%s", ts, string(body))
		mac := hmac.New(sha256.New, []byte(c.webhookSecret))
		mac.Write([]byte(signedPayload))
		expectedMAC := hex.EncodeToString(mac.Sum(nil))
		return hmac.Equal([]byte(sig), []byte(expectedMAC))
	}

	// Fall back to simple signature verification
	sig := trimSignaturePrefix(signature)
	mac := hmac.New(sha256.New, []byte(c.webhookSecret))
	mac.Write(body)
	expectedMAC := hex.EncodeToString(mac.Sum(nil))
	return hmac.Equal([]byte(sig), []byte(expectedMAC))
}

// parseHaxSignature parses "t=timestamp,v1=signature" format.
func parseHaxSignature(sig string) (timestamp, signature string, ok bool) {
	parts := make(map[string]string)
	for _, part := range splitSignatureParts(sig) {
		if idx := indexOf(part, '='); idx > 0 {
			parts[part[:idx]] = part[idx+1:]
		}
	}
	ts, hasT := parts["t"]
	v1, hasV1 := parts["v1"]
	if hasT && hasV1 {
		return ts, v1, true
	}
	return "", "", false
}

func splitSignatureParts(s string) []string {
	var parts []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	parts = append(parts, s[start:])
	return parts
}

func indexOf(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}

// trimSignaturePrefix removes common signature prefixes like "sha256=".
func trimSignaturePrefix(sig string) string {
	if len(sig) > 7 && sig[:7] == "sha256=" {
		return sig[7:]
	}
	return sig
}

// notifyApprovalCallback POSTs the approval result to the agent's registered callback URL.
// Called asynchronously (go routine) — best-effort, does not block the webhook response.
func (c *webhookApprovalController) notifyApprovalCallback(callbackURL, executionID, decision, newStatus, feedback string, response *string, approvalRequestID string) {
	callbackPayload := map[string]interface{}{
		"execution_id":        executionID,
		"decision":            decision,
		"new_status":          newStatus,
		"feedback":            feedback,
		"approval_request_id": approvalRequestID,
	}
	if response != nil {
		callbackPayload["response"] = *response
	}

	body, err := json.Marshal(callbackPayload)
	if err != nil {
		logger.Logger.Error().Err(err).Str("callback_url", callbackURL).Msg("failed to marshal approval callback payload")
		return
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(callbackURL, "application/json", bytes.NewReader(body))
	if err != nil {
		logger.Logger.Warn().Err(err).Str("callback_url", callbackURL).Str("execution_id", executionID).Msg("approval callback delivery failed")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		logger.Logger.Warn().Int("status", resp.StatusCode).Str("callback_url", callbackURL).Str("execution_id", executionID).Msg("approval callback returned error")
	} else {
		logger.Logger.Info().Str("callback_url", callbackURL).Str("execution_id", executionID).Msg("approval callback delivered")
	}
}
