package handlers

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
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
	webhookSecret string        // HMAC-SHA256 secret for signature verification
	scope         ApprovalScope // nil: every authenticated caller may answer
}

// ApprovalWebhookHandler receives approval responses via webhook callback.
// Can be called by external services (e.g. hax-sdk) or by agents directly.
// Requests are authenticated via HMAC-SHA256 signature verification using webhookSecret.
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

	// Phase 1 auth posture: webhook endpoint is HMAC-only.
	// If no secret is configured, treat the webhook endpoint as disabled.
	if c.webhookSecret == "" {
		logger.Logger.Error().Msg("approval webhook secret is not configured")
		ctx.JSON(http.StatusServiceUnavailable, gin.H{
			"error":   "approval webhook not configured",
			"message": "set agentfield.approval.webhook_secret or AGENTFIELD_APPROVAL_WEBHOOK_SECRET to enable /api/v1/webhooks/approval-response",
		})
		return
	}

	signature := ctx.GetHeader("X-Hax-Signature")
	if signature == "" {
		signature = ctx.GetHeader("X-Webhook-Signature")
	}
	if signature == "" {
		signature = ctx.GetHeader("X-Hub-Signature-256")
	}
	if signature == "" {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "missing webhook signature"})
		return
	}
	if !c.verifySignature(bodyBytes, signature) {
		logger.Logger.Warn().Msg("approval webhook signature verification failed")
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid webhook signature"})
		return
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

	c.applyApprovalDecision(ctx, executionID, wfExec, payload, decision)
}

// applyApprovalDecision applies a validated approval decision to a waiting
// execution: idempotency check, state transitions, approval bookkeeping,
// observability events, and the async agent callback. Shared by the
// HMAC-signed webhook and the authenticated
// POST /executions/:execution_id/approval-response endpoint so both paths
// resolve approvals identically. decision must already be normalized to one
// of approved, rejected, request_changes, or expired.
func (c *webhookApprovalController) applyApprovalDecision(ctx *gin.Context, executionID string, wfExec *types.WorkflowExecution, payload *ApprovalWebhookPayload, decision string) {
	reqCtx := ctx.Request.Context()

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
	err := c.store.UpdateWorkflowExecution(reqCtx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
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
		//
		// "request_changes" keeps its ApprovalRequestID too: approval-status
		// must still describe the round just answered — its decision and the
		// person's words — so an agent whose callback was lost resumes from a
		// poll with the feedback. A non-pending status already lets the next
		// request-approval start a new round.
		if decision == "request_changes" || decision == "approved" {
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
		RunID:        wfExec.RunID,
		EventType:    eventType,
		Status:       &newStatus,
		StatusReason: newStatusReason,
		Payload:      eventPayload,
		EmittedAt:    now,
	}
	if storeErr := c.store.StoreWorkflowExecutionEvent(reqCtx, event); storeErr != nil {
		logger.Logger.Warn().Err(storeErr).Str("execution_id", executionID).Msg("failed to store approval resolved event (non-fatal)")
	}

	// And on the global bus, so the node link's pipe pushes the resolved
	// state to the run's owner (see execute_approval.go).
	events.PublishExecutionApprovalResolved(executionID, wfExec.WorkflowID, wfExec.AgentNodeID, newStatus, map[string]interface{}{
		"approval_decision":   decision,
		"approval_request_id": payload.RequestID,
	})

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
		go c.notifyApprovalCallback(approvalCallback{
			url: savedCallbackURL, executionID: executionID, workflowID: wfExec.WorkflowID, runID: wfExec.RunID,
			decision: decision, newStatus: newStatus, feedback: payload.Feedback,
			response: responseStr, approvalRequestID: payload.RequestID,
		})
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
	// Callers should ensure webhookSecret is configured before invoking.
	if c.webhookSecret == "" {
		return false
	}
	if signature == "" {
		return false
	}

	// Try hax-sdk format: "t=timestamp,v1=signature"
	if ts, sig, ok := parseHaxSignature(signature); ok {
		// Enforce timestamp freshness to prevent replay attacks.
		// Hard-coded to 5 minutes; making this configurable would require
		// plumbing config through the handler constructor (future enhancement).
		if tsInt, err := strconv.ParseInt(ts, 10, 64); err == nil {
			diff := time.Now().Unix() - tsInt
			const maxSkewSeconds = 300 // 5 minutes
			if diff > maxSkewSeconds || diff < -maxSkewSeconds {
				return false
			}
		} else {
			// Non-numeric timestamp — reject
			return false
		}

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

// approvalCallbackBackoff is the wait before each retry of the agent
// callback: about five minutes in all. Tests shorten it.
var approvalCallbackBackoff = []time.Duration{
	1 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second,
	16 * time.Second, 30 * time.Second, 30 * time.Second, 30 * time.Second,
	30 * time.Second, 30 * time.Second, 30 * time.Second, 30 * time.Second,
	30 * time.Second, 30 * time.Second,
}

// approvalCallbackClient is the client one callback attempt goes through.
// A link host ("<node>.nodelink") is a node with no address: its request is
// carried down the node's link as an http delivery by agenturl's client,
// which waits for the link to come up before it gives up, so it gets the
// longer budget. Every other host goes through the SSRF-safe client, which
// refuses private and metadata addresses (CWE-918).
var approvalCallbackClient = func(callbackURL string) *http.Client {
	if u, err := url.Parse(callbackURL); err == nil {
		if _, ok := agenturl.LinkNode(u.Host); ok {
			return agenturl.NewClient(45 * time.Second)
		}
	}
	return services.NewSSRFSafeClient(10 * time.Second)
}

// approvalCallback is one resume the control plane owes a paused agent.
type approvalCallback struct {
	url, executionID, workflowID, decision, newStatus, feedback, approvalRequestID string
	runID                                                                          *string
	response                                                                       *string
}

// notifyApprovalCallback POSTs the approval result to the agent's registered
// callback URL, retrying with backoff. Called asynchronously — it does not
// block the answer.
//
// A LOST CALLBACK USED TO LEAVE THE REASONER BLOCKED UNTIL ITS APPROVAL
// EXPIRED: one attempt, five seconds, nothing recorded. Now a transport error,
// a 429 or a 5xx is retried for about five minutes, and an attempt that never
// lands is stored as an execution.approval_callback_failed event, so the
// failure is visible on the execution. The Go SDK's Pause also polls
// approval-status, so a node still resumes when every attempt is lost.
func (c *webhookApprovalController) notifyApprovalCallback(cb approvalCallback) {
	callbackPayload := map[string]interface{}{
		"execution_id":        cb.executionID,
		"decision":            cb.decision,
		"new_status":          cb.newStatus,
		"feedback":            cb.feedback,
		"approval_request_id": cb.approvalRequestID,
	}
	if cb.response != nil {
		callbackPayload["response"] = *cb.response
	}

	body, err := json.Marshal(callbackPayload)
	if err != nil {
		logger.Logger.Error().Err(err).Str("callback_url", cb.url).Msg("failed to marshal approval callback payload")
		return
	}

	var lastErr string
	attempts := 0
	for i := 0; i <= len(approvalCallbackBackoff); i++ {
		if i > 0 {
			time.Sleep(approvalCallbackBackoff[i-1])
		}
		attempts++
		retry, errWords := c.attemptApprovalCallback(cb, body)
		if errWords == "" {
			logger.Logger.Info().Str("callback_url", cb.url).Str("execution_id", cb.executionID).Int("attempts", attempts).Msg("approval callback delivered")
			return
		}
		lastErr = errWords
		logger.Logger.Warn().Str("callback_url", cb.url).Str("execution_id", cb.executionID).Int("attempt", attempts).Str("error", errWords).Msg("approval callback attempt failed")
		if !retry {
			break
		}
	}
	c.recordApprovalCallbackFailure(cb, attempts, lastErr)
}

// attemptApprovalCallback makes one delivery. It answers whether a failure is
// worth retrying and the failure's words ("" when delivered).
func (c *webhookApprovalController) attemptApprovalCallback(cb approvalCallback, body []byte) (bool, string) {
	client := approvalCallbackClient(cb.url)
	req, err := http.NewRequest(http.MethodPost, cb.url, bytes.NewReader(body))
	if err != nil {
		return false, err.Error()
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		// A refused destination stays refused; only a transient failure
		// (the link not up yet, a timeout, a reset) is worth another try.
		return !strings.Contains(err.Error(), "private/internal"), err.Error()
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	switch {
	case resp.StatusCode < 400:
		return false, ""
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
		return true, fmt.Sprintf("the agent answered %d", resp.StatusCode)
	default:
		return false, fmt.Sprintf("the agent answered %d", resp.StatusCode)
	}
}

func (c *webhookApprovalController) recordApprovalCallbackFailure(cb approvalCallback, attempts int, lastErr string) {
	logger.Logger.Error().Str("callback_url", cb.url).Str("execution_id", cb.executionID).Int("attempts", attempts).Str("error", lastErr).Msg("approval callback never delivered")
	payload, _ := json.Marshal(map[string]interface{}{
		"approval_request_id": cb.approvalRequestID,
		"decision":            cb.decision,
		"callback_url":        cb.url,
		"attempts":            attempts,
		"error":               lastErr,
	})
	status := cb.newStatus
	event := &types.WorkflowExecutionEvent{
		ExecutionID: cb.executionID,
		WorkflowID:  cb.workflowID,
		RunID:       cb.runID,
		EventType:   "execution.approval_callback_failed",
		Status:      &status,
		Payload:     payload,
		EmittedAt:   time.Now().UTC(),
	}
	if c.store == nil {
		return
	}
	if err := c.store.StoreWorkflowExecutionEvent(context.Background(), event); err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", cb.executionID).Msg("failed to store approval callback failure (non-fatal)")
	}
}
