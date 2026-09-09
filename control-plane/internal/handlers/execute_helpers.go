package handlers

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
)

func readExecutionHeaders(ctx *gin.Context) executionHeaders {
	runID := strings.TrimSpace(ctx.GetHeader("X-Run-ID"))
	parent := strings.TrimSpace(ctx.GetHeader("X-Parent-Execution-ID"))
	session := strings.TrimSpace(ctx.GetHeader("X-Session-ID"))
	actor := strings.TrimSpace(ctx.GetHeader("X-Actor-ID"))
	replaySourceRunID := strings.TrimSpace(ctx.GetHeader("X-AgentField-Replay-Source-Run-ID"))
	replayBeforeExecutionID := strings.TrimSpace(ctx.GetHeader("X-AgentField-Replay-Before-Execution-ID"))
	replayMode := strings.TrimSpace(ctx.GetHeader("X-AgentField-Replay-Mode"))

	var parentPtr *string
	if parent != "" {
		parentPtr = &parent
	}

	var sessionPtr *string
	if session != "" {
		sessionPtr = &session
	}

	var actorPtr *string
	if actor != "" {
		actorPtr = &actor
	}

	return executionHeaders{
		runID:                   runID,
		parentExecutionID:       parentPtr,
		sessionID:               sessionPtr,
		actorID:                 actorPtr,
		replaySourceRunID:       replaySourceRunID,
		replayBeforeExecutionID: replayBeforeExecutionID,
		replayMode:              replayMode,
	}
}

type parsedTarget struct {
	NodeID     string
	TargetName string
	TargetType string
}

func parseTarget(value string) (*parsedTarget, error) {
	if value == "" {
		return nil, errors.New("target is required")
	}
	parts := strings.Split(value, ".")
	if len(parts) != 2 {
		return nil, fmt.Errorf("target must be in format 'node_id.reasoner_name'")
	}
	return &parsedTarget{
		NodeID:     parts[0],
		TargetName: parts[1],
	}, nil
}

func determineTargetType(agent *types.AgentNode, name string) (string, error) {
	for _, reasoner := range agent.Reasoners {
		if reasoner.ID == name {
			return "reasoner", nil
		}
	}
	for _, skill := range agent.Skills {
		if skill.ID == name {
			return "skill", nil
		}
	}
	return "", fmt.Errorf("target '%s' not found on agent '%s'", name, agent.ID)
}

func buildAgentURL(agent *types.AgentNode, target *parsedTarget) string {
	if agent == nil {
		return ""
	}
	if agent.InvocationURL != nil && *agent.InvocationURL != "" {
		return *agent.InvocationURL
	}
	if agent.DeploymentType == "serverless" {
		base := strings.TrimSuffix(agent.BaseURL, "/")
		if base == "" {
			return ""
		}
		return fmt.Sprintf("%s/execute", base)
	}

	base := strings.TrimSuffix(agent.BaseURL, "/")
	if target.TargetType == "skill" {
		return fmt.Sprintf("%s/skills/%s", base, target.TargetName)
	}
	return fmt.Sprintf("%s/reasoners/%s", base, target.TargetName)
}

// versionRoundRobinCounter is used for round-robin selection across versioned agents.
var versionRoundRobinCounter uint64

// selectVersionedAgent picks a healthy agent from the versioned list using
// weighted round-robin. Returns the selected agent and its version string.
func selectVersionedAgent(versions []*types.AgentNode) (*types.AgentNode, string) {
	// Filter to healthy nodes
	var healthy []*types.AgentNode
	for _, v := range versions {
		if v.HealthStatus == types.HealthStatusActive && v.LifecycleStatus == types.AgentStatusReady {
			healthy = append(healthy, v)
		}
	}
	if len(healthy) == 0 {
		// Fallback: accept any non-offline, non-pending-approval node
		for _, v := range versions {
			if v.LifecycleStatus != types.AgentStatusOffline && v.LifecycleStatus != types.AgentStatusPendingApproval {
				healthy = append(healthy, v)
			}
		}
	}
	if len(healthy) == 0 {
		return nil, ""
	}

	// Check if all weights are equal (use simple round-robin)
	allEqual := true
	firstWeight := healthy[0].TrafficWeight
	totalWeight := 0
	for _, v := range healthy {
		w := v.TrafficWeight
		if w <= 0 {
			w = 100
		}
		totalWeight += w
		if w != firstWeight {
			allEqual = false
		}
	}

	if allEqual || totalWeight == 0 {
		// Simple round-robin
		n := atomic.AddUint64(&versionRoundRobinCounter, 1) - 1
		idx := n % uint64(len(healthy))
		selected := healthy[idx]
		return selected, selected.Version
	}

	// Weighted selection
	n := atomic.AddUint64(&versionRoundRobinCounter, 1) - 1
	counter := n % uint64(totalWeight)
	cumulative := 0
	for _, v := range healthy {
		w := v.TrafficWeight
		if w <= 0 {
			w = 100
		}
		cumulative += w
		if uint64(cumulative) > counter {
			return v, v.Version
		}
	}

	// Fallback
	return healthy[0], healthy[0].Version
}

func buildServerlessPayload(target *parsedTarget, exec *types.Execution, headers executionHeaders, input map[string]interface{}) map[string]interface{} {
	if target == nil || exec == nil {
		return map[string]interface{}{
			"input": input,
		}
	}

	execCtx := map[string]interface{}{
		"execution_id": exec.ExecutionID,
		"run_id":       exec.RunID,
		"workflow_id":  exec.RunID,
	}

	if headers.parentExecutionID != nil && *headers.parentExecutionID != "" {
		execCtx["parent_execution_id"] = *headers.parentExecutionID
	}
	if headers.sessionID != nil && *headers.sessionID != "" {
		execCtx["session_id"] = *headers.sessionID
	}
	if headers.actorID != nil && *headers.actorID != "" {
		execCtx["actor_id"] = *headers.actorID
	}

	payload := map[string]interface{}{
		"path":              fmt.Sprintf("/execute/%s", target.TargetName),
		"target":            target.TargetName,
		"reasoner":          target.TargetName,
		"input":             input,
		"execution_context": execCtx,
	}

	if target.TargetType != "" {
		payload["type"] = target.TargetType
		if target.TargetType == "skill" {
			payload["skill"] = target.TargetName
		}
	}

	return payload
}

type normalizedWebhookConfig struct {
	URL     string
	Secret  *string
	Headers map[string]string
}

func normalizeWebhookRequest(req *WebhookRequest) (*normalizedWebhookConfig, error) {
	if req == nil {
		return nil, nil
	}

	trimmedURL := strings.TrimSpace(req.URL)
	if trimmedURL == "" {
		return nil, fmt.Errorf("webhook.url is required")
	}

	parsed, err := url.Parse(trimmedURL)
	if err != nil {
		return nil, fmt.Errorf("invalid webhook url: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("webhook url must include scheme and host")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "https", "http":
	default:
		return nil, fmt.Errorf("webhook url must use http or https")
	}
	if parsed.User != nil {
		return nil, fmt.Errorf("webhook url must not contain embedded credentials")
	}
	if err := services.ValidateWebhookURL(trimmedURL); err != nil {
		return nil, fmt.Errorf("webhook url rejected: %w", err)
	}
	parsed.Fragment = ""

	normalizedHeaders := make(map[string]string)
	if len(req.Headers) > 0 {
		for key, value := range req.Headers {
			trimmedKey := strings.TrimSpace(key)
			trimmedValue := strings.TrimSpace(value)
			if trimmedKey == "" {
				continue
			}
			if len(normalizedHeaders) >= maxWebhookHeaders {
				return nil, fmt.Errorf("webhook.headers supports at most %d entries", maxWebhookHeaders)
			}
			if len(trimmedKey) > maxWebhookHeaderLength {
				return nil, fmt.Errorf("webhook header name '%s' is too long", trimmedKey)
			}
			if len(trimmedValue) > maxWebhookHeaderLength {
				return nil, fmt.Errorf("webhook header '%s' value is too long", trimmedKey)
			}
			normalizedHeaders[trimmedKey] = trimmedValue
		}
	}

	var secretPtr *string
	if trimmedSecret := strings.TrimSpace(req.Secret); trimmedSecret != "" {
		if len(trimmedSecret) > maxWebhookSecretLength {
			return nil, fmt.Errorf("webhook secret exceeds %d characters", maxWebhookSecretLength)
		}
		secretCopy := trimmedSecret
		secretPtr = &secretCopy
	}

	return &normalizedWebhookConfig{
		URL:     parsed.String(),
		Secret:  secretPtr,
		Headers: normalizedHeaders,
	}, nil
}

func decodeJSON(payload []byte) interface{} {
	if len(payload) == 0 {
		return nil
	}
	var v interface{}
	if err := json.Unmarshal(payload, &v); err == nil {
		return v
	}
	return string(payload)
}

func renderStatus(exec *types.Execution) ExecutionStatusResponse {
	var completedAt *string
	if exec.CompletedAt != nil {
		formatted := exec.CompletedAt.UTC().Format(time.RFC3339)
		completedAt = &formatted
	}

	resp := ExecutionStatusResponse{
		ExecutionID:       exec.ExecutionID,
		RunID:             exec.RunID,
		AgentNodeID:       exec.AgentNodeID,
		InstanceID:        exec.InstanceID,
		Status:            exec.Status,
		StatusReason:      exec.StatusReason,
		Result:            decodeJSON(exec.ResultPayload),
		Error:             exec.ErrorMessage,
		StartedAt:         exec.StartedAt.UTC().Format(time.RFC3339),
		CompletedAt:       completedAt,
		DurationMS:        exec.DurationMS,
		WebhookRegistered: exec.WebhookRegistered,
		WebhookEvents:     exec.WebhookEvents,
	}
	// For failed executions, expose the agent's raw response as error_details
	// so callers can access structured error data (e.g., permission_denied fields).
	if exec.Status == types.ExecutionStatusFailed && len(exec.ResultPayload) > 0 {
		resp.ErrorDetails = decodeJSON(exec.ResultPayload)
	}
	return resp
}

// renderStatusWithApproval enriches the base status response with approval
// fields from the corresponding WorkflowExecution record, if one exists.
func (c *executionController) renderStatusWithApproval(ctx context.Context, exec *types.Execution) ExecutionStatusResponse {
	resp := renderStatus(exec)

	// Resolve webhook_registered from the execution_webhooks table since the
	// field is not persisted on the execution record itself (db:"-").
	if hasWH, err := c.store.HasExecutionWebhook(ctx, exec.ExecutionID); err == nil {
		resp.WebhookRegistered = hasWH
	}

	// Best-effort enrichment — if the lookup fails we still return the base response.
	wfExec, err := c.store.GetWorkflowExecution(ctx, exec.ExecutionID)
	if err != nil || wfExec == nil {
		return resp
	}

	resp.ApprovalRequestID = wfExec.ApprovalRequestID
	resp.ApprovalStatus = wfExec.ApprovalStatus
	resp.ApprovalRequestURL = wfExec.ApprovalRequestURL
	return resp
}

func (c *executionController) ensureWorkflowExecutionRecord(ctx context.Context, exec *types.Execution, target *parsedTarget, payload []byte) {
	workflowExec := c.buildWorkflowExecutionRecord(ctx, exec, target, payload)
	if workflowExec == nil {
		return
	}

	if err := c.store.StoreWorkflowExecution(ctx, workflowExec); err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", exec.ExecutionID).
			Msg("failed to persist workflow execution state")
	}
}

func (c *executionController) buildWorkflowExecutionRecord(ctx context.Context, exec *types.Execution, target *parsedTarget, payload []byte) *types.WorkflowExecution {
	if exec == nil || target == nil {
		return nil
	}

	runID := exec.RunID
	if runID == "" {
		runID = utils.GenerateRunID()
	}

	rootWorkflowID, parentWorkflowID, depth := c.deriveWorkflowHierarchy(ctx, exec)

	startTime := exec.StartedAt
	if startTime.IsZero() {
		startTime = time.Now().UTC()
	}

	workflowName := fmt.Sprintf("%s.%s", exec.NodeID, exec.ReasonerID)
	runIDCopy := runID
	workflowExec := &types.WorkflowExecution{
		WorkflowID:          runID,
		ExecutionID:         exec.ExecutionID,
		AgentFieldRequestID: utils.GenerateAgentFieldRequestID(),
		RunID:               &runIDCopy,
		SessionID:           exec.SessionID,
		ActorID:             exec.ActorID,
		AgentNodeID:         exec.AgentNodeID,
		InstanceID:          exec.InstanceID,
		ParentWorkflowID:    parentWorkflowID,
		ParentExecutionID:   exec.ParentExecutionID,
		RootWorkflowID:      rootWorkflowID,
		WorkflowDepth:       depth,
		ReasonerID:          exec.ReasonerID,
		Status:              string(exec.Status),
		WorkflowName:        &workflowName,
		StartedAt:           startTime,
		CreatedAt:           startTime,
		UpdatedAt:           startTime,
		Notes:               []types.ExecutionNote{},
	}

	if len(payload) > 0 {
		cloned := cloneBytes(payload)
		workflowExec.InputData = json.RawMessage(cloned)
		workflowExec.InputSize = len(cloned)
	}

	if target.TargetType != "" {
		workflowExec.WorkflowTags = []string{target.TargetType}
	} else {
		workflowExec.WorkflowTags = []string{}
	}

	return workflowExec
}

func (c *executionController) deriveWorkflowHierarchy(ctx context.Context, exec *types.Execution) (*string, *string, int) {
	runID := exec.RunID
	rootWorkflowID := pointerString(runID)
	var parentWorkflowID *string
	depth := 0

	if exec.ParentExecutionID != nil {
		parentExecution, err := c.store.GetWorkflowExecution(ctx, *exec.ParentExecutionID)
		if err != nil {
			logger.Logger.Debug().
				Err(err).
				Str("execution_id", exec.ExecutionID).
				Str("parent_execution_id", *exec.ParentExecutionID).
				Msg("failed to load parent workflow execution")
		}
		if parentExecution != nil {
			parentWorkflowID = pointerString(parentExecution.WorkflowID)
			if parentExecution.RootWorkflowID != nil {
				rootWorkflowID = parentExecution.RootWorkflowID
			} else {
				rootWorkflowID = pointerString(parentExecution.WorkflowID)
			}
			depth = parentExecution.WorkflowDepth + 1
		} else {
			depth = 1
		}
	}

	return rootWorkflowID, parentWorkflowID, depth
}

func (c *executionController) updateWorkflowExecutionFinalState(
	ctx context.Context,
	executionID string,
	status types.ExecutionStatus,
	result []byte,
	elapsed time.Duration,
	errorMessage *string,
) {
	err := c.store.UpdateWorkflowExecution(ctx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution with ID %s not found", executionID)
		}
		now := time.Now().UTC()
		current.Status = string(status)
		current.UpdatedAt = now
		completedAt := now
		current.CompletedAt = &completedAt
		duration := elapsed.Milliseconds()
		current.DurationMS = &duration
		if len(result) > 0 {
			cloned := cloneBytes(result)
			current.OutputData = json.RawMessage(cloned)
			current.OutputSize = len(cloned)
		} else {
			current.OutputData = nil
			current.OutputSize = 0
		}
		current.ErrorMessage = errorMessage
		return current, nil
	})
	if err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", executionID).
			Msg("failed to update workflow execution state")
	}
}

func cloneBytes(src []byte) []byte {
	if src == nil {
		return nil
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}

// callError wraps an upstream agent HTTP error, preserving the original status
// code and response body for structured error propagation.
type callError struct {
	statusCode int
	message    string
	body       []byte
}

func (e *callError) Error() string {
	return e.message
}

func writeExecutionError(ctx *gin.Context, err error) {
	if err == nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "unknown error", "error_category": string(ErrorCategoryInternal)})
		return
	}

	var maxBytesErr *http.MaxBytesError
	if errors.As(err, &maxBytesErr) {
		ctx.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body too large"})
		return
	}

	var ce *callError
	if errors.As(err, &ce) {
		category := classifyCallError(ce, err)
		response := gin.H{
			"error":          ce.message,
			"error_category": string(category),
			"status":         "failed",
		}
		// Preserve structured error data from the agent's response body.
		if len(ce.body) > 0 {
			var parsed interface{}
			if json.Unmarshal(ce.body, &parsed) == nil {
				response["error_details"] = parsed
			}
		}
		// Propagate 4xx status codes from the agent (client-facing errors);
		// use 502 Bad Gateway for 5xx (upstream server failure).
		httpStatus := http.StatusBadGateway
		if ce.statusCode >= 400 && ce.statusCode < 500 {
			httpStatus = ce.statusCode
		}
		ctx.JSON(httpStatus, response)
		return
	}

	var pe *executionPreconditionError
	if errors.As(err, &pe) {
		body, retryAfter := renderExecutionPreconditionError(pe)
		if retryAfter > 0 {
			ctx.Header("Retry-After", strconv.Itoa(retryAfter))
		}
		ctx.JSON(pe.HTTPStatusCode(), body)
		return
	}

	// Classify untyped errors (timeouts, connection failures, etc.)
	category := classifyRawError(err)
	httpStatus := http.StatusBadRequest
	if category == ErrorCategoryAgentTimeout || category == ErrorCategoryAgentUnreachable {
		httpStatus = http.StatusGatewayTimeout
	}
	ctx.JSON(httpStatus, gin.H{
		"error":          err.Error(),
		"error_category": string(category),
	})
}

func renderExecutionPreconditionError(pe *executionPreconditionError) (gin.H, int) {
	body := gin.H{
		"error":          pe.Error(),
		"error_category": string(pe.Category()),
	}
	// When a stable machine code is set, promote it to `error` and move the
	// human-readable text to `message` — matching the contract used by sibling
	// handlers.
	if code := pe.ErrorCode(); code != "" {
		body["error"] = code
		body["message"] = pe.Error()
	}
	retryAfter := pe.retryAfter
	if retryAfter <= 0 {
		retryAfter = map[ErrorCategory]int{
			ErrorCategoryConcurrencyLimit:     1,
			ErrorCategoryNodeUnavailable:      1,
			ErrorCategoryLLMUnavailable:       30,
			ErrorCategoryControlPlaneShutdown: 1,
		}[pe.Category()]
	}
	if retryAfter > 0 {
		body["retry_after"] = retryAfter
	}
	return body, retryAfter
}

// classifyExecutionError determines the error category from any execution error.
func classifyExecutionError(err error) ErrorCategory {
	if err == nil {
		return ErrorCategoryInternal
	}

	var ce *callError
	if errors.As(err, &ce) {
		return classifyCallError(ce, err)
	}

	var pe *executionPreconditionError
	if errors.As(err, &pe) {
		return pe.Category()
	}

	return classifyRawError(err)
}

// classifyCallError determines the error category for an agent call error.
func classifyCallError(ce *callError, original error) ErrorCategory {
	if ce.statusCode >= 500 {
		return ErrorCategoryAgentError
	}
	if ce.statusCode == 408 {
		return ErrorCategoryAgentTimeout
	}
	// Check if the body is valid JSON — if not, it's a bad response
	if len(ce.body) > 0 {
		var js json.RawMessage
		if json.Unmarshal(ce.body, &js) != nil {
			return ErrorCategoryBadResponse
		}
	}
	return ErrorCategoryAgentError
}

// classifyRawError inspects an untyped error for timeout/connection patterns.
func classifyRawError(err error) ErrorCategory {
	if err == nil {
		return ErrorCategoryInternal
	}

	errStr := err.Error()

	// Context deadline exceeded = timeout
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(errStr, "context deadline exceeded") {
		return ErrorCategoryAgentTimeout
	}

	// Connection refused / reset = agent unreachable
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "i/o timeout") {
		return ErrorCategoryAgentUnreachable
	}

	// Cancelled context
	if errors.Is(err, context.Canceled) || strings.Contains(errStr, "context canceled") {
		return ErrorCategoryInternal
	}

	// An agent or reasoner that does not exist. Matched on the messages
	// prepareExecutionForTarget and determineTargetType produce, in the same
	// style as the transport patterns above.
	if strings.Contains(errStr, "not found") &&
		(strings.HasPrefix(errStr, "agent '") || strings.HasPrefix(errStr, "target '")) {
		return ErrorCategoryTargetNotFound
	}

	return ErrorCategoryInternal
}

// httpStatusForFailedExecution determines the HTTP status code to return to
// callers for a failed execution record. It replaces the hardcoded 502 in the
// async-completion branch with context-aware classification:
//
//  1. If StatusReason encodes a client error ("agent_client_error:<code>"), use that code.
//  2. If StatusReason maps to a known server-side category, use the appropriate 5xx.
//  3. Parse ErrorMessage for the "agent error (NNN):" pattern produced by the sync lane.
//  4. Default to 502 Bad Gateway.
func httpStatusForFailedExecution(exec *types.Execution) int {
	// Check StatusReason for an encoded client error status code.
	if exec.StatusReason != nil {
		reason := *exec.StatusReason
		if strings.HasPrefix(reason, "agent_client_error:") {
			codeStr := strings.TrimPrefix(reason, "agent_client_error:")
			if code, err := strconv.Atoi(codeStr); err == nil && code >= 400 && code < 500 {
				return code
			}
		}
		// Map known server-side categories.
		switch ErrorCategory(reason) {
		case ErrorCategoryAgentTimeout:
			return http.StatusGatewayTimeout
		case ErrorCategoryAgentUnreachable:
			return http.StatusBadGateway
		case ErrorCategoryTargetNotFound:
			return http.StatusNotFound
		case ErrorCategoryNodeUnavailable:
			return http.StatusServiceUnavailable
		case ErrorCategoryConcurrencyLimit:
			return http.StatusTooManyRequests
		}
	}

	// Fallback: parse ErrorMessage for the "agent error (NNN):" pattern that
	// the sync lane produces when the agent returns an HTTP error directly.
	if exec.ErrorMessage != nil {
		msg := *exec.ErrorMessage
		if strings.HasPrefix(msg, "agent error (") {
			if idx := strings.Index(msg, "):"); idx > 13 {
				codeStr := msg[13:idx]
				if code, err := strconv.Atoi(codeStr); err == nil && code >= 400 && code < 500 {
					return code
				}
			}
		}
	}

	return http.StatusBadGateway
}

func pointerTime(t time.Time) *time.Time {
	return &t
}

func pointerString(v string) *string {
	return &v
}

func pointerInt64(v int64) *int64 {
	return &v
}

func truncateForLog(body []byte) string {
	const limit = 1024
	if len(body) <= limit {
		return string(body)
	}
	return string(body[:limit]) + "..."
}

const (
	// bodyDigestPrefixLen is how many hex characters of the keyed body digest
	// reach the log. Long enough that two distinct bodies colliding is not a
	// practical concern for correlation.
	bodyDigestPrefixLen = 16

	// maxLoggedContentTypeLen bounds the agent-supplied content type. Response
	// headers are attacker-influenced and Go accepts them up to megabytes; the
	// log line must stay a log line.
	maxLoggedContentTypeLen = 128
)

// bodyDigestKey is a random key minted once per process for the redacted body
// digests below. A bare hash of the body would be a guessable commitment to it:
// a short, low-entropy response — a one-time code, an email address, a bare
// token, "true" — could be recovered offline by hashing candidates and matching
// the logged digest, with body_bytes to prune the search. Keying the digest
// removes that while keeping the property operators actually use, namely that
// the same body logs the same digest within one run of the control plane.
var bodyDigestKey = sync.OnceValue(func() []byte {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		// No entropy source: fail closed and log no digest at all rather than
		// fall back to an unkeyed one.
		return nil
	}
	return key
})

// redactedBodyDigest returns the logged digest prefix for a body, or "" when no
// key could be minted.
func redactedBodyDigest(body []byte) string {
	key := bodyDigestKey()
	if key == nil {
		return ""
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))[:bodyDigestPrefixLen]
}

// logSafeContentType reduces an agent-supplied Content-Type header to something
// bounded: the media type without its parameters, truncated as a last resort.
func logSafeContentType(contentType string) string {
	if mediaType, _, err := mime.ParseMediaType(contentType); err == nil && mediaType != "" {
		contentType = mediaType
	}
	if len(contentType) > maxLoggedContentTypeLen {
		return contentType[:maxLoggedContentTypeLen] + "..."
	}
	return contentType
}

// annotateBodyForLog describes an agent response body on a log event.
//
// Agent responses are caller data, so they follow the same redaction switch as
// execution payloads (logging.redact_payloads / AGENTFIELD_LOG_REDACT_PAYLOADS,
// see SetRedactPayloads); redact is the caller's view of that switch. With
// redaction on — the default — only non-reversible metadata is attached: the
// response media type, the body length in bytes and a keyed digest prefix. That
// is enough to recognise "same body as before" and to find the full payload in
// the database, without the body itself reaching stdout. With redaction
// explicitly disabled the previous truncated preview is attached instead.
func annotateBodyForLog(event *zerolog.Event, contentType string, body []byte, redact bool) *zerolog.Event {
	if event == nil {
		return nil
	}
	if mediaType := logSafeContentType(contentType); mediaType != "" {
		event = event.Str("content_type", mediaType)
	}
	event = event.Int("body_bytes", len(body))
	if !redact {
		return event.Str("body", truncateForLog(body))
	}
	event = event.Bool("body_redacted", true)
	if digest := redactedBodyDigest(body); digest != "" {
		event = event.Str("body_digest", digest)
	}
	return event
}

func (c *executionController) savePayload(ctx context.Context, data []byte) *string {
	if c.payloads == nil || len(data) == 0 {
		return nil
	}
	record, err := c.payloads.SaveBytes(ctx, data)
	if err != nil {
		logger.Logger.Warn().Err(err).Int("bytes", len(data)).Msg("failed to persist payload; proceeding without URI")
		return nil
	}
	if record == nil {
		return nil
	}
	uri := record.URI
	return &uri
}

func (j asyncExecutionJob) process() {
	j.processWithContext(context.Background())
}

func (j asyncExecutionJob) processWithContext(workerCtx context.Context) {
	// Release the per-agent concurrency slot when this job finishes
	defer j.plan.releaseSlot()

	// Use a bounded context so that paused executions do not block goroutines
	// indefinitely if the resume/cancel event is never delivered (e.g. event bus
	// crash, server restart). 24 hours is generous but prevents permanent leaks.
	bgCtx, cancel := context.WithTimeout(workerCtx, 24*time.Hour)
	defer cancel()

	currentExec, err := j.controller.store.GetExecutionRecord(bgCtx, j.plan.exec.ExecutionID)
	if err == nil && currentExec != nil {
		if currentExec.Status == types.ExecutionStatusCancelled {
			logger.Logger.Info().
				Str("execution_id", j.plan.exec.ExecutionID).
				Msg("skipping async agent call; execution cancelled")
			return
		}
		if currentExec.Status == types.ExecutionStatusPaused {
			if waitErr := j.controller.waitForResume(bgCtx, j.plan.exec.ExecutionID); waitErr != nil {
				logger.Logger.Info().
					Str("execution_id", j.plan.exec.ExecutionID).
					Err(waitErr).
					Msg("aborting async agent call while paused")
				return
			}
		}
	}

	resultBody, elapsed, asyncAccepted, callErr := j.controller.callAgent(bgCtx, &j.plan)
	if workerCtx.Err() != nil {
		persistCtx, persistCancel := shutdownPersistenceContext()
		j.failForControlPlaneShutdown(persistCtx, newControlPlaneShutdownError("execution was interrupted because the control plane shut down"))
		persistCancel()
		return
	}
	if callErr == nil && asyncAccepted {
		logger.Logger.Info().
			Str("execution_id", j.plan.exec.ExecutionID).
			Msg("agent accepted execution for async processing")
		return
	}

	// Extract, persist (best-effort), and strip token/cost usage from the
	// synchronous result envelope so it never leaks into the stored payload.
	if callErr == nil {
		if usageRaw, stripped := extractUsageFromResult(resultBody); usageRaw != nil {
			resultBody = stripped
			j.controller.ingestUsage(bgCtx, j.plan.exec, usageRaw)
		}
	}

	job := completionJob{
		controller: j.controller,
		plan:       &j.plan,
		result:     resultBody,
		elapsed:    elapsed,
		callErr:    callErr,
	}
	if err := enqueueCompletion(job); err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", j.plan.exec.ExecutionID).
			Msg("failed to enqueue completion job for async execution")
		if callErr != nil {
			if updateErr := j.controller.failExecution(bgCtx, &j.plan, callErr, elapsed, resultBody); updateErr != nil {
				logger.Logger.Error().
					Err(updateErr).
					Str("execution_id", j.plan.exec.ExecutionID).
					Msg("fallback async failure persistence failed")
			}
		} else {
			if updateErr := j.controller.completeExecution(bgCtx, &j.plan, resultBody, elapsed); updateErr != nil {
				logger.Logger.Error().
					Err(updateErr).
					Str("execution_id", j.plan.exec.ExecutionID).
					Msg("fallback async completion persistence failed")
			}
		}
	}
}

func newAsyncWorkerPool(workerCount, queueCapacity int) *asyncWorkerPool {
	workerCtx, cancelWorkers := context.WithCancel(context.Background())
	admissionCapacity := workerCount + queueCapacity
	pool := &asyncWorkerPool{
		queue:         make(chan asyncExecutionJob, admissionCapacity),
		reservations:  make(chan struct{}, admissionCapacity),
		workerCtx:     workerCtx,
		cancelWorkers: cancelWorkers,
	}

	for i := 0; i < workerCount; i++ {
		go func(workerID int) {
			for job := range pool.queue {
				func() {
					defer pool.releaseReservation()
					defer pool.jobs.Done()
					pool.mu.RLock()
					stopped := pool.stopped
					pool.mu.RUnlock()
					if stopped {
						persistCtx, cancel := shutdownPersistenceContext()
						job.terminateForControlPlaneShutdownWithContext(persistCtx, newControlPlaneShutdownError("execution was not started before the control plane shut down"))
						cancel()
					} else {
						job.processWithContext(pool.workerCtx)
					}
				}()
			}
		}(i)
	}

	logger.Logger.Info().
		Int("workers", workerCount).
		Int("queue_capacity", queueCapacity).
		Msg("async execution worker pool initialized")

	return pool
}

func (p *asyncWorkerPool) submit(job asyncExecutionJob) bool {
	if !p.reserve() {
		return false
	}
	if !p.submitReserved(job) {
		p.releaseReservation()
		return false
	}
	return true
}

func (p *asyncWorkerPool) reserve() bool {
	reserved, _ := p.reserveForAdmission()
	return reserved
}

// reserveForAdmission distinguishes saturation from a pool that has stopped so
// callers can return the stable control_plane_shutdown contract.
func (p *asyncWorkerPool) reserveForAdmission() (reserved, stopped bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.stopped {
		return false, true
	}
	select {
	case p.reservations <- struct{}{}:
		return true, false
	default:
		return false, false
	}
}

func (p *asyncWorkerPool) releaseReservation() {
	select {
	case <-p.reservations:
	default:
	}
}

func (p *asyncWorkerPool) submitReserved(job asyncExecutionJob) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.stopped {
		return false
	}
	p.jobs.Add(1)
	p.queue <- job
	return true
}

// StopAsyncWorkerPool prevents new admission and drains the process-wide pool.
func StopAsyncWorkerPool(ctx context.Context) {
	if asyncPool != nil {
		asyncPool.Stop(ctx)
	}
}

// Stop rejects new work, lets accepted work finish until ctx expires, then
// honestly terminates any jobs which have not started.
func (p *asyncWorkerPool) Stop(ctx context.Context) {
	p.mu.Lock()
	if !p.stopped {
		p.stopped = true
		close(p.queue)
	}
	p.mu.Unlock()

	done := make(chan struct{})
	go func() {
		p.jobs.Wait()
		close(done)
	}()
	select {
	case <-done:
		return
	case <-ctx.Done():
	}

	p.cancelWorkers()
	persistCtx, cancel := shutdownPersistenceContext()
	defer cancel()
	for job := range p.queue {
		p.releaseReservation()
		job.terminateForControlPlaneShutdownWithContext(persistCtx, newControlPlaneShutdownError("execution was not started before the control plane shut down"))
		p.jobs.Done()
	}
	select {
	case <-done:
	case <-persistCtx.Done():
	}
}

func shutdownPersistenceContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

func (j *asyncExecutionJob) terminateForControlPlaneShutdown(shutdownErr *executionPreconditionError) {
	persistCtx, cancel := shutdownPersistenceContext()
	defer cancel()
	j.terminateForControlPlaneShutdownWithContext(persistCtx, shutdownErr)
}

func (j *asyncExecutionJob) terminateForControlPlaneShutdownWithContext(ctx context.Context, shutdownErr *executionPreconditionError) {
	defer j.plan.releaseSlot()
	j.failForControlPlaneShutdown(ctx, shutdownErr)
}

func (j asyncExecutionJob) failForControlPlaneShutdown(ctx context.Context, shutdownErr *executionPreconditionError) {
	nodeID := ""
	if j.plan.target != nil {
		nodeID = j.plan.target.NodeID
	}
	if shutdownErr == nil {
		shutdownErr = newControlPlaneShutdownError("execution was not started before the control plane shut down")
	}
	if err := j.controller.failExecution(ctx, &j.plan, shutdownErr, 0, nil); err != nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Str("execution_id", j.plan.exec.ExecutionID).Msg("failed to terminate queued execution during shutdown")
		return
	}
	// failExecution deliberately preserves cancellation/waiting. Re-read the
	// execution after that atomic update and mirror the state that won the race
	// instead of blindly overwriting workflow_executions with failed.
	updatedExec, err := j.controller.store.GetExecutionRecord(ctx, j.plan.exec.ExecutionID)
	if err != nil || updatedExec == nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Str("execution_id", j.plan.exec.ExecutionID).Msg("failed to load terminal execution during shutdown reconciliation")
		return
	}
	if err := j.controller.store.UpdateWorkflowExecution(ctx, j.plan.exec.ExecutionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("workflow execution %s not found", j.plan.exec.ExecutionID)
		}
		current.Status = string(updatedExec.Status)
		current.StatusReason = updatedExec.StatusReason
		current.CompletedAt = updatedExec.CompletedAt
		current.UpdatedAt = updatedExec.UpdatedAt
		return current, nil
	}); err != nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Str("execution_id", j.plan.exec.ExecutionID).Msg("failed to record shutdown reason on queued workflow execution")
	}
}

func writeAsyncAdmissionError(ctx *gin.Context, status int, message string) {
	ctx.Header("Retry-After", "1")
	ctx.JSON(status, gin.H{"error": message, "error_category": "concurrency_limit", "retry_after": 1})
}

func getAsyncWorkerPool() *asyncWorkerPool {
	asyncPoolOnce.Do(func() {
		workerCount := resolveIntFromEnv("AGENTFIELD_EXEC_ASYNC_WORKERS", max(runtime.NumCPU(), 16))
		if workerCount <= 0 {
			workerCount = max(runtime.NumCPU(), 16)
		}

		queueCapacity := resolveIntFromEnv("AGENTFIELD_EXEC_ASYNC_QUEUE_CAPACITY", 1024)
		if queueCapacity <= 0 {
			queueCapacity = 1024
		}

		asyncPool = newAsyncWorkerPool(workerCount, queueCapacity)
	})
	return asyncPool
}

func resolveIntFromEnv(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		logger.Logger.Warn().
			Str("key", key).
			Str("value", raw).
			Msg("invalid integer environment override; using fallback")
		return fallback
	}
	return value
}

func ensureCompletionWorker() {
	completionOnce.Do(func() {
		size := resolveIntFromEnv("AGENTFIELD_EXEC_COMPLETION_QUEUE", 2048)
		if size <= 0 {
			size = 2048
		}
		completionQueue = make(chan completionJob, size)
		go func() {
			for job := range completionQueue {
				err := processCompletionJob(job)
				if job.done != nil {
					job.done <- err
					close(job.done)
				}
			}
		}()
	})
}

func processCompletionJob(job completionJob) error {
	ctx := context.Background()
	if job.callErr != nil {
		return job.controller.failExecution(ctx, job.plan, job.callErr, job.elapsed, job.result)
	}
	return job.controller.completeExecution(ctx, job.plan, job.result, job.elapsed)
}

func enqueueCompletion(job completionJob) error {
	ensureCompletionWorker()
	select {
	case completionQueue <- job:
		return nil
	default:
		return fmt.Errorf("completion queue is full")
	}
}
