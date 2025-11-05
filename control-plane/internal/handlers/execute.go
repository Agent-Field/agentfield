package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ExecutionStore captures the storage operations required by the simplified execution handlers.
type ExecutionStore interface {
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
	CreateExecutionRecord(ctx context.Context, execution *types.Execution) error
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error
}

// ExecuteRequest represents an execution request from an agent client.
type ExecuteRequest struct {
	Input   map[string]interface{} `json:"input" binding:"required"`
	Context map[string]interface{} `json:"context,omitempty"`
	Webhook *WebhookRequest        `json:"webhook,omitempty"`
}

// WebhookRequest represents webhook registration parameters supplied by the client.
type WebhookRequest struct {
	URL     string            `json:"url"`
	Secret  string            `json:"secret,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
}

// ExecuteResponse is returned for synchronous executions.
type ExecuteResponse struct {
	ExecutionID       string      `json:"execution_id"`
	RunID             string      `json:"run_id"`
	Status            string      `json:"status"`
	Result            interface{} `json:"result,omitempty"`
	ErrorMessage      *string     `json:"error_message,omitempty"`
	DurationMS        int64       `json:"duration_ms"`
	FinishedAt        string      `json:"finished_at"`
	WebhookRegistered bool        `json:"webhook_registered,omitempty"`
}

// AsyncExecuteResponse is returned when callers request asynchronous execution.
type AsyncExecuteResponse struct {
	ExecutionID       string  `json:"execution_id"`
	RunID             string  `json:"run_id"`
	WorkflowID        string  `json:"workflow_id"`
	Status            string  `json:"status"`
	Target            string  `json:"target"`
	Type              string  `json:"type"`
	CreatedAt         string  `json:"created_at"`
	EnqueuedAt        string  `json:"enqueued_at,omitempty"`
	WebhookRegistered bool    `json:"webhook_registered"`
	WebhookError      *string `json:"webhook_error,omitempty"`
}

// ExecutionStatusResponse mirrors the data required by the UI to render execution state.
type ExecutionStatusResponse struct {
	ExecutionID       string                         `json:"execution_id"`
	RunID             string                         `json:"run_id"`
	Status            string                         `json:"status"`
	Result            interface{}                    `json:"result,omitempty"`
	Error             *string                        `json:"error,omitempty"`
	StartedAt         string                         `json:"started_at"`
	CompletedAt       *string                        `json:"completed_at,omitempty"`
	DurationMS        *int64                         `json:"duration_ms,omitempty"`
	WebhookRegistered bool                           `json:"webhook_registered"`
	WebhookEvents     []*types.ExecutionWebhookEvent `json:"webhook_events,omitempty"`
}

// BatchStatusRequest allows the UI to fetch multiple execution statuses at once.
type BatchStatusRequest struct {
	ExecutionIDs []string `json:"execution_ids" binding:"required"`
}

// BatchStatusResponse is the batched counterpart to ExecutionStatusResponse.
type BatchStatusResponse map[string]ExecutionStatusResponse

type executionController struct {
	store      ExecutionStore
	httpClient *http.Client
	payloads   services.PayloadStore
	webhooks   services.WebhookDispatcher
}

type asyncExecutionJob struct {
	controller *executionController
	plan       preparedExecution
}

type asyncWorkerPool struct {
	queue chan asyncExecutionJob
}

type completionJob struct {
	controller *executionController
	plan       *preparedExecution
	result     []byte
	elapsed    time.Duration
	callErr    error
	done       chan error
}

var (
	asyncPoolOnce sync.Once
	asyncPool     *asyncWorkerPool

	completionOnce  sync.Once
	completionQueue chan completionJob
)

const (
	maxWebhookHeaders      = 20
	maxWebhookHeaderLength = 512
	maxWebhookSecretLength = 4096
)

// ExecuteHandler handles synchronous execution requests.
func ExecuteHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks)
	return controller.handleSync
}

// ExecuteAsyncHandler handles asynchronous execution requests.
func ExecuteAsyncHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks)
	return controller.handleAsync
}

// GetExecutionStatusHandler resolves a single execution record.
func GetExecutionStatusHandler(store ExecutionStore) gin.HandlerFunc {
	controller := newExecutionController(store, nil, nil)
	return controller.handleStatus
}

// BatchExecutionStatusHandler resolves multiple execution records.
func BatchExecutionStatusHandler(store ExecutionStore) gin.HandlerFunc {
	controller := newExecutionController(store, nil, nil)
	return controller.handleBatchStatus
}

func newExecutionController(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher) *executionController {
	return &executionController{
		store: store,
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
		},
		payloads: payloads,
		webhooks: webhooks,
	}
}

func (c *executionController) handleSync(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	plan, err := c.prepareExecution(reqCtx, ctx)
	if err != nil {
		writeExecutionError(ctx, err)
		return
	}

	resultBody, elapsed, callErr := c.callAgent(reqCtx, plan)
	job := completionJob{
		controller: c,
		plan:       plan,
		result:     resultBody,
		elapsed:    elapsed,
		callErr:    callErr,
		done:       make(chan error, 1),
	}
	if err := enqueueCompletion(job); err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", plan.exec.ExecutionID).Msg("failed to enqueue completion job")
		writeExecutionError(ctx, err)
		return
	}
	if err := <-job.done; err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", plan.exec.ExecutionID).Msg("completion processing failed")
		writeExecutionError(ctx, err)
		return
	}
	if callErr != nil {
		writeExecutionError(ctx, callErr)
		return
	}

	response := ExecuteResponse{
		ExecutionID:       plan.exec.ExecutionID,
		RunID:             plan.exec.RunID,
		Status:            types.ExecutionStatusSucceeded,
		Result:            decodeJSON(resultBody),
		DurationMS:        elapsed.Milliseconds(),
		FinishedAt:        time.Now().UTC().Format(time.RFC3339),
		WebhookRegistered: plan.webhookRegistered,
	}

	ctx.Header("X-Execution-ID", plan.exec.ExecutionID)
	ctx.Header("X-Run-ID", plan.exec.RunID)
	ctx.JSON(http.StatusOK, response)
}

func (c *executionController) handleAsync(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	plan, err := c.prepareExecution(reqCtx, ctx)
	if err != nil {
		writeExecutionError(ctx, err)
		return
	}

	pool := getAsyncWorkerPool()
	job := asyncExecutionJob{
		controller: c,
		plan:       *plan,
	}

	if ok := pool.submit(job); !ok {
		queueErr := errors.New("async execution queue is full; retry later")
		if updateErr := c.failExecution(reqCtx, plan, queueErr, 0, nil); updateErr != nil {
			logger.Logger.Error().
				Err(updateErr).
				Str("execution_id", plan.exec.ExecutionID).
				Msg("failed to persist execution failure after queue saturation")
		}
		logger.Logger.Warn().
			Str("execution_id", plan.exec.ExecutionID).
			Msg("async execution rejected due to queue saturation")
		ctx.JSON(http.StatusServiceUnavailable, gin.H{"error": queueErr.Error()})
		return
	}

	createdAt := plan.exec.CreatedAt.UTC().Format(time.RFC3339)
	targetLabel := fmt.Sprintf("%s.%s", plan.target.NodeID, plan.target.TargetName)
	response := AsyncExecuteResponse{
		ExecutionID:       plan.exec.ExecutionID,
		RunID:             plan.exec.RunID,
		WorkflowID:        plan.exec.RunID,
		Status:            string(types.ExecutionStatusQueued),
		Target:            targetLabel,
		Type:              plan.targetType,
		CreatedAt:         createdAt,
		EnqueuedAt:        createdAt,
		WebhookRegistered: plan.webhookRegistered,
	}
	if plan.webhookError != nil {
		response.WebhookError = plan.webhookError
	}

	ctx.Header("X-Execution-ID", plan.exec.ExecutionID)
	ctx.Header("X-Run-ID", plan.exec.RunID)
	ctx.JSON(http.StatusAccepted, response)
}

func (c *executionController) handleStatus(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	executionID := ctx.Param("execution_id")
	if executionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	exec, err := c.store.GetExecutionRecord(reqCtx, executionID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to load execution: %v", err)})
		return
	}
	if exec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": "execution not found"})
		return
	}

	ctx.JSON(http.StatusOK, renderStatus(exec))
}

func (c *executionController) handleBatchStatus(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	var request BatchStatusRequest
	if err := ctx.ShouldBindJSON(&request); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	response := make(BatchStatusResponse, len(request.ExecutionIDs))
	for _, id := range request.ExecutionIDs {
		exec, err := c.store.GetExecutionRecord(reqCtx, id)
		if err != nil {
			response[id] = ExecutionStatusResponse{
				ExecutionID: id,
				Status:      "error",
				Error:       pointerString(fmt.Sprintf("load execution: %v", err)),
			}
			continue
		}
		if exec == nil {
			response[id] = ExecutionStatusResponse{
				ExecutionID: id,
				Status:      "not_found",
			}
			continue
		}
		response[id] = renderStatus(exec)
	}

	ctx.JSON(http.StatusOK, response)
}

type preparedExecution struct {
	exec              *types.Execution
	requestBody       []byte
	agent             *types.AgentNode
	target            *parsedTarget
	targetType        string
	webhookRegistered bool
	webhookError      *string
}

func (c *executionController) prepareExecution(ctx context.Context, ginCtx *gin.Context) (*preparedExecution, error) {
	targetParam := ginCtx.Param("target")
	target, err := parseTarget(targetParam)
	if err != nil {
		return nil, fmt.Errorf("invalid target: %w", err)
	}

	var req ExecuteRequest
	if err := ginCtx.ShouldBindJSON(&req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	if len(req.Input) == 0 {
		return nil, errors.New("input is required")
	}

	var (
		sanitizedWebhook *normalizedWebhookConfig
		webhookError     *string
	)

	if req.Webhook != nil {
		cfg, err := normalizeWebhookRequest(req.Webhook)
		if err != nil {
			errMsg := err.Error()
			webhookError = &errMsg
		} else if cfg != nil {
			sanitizedWebhook = cfg
		}
	}

	agent, err := c.store.GetAgent(ctx, target.NodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to load agent '%s': %w", target.NodeID, err)
	}
	if agent == nil {
		return nil, fmt.Errorf("agent '%s' not found", target.NodeID)
	}

	targetType, err := determineTargetType(agent, target.TargetName)
	if err != nil {
		return nil, err
	}
	target.TargetType = targetType

	headers := readExecutionHeaders(ginCtx)
	runID := headers.runID
	if runID == "" {
		runID = utils.GenerateRunID()
	}

	executionID := utils.GenerateExecutionID()
	now := time.Now().UTC()

	clientPayload := map[string]interface{}{
		"input": req.Input,
	}
	if len(req.Context) > 0 {
		clientPayload["context"] = req.Context
	}

	storedPayload, err := json.Marshal(clientPayload)
	if err != nil {
		return nil, fmt.Errorf("encode execution payload: %w", err)
	}

	agentPayload := make(map[string]interface{}, len(req.Input))
	for key, value := range req.Input {
		agentPayload[key] = value
	}
	agentPayloadBytes, err := json.Marshal(agentPayload)
	if err != nil {
		return nil, fmt.Errorf("encode agent payload: %w", err)
	}

	inputURI := c.savePayload(ctx, storedPayload)

	exec := &types.Execution{
		ExecutionID:       executionID,
		RunID:             runID,
		ParentExecutionID: headers.parentExecutionID,
		AgentNodeID:       agent.ID,
		ReasonerID:        target.TargetName,
		NodeID:            target.NodeID,
		Status:            types.ExecutionStatusRunning,
		InputPayload:      json.RawMessage(storedPayload),
		InputURI:          inputURI,
		StartedAt:         now,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	if headers.sessionID != nil {
		exec.SessionID = headers.sessionID
	}
	if headers.actorID != nil {
		exec.ActorID = headers.actorID
	}

	if err := c.store.CreateExecutionRecord(ctx, exec); err != nil {
		return nil, fmt.Errorf("create execution record: %w", err)
	}

	var webhookRegistered bool
	if sanitizedWebhook != nil && webhookError == nil {
		registration := &types.ExecutionWebhook{
			ExecutionID:   executionID,
			URL:           sanitizedWebhook.URL,
			Headers:       sanitizedWebhook.Headers,
			Status:        types.ExecutionWebhookStatusPending,
			AttemptCount:  0,
			NextAttemptAt: pointerTime(now),
		}
		if sanitizedWebhook.Secret != nil {
			registration.Secret = sanitizedWebhook.Secret
		}
		if err := c.store.RegisterExecutionWebhook(ctx, registration); err != nil {
			logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to register execution webhook")
			errMsg := err.Error()
			webhookError = &errMsg
		} else {
			webhookRegistered = true
			exec.WebhookRegistered = true
		}
	}

	if !webhookRegistered {
		exec.WebhookRegistered = false
	}

	return &preparedExecution{
		exec:              exec,
		requestBody:       agentPayloadBytes,
		agent:             agent,
		target:            target,
		targetType:        targetType,
		webhookRegistered: webhookRegistered,
		webhookError:      webhookError,
	}, nil
}

func (c *executionController) callAgent(ctx context.Context, plan *preparedExecution) ([]byte, time.Duration, error) {
	start := time.Now()
	url := buildAgentURL(plan.agent, plan.target)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(plan.requestBody))
	if err != nil {
		return nil, 0, fmt.Errorf("create agent request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", plan.exec.RunID)
	req.Header.Set("X-Execution-ID", plan.exec.ExecutionID)
	if plan.exec.ParentExecutionID != nil {
		req.Header.Set("X-Parent-Execution-ID", *plan.exec.ParentExecutionID)
	}
	if plan.exec.SessionID != nil {
		req.Header.Set("X-Session-ID", *plan.exec.SessionID)
	}
	if plan.exec.ActorID != nil {
		req.Header.Set("X-Actor-ID", *plan.exec.ActorID)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, time.Since(start), fmt.Errorf("agent call failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, time.Since(start), fmt.Errorf("read agent response: %w", err)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return body, time.Since(start), fmt.Errorf("agent error (%d): %s", resp.StatusCode, truncateForLog(body))
	}

	return body, time.Since(start), nil
}

func (c *executionController) completeExecution(ctx context.Context, plan *preparedExecution, result []byte, elapsed time.Duration) error {
	resultURI := c.savePayload(ctx, result)

	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		updated, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
			}
			now := time.Now().UTC()
			current.Status = types.ExecutionStatusSucceeded
			current.ResultPayload = json.RawMessage(result)
			current.ErrorMessage = nil
			current.CompletedAt = pointerTime(now)
			duration := elapsed.Milliseconds()
			current.DurationMS = &duration
			current.UpdatedAt = now
			current.ResultURI = resultURI
			return current, nil
		})
		if err == nil {
			if plan.webhookRegistered || (updated != nil && updated.WebhookRegistered) {
				c.triggerWebhook(plan.exec.ExecutionID)
			}
			return nil
		}
		lastErr = err
		if isRetryableDBError(err) {
			time.Sleep(backoffDelay(attempt))
			continue
		}
		return err
	}
	return lastErr
}

func (c *executionController) failExecution(ctx context.Context, plan *preparedExecution, callErr error, elapsed time.Duration, result []byte) error {
	errMsg := callErr.Error()
	resultURI := c.savePayload(ctx, result)
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		updated, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
			}
			now := time.Now().UTC()
			current.Status = types.ExecutionStatusFailed
			current.ErrorMessage = &errMsg
			current.CompletedAt = pointerTime(now)
			duration := elapsed.Milliseconds()
			current.DurationMS = &duration
			current.UpdatedAt = now
			if len(result) > 0 {
				current.ResultPayload = json.RawMessage(result)
			}
			current.ResultURI = resultURI
			return current, nil
		})
		if err == nil {
			if plan.webhookRegistered || (updated != nil && updated.WebhookRegistered) {
				c.triggerWebhook(plan.exec.ExecutionID)
			}
			return nil
		}
		lastErr = err
		if isRetryableDBError(err) {
			time.Sleep(backoffDelay(attempt))
			continue
		}
		return err
	}
	return lastErr
}

func (c *executionController) triggerWebhook(executionID string) {
	if c.webhooks == nil || executionID == "" {
		return
	}
	if err := c.webhooks.Notify(context.Background(), executionID); err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to enqueue webhook delivery")
	}
}

type executionHeaders struct {
	runID             string
	parentExecutionID *string
	sessionID         *string
	actorID           *string
}

func readExecutionHeaders(ctx *gin.Context) executionHeaders {
	runID := strings.TrimSpace(ctx.GetHeader("X-Run-ID"))
	parent := strings.TrimSpace(ctx.GetHeader("X-Parent-Execution-ID"))
	session := strings.TrimSpace(ctx.GetHeader("X-Session-ID"))
	actor := strings.TrimSpace(ctx.GetHeader("X-Actor-ID"))

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
		runID:             runID,
		parentExecutionID: parentPtr,
		sessionID:         sessionPtr,
		actorID:           actorPtr,
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
	if agent.DeploymentType == "serverless" && agent.InvocationURL != nil && *agent.InvocationURL != "" {
		return *agent.InvocationURL
	}

	base := strings.TrimSuffix(agent.BaseURL, "/")
	if target.TargetType == "skill" {
		return fmt.Sprintf("%s/skills/%s", base, target.TargetName)
	}
	return fmt.Sprintf("%s/reasoners/%s", base, target.TargetName)
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

	return ExecutionStatusResponse{
		ExecutionID:       exec.ExecutionID,
		RunID:             exec.RunID,
		Status:            exec.Status,
		Result:            decodeJSON(exec.ResultPayload),
		Error:             exec.ErrorMessage,
		StartedAt:         exec.StartedAt.UTC().Format(time.RFC3339),
		CompletedAt:       completedAt,
		DurationMS:        exec.DurationMS,
		WebhookRegistered: exec.WebhookRegistered,
		WebhookEvents:     exec.WebhookEvents,
	}
}

func writeExecutionError(ctx *gin.Context, err error) {
	if err == nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "unknown error"})
		return
	}
	ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
}

func pointerTime(t time.Time) *time.Time {
	return &t
}

func pointerString(v string) *string {
	return &v
}

func truncateForLog(body []byte) string {
	const limit = 1024
	if len(body) <= limit {
		return string(body)
	}
	return string(body[:limit]) + "..."
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
	uri := record.URI
	return &uri
}

func (j asyncExecutionJob) process() {
	bgCtx := context.Background()
	resultBody, elapsed, callErr := j.controller.callAgent(bgCtx, &j.plan)
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
	pool := &asyncWorkerPool{
		queue: make(chan asyncExecutionJob, queueCapacity),
	}

	for i := 0; i < workerCount; i++ {
		go func(workerID int) {
			for job := range pool.queue {
				job.process()
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
	select {
	case p.queue <- job:
		return true
	default:
		return false
	}
}

func getAsyncWorkerPool() *asyncWorkerPool {
	asyncPoolOnce.Do(func() {
		workerCount := resolveIntFromEnv("AGENTFIELD_EXEC_ASYNC_WORKERS", runtime.NumCPU())
		if workerCount <= 0 {
			workerCount = runtime.NumCPU()
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
