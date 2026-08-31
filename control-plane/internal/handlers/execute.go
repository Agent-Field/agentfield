package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ExecutionStore captures the storage operations required by the simplified execution handlers.
type ExecutionStore interface {
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
	ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error)
	CreateExecutionRecord(ctx context.Context, execution *types.Execution) error
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	GetExecutionRecordsBatch(ctx context.Context, executionIDs []string) (map[string]*types.Execution, error)
	UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error
	HasExecutionWebhook(ctx context.Context, executionID string) (bool, error)
	StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error
	UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error
	GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error)
	QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error)
	StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error
	GetExecutionEventBus() *events.ExecutionEventBus
}

// ExecuteRequest represents an execution request from an agent client.
type ExecuteRequest struct {
	Input   map[string]interface{} `json:"input"`
	Context map[string]interface{} `json:"context,omitempty"`
	Webhook *WebhookRequest        `json:"webhook,omitempty"`
	// RunMetadata names, labels or links the run started by this request. It is
	// excluded from replay dedupe and ignored on child executions.
	RunMetadata *RunMetadataInput `json:"run_metadata,omitempty"`
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
	ErrorDetails      interface{} `json:"error_details,omitempty"`
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
	StatusReason      *string                        `json:"status_reason,omitempty"`
	Result            interface{}                    `json:"result,omitempty"`
	Error             *string                        `json:"error,omitempty"`
	ErrorDetails      interface{}                    `json:"error_details,omitempty"`
	StartedAt         string                         `json:"started_at"`
	CompletedAt       *string                        `json:"completed_at,omitempty"`
	DurationMS        *int64                         `json:"duration_ms,omitempty"`
	WebhookRegistered bool                           `json:"webhook_registered"`
	WebhookEvents     []*types.ExecutionWebhookEvent `json:"webhook_events,omitempty"`
	// Approval fields (populated when execution has an active approval request)
	ApprovalRequestID  *string `json:"approval_request_id,omitempty"`
	ApprovalStatus     *string `json:"approval_status,omitempty"`
	ApprovalRequestURL *string `json:"approval_request_url,omitempty"`
}

// BatchStatusRequest allows the UI to fetch multiple execution statuses at once.
type BatchStatusRequest struct {
	ExecutionIDs []string `json:"execution_ids" binding:"required"`
}

// BatchStatusResponse is the batched counterpart to ExecutionStatusResponse.
type BatchStatusResponse map[string]ExecutionStatusResponse

type executionStatusUpdateRequest struct {
	Status       string                 `json:"status" binding:"required"`
	StatusReason *string                `json:"status_reason,omitempty"`
	Result       map[string]interface{} `json:"result,omitempty"`
	Error        string                 `json:"error,omitempty"`
	DurationMS   *int64                 `json:"duration_ms,omitempty"`
	CompletedAt  *time.Time             `json:"completed_at,omitempty"`
	Progress     *int                   `json:"progress,omitempty"`
	// ErrorStatusCode is the optional HTTP status code the agent SDK sends to
	// indicate whether the failure is client-facing (4xx) or an upstream error
	// (5xx). When present and in the 4xx range, the control plane propagates it
	// to callers instead of returning a blanket 502 Bad Gateway.
	ErrorStatusCode *int `json:"error_status_code,omitempty"`
	// Usage is the optional token/cost usage object the agent SDK attaches at
	// the top level of the status-callback body. It is a sibling of Result, so
	// it is never persisted into the result payload. Absent = no-op.
	Usage map[string]interface{} `json:"usage,omitempty"`
}

type replayHit struct {
	SourceExecutionID string
	SourceRunID       string
	Result            json.RawMessage
}

type executionController struct {
	store          ExecutionStore
	httpClient     *http.Client
	payloads       services.PayloadStore
	webhooks       services.WebhookDispatcher
	eventBus       *events.ExecutionEventBus
	timeout        time.Duration
	internalToken  string // sent as Authorization header when forwarding to agents
	readARDConfig  func() config.ARDConfig
	redactPayloads bool
}

type asyncExecutionJob struct {
	controller *executionController
	plan       preparedExecution
}

type asyncWorkerPool struct {
	queue         chan asyncExecutionJob
	reservations  chan struct{}
	workerCtx     context.Context
	cancelWorkers context.CancelFunc
	mu            sync.RWMutex
	stopped       bool
	jobs          sync.WaitGroup
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

	// defaultRedactPayloads controls whether execution input/output data is
	// excluded from internal event bus payloads. Set at server startup from
	// config.Logging.ShouldRedactPayloads(). Default true (safe).
	defaultRedactPayloads = true
)

// SetRedactPayloads configures the package-level default for payload redaction.
// Call this once at server startup after loading config.
func SetRedactPayloads(redact bool) {
	defaultRedactPayloads = redact
}

const (
	maxWebhookHeaders      = 20
	maxWebhookHeaderLength = 512
	maxWebhookSecretLength = 4096

	// maxBatchStatusIDs caps the number of execution IDs a single
	// batch-status request may fetch, matching the storage-layer cap.
	maxBatchStatusIDs = 500
)

// ExecuteHandler handles synchronous execution requests.
func ExecuteHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks, timeout, internalToken, nil)
	return controller.handleSync
}

// ExecuteHandlerWithARD handles synchronous execution requests and can route
// explicitly-callable imported ARD resources through the same SDK app.call path.
func ExecuteHandlerWithARD(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string, readARDConfig func() config.ARDConfig) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks, timeout, internalToken, readARDConfig)
	return controller.handleSync
}

// ExecuteAsyncHandler handles asynchronous execution requests.
func ExecuteAsyncHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks, timeout, internalToken, nil)
	return controller.handleAsync
}

// GetExecutionStatusHandler resolves a single execution record.
func GetExecutionStatusHandler(store ExecutionStore) gin.HandlerFunc {
	controller := newExecutionController(store, nil, nil, 0, "", nil)
	return controller.handleStatus
}

// BatchExecutionStatusHandler resolves multiple execution records.
func BatchExecutionStatusHandler(store ExecutionStore) gin.HandlerFunc {
	controller := newExecutionController(store, nil, nil, 0, "", nil)
	return controller.handleBatchStatus
}

// UpdateExecutionStatusHandler ingests status callbacks from agent nodes.
func UpdateExecutionStatusHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks, timeout, "", nil)
	return controller.handleStatusUpdate
}

func newExecutionController(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string, readARDConfig ...func() config.ARDConfig) *executionController {
	var ardConfigReader func() config.ARDConfig
	if len(readARDConfig) > 0 {
		ardConfigReader = readARDConfig[0]
	}
	return &executionController{
		store: store,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		payloads:       payloads,
		webhooks:       webhooks,
		eventBus:       store.GetExecutionEventBus(),
		timeout:        timeout,
		internalToken:  internalToken,
		readARDConfig:  ardConfigReader,
		redactPayloads: defaultRedactPayloads,
	}
}

func (c *executionController) handleSync(ctx *gin.Context) {
	if c.tryHandleExternalARDCall(ctx) {
		return
	}

	reqCtx := ctx.Request.Context()
	plan, err := c.prepareExecution(reqCtx, ctx)
	if err != nil {
		writeExecutionError(ctx, err)
		return
	}
	plan.executionMode = "sync"

	if plan.replayHit != nil {
		if err := c.completeReplayHit(reqCtx, plan); err != nil {
			writeExecutionError(ctx, err)
			return
		}
		ctx.Header("X-Execution-ID", plan.exec.ExecutionID)
		ctx.Header("X-Run-ID", plan.exec.RunID)
		ctx.Header("X-AgentField-Replay-Hit", plan.replayHit.SourceExecutionID)
		ctx.JSON(http.StatusOK, ExecuteResponse{
			ExecutionID:       plan.exec.ExecutionID,
			RunID:             plan.exec.RunID,
			Status:            types.ExecutionStatusSucceeded,
			Result:            decodeJSON(plan.replayHit.Result),
			DurationMS:        0,
			FinishedAt:        time.Now().UTC().Format(time.RFC3339),
			WebhookRegistered: plan.webhookRegistered,
		})
		return
	}

	// Check LLM health and per-agent concurrency limits before proceeding
	if err := CheckExecutionPreconditions(plan.target.NodeID, plan.llmEndpoint); err != nil {
		_ = c.failExecution(reqCtx, plan, err, 0, nil)
		writeExecutionError(ctx, err)
		return
	}
	defer ReleaseExecutionSlot(plan.target.NodeID)

	// Emit execution started event with full reasoner context
	c.publishExecutionStartedEvent(plan)

	resultBody, elapsed, asyncAccepted, callErr := c.callAgent(reqCtx, plan)

	// If agent returned HTTP 202 (async acknowledgment), wait for callback completion
	if callErr == nil && asyncAccepted {
		logger.Logger.Info().
			Str("execution_id", plan.exec.ExecutionID).
			Str("agent", plan.target.NodeID).
			Str("reasoner", plan.target.TargetName).
			Msg("agent returned async acknowledgment, waiting for completion")

		// Wait for agent to call back and complete the execution
		// Use configured timeout to match the HTTP client timeout
		exec, waitErr := c.waitForExecutionCompletion(reqCtx, plan.exec.ExecutionID, c.timeout)
		if waitErr != nil {
			logger.Logger.Error().
				Err(waitErr).
				Str("execution_id", plan.exec.ExecutionID).
				Msg("failed to wait for async execution completion")
			writeExecutionError(ctx, waitErr)
			return
		}

		// Build response from completed execution
		var result interface{}
		if exec.ResultPayload != nil {
			result = decodeJSON(exec.ResultPayload)
		}

		var durationMS int64
		if exec.DurationMS != nil {
			durationMS = *exec.DurationMS
		}

		var finishedAt string
		if exec.CompletedAt != nil {
			finishedAt = exec.CompletedAt.UTC().Format(time.RFC3339)
		} else {
			finishedAt = time.Now().UTC().Format(time.RFC3339)
		}

		// Check if execution failed
		if exec.Status == types.ExecutionStatusFailed {
			errMsg := "execution failed"
			if exec.ErrorMessage != nil {
				errMsg = *exec.ErrorMessage
			}
			response := ExecuteResponse{
				ExecutionID:       exec.ExecutionID,
				RunID:             exec.RunID,
				Status:            string(exec.Status),
				ErrorMessage:      &errMsg,
				ErrorDetails:      decodeJSON(exec.ResultPayload),
				DurationMS:        durationMS,
				FinishedAt:        finishedAt,
				WebhookRegistered: exec.WebhookRegistered,
			}
			ctx.Header("X-Execution-ID", exec.ExecutionID)
			ctx.Header("X-Run-ID", exec.RunID)
			ctx.JSON(httpStatusForFailedExecution(exec), response)
			return
		}

		// Return successful execution result
		response := ExecuteResponse{
			ExecutionID:       exec.ExecutionID,
			RunID:             exec.RunID,
			Status:            string(exec.Status),
			Result:            result,
			DurationMS:        durationMS,
			FinishedAt:        finishedAt,
			WebhookRegistered: exec.WebhookRegistered,
		}
		ctx.Header("X-Execution-ID", exec.ExecutionID)
		ctx.Header("X-Run-ID", exec.RunID)
		if plan.routedVersion != "" {
			ctx.Header("X-Routed-Version", plan.routedVersion)
		}
		ctx.JSON(http.StatusOK, response)
		return
	}

	// Agent returned HTTP 200 (synchronous result). Extract any token/cost
	// usage the SDK attached to the result envelope, persist it (best-effort),
	// and strip it so it never leaks into the stored/returned result payload.
	if callErr == nil {
		if usageRaw, stripped := extractUsageFromResult(resultBody); usageRaw != nil {
			resultBody = stripped
			c.ingestUsage(reqCtx, plan.exec, usageRaw)
		}
	}

	// Process completion normally
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
	if plan.routedVersion != "" {
		ctx.Header("X-Routed-Version", plan.routedVersion)
	}
	ctx.JSON(http.StatusOK, response)
}

func (c *executionController) handleAsync(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	pool := getAsyncWorkerPool()
	// A reservation covers both preparation and time waiting in the queue. The
	// worker releases it on dequeue so only not-yet-started work consumes capacity.
	if !pool.reserve() {
		writeAsyncAdmissionError(ctx, http.StatusServiceUnavailable, "async execution queue is full; retry later")
		return
	}
	reserved := true
	defer func() {
		if reserved {
			pool.releaseReservation()
		}
	}()

	plan, err := c.prepareAsyncExecution(reqCtx, ctx)
	if err != nil {
		writeExecutionError(ctx, err)
		return
	}
	plan.executionMode = "async"

	if plan.replayHit != nil {
		ReleaseExecutionSlot(plan.target.NodeID)
		if err := c.completeReplayHit(reqCtx, plan); err != nil {
			writeExecutionError(ctx, err)
			return
		}

		createdAt := plan.exec.CreatedAt.UTC().Format(time.RFC3339)
		targetLabel := fmt.Sprintf("%s.%s", plan.target.NodeID, plan.target.TargetName)
		response := AsyncExecuteResponse{
			ExecutionID:       plan.exec.ExecutionID,
			RunID:             plan.exec.RunID,
			WorkflowID:        plan.exec.RunID,
			Status:            string(types.ExecutionStatusSucceeded),
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
		ctx.Header("X-AgentField-Replay-Hit", plan.replayHit.SourceExecutionID)
		ctx.JSON(http.StatusAccepted, response)
		return
	}

	// The slot was acquired before persistence. process releases it when the
	// agent call returns (including an HTTP 202 acknowledgement).

	// Emit execution started event with full reasoner context
	c.publishExecutionStartedEvent(plan)

	job := asyncExecutionJob{
		controller: c,
		plan:       *plan,
	}

	if ok := pool.submitReserved(job); !ok {
		ReleaseExecutionSlot(plan.target.NodeID) // Release since process() won't run
		writeAsyncAdmissionError(ctx, http.StatusServiceUnavailable, "async execution queue stopped; retry later")
		return
	}
	reserved = false

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

	ctx.JSON(http.StatusOK, c.renderStatusWithApproval(reqCtx, exec))
}

func (c *executionController) handleBatchStatus(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	var request BatchStatusRequest
	if err := ctx.ShouldBindJSON(&request); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(request.ExecutionIDs) > maxBatchStatusIDs {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("batch status supports at most %d execution IDs, got %d", maxBatchStatusIDs, len(request.ExecutionIDs))})
		return
	}

	// Use one storage fetch for the normal path. If it fails, fall back to
	// individual reads so the established per-ID error contract is preserved.
	records, err := c.store.GetExecutionRecordsBatch(reqCtx, request.ExecutionIDs)

	response := make(BatchStatusResponse, len(request.ExecutionIDs))
	for _, id := range request.ExecutionIDs {
		if err != nil {
			exec, getErr := c.store.GetExecutionRecord(reqCtx, id)
			if getErr != nil {
				response[id] = ExecutionStatusResponse{
					ExecutionID: id,
					Status:      "error",
					Error:       pointerString(fmt.Sprintf("load execution: %v", getErr)),
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
			response[id] = c.renderStatusWithApproval(reqCtx, exec)
			continue
		}

		exec, ok := records[id]
		if !ok || exec == nil {
			// Missing IDs preserve the prior per-ID response behavior.
			response[id] = ExecutionStatusResponse{
				ExecutionID: id,
				Status:      "not_found",
			}
			continue
		}
		response[id] = c.renderStatusWithApproval(reqCtx, exec)
	}

	ctx.JSON(http.StatusOK, response)
}

// errTerminalStatusConflict marks a callback that tries to rewrite one
// terminal status as a different one (e.g. succeeded → failed). The handler
// answers it with 409 so bounded SDK retries fail fast instead of looking
// like a server fault.
var errTerminalStatusConflict = errors.New("terminal status conflict")

func (c *executionController) handleStatusUpdate(ctx *gin.Context) {
	reqCtx := ctx.Request.Context()
	executionID := ctx.Param("execution_id")
	if executionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	var req executionStatusUpdateRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}

	normalizedStatus := types.NormalizeExecutionStatus(req.Status)
	if normalizedStatus == "" || normalizedStatus == string(types.ExecutionStatusUnknown) {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("unsupported status '%s'", req.Status)})
		return
	}

	var (
		resultBytes []byte
		err         error
	)
	if len(req.Result) > 0 {
		resultBytes, err = json.Marshal(req.Result)
		if err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("failed to encode result: %v", err)})
			return
		}
	}

	resultURI := c.savePayload(reqCtx, resultBytes)
	isTerminal := types.IsTerminalExecutionStatus(normalizedStatus)
	var elapsed time.Duration
	var errorMsg *string
	var terminalNoop bool

	updated, err := c.store.UpdateExecutionRecord(reqCtx, executionID, func(current *types.Execution) (*types.Execution, error) {
		terminalNoop = false
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}

		// Guard: executions in "waiting" state can only transition to
		// running, cancelled, or failed. The approval webhook handler
		// manages the waiting→running transition; direct jumps to
		// succeeded or timeout would desync the executions and
		// workflow_executions tables.
		if current.Status == types.ExecutionStatusWaiting {
			switch normalizedStatus {
			case string(types.ExecutionStatusRunning),
				string(types.ExecutionStatusCancelled),
				string(types.ExecutionStatusFailed):
				// allowed
			default:
				logger.Logger.Warn().
					Str("execution_id", executionID).
					Str("current_status", string(current.Status)).
					Str("requested_status", normalizedStatus).
					Msg("rejecting status update: execution is waiting for approval")
				return nil, fmt.Errorf("execution %s is in 'waiting' state; only running, cancelled, or failed transitions are allowed", executionID)
			}
		}

		// Terminal-state guard. Once an execution has reached a terminal
		// state, the only accepted write is an idempotent re-delivery of
		// that same status (so callers can retry their own callback); it is
		// acknowledged as a no-op so the record is not rewritten and none of
		// the side effects (lifecycle event, webhook, usage ingestion) run a
		// second time. A non-terminal write is rejected — a late retried
		// fire-and-forget update must not stomp the status back to "running"
		// and strand the caller's poll loop. A different terminal status is
		// rejected too: a duplicate callback must not flip "succeeded" to
		// "failed" after the outcome was already observed.
		if types.IsTerminalExecutionStatus(string(current.Status)) {
			if normalizedStatus == string(current.Status) {
				terminalNoop = true
				return current, nil
			}
			logger.Logger.Warn().
				Str("execution_id", executionID).
				Str("current_status", string(current.Status)).
				Str("requested_status", normalizedStatus).
				Msg("rejecting status update: execution is already in a terminal state")
			if types.IsTerminalExecutionStatus(normalizedStatus) {
				return nil, fmt.Errorf("execution %s is already in terminal state '%s'; cannot transition to '%s': %w", executionID, current.Status, normalizedStatus, errTerminalStatusConflict)
			}
			return nil, fmt.Errorf("execution %s is already in terminal state '%s'; cannot transition to '%s'", executionID, current.Status, normalizedStatus)
		}

		current.Status = normalizedStatus
		current.StatusReason = req.StatusReason
		// When the SDK sends an error_status_code in the 4xx range, record it in
		// StatusReason so the async-completion branch can propagate the correct
		// HTTP status to the caller (instead of a blanket 502).
		if req.ErrorStatusCode != nil && *req.ErrorStatusCode >= 400 && *req.ErrorStatusCode < 500 {
			code := fmt.Sprintf("agent_client_error:%d", *req.ErrorStatusCode)
			current.StatusReason = &code
		} else if req.StatusReason == nil && req.ErrorStatusCode != nil && *req.ErrorStatusCode >= 500 {
			reason := string(ErrorCategoryAgentError)
			current.StatusReason = &reason
		}
		if len(resultBytes) > 0 {
			current.ResultPayload = json.RawMessage(resultBytes)
			current.ResultURI = resultURI
		}

		if req.Error != "" {
			errCopy := req.Error
			current.ErrorMessage = &errCopy
			errorMsg = &errCopy
		} else if normalizedStatus == string(types.ExecutionStatusSucceeded) {
			current.ErrorMessage = nil
			errorMsg = nil
		}

		if req.DurationMS != nil {
			current.DurationMS = req.DurationMS
			elapsed = time.Duration(*req.DurationMS) * time.Millisecond
		} else if isTerminal && !current.StartedAt.IsZero() {
			var completed time.Time
			if req.CompletedAt != nil && !req.CompletedAt.IsZero() {
				completed = req.CompletedAt.UTC()
			} else {
				completed = time.Now().UTC()
			}
			elapsed = completed.Sub(current.StartedAt)
			duration := elapsed.Milliseconds()
			current.DurationMS = pointerInt64(duration)
		}

		if normalizedStatus == string(types.ExecutionStatusSucceeded) || normalizedStatus == string(types.ExecutionStatusFailed) || normalizedStatus == string(types.ExecutionStatusCancelled) || normalizedStatus == string(types.ExecutionStatusTimeout) {
			if req.CompletedAt != nil && !req.CompletedAt.IsZero() {
				completed := req.CompletedAt.UTC()
				current.CompletedAt = &completed
			} else {
				now := time.Now().UTC()
				current.CompletedAt = &now
			}
		} else if req.CompletedAt != nil && !req.CompletedAt.IsZero() {
			completed := req.CompletedAt.UTC()
			current.CompletedAt = &completed
		} else {
			current.CompletedAt = nil
		}

		return current, nil
	})
	if err != nil {
		if errors.Is(err, errTerminalStatusConflict) {
			ctx.JSON(http.StatusConflict, gin.H{"error": fmt.Sprintf("failed to update execution: %v", err)})
			return
		}
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to update execution: %v", err)})
		return
	}
	if updated == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": "execution not found"})
		return
	}
	if terminalNoop {
		ctx.JSON(http.StatusOK, c.renderStatusWithApproval(reqCtx, updated))
		return
	}
	if elapsed == 0 && updated.DurationMS != nil {
		elapsed = time.Duration(*updated.DurationMS) * time.Millisecond
	}

	// Persist token/cost usage reported alongside the status callback.
	// Best-effort: failures are logged and never fail the status update.
	c.ingestUsage(reqCtx, updated, req.Usage)

	c.updateWorkflowExecutionStatus(reqCtx, executionID, normalizedStatus, req.StatusReason)

	if isTerminal {
		c.updateWorkflowExecutionFinalState(reqCtx, executionID, types.ExecutionStatus(normalizedStatus), updated.ResultPayload, elapsed, errorMsg)
		if hasWH, _ := c.store.HasExecutionWebhook(reqCtx, executionID); hasWH {
			c.triggerWebhook(executionID)
		}
	}

	eventData := map[string]interface{}{
		"error":             req.Error,
		"progress":          req.Progress,
		"transition_source": "status_callback",
	}
	if req.StatusReason != nil && strings.TrimSpace(*req.StatusReason) != "" {
		eventData["status_reason"] = strings.TrimSpace(*req.StatusReason)
	}
	if !c.redactPayloads {
		eventData["result"] = req.Result
		if inputPayload := decodeJSON(updated.InputPayload); inputPayload != nil {
			eventData["input"] = inputPayload
		}
	}
	c.publishExecutionEvent(updated, normalizedStatus, eventData)

	ctx.JSON(http.StatusOK, c.renderStatusWithApproval(reqCtx, updated))
}
