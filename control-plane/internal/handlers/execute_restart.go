package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type restartExecutionRequest struct {
	Scope   string                 `json:"scope,omitempty"`
	Reuse   string                 `json:"reuse,omitempty"`
	Fork    bool                   `json:"fork,omitempty"`
	Reason  string                 `json:"reason,omitempty"`
	Input   map[string]interface{} `json:"input,omitempty"`
	Context map[string]interface{} `json:"context,omitempty"`
	Webhook *WebhookRequest        `json:"webhook,omitempty"`
}

type restartExecutionResponse struct {
	ExecutionID             string  `json:"execution_id"`
	RunID                   string  `json:"run_id"`
	WorkflowID              string  `json:"workflow_id"`
	Status                  string  `json:"status"`
	Target                  string  `json:"target"`
	Type                    string  `json:"type"`
	CreatedAt               string  `json:"created_at"`
	EnqueuedAt              string  `json:"enqueued_at,omitempty"`
	SourceExecutionID       string  `json:"source_execution_id"`
	SourceRunID             string  `json:"source_run_id"`
	RestartedExecutionID    string  `json:"restarted_execution_id"`
	ReplayBeforeExecutionID *string `json:"replay_before_execution_id,omitempty"`
	ReplayMode              string  `json:"replay_mode"`
	Scope                   string  `json:"scope"`
	Kind                    string  `json:"kind"`
	WebhookRegistered       bool    `json:"webhook_registered"`
	WebhookError            *string `json:"webhook_error,omitempty"`
}

type workflowRunMetadataStore interface {
	UpdateWorkflowRunMetadata(context.Context, string, func(map[string]json.RawMessage) error) error
}

type workflowRunMetadataReader interface {
	GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
}

type executionRestartPointerStore interface {
	SetExecutionRestartedAs(context.Context, string, string) error
}

type restartOptions struct {
	Scope         string
	Reuse         string
	Reason        string
	Fork          bool
	Input         map[string]interface{}
	Context       map[string]interface{}
	Webhook       *WebhookRequest
	Kind          string
	ResumeAttempt int
}

type restartResponseError struct {
	status  int
	message string
	cause   error
}

func (e *restartResponseError) Error() string {
	if e.cause != nil {
		return e.cause.Error()
	}
	return e.message
}

type restartQueueError struct{ stopped bool }

func (e *restartQueueError) Error() string {
	if e.stopped {
		return "async execution queue stopped; retry later"
	}
	return "async execution queue is full; retry later"
}

// RestartExecutionHandler starts a new execution/run from an existing workflow
// point. The restarted code runs normally, while downstream app.call requests can
// reuse matching successful child outputs from the source run.
func RestartExecutionHandler(store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string) gin.HandlerFunc {
	controller := newExecutionController(store, payloads, webhooks, timeout, internalToken)
	return controller.handleRestart
}

func (c *executionController) handleRestart(ctx *gin.Context) {
	sourceExecutionID := strings.TrimSpace(ctx.Param("execution_id"))
	if sourceExecutionID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	var req restartExecutionRequest
	if err := ctx.ShouldBindJSON(&req); err != nil && !errors.Is(err, io.EOF) {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}

	scope := strings.TrimSpace(req.Scope)
	if scope == "" {
		scope = "workflow"
	}
	if scope != "workflow" && scope != "execution" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "scope must be one of: workflow, execution"})
		return
	}

	reuse := strings.TrimSpace(req.Reuse)
	if reuse == "" {
		reuse = "succeeded-before"
	}
	if reuse != "succeeded-before" && reuse != "all-succeeded" && reuse != "none" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "reuse must be one of: succeeded-before, all-succeeded, none"})
		return
	}

	reqCtx := ctx.Request.Context()
	sourceExec, err := c.store.GetExecutionRecord(reqCtx, sourceExecutionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", sourceExecutionID).Msg("restart: failed to load source execution")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load source execution"})
		return
	}
	if sourceExec == nil {
		ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("execution %s not found", sourceExecutionID)})
		return
	}

	response, err := c.startRestart(reqCtx, sourceExec, restartOptions{
		Scope: scope, Reuse: reuse, Reason: req.Reason, Fork: req.Fork,
		Input: req.Input, Context: req.Context, Webhook: req.Webhook,
	})
	if err != nil {
		var responseErr *restartResponseError
		if errors.As(err, &responseErr) {
			if responseErr.cause != nil {
				logger.Logger.Error().Err(responseErr.cause).Str("run_id", sourceExec.RunID).Msg("restart: failed to find workflow root")
			}
			ctx.JSON(responseErr.status, gin.H{"error": responseErr.message})
			return
		}
		var queueErr *restartQueueError
		if errors.As(err, &queueErr) {
			if queueErr.stopped {
				writeExecutionError(ctx, newControlPlaneShutdownError(queueErr.Error()))
				return
			}
			writeAsyncAdmissionError(ctx, http.StatusServiceUnavailable, queueErr.Error())
			return
		}
		writeExecutionError(ctx, err)
		return
	}
	ctx.Header("X-Execution-ID", response.ExecutionID)
	ctx.Header("X-Run-ID", response.RunID)
	ctx.JSON(http.StatusAccepted, response)
}

func (c *executionController) startRestart(ctx context.Context, sourceExec *types.Execution, opts restartOptions) (*restartExecutionResponse, error) {
	if sourceExec == nil {
		return nil, fmt.Errorf("source execution is required")
	}
	restartExec := sourceExec
	if opts.Scope == "workflow" {
		root, err := c.findWorkflowRestartRoot(ctx, sourceExec.RunID)
		if err != nil {
			return nil, &restartResponseError{status: http.StatusInternalServerError, message: "failed to load workflow root", cause: err}
		}
		if root == nil {
			return nil, &restartResponseError{status: http.StatusNotFound, message: fmt.Sprintf("run %s not found", sourceExec.RunID)}
		}
		restartExec = root
	}

	stored := types.DecodeStoredExecutionPayload(restartExec.InputPayload)
	input := stored.Input
	if input == nil {
		input = map[string]interface{}{}
	}
	if opts.Input != nil {
		input = opts.Input
	}
	contextPayload := stored.Context
	if opts.Context != nil {
		contextPayload = opts.Context
	}

	headers := executionHeaders{
		runID: utils.GenerateRunID(), sessionID: restartExec.SessionID, actorID: restartExec.ActorID,
		replaySourceRunID: sourceExec.RunID, replayBeforeExecutionID: sourceExec.ExecutionID, replayMode: opts.Reuse,
	}
	if opts.Reuse == "none" {
		headers.replaySourceRunID = ""
		headers.replayBeforeExecutionID = ""
	}
	if opts.Scope == "execution" && opts.Reuse == "succeeded-before" {
		headers.replayMode = "all-succeeded"
		headers.replayBeforeExecutionID = ""
	}

	target := fmt.Sprintf("%s.%s", restartExec.NodeID, restartExec.ReasonerID)
	pool := getAsyncWorkerPool()
	if admitted, stopped := pool.reserveForAdmission(); !admitted {
		return nil, &restartQueueError{stopped: stopped}
	}
	reserved := true
	defer func() {
		if reserved {
			pool.releaseReservation()
		}
	}()

	plan, err := c.prepareExecutionForTargetWithAdmission(ctx, target, ExecuteRequest{
		Input: input, Context: contextPayload, Webhook: opts.Webhook,
	}, headers, "", "", true)
	if err != nil {
		return nil, err
	}
	defer plan.releaseSlot()

	kind := strings.TrimSpace(opts.Kind)
	if kind == "" {
		kind = "restart"
		if opts.Fork || opts.Input != nil || opts.Context != nil {
			kind = "fork"
		}
	}
	c.persistRestartRunMetadataWithOptions(ctx, plan, sourceExec, restartExec, opts, kind)
	c.publishExecutionStartedEvent(plan)

	job := asyncExecutionJob{controller: c, plan: *plan}
	plan.slotHeld = false
	submitted := false
	defer func() {
		if !submitted {
			job.plan.releaseSlot()
		}
	}()
	if ok := pool.submitReserved(job); !ok {
		shutdownErr := newControlPlaneShutdownError("async execution queue stopped; retry later")
		job.terminateForControlPlaneShutdown(shutdownErr)
		return nil, &restartQueueError{stopped: true}
	}
	submitted = true
	reserved = false

	c.persistRestartForwardPointers(ctx, sourceExec, restartExec, plan.exec)

	createdAt := plan.exec.CreatedAt.UTC().Format(time.RFC3339)
	var replayBefore *string
	if headers.replayBeforeExecutionID != "" {
		replayBefore = &headers.replayBeforeExecutionID
	}
	return &restartExecutionResponse{
		ExecutionID: plan.exec.ExecutionID, RunID: plan.exec.RunID, WorkflowID: plan.exec.RunID,
		Status: string(types.ExecutionStatusQueued), Target: target, Type: plan.targetType,
		CreatedAt: createdAt, EnqueuedAt: createdAt,
		SourceExecutionID: sourceExec.ExecutionID, SourceRunID: sourceExec.RunID,
		RestartedExecutionID: restartExec.ExecutionID, ReplayBeforeExecutionID: replayBefore,
		ReplayMode: headers.replayMode, Scope: opts.Scope, Kind: kind,
		WebhookRegistered: plan.webhookRegistered, WebhookError: plan.webhookError,
	}, nil
}

func (c *executionController) findWorkflowRestartRoot(ctx context.Context, runID string) (*types.Execution, error) {
	executions, err := c.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		RunID:          &runID,
		SortBy:         "started_at",
		SortDescending: false,
	})
	if err != nil || len(executions) == 0 {
		return nil, err
	}
	sort.SliceStable(executions, func(i, j int) bool {
		return executions[i].StartedAt.Before(executions[j].StartedAt)
	})
	for _, exec := range executions {
		if exec != nil && (exec.ParentExecutionID == nil || strings.TrimSpace(*exec.ParentExecutionID) == "") {
			return exec, nil
		}
	}
	return executions[0], nil
}

func (c *executionController) persistRestartRunMetadata(ctx context.Context, plan *preparedExecution, sourceExec, restartExec *types.Execution, scope, reuse, kind, reason string) {
	c.persistRestartRunMetadataWithOptions(ctx, plan, sourceExec, restartExec, restartOptions{
		Scope: scope, Reuse: reuse, Reason: reason,
	}, kind)
}

func (c *executionController) persistRestartRunMetadataWithOptions(ctx context.Context, plan *preparedExecution, sourceExec, restartExec *types.Execution, opts restartOptions, kind string) {
	if plan == nil || plan.exec == nil || sourceExec == nil || restartExec == nil {
		return
	}
	store, ok := c.store.(workflowRunMetadataStore)
	if !ok {
		return
	}
	lineage := map[string]interface{}{
		"kind":                   kind,
		"source_run_id":          sourceExec.RunID,
		"source_execution_id":    sourceExec.ExecutionID,
		"restarted_execution_id": restartExec.ExecutionID,
		"reuse":                  opts.Reuse,
		"scope":                  opts.Scope,
	}
	if kind == "resume" {
		lineage["resume_attempt"] = opts.ResumeAttempt
	}
	var encodedReason json.RawMessage
	if trimmed := strings.TrimSpace(opts.Reason); trimmed != "" {
		encodedReason, _ = json.Marshal(trimmed)
	}
	encoded, err := json.Marshal(lineage)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", plan.exec.RunID).Msg("failed to encode restart run metadata")
		return
	}
	if err := store.UpdateWorkflowRunMetadata(ctx, plan.exec.RunID, func(namespaces map[string]json.RawMessage) error {
		namespaces["lineage"] = encoded
		if encodedReason != nil {
			namespaces["reason"] = encodedReason
		} else {
			delete(namespaces, "reason")
		}
		return nil
	}); err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", plan.exec.RunID).Msg("failed to persist restart run metadata")
	}
}

func (c *executionController) persistRestartForwardPointers(ctx context.Context, sourceExec, restartExec, successor *types.Execution) {
	if sourceExec == nil || restartExec == nil || successor == nil {
		return
	}
	if store, ok := c.store.(executionRestartPointerStore); ok {
		sourceIDs := []string{sourceExec.ExecutionID}
		if restartExec.ExecutionID != sourceExec.ExecutionID {
			sourceIDs = append(sourceIDs, restartExec.ExecutionID)
		}
		for _, sourceID := range sourceIDs {
			if sourceID == "" {
				continue
			}
			if err := store.SetExecutionRestartedAs(ctx, sourceID, successor.ExecutionID); err != nil {
				logger.Logger.Warn().Err(err).Str("execution_id", sourceID).Str("restarted_as", successor.ExecutionID).Msg("failed to persist restart forward pointer")
			}
		}
	}

	metadata, ok := c.store.(workflowRunMetadataStore)
	if !ok {
		return
	}
	at := time.Now().UTC().Format(time.RFC3339)
	if err := metadata.UpdateWorkflowRunMetadata(ctx, sourceExec.RunID, func(namespaces map[string]json.RawMessage) error {
		lineage := make(map[string]interface{})
		if raw := namespaces["lineage"]; len(raw) > 0 {
			_ = json.Unmarshal(raw, &lineage)
		}
		if lineage == nil {
			lineage = make(map[string]interface{})
		}
		lineage["restarted_as"] = map[string]interface{}{
			"execution_id": successor.ExecutionID,
			"run_id":       successor.RunID,
			"at":           at,
		}
		encoded, err := json.Marshal(lineage)
		if err != nil {
			return err
		}
		namespaces["lineage"] = encoded
		return nil
	}); err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", sourceExec.RunID).Str("restarted_as", successor.ExecutionID).Msg("failed to persist restart forward lineage")
	}
}
