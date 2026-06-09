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
	StoreWorkflowRun(ctx context.Context, run *types.WorkflowRun) error
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

	restartExec := sourceExec
	if scope == "workflow" {
		root, rootErr := c.findWorkflowRestartRoot(reqCtx, sourceExec.RunID)
		if rootErr != nil {
			logger.Logger.Error().Err(rootErr).Str("run_id", sourceExec.RunID).Msg("restart: failed to find workflow root")
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load workflow root"})
			return
		}
		if root == nil {
			ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("run %s not found", sourceExec.RunID)})
			return
		}
		restartExec = root
	}

	stored := types.DecodeStoredExecutionPayload(restartExec.InputPayload)
	input := stored.Input
	if input == nil {
		input = map[string]interface{}{}
	}
	if req.Input != nil {
		input = req.Input
	}
	contextPayload := stored.Context
	if req.Context != nil {
		contextPayload = req.Context
	}

	newRunID := utils.GenerateRunID()
	headers := executionHeaders{
		runID:                   newRunID,
		sessionID:               restartExec.SessionID,
		actorID:                 restartExec.ActorID,
		replaySourceRunID:       sourceExec.RunID,
		replayBeforeExecutionID: sourceExec.ExecutionID,
		replayMode:              reuse,
	}
	if reuse == "none" {
		headers.replaySourceRunID = ""
		headers.replayBeforeExecutionID = ""
	}
	if scope == "execution" && reuse == "succeeded-before" {
		headers.replayMode = "all-succeeded"
		headers.replayBeforeExecutionID = ""
	}

	target := fmt.Sprintf("%s.%s", restartExec.NodeID, restartExec.ReasonerID)
	plan, err := c.prepareExecutionForTarget(reqCtx, target, ExecuteRequest{
		Input:   input,
		Context: contextPayload,
		Webhook: req.Webhook,
	}, headers, "", "")
	if err != nil {
		writeExecutionError(ctx, err)
		return
	}

	if err := CheckExecutionPreconditions(plan.target.NodeID, plan.llmEndpoint); err != nil {
		_ = c.failExecution(reqCtx, plan, err, 0, nil)
		writeExecutionError(ctx, err)
		return
	}

	kind := "restart"
	if req.Fork || req.Input != nil || req.Context != nil {
		kind = "fork"
	}
	c.persistRestartRunMetadata(reqCtx, plan, sourceExec, restartExec, scope, reuse, kind, req.Reason)

	c.publishExecutionStartedEvent(plan)

	pool := getAsyncWorkerPool()
	job := asyncExecutionJob{
		controller: c,
		plan:       *plan,
	}
	if ok := pool.submit(job); !ok {
		ReleaseExecutionSlot(plan.target.NodeID)
		queueErr := errors.New("async execution queue is full; retry later")
		if updateErr := c.failExecution(reqCtx, plan, queueErr, 0, nil); updateErr != nil {
			logger.Logger.Error().Err(updateErr).Str("execution_id", plan.exec.ExecutionID).Msg("restart: failed to persist queue saturation")
		}
		ctx.JSON(http.StatusServiceUnavailable, gin.H{"error": queueErr.Error(), "error_category": "concurrency_limit"})
		return
	}

	createdAt := plan.exec.CreatedAt.UTC().Format(time.RFC3339)
	var replayBefore *string
	if headers.replayBeforeExecutionID != "" {
		replayBefore = &headers.replayBeforeExecutionID
	}
	response := restartExecutionResponse{
		ExecutionID:             plan.exec.ExecutionID,
		RunID:                   plan.exec.RunID,
		WorkflowID:              plan.exec.RunID,
		Status:                  string(types.ExecutionStatusQueued),
		Target:                  target,
		Type:                    plan.targetType,
		CreatedAt:               createdAt,
		EnqueuedAt:              createdAt,
		SourceExecutionID:       sourceExec.ExecutionID,
		SourceRunID:             sourceExec.RunID,
		RestartedExecutionID:    restartExec.ExecutionID,
		ReplayBeforeExecutionID: replayBefore,
		ReplayMode:              headers.replayMode,
		Scope:                   scope,
		Kind:                    kind,
		WebhookRegistered:       plan.webhookRegistered,
		WebhookError:            plan.webhookError,
	}
	ctx.Header("X-Execution-ID", plan.exec.ExecutionID)
	ctx.Header("X-Run-ID", plan.exec.RunID)
	ctx.JSON(http.StatusAccepted, response)
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
	if plan == nil || plan.exec == nil || sourceExec == nil || restartExec == nil {
		return
	}
	store, ok := c.store.(workflowRunMetadataStore)
	if !ok {
		return
	}
	now := time.Now().UTC()
	metadata := map[string]interface{}{
		"lineage": map[string]interface{}{
			"kind":                   kind,
			"source_run_id":          sourceExec.RunID,
			"source_execution_id":    sourceExec.ExecutionID,
			"restarted_execution_id": restartExec.ExecutionID,
			"reuse":                  reuse,
			"scope":                  scope,
		},
	}
	if trimmed := strings.TrimSpace(reason); trimmed != "" {
		metadata["reason"] = trimmed
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", plan.exec.RunID).Msg("failed to encode restart run metadata")
		return
	}
	if err := store.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:           plan.exec.RunID,
		RootWorkflowID:  plan.exec.RunID,
		RootExecutionID: &plan.exec.ExecutionID,
		Status:          string(types.ExecutionStatusQueued),
		TotalSteps:      1,
		Metadata:        json.RawMessage(encoded),
		CreatedAt:       now,
		UpdatedAt:       now,
	}); err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", plan.exec.RunID).Msg("failed to persist restart run metadata")
	}
}
