package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

func (c *executionController) prepareExecution(ctx context.Context, ginCtx *gin.Context) (*preparedExecution, error) {
	return c.prepareExecutionWithAdmission(ctx, ginCtx, false)
}

func (c *executionController) prepareAsyncExecution(ctx context.Context, ginCtx *gin.Context) (*preparedExecution, error) {
	return c.prepareExecutionWithAdmission(ctx, ginCtx, true)
}

func (c *executionController) prepareExecutionWithAdmission(ctx context.Context, ginCtx *gin.Context, acquireSlot bool) (*preparedExecution, error) {
	targetParam := ginCtx.Param("target")
	var req ExecuteRequest
	if err := ginCtx.ShouldBindJSON(&req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	return c.prepareExecutionForTargetWithAdmission(
		ctx,
		targetParam,
		req,
		readExecutionHeaders(ginCtx),
		middleware.GetVerifiedCallerDID(ginCtx),
		middleware.GetTargetDID(ginCtx),
		acquireSlot,
	)
}

func (c *executionController) prepareExecutionForTarget(ctx context.Context, targetParam string, req ExecuteRequest, headers executionHeaders, callerDID, targetDID string) (*preparedExecution, error) {
	return c.prepareExecutionForTargetWithAdmission(ctx, targetParam, req, headers, callerDID, targetDID, false)
}

func (c *executionController) prepareExecutionForTargetWithAdmission(ctx context.Context, targetParam string, req ExecuteRequest, headers executionHeaders, callerDID, targetDID string, acquireSlot bool) (_ *preparedExecution, retErr error) {
	target, err := parseTarget(targetParam)
	if err != nil {
		return nil, fmt.Errorf("invalid target: %w", err)
	}

	// Allow empty input for skills/reasoners that take no parameters (issue #196).
	if req.Input == nil {
		req.Input = map[string]interface{}{}
	}
	if req.RunMetadata != nil && headers.parentExecutionID == nil {
		if _, err := applyRunMetadataInput(types.RunMetadata{}, *req.RunMetadata); err != nil {
			return nil, fmt.Errorf("invalid run_metadata: %w", err)
		}
		if _, err := normalizeRunMetadataActor(pointerValue(headers.actorID)); err != nil {
			return nil, fmt.Errorf("invalid run_metadata: %w", err)
		}
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

	// Version-aware agent resolution:
	// 1. Try GetAgent (default unversioned agent, version='')
	// 2. If not found, fall back to ListAgentVersions and select via weighted round-robin
	var agent *types.AgentNode
	var routedVersion string

	agent, err = c.store.GetAgent(ctx, target.NodeID)
	if err != nil {
		// GetAgent returns error for "not found" — check if versioned agents exist
		versions, listErr := c.store.ListAgentVersions(ctx, target.NodeID)
		if listErr != nil || len(versions) == 0 {
			return nil, fmt.Errorf("agent '%s' not found", target.NodeID)
		}
		// Filter to healthy nodes
		agent, routedVersion = selectVersionedAgent(versions)
		if agent == nil {
			return nil, fmt.Errorf("agent '%s' has no healthy versioned nodes", target.NodeID)
		}
	}

	// Block calls to agents that are pending approval (e.g. tags revoked).
	// Matches the contract used by reasoners.go / skills.go / permission
	// middleware: stable machine code in `error`, friendly text in `message`.
	if agent.LifecycleStatus == types.AgentStatusPendingApproval {
		return nil, &executionPreconditionError{
			code:      http.StatusServiceUnavailable,
			message:   fmt.Sprintf("agent node '%s' is awaiting tag approval and cannot execute", target.NodeID),
			category:  ErrorCategoryAgentError,
			errorCode: "agent_pending_approval",
		}
	}

	if agent.DeploymentType == "" && agent.Metadata.Custom != nil {
		if v, ok := agent.Metadata.Custom["serverless"]; ok && fmt.Sprint(v) == "true" {
			agent.DeploymentType = "serverless"
		}
	}

	// Reject a call to a node we already know is down BEFORE the execution
	// record is created. Dispatching into a dead node would persist a row and
	// then fail it, charging the caller for a failed execution when nothing
	// was ever attempted. See execute_agent_restart.go.
	//
	// Serverless nodes are exempt: they have no heartbeat loop and the health
	// monitor never polls them, so the presence sweep marks every serverless
	// node inactive shortly after registration — their recorded health says
	// nothing about whether an invocation would succeed. Replay requests are
	// also exempt: a replay hit is served from the recorded run without ever
	// contacting the agent, so the node being down must not reject it (a
	// replay miss simply dials and fails exactly as it did before this gate).
	if agent.DeploymentType != "serverless" && strings.TrimSpace(headers.replaySourceRunID) == "" {
		if agentIsDraining(agent) {
			agent, err = c.waitForDrainingAgent(ctx, agent, target.NodeID)
			if err != nil {
				return nil, err
			}
		}
		if err := ensureAgentDispatchable(agent, target.NodeID); err != nil {
			return nil, err
		}
	}
	if agent.DeploymentType == "serverless" && (agent.InvocationURL == nil || strings.TrimSpace(*agent.InvocationURL) == "") {
		if trimmed := strings.TrimSpace(agent.BaseURL); trimmed != "" {
			execURL := strings.TrimSuffix(trimmed, "/") + "/execute"
			agent.InvocationURL = &execURL
		}
	}

	targetType, err := determineTargetType(agent, target.TargetName)
	if err != nil {
		return nil, err
	}
	target.TargetType = targetType

	llmEndpoint := extractRequestedLLMEndpoint(req)
	slotAcquired := false
	if acquireSlot {
		if err := CheckExecutionPreconditions(target.NodeID, llmEndpoint); err != nil {
			return nil, err
		}
		slotAcquired = true
		defer func() {
			if retErr != nil && slotAcquired {
				ReleaseExecutionSlot(target.NodeID)
			}
		}()
	}

	runID := headers.runID
	if runID == "" {
		runID = utils.GenerateRunID()
	}

	executionID := utils.GenerateExecutionID()
	now := time.Now().UTC()

	storedPayload, err := json.Marshal(buildClientPayload(req))
	if err != nil {
		return nil, fmt.Errorf("encode execution payload: %w", err)
	}

	exec := &types.Execution{
		ExecutionID:       executionID,
		RunID:             runID,
		ParentExecutionID: headers.parentExecutionID,
		AgentNodeID:       agent.ID,
		InstanceID:        agent.InstanceID,
		ReasonerID:        target.TargetName,
		NodeID:            target.NodeID,
		Status:            types.ExecutionStatusRunning,
		InputPayload:      json.RawMessage(storedPayload),
		StartedAt:         now,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	agentPayload := make(map[string]interface{}, len(req.Input))
	for key, value := range req.Input {
		agentPayload[key] = value
	}

	var agentPayloadBytes []byte
	if agent.DeploymentType == "serverless" {
		agentPayloadBytes, err = json.Marshal(buildServerlessPayload(target, exec, headers, agentPayload))
	} else {
		agentPayloadBytes, err = json.Marshal(agentPayload)
	}
	if err != nil {
		return nil, fmt.Errorf("encode agent payload: %w", err)
	}

	inputURI := c.savePayload(ctx, storedPayload)
	exec.InputURI = inputURI

	if headers.sessionID != nil {
		exec.SessionID = headers.sessionID
	}
	if headers.actorID != nil {
		exec.ActorID = headers.actorID
	}

	if err := c.store.CreateExecutionRecord(ctx, exec); err != nil {
		return nil, fmt.Errorf("create execution record: %w", err)
	}
	if headers.parentExecutionID == nil && req.RunMetadata != nil {
		c.persistExecuteRunMetadata(ctx, runID, *req.RunMetadata, headers.actorID)
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

	c.ensureWorkflowExecutionRecord(ctx, exec, target, storedPayload)

	hit, err := c.findReplayHit(ctx, headers, target, storedPayload)
	if err != nil {
		return nil, err
	}

	return &preparedExecution{
		exec:                    exec,
		requestBody:             agentPayloadBytes,
		agent:                   agent,
		target:                  target,
		targetType:              targetType,
		llmEndpoint:             llmEndpoint,
		webhookRegistered:       webhookRegistered,
		webhookError:            webhookError,
		callerDID:               callerDID,
		targetDID:               targetDID,
		routedVersion:           routedVersion,
		replaySourceRunID:       headers.replaySourceRunID,
		replayBeforeExecutionID: headers.replayBeforeExecutionID,
		replayMode:              headers.replayMode,
		replayHit:               hit,
	}, nil
}

// buildClientPayload builds the blob persisted as executions.input_payload,
// which canonicalReplayPayload then hashes into the replay dedupe key.
//
// Only input and context belong in it. run_metadata is deliberately excluded:
// it names the run for humans and has no bearing on what the reasoner is asked
// to compute, so two executes that differ only in run_metadata must still
// replay-match each other. Adding a field here changes the dedupe key for every
// caller and silently turns existing replay hits into misses.
func buildClientPayload(req ExecuteRequest) map[string]interface{} {
	payload := map[string]interface{}{
		"input": req.Input,
	}
	if len(req.Context) > 0 {
		payload["context"] = req.Context
	}
	return payload
}

// persistExecuteRunMetadata stores the run_metadata a root execute carried, by
// merging it into the run's "run" namespace. Best effort on purpose: an execute
// must not fail because a display name could not be recorded, so a failure is
// logged and swallowed — the same contract persistRestartRunMetadata uses for
// the lineage seed. Callers must have already checked that this is a root
// execute; only the run root establishes run identity.
func (c *executionController) persistExecuteRunMetadata(ctx context.Context, runID string, input RunMetadataInput, actorID *string) {
	writer, ok := c.store.(workflowRunMetadataWriter)
	if !ok {
		return
	}
	actor, err := normalizeRunMetadataActor(pointerValue(actorID))
	if err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", runID).Msg("failed to persist execute run metadata")
		return
	}
	if err := writer.UpdateWorkflowRunMetadata(ctx, runID, func(namespaces map[string]json.RawMessage) error {
		current := types.RunMetadata{}
		if raw := namespaces[types.RunMetadataNamespace]; raw != nil {
			_ = json.Unmarshal(raw, &current)
		}
		merged, err := applyRunMetadataInput(current, input)
		if err != nil {
			return err
		}
		merged.SetBy = actor
		merged.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
		namespaces[types.RunMetadataNamespace], err = json.Marshal(merged)
		return err
	}); err != nil {
		logger.Logger.Warn().Err(err).Str("run_id", runID).Msg("failed to persist execute run metadata")
	}
}

func pointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

// findReplayHit returns a previously-succeeded child output to reuse for the
// current app.call, or nil to run it normally. Only child executions (those with
// a parent) are eligible — the restarted root always re-runs.
//
// Matching is keyed solely on (node id, reasoner id, canonical input+context);
// among matches the earliest-started succeeded source execution wins. This is
// intentionally position- and ordering-agnostic, so two calls to the same
// reasoner with identical input+context within a run will both reuse the first
// source result. That is correct for deterministic graphs; callers that need a
// distinct result per identical call should vary the input/context or restart
// with reuse=none.
func (c *executionController) findReplayHit(ctx context.Context, headers executionHeaders, target *parsedTarget, storedPayload []byte) (*replayHit, error) {
	if target == nil || headers.parentExecutionID == nil {
		return nil, nil
	}
	sourceRunID := strings.TrimSpace(headers.replaySourceRunID)
	if sourceRunID == "" {
		return nil, nil
	}
	mode := strings.TrimSpace(headers.replayMode)
	if mode == "" {
		mode = "succeeded-before"
	}
	if mode == "none" {
		return nil, nil
	}
	if mode != "succeeded-before" && mode != "all-succeeded" {
		return nil, fmt.Errorf("unsupported replay mode %q", mode)
	}

	executions, err := c.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		RunID:          &sourceRunID,
		SortBy:         "started_at",
		SortDescending: false,
	})
	if err != nil {
		return nil, fmt.Errorf("query replay source run: %w", err)
	}
	if len(executions) == 0 {
		return nil, nil
	}

	var beforeTime *time.Time
	if mode == "succeeded-before" && strings.TrimSpace(headers.replayBeforeExecutionID) != "" {
		for _, exec := range executions {
			if exec != nil && exec.ExecutionID == headers.replayBeforeExecutionID {
				t := exec.StartedAt
				beforeTime = &t
				break
			}
		}
		if beforeTime == nil {
			return nil, nil
		}
	}

	newKey, ok := canonicalReplayPayload(storedPayload)
	if !ok {
		return nil, nil
	}
	for _, exec := range executions {
		if exec == nil {
			continue
		}
		if beforeTime != nil && !exec.StartedAt.Before(*beforeTime) {
			continue
		}
		if exec.Status != types.ExecutionStatusSucceeded {
			continue
		}
		if exec.NodeID != target.NodeID || exec.ReasonerID != target.TargetName {
			continue
		}
		if len(exec.ResultPayload) == 0 {
			continue
		}
		oldKey, oldOK := canonicalReplayPayload(exec.InputPayload)
		if !oldOK || oldKey != newKey {
			continue
		}
		return &replayHit{
			SourceExecutionID: exec.ExecutionID,
			SourceRunID:       exec.RunID,
			Result:            json.RawMessage(cloneBytes(exec.ResultPayload)),
		}, nil
	}
	return nil, nil
}

func canonicalReplayPayload(raw []byte) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var v interface{}
	if err := json.Unmarshal(raw, &v); err != nil {
		return "", false
	}
	encoded, err := json.Marshal(v)
	if err != nil {
		return "", false
	}
	return string(encoded), true
}

func (c *executionController) completeReplayHit(ctx context.Context, plan *preparedExecution) error {
	if plan == nil || plan.exec == nil || plan.replayHit == nil {
		return fmt.Errorf("missing replay execution plan")
	}
	reason := "replayed_from_execution:" + plan.replayHit.SourceExecutionID
	now := time.Now().UTC()
	duration := int64(0)
	result := cloneBytes(plan.replayHit.Result)
	resultURI := c.savePayload(ctx, result)

	updated, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
		}
		current.Status = types.ExecutionStatusSucceeded
		current.StatusReason = &reason
		current.ResultPayload = json.RawMessage(result)
		current.ResultURI = resultURI
		current.ErrorMessage = nil
		current.CompletedAt = &now
		current.DurationMS = &duration
		current.UpdatedAt = now
		return current, nil
	})
	if err != nil {
		return err
	}

	c.updateWorkflowExecutionFinalState(ctx, plan.exec.ExecutionID, types.ExecutionStatusSucceeded, result, 0, nil)
	c.updateWorkflowExecutionStatus(ctx, plan.exec.ExecutionID, types.ExecutionStatusSucceeded, &reason)
	if plan.webhookRegistered || (updated != nil && updated.WebhookRegistered) {
		c.triggerWebhook(plan.exec.ExecutionID)
	}

	eventData := map[string]interface{}{
		"target_type":       plan.targetType,
		"execution_mode":    plan.executionMode,
		"transition_source": "replay",
		"replay": map[string]interface{}{
			"source_execution_id": plan.replayHit.SourceExecutionID,
			"source_run_id":       plan.replayHit.SourceRunID,
		},
	}
	if !c.redactPayloads {
		eventData["result"] = decodeJSON(result)
		if inputPayload := decodeJSON(plan.exec.InputPayload); inputPayload != nil {
			eventData["input"] = inputPayload
		}
	}
	c.publishExecutionEventWithReasonerInfo(updated, string(types.ExecutionStatusSucceeded), eventData, plan.agent, &plan.target.TargetName)
	return nil
}

func extractRequestedLLMEndpoint(req ExecuteRequest) string {
	for _, key := range []string{"llm_endpoint", "llm_backend", "backend", "provider", "model_provider"} {
		if value, ok := req.Context[key]; ok {
			if endpoint := strings.TrimSpace(fmt.Sprint(value)); endpoint != "" {
				return endpoint
			}
		}
	}
	return ""
}
