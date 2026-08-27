package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// completionPollInterval is how often waitForExecutionCompletion re-reads the
// execution record while it waits on the event bus. Var, not const, so tests
// can shrink it.
var completionPollInterval = 500 * time.Millisecond

// waitForExecutionCompletion waits for an execution to complete by subscribing to the event bus.
// It returns the completed execution record or an error if the execution fails or times out.
// This is used when agents return HTTP 202 (async acknowledgment) but the sync endpoint needs to wait for completion.
//
// The event bus is the fast path, but not every writer of a terminal status
// publishes a lifecycle event: the SDK's reasoner.completed workflow event
// persists the terminal state through WorkflowExecutionEventHandler without
// one, and when the authoritative /status callback lands afterwards it is an
// idempotent terminal->terminal update. If that ordering wins the race, the
// only signal a synchronous caller would ever get was the timeout, ninety
// seconds after its result had been stored. The store is therefore polled as
// a fallback, so the wait is bounded by completionPollInterval rather than by
// whichever callback happened to arrive first.
func (c *executionController) waitForExecutionCompletion(ctx context.Context, executionID string, timeout time.Duration) (*types.Execution, error) {
	if c.eventBus == nil {
		return nil, fmt.Errorf("event bus not available")
	}

	// Create unique subscriber ID for this wait operation
	subscriberID := fmt.Sprintf("sync-wait-%s", executionID)

	// Subscribe to events
	eventChan := c.eventBus.Subscribe(subscriberID)
	defer c.eventBus.Unsubscribe(subscriberID)

	// Create timeout timer
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	// Fallback for terminal states that reach the store without an event.
	poll := time.NewTicker(completionPollInterval)
	defer poll.Stop()

	logger.Logger.Debug().
		Str("execution_id", executionID).
		Dur("timeout", timeout).
		Msg("waiting for execution completion via event bus")

	// Check if execution already completed before we subscribed (race condition:
	// fast agents may POST the callback before we subscribe to the event bus).
	if existing, err := c.store.GetExecutionRecord(ctx, executionID); err == nil && existing != nil {
		if types.IsTerminalExecutionStatus(existing.Status) {
			logger.Logger.Debug().
				Str("execution_id", executionID).
				Str("status", existing.Status).
				Msg("execution already completed before event subscription")
			return existing, nil
		}
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		case <-timer.C:
			logger.Logger.Warn().
				Str("execution_id", executionID).
				Dur("timeout", timeout).
				Msg("execution completion timeout")
			return nil, fmt.Errorf("execution timeout after %v", timeout)

		case <-poll.C:
			existing, err := c.store.GetExecutionRecord(ctx, executionID)
			if err != nil || existing == nil || !types.IsTerminalExecutionStatus(existing.Status) {
				continue
			}
			logger.Logger.Debug().
				Str("execution_id", executionID).
				Str("status", existing.Status).
				Msg("execution reached a terminal state without a terminal event; completing from the stored record")
			return existing, nil

		case event := <-eventChan:
			// Only process events for this specific execution
			if event.ExecutionID != executionID {
				continue
			}

			// Check if this is a terminal event
			if event.Type == events.ExecutionCompleted || event.Type == events.ExecutionFailed {
				logger.Logger.Debug().
					Str("execution_id", executionID).
					Str("event_type", string(event.Type)).
					Msg("received terminal execution event")

				// Fetch the updated execution record
				exec, err := c.store.GetExecutionRecord(ctx, executionID)
				if err != nil {
					return nil, fmt.Errorf("failed to fetch execution after completion: %w", err)
				}
				if exec == nil {
					return nil, fmt.Errorf("execution %s not found after completion event", executionID)
				}

				return exec, nil
			}

			// Continue waiting for other event types (ExecutionUpdated, etc.)
		}
	}
}

// waitForResume waits for a paused execution to be resumed or cancelled.
// It returns nil when resumed and an error when cancelled or context is cancelled.
func (c *executionController) waitForResume(ctx context.Context, executionID string) error {
	if c.eventBus == nil {
		return fmt.Errorf("event bus not available")
	}

	// Create unique subscriber ID for this wait operation. Include a monotonic
	// counter so that multiple goroutines waiting on the same execution (e.g.
	// parallel DAG branches) each get their own event channel.
	subscriberID := fmt.Sprintf("pause-wait-%s-%d", executionID, time.Now().UnixNano())

	// Subscribe to events.
	eventChan := c.eventBus.Subscribe(subscriberID)
	defer c.eventBus.Unsubscribe(subscriberID)

	logger.Logger.Debug().
		Str("execution_id", executionID).
		Msg("waiting for execution resume via event bus")

	// Check if execution already resumed/cancelled before we subscribed (race condition:
	// fast status transitions may happen before we subscribe to the event bus).
	if existing, err := c.store.GetExecutionRecord(ctx, executionID); err == nil && existing != nil {
		if existing.Status == types.ExecutionStatusCancelled {
			return fmt.Errorf("execution cancelled")
		}
		if existing.Status != types.ExecutionStatusPaused {
			logger.Logger.Debug().
				Str("execution_id", executionID).
				Str("status", existing.Status).
				Msg("execution already resumed before event subscription")
			return nil
		}
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case event := <-eventChan:
			// Only process events for this specific execution.
			if event.ExecutionID != executionID {
				continue
			}

			switch event.Type {
			case events.ExecutionResumed:
				logger.Logger.Debug().
					Str("execution_id", executionID).
					Msg("received execution resumed event")
				return nil
			case events.ExecutionCancelledEvent:
				return fmt.Errorf("execution cancelled")
			}

			// Continue waiting for other event types (ExecutionUpdated, etc.)
		}
	}
}

type preparedExecution struct {
	exec              *types.Execution
	requestBody       []byte
	agent             *types.AgentNode
	target            *parsedTarget
	targetType        string
	executionMode     string
	llmEndpoint       string
	webhookRegistered bool
	webhookError      *string
	// DID context forwarded to the target agent.
	callerDID string
	targetDID string
	// Version that was selected during routing (empty if default/unversioned agent)
	routedVersion           string
	replaySourceRunID       string
	replayBeforeExecutionID string
	replayMode              string
	replayHit               *replayHit
}

func (c *executionController) callAgent(ctx context.Context, plan *preparedExecution) ([]byte, time.Duration, bool, error) {
	start := time.Now()

	if plan.target != nil && plan.exec != nil {
		PublishExecutionLog(plan.exec.ExecutionID, plan.exec.RunID, plan.target.NodeID,
			"info", "calling agent", map[string]interface{}{
				"agent":    plan.target.NodeID,
				"reasoner": plan.target.TargetName,
				"base_url": plan.agent.BaseURL,
			})
	}

	// Check execution state before calling agent.
	currentExec, err := c.store.GetExecutionRecord(ctx, plan.exec.ExecutionID)
	if err == nil && currentExec != nil {
		if currentExec.Status == types.ExecutionStatusCancelled {
			return nil, 0, false, fmt.Errorf("execution cancelled")
		}
		if currentExec.Status == types.ExecutionStatusPaused {
			if err := c.waitForResume(ctx, plan.exec.ExecutionID); err != nil {
				return nil, 0, false, fmt.Errorf("execution paused and then cancelled or timed out: %w", err)
			}
		}
	}

	resp, err := c.dispatchAgentRequest(ctx, plan)
	if err != nil {
		return nil, time.Since(start), false, fmt.Errorf("agent call failed: %w", err)
	}
	defer resp.Body.Close()

	url := buildAgentURL(plan.agent, plan.target)
	if resp.StatusCode == http.StatusAccepted {
		logger.Logger.Info().
			Str("execution_id", plan.exec.ExecutionID).
			Str("agent", plan.target.NodeID).
			Str("reasoner", plan.target.TargetName).
			Msg("agent acknowledged async execution")
		return nil, time.Since(start), true, nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, time.Since(start), false, fmt.Errorf("read agent response: %w", err)
	}

	if plan.agent.DeploymentType == "serverless" {
		annotateBodyForLog(
			logger.Logger.Debug().
				Str("agent", plan.target.NodeID).
				Str("reasoner", plan.target.TargetName).
				Str("url", url).
				Int("status", resp.StatusCode),
			resp.Header.Get("Content-Type"),
			body,
			c.redactPayloads,
		).Msg("serverless response")
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return body, time.Since(start), false, &callError{
			statusCode: resp.StatusCode,
			message:    fmt.Sprintf("agent error (%d): %s", resp.StatusCode, truncateForLog(body)),
			body:       body,
		}
	}

	return body, time.Since(start), false, nil
}

func (c *executionController) completeExecution(ctx context.Context, plan *preparedExecution, result []byte, elapsed time.Duration) error {
	if plan.target != nil && plan.exec != nil {
		PublishExecutionLog(plan.exec.ExecutionID, plan.exec.RunID, plan.target.NodeID,
			"info", "execution completed", map[string]interface{}{
				"duration_ms": elapsed.Milliseconds(),
			})
	}

	resultURI := c.savePayload(ctx, result)

	var lastErr error
	var alreadyCancelled bool
	for attempt := 0; attempt < 5; attempt++ {
		updated, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
			}
			// Guard: don't overwrite if already cancelled (e.g. by approval rejection webhook)
			// or waiting for approval — the approval webhook handler manages the transition.
			if current.Status == types.ExecutionStatusCancelled || current.Status == types.ExecutionStatusWaiting {
				logger.Logger.Info().
					Str("execution_id", plan.exec.ExecutionID).
					Str("current_status", string(current.Status)).
					Msg("skipping completion update; execution already cancelled or waiting for approval")
				alreadyCancelled = true
				return current, nil
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
			if alreadyCancelled {
				return nil
			}
			c.updateWorkflowExecutionFinalState(
				ctx,
				plan.exec.ExecutionID,
				types.ExecutionStatusSucceeded,
				result,
				elapsed,
				nil,
			)
			if plan.webhookRegistered || (updated != nil && updated.WebhookRegistered) {
				c.triggerWebhook(plan.exec.ExecutionID)
			}
			eventData := map[string]interface{}{
				"target_type":       plan.targetType,
				"execution_mode":    plan.executionMode,
				"transition_source": "execution_controller",
			}
			if !c.redactPayloads {
				if payload := decodeJSON(result); payload != nil {
					eventData["result"] = payload
				}
				if inputPayload := decodeJSON(plan.exec.InputPayload); inputPayload != nil {
					eventData["input"] = inputPayload
				}
			}
			c.publishExecutionEventWithReasonerInfo(updated, string(types.ExecutionStatusSucceeded), eventData, plan.agent, &plan.target.TargetName)
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
	// Classify the error for user-facing diagnostics
	category := classifyExecutionError(callErr)

	if plan.target != nil && plan.exec != nil {
		PublishExecutionLog(plan.exec.ExecutionID, plan.exec.RunID, plan.target.NodeID,
			"error", "execution failed", map[string]interface{}{
				"error":          callErr.Error(),
				"error_category": string(category),
				"duration_ms":    elapsed.Milliseconds(),
			})
	}

	errMsg := callErr.Error()
	resultURI := c.savePayload(ctx, result)
	var lastErr error
	var alreadyCancelled bool
	for attempt := 0; attempt < 5; attempt++ {
		updated, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
			}
			// Guard: don't overwrite if already cancelled (e.g. by approval rejection webhook)
			// or waiting for approval — the approval webhook handler manages the transition.
			if current.Status == types.ExecutionStatusCancelled || current.Status == types.ExecutionStatusWaiting {
				logger.Logger.Info().
					Str("execution_id", plan.exec.ExecutionID).
					Str("current_status", string(current.Status)).
					Msg("skipping failure update; execution already cancelled or waiting for approval")
				alreadyCancelled = true
				return current, nil
			}
			now := time.Now().UTC()
			current.Status = types.ExecutionStatusFailed
			current.ErrorMessage = &errMsg
			categoryStr := string(category)
			current.StatusReason = &categoryStr
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
			if alreadyCancelled {
				return nil
			}
			c.updateWorkflowExecutionFinalState(
				ctx,
				plan.exec.ExecutionID,
				types.ExecutionStatusFailed,
				result,
				elapsed,
				&errMsg,
			)
			if plan.webhookRegistered || (updated != nil && updated.WebhookRegistered) {
				c.triggerWebhook(plan.exec.ExecutionID)
			}
			eventData := map[string]interface{}{
				"error":             errMsg,
				"target_type":       plan.targetType,
				"execution_mode":    plan.executionMode,
				"failure_category":  string(category),
				"transition_source": "execution_controller",
			}
			if !c.redactPayloads {
				if payload := decodeJSON(result); payload != nil {
					eventData["result"] = payload
				}
				if inputPayload := decodeJSON(plan.exec.InputPayload); inputPayload != nil {
					eventData["input"] = inputPayload
				}
			}
			c.publishExecutionEventWithReasonerInfo(updated, string(types.ExecutionStatusFailed), eventData, plan.agent, &plan.target.TargetName)
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
	runID                   string
	parentExecutionID       *string
	sessionID               *string
	actorID                 *string
	replaySourceRunID       string
	replayBeforeExecutionID string
	replayMode              string
}
