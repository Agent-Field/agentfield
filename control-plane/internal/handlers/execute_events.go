package handlers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func (c *executionController) updateWorkflowExecutionStatus(
	ctx context.Context,
	executionID string,
	status string,
	statusReason *string,
) {
	if c.store == nil {
		return
	}

	var normalizedReason *string
	if statusReason != nil {
		trimmed := strings.TrimSpace(*statusReason)
		if trimmed != "" {
			normalizedReason = &trimmed
		}
	}

	err := c.store.UpdateWorkflowExecution(ctx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution with ID %s not found", executionID)
		}

		current.Status = status
		current.StatusReason = normalizedReason
		current.UpdatedAt = time.Now().UTC()

		if !types.IsTerminalExecutionStatus(status) {
			current.CompletedAt = nil
			if current.DurationMS != nil {
				current.DurationMS = nil
			}
		}

		return current, nil
	})
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "not found") {
			return
		}
		logger.Logger.Error().
			Err(err).
			Str("execution_id", executionID).
			Str("status", status).
			Msg("failed to update workflow execution status")
	}
}

func (c *executionController) publishExecutionEvent(exec *types.Execution, status string, data map[string]interface{}) {
	c.publishExecutionEventWithReasonerInfo(exec, status, data, nil, nil)
}

// enrichExecutionLifecycleData adds low-cardinality lifecycle dimensions used by
// observability consumers. It does not mutate execution state or include payloads.
func enrichExecutionLifecycleData(data map[string]interface{}, exec *types.Execution, status string) {
	if data == nil || exec == nil {
		return
	}

	data["is_root_execution"] = exec.ParentExecutionID == nil || strings.TrimSpace(*exec.ParentExecutionID) == ""
	if _, ok := data["workflow_depth"]; !ok {
		if data["is_root_execution"] == true {
			data["workflow_depth"] = 0
		}
	}
	if exec.DurationMS != nil {
		data["duration_ms"] = *exec.DurationMS
	}

	switch status {
	case string(types.ExecutionStatusSucceeded):
		data["outcome"] = "succeeded"
	case string(types.ExecutionStatusFailed):
		data["outcome"] = "failed"
		data["failure_category"] = canonicalFailureCategory(exec.StatusReason, "unknown")
	case string(types.ExecutionStatusCancelled):
		data["outcome"] = "cancelled"
		data["failure_category"] = "cancelled"
	case string(types.ExecutionStatusTimeout):
		data["outcome"] = "timeout"
		data["failure_category"] = "timeout"
	}
}

func canonicalFailureCategory(statusReason *string, fallback string) string {
	if statusReason == nil {
		return fallback
	}
	category := strings.TrimSpace(*statusReason)
	if separator := strings.Index(category, ":"); separator >= 0 {
		category = strings.TrimSpace(category[:separator])
	}
	switch category {
	case string(ErrorCategoryLLMUnavailable),
		string(ErrorCategoryConcurrencyLimit),
		string(ErrorCategoryAgentTimeout),
		string(ErrorCategoryAgentError),
		string(ErrorCategoryAgentUnreachable),
		string(ErrorCategoryBadResponse),
		string(ErrorCategoryInternal),
		"agent_restart_orphaned",
		"validation",
		"permission_denied",
		"node_unavailable",
		"target_not_found":
		return category
	default:
		return fallback
	}
}

func (c *executionController) publishExecutionEventWithReasonerInfo(exec *types.Execution, status string, data map[string]interface{}, agent *types.AgentNode, reasonerID *string) {
	if exec == nil {
		return
	}

	eventType := events.ExecutionUpdated
	switch status {
	case string(types.ExecutionStatusSucceeded):
		eventType = events.ExecutionCompleted
	case string(types.ExecutionStatusFailed):
		eventType = events.ExecutionFailed
	case string(types.ExecutionStatusRunning):
		eventType = events.ExecutionStarted
	case "created":
		eventType = events.ExecutionCreated
	}

	// Ensure data map exists
	if data == nil {
		data = make(map[string]interface{})
	}
	enrichExecutionLifecycleData(data, exec, status)

	// Add reasoner_id to the event data
	rID := exec.ReasonerID
	if reasonerID != nil && *reasonerID != "" {
		rID = *reasonerID
	}
	if rID != "" {
		data["reasoner_id"] = rID
	}

	// Add node_id to the event data
	if exec.NodeID != "" {
		data["node_id"] = exec.NodeID
	}
	if exec.AgentNodeID != "" {
		data["agent_node_id"] = exec.AgentNodeID
	}
	if exec.StatusReason != nil && *exec.StatusReason != "" {
		data["status_reason"] = *exec.StatusReason
		data["error_category"] = *exec.StatusReason
	}
	data["started_at"] = exec.StartedAt.UTC().Format(time.RFC3339)
	if exec.CompletedAt != nil {
		data["completed_at"] = exec.CompletedAt.UTC().Format(time.RFC3339)
	}
	if exec.DurationMS != nil {
		data["duration_ms"] = *exec.DurationMS
	}
	if exec.SessionID != nil && *exec.SessionID != "" {
		data["session_id"] = *exec.SessionID
	}
	if exec.ActorID != nil && *exec.ActorID != "" {
		data["actor_id"] = *exec.ActorID
	}
	storedPayload := types.DecodeStoredExecutionPayload(exec.InputPayload)
	if !c.redactPayloads && storedPayload.Context != nil {
		data["context"] = storedPayload.Context
	}
	if workflowExec, err := c.store.GetWorkflowExecution(context.Background(), exec.ExecutionID); err == nil && workflowExec != nil {
		data["retry_count"] = workflowExec.RetryCount
		data["workflow_depth"] = workflowExec.WorkflowDepth
	}

	// Add reasoner definitions if agent info is available
	if agent != nil {
		// Find the specific reasoner being executed
		for _, r := range agent.Reasoners {
			if r.ID == rID {
				data["reasoner"] = map[string]interface{}{
					"id":            r.ID,
					"input_schema":  r.InputSchema,
					"output_schema": r.OutputSchema,
				}
				break
			}
		}

		// Find the specific skill being executed
		for _, s := range agent.Skills {
			if s.ID == rID {
				data["skill"] = map[string]interface{}{
					"id":           s.ID,
					"input_schema": s.InputSchema,
					"tags":         s.Tags,
				}
				data["skill_id"] = s.ID
				break
			}
		}

		// Include all reasoners on this agent node for back-population
		if len(agent.Reasoners) > 0 {
			reasonerList := make([]map[string]interface{}, 0, len(agent.Reasoners))
			for _, r := range agent.Reasoners {
				reasonerList = append(reasonerList, map[string]interface{}{
					"id":            r.ID,
					"input_schema":  r.InputSchema,
					"output_schema": r.OutputSchema,
				})
			}
			data["agent_reasoners"] = reasonerList
		}

		// Include all skills on this agent node for back-population
		if len(agent.Skills) > 0 {
			skillList := make([]map[string]interface{}, 0, len(agent.Skills))
			for _, s := range agent.Skills {
				skillList = append(skillList, map[string]interface{}{
					"id":           s.ID,
					"input_schema": s.InputSchema,
					"tags":         s.Tags,
				})
			}
			data["agent_skills"] = skillList
		}

		// Include agent node info
		data["agent_node"] = map[string]interface{}{
			"id":              agent.ID,
			"base_url":        agent.BaseURL,
			"version":         agent.Version,
			"deployment_type": agent.DeploymentType,
		}
	}

	event := events.ExecutionEvent{
		Type:        eventType,
		ExecutionID: exec.ExecutionID,
		WorkflowID:  exec.RunID,
		AgentNodeID: exec.AgentNodeID,
		Status:      status,
		Timestamp:   time.Now(),
		Data:        data,
	}
	if c.eventBus != nil {
		c.eventBus.Publish(event)
	}
	events.GlobalExecutionEventBus.Publish(event)
}

// publishExecutionStartedEvent emits the ExecutionStarted event with full reasoner context
func (c *executionController) publishExecutionStartedEvent(plan *preparedExecution) {
	if plan == nil || plan.exec == nil {
		return
	}

	data := map[string]interface{}{
		"target_type":       plan.targetType,
		"execution_mode":    plan.executionMode,
		"transition_source": "execution_controller",
	}

	// Include input payload info (not the full payload, just metadata)
	if len(plan.exec.InputPayload) > 0 {
		data["input_size"] = len(plan.exec.InputPayload)
	}

	c.publishExecutionEventWithReasonerInfo(
		plan.exec,
		string(types.ExecutionStatusRunning),
		data,
		plan.agent,
		&plan.target.TargetName,
	)
}
