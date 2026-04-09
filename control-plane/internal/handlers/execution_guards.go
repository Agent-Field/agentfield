package handlers

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
)

// Package-level reference to LLM health monitor for execution guards.
// Set during server startup via SetLLMHealthMonitor.
var (
	llmHealthMonitor     *services.LLMHealthMonitor
	llmHealthMonitorOnce sync.Once
	llmHealthMonitorMu   sync.RWMutex
)

// SetLLMHealthMonitor registers the LLM health monitor for execution guards.
func SetLLMHealthMonitor(monitor *services.LLMHealthMonitor) {
	llmHealthMonitorMu.Lock()
	defer llmHealthMonitorMu.Unlock()
	llmHealthMonitor = monitor
}

// GetLLMHealthMonitor returns the registered LLM health monitor, or nil.
func GetLLMHealthMonitor() *services.LLMHealthMonitor {
	llmHealthMonitorMu.RLock()
	defer llmHealthMonitorMu.RUnlock()
	return llmHealthMonitor
}

// CheckExecutionPreconditions verifies that shared execution preconditions are met:
// - LLM backend is reachable (circuit breaker is not open)
// - Agent has not exceeded its concurrent execution limit
//
// Returns nil if all checks pass, or an error describing the blocking condition.
func CheckExecutionPreconditions(agentNodeID, llmEndpoint string) error {
	// Check LLM health
	monitor := GetLLMHealthMonitor()
	if monitor != nil {
		if err := checkLLMEndpointHealth(monitor, llmEndpoint); err != nil {
			return err
		}
	}

	// Check per-agent concurrency limit
	limiter := GetConcurrencyLimiter()
	if err := limiter.Acquire(agentNodeID); err != nil {
		return &executionPreconditionError{
			code:     429,
			message:  err.Error(),
			category: ErrorCategoryConcurrencyLimit,
		}
	}

	return nil
}

func checkLLMEndpointHealth(monitor *services.LLMHealthMonitor, llmEndpoint string) error {
	if monitor == nil || !monitor.Enabled() || monitor.EndpointCount() == 0 {
		return nil
	}

	llmEndpoint = strings.TrimSpace(llmEndpoint)
	if llmEndpoint != "" {
		if status, ok := monitor.GetStatus(llmEndpoint); ok {
			if status.CircuitState != services.CircuitOpen {
				return nil
			}
			return newLLMUnavailableError(fmt.Sprintf("LLM backend %q unavailable", status.Name), status.LastError)
		}
	}

	statuses := monitor.GetAllStatuses()
	if monitor.EndpointCount() == 1 {
		if unavailable := firstUnavailableEndpoint(statuses); unavailable != nil {
			return newLLMUnavailableError(fmt.Sprintf("LLM backend %q unavailable", unavailable.Name), unavailable.LastError)
		}
		return nil
	}

	if unavailable := firstUnavailableEndpoint(statuses); unavailable != nil {
		return newLLMUnavailableError(
			fmt.Sprintf("LLM backend health is degraded and request backend could not be determined (endpoint %q unavailable)", unavailable.Name),
			unavailable.LastError,
		)
	}

	return nil
}

func firstUnavailableEndpoint(statuses []services.LLMEndpointStatus) *services.LLMEndpointStatus {
	for i := range statuses {
		if statuses[i].CircuitState == services.CircuitOpen {
			return &statuses[i]
		}
	}
	return nil
}

func newLLMUnavailableError(message, lastErr string) error {
	if strings.TrimSpace(lastErr) != "" {
		message += ": " + lastErr
	}
	return &executionPreconditionError{
		code:     503,
		message:  message,
		category: ErrorCategoryLLMUnavailable,
	}
}

// ReleaseExecutionSlot releases the concurrency slot for the given agent.
// Safe to call even if concurrency limiting is disabled.
func ReleaseExecutionSlot(agentNodeID string) {
	limiter := GetConcurrencyLimiter()
	limiter.Release(agentNodeID)
}

// ErrorCategory classifies execution failures for user-facing diagnostics.
type ErrorCategory string

const (
	ErrorCategoryLLMUnavailable   ErrorCategory = "llm_unavailable"
	ErrorCategoryConcurrencyLimit ErrorCategory = "concurrency_limit"
	ErrorCategoryAgentTimeout     ErrorCategory = "agent_timeout"
	ErrorCategoryAgentError       ErrorCategory = "agent_error"
	ErrorCategoryAgentUnreachable ErrorCategory = "agent_unreachable"
	ErrorCategoryBadResponse      ErrorCategory = "bad_response"
	ErrorCategoryInternal         ErrorCategory = "internal_error"
)

// executionPreconditionError carries both an HTTP status code and message.
//
// When errorCode is non-empty, the JSON renderer will emit the stable machine
// code in the top-level "error" field and move the human-readable message to
// "message" — matching the contract already used by sibling handlers
// (reasoners, skills, permission middleware) for conditions like
// agent_pending_approval.
type executionPreconditionError struct {
	code      int
	message   string
	category  ErrorCategory
	errorCode string
}

func (e *executionPreconditionError) Error() string {
	return e.message
}

// HTTPStatusCode returns the appropriate HTTP status code for this error.
func (e *executionPreconditionError) HTTPStatusCode() int {
	return e.code
}

// Category returns the error classification.
func (e *executionPreconditionError) Category() ErrorCategory {
	return e.category
}

// ErrorCode returns the stable machine-readable error code (if set).
// Clients should key on this rather than the human-readable message.
func (e *executionPreconditionError) ErrorCode() string {
	return e.errorCode
}

// PublishExecutionLog publishes a structured log event for an execution.
// These events are streamed to SSE clients watching the execution.
func PublishExecutionLog(executionID, workflowID, agentNodeID, level, message string, metadata map[string]interface{}) {
	data := map[string]interface{}{
		"level":   level,
		"message": message,
	}
	if metadata != nil {
		data["metadata"] = metadata
	}

	events.GlobalExecutionEventBus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionLogEntry,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      level,
		Timestamp:   time.Now(),
		Data:        data,
	})
}
