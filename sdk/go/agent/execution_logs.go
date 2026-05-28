package agent

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	executionLogVersion       = 1
	executionLogSourceUser    = "sdk.user"
	executionLogSourceRuntime = "sdk.runtime"
)

var executionLogStdoutMu sync.Mutex

// ExecutionLogEntry is the canonical structured execution log envelope.
type ExecutionLogEntry struct {
	V                 int            `json:"v"`
	TS                string         `json:"ts"`
	ExecutionID       string         `json:"execution_id"`
	WorkflowID        string         `json:"workflow_id"`
	RunID             string         `json:"run_id"`
	RootWorkflowID    string         `json:"root_workflow_id"`
	ParentExecutionID string         `json:"parent_execution_id"`
	AgentNodeID       string         `json:"agent_node_id"`
	ReasonerID        string         `json:"reasoner_id"`
	Level             string         `json:"level"`
	Source            string         `json:"source"`
	EventType         string         `json:"event_type"`
	Message           string         `json:"message"`
	Attributes        map[string]any `json:"attributes"`
	SystemGenerated   bool           `json:"system_generated"`
	SessionID         string         `json:"session_id,omitempty"`
	ActorID           string         `json:"actor_id,omitempty"`
	ParentWorkflowID  string         `json:"parent_workflow_id,omitempty"`
	Depth             int            `json:"depth,omitempty"`
	CallerDID         string         `json:"caller_did,omitempty"`
	TargetDID         string         `json:"target_did,omitempty"`
	AgentNodeDID      string         `json:"agent_node_did,omitempty"`
}

// ExecutionLogger emits structured execution logs and mirrors them to stdout.
type ExecutionLogger struct {
	agent   *Agent
	execCtx ExecutionContext
	source  string
}

// ExecutionLogger returns a context-aware logger for the provided execution context.
// Logs are mirrored to stdout and, when an execution ID is present, forwarded to
// the control plane ingestion API.
func (a *Agent) ExecutionLogger(ctx context.Context) *ExecutionLogger {
	return a.executionLogger(ctx, executionLogSourceUser)
}

func (a *Agent) executionLogger(ctx context.Context, source string) *ExecutionLogger {
	if a == nil {
		return &ExecutionLogger{source: source}
	}
	execCtx := a.normalizeExecutionContext(executionContextFrom(ctx))
	return &ExecutionLogger{
		agent:   a,
		execCtx: execCtx,
		source:  strings.TrimSpace(source),
	}
}

// WithSource returns a copy of the logger with the provided source label.
func (l *ExecutionLogger) WithSource(source string) *ExecutionLogger {
	if l == nil {
		return nil
	}
	clone := *l
	clone.source = strings.TrimSpace(source)
	return &clone
}

// Debug emits a debug-level structured log entry.
func (l *ExecutionLogger) Debug(eventType, message string, attributes map[string]any) {
	l.Emit("debug", eventType, message, attributes, false)
}

// Info emits an info-level structured log entry.
func (l *ExecutionLogger) Info(eventType, message string, attributes map[string]any) {
	l.Emit("info", eventType, message, attributes, false)
}

// Warn emits a warning-level structured log entry.
func (l *ExecutionLogger) Warn(eventType, message string, attributes map[string]any) {
	l.Emit("warn", eventType, message, attributes, false)
}

// Error emits an error-level structured log entry.
func (l *ExecutionLogger) Error(eventType, message string, attributes map[string]any) {
	l.Emit("error", eventType, message, attributes, false)
}

// System emits a system-generated info-level structured log entry.
func (l *ExecutionLogger) System(eventType, message string, attributes map[string]any) {
	l.Emit("info", eventType, message, attributes, true)
}

// Emit writes a structured execution log entry.
func (l *ExecutionLogger) Emit(level, eventType, message string, attributes map[string]any, systemGenerated bool) {
	if l == nil {
		return
	}

	entry := l.entry(level, eventType, message, attributes, systemGenerated)
	writeStructuredExecutionLog(entry)
	l.dispatch(entry)
}

func (l *ExecutionLogger) entry(level, eventType, message string, attributes map[string]any, systemGenerated bool) ExecutionLogEntry {
	if attributes == nil {
		attributes = map[string]any{}
	}

	level = strings.ToLower(strings.TrimSpace(level))
	if level == "" {
		level = "info"
	}
	eventType = strings.TrimSpace(eventType)
	if eventType == "" {
		eventType = "log"
	}
	message = strings.TrimSpace(message)
	if message == "" {
		message = eventType
	}

	execCtx := l.execCtx
	rootWorkflowID := execCtx.RootWorkflowID
	if rootWorkflowID == "" {
		rootWorkflowID = execCtx.WorkflowID
	}
	if rootWorkflowID == "" {
		rootWorkflowID = execCtx.RunID
	}

	return ExecutionLogEntry{
		V:                 executionLogVersion,
		TS:                time.Now().UTC().Format(time.RFC3339Nano),
		ExecutionID:       execCtx.ExecutionID,
		WorkflowID:        execCtx.WorkflowID,
		RunID:             execCtx.RunID,
		RootWorkflowID:    rootWorkflowID,
		ParentExecutionID: execCtx.ParentExecutionID,
		AgentNodeID:       execCtx.AgentNodeID,
		ReasonerID:        execCtx.ReasonerName,
		Level:             level,
		Source:            l.source,
		EventType:         eventType,
		Message:           message,
		Attributes:        attributes,
		SystemGenerated:   systemGenerated,
		SessionID:         execCtx.SessionID,
		ActorID:           execCtx.ActorID,
		ParentWorkflowID:  execCtx.ParentWorkflowID,
		Depth:             execCtx.Depth,
		CallerDID:         execCtx.CallerDID,
		TargetDID:         execCtx.TargetDID,
		AgentNodeDID:      execCtx.AgentNodeDID,
	}
}

func (l *ExecutionLogger) dispatch(entry ExecutionLogEntry) {
	if l == nil || l.agent == nil || l.agent.client == nil || strings.TrimSpace(entry.ExecutionID) == "" {
		return
	}

	go func(entry ExecutionLogEntry) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := l.agent.client.PostExecutionLogs(ctx, entry.ExecutionID, entry); err != nil {
			l.agent.logger.Printf("warn: execution log send failed: %v", err)
		}
	}(entry)
}

func writeStructuredExecutionLog(entry ExecutionLogEntry) {
	executionLogStdoutMu.Lock()
	defer executionLogStdoutMu.Unlock()

	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(entry)
}

func (a *Agent) logExecution(ctx context.Context, level, eventType, message string, attributes map[string]any, systemGenerated bool) {
	if a == nil {
		return
	}
	a.executionLogger(ctx, executionLogSourceRuntime).Emit(level, eventType, message, attributes, systemGenerated)
}

func (a *Agent) logExecutionInfo(ctx context.Context, eventType, message string, attributes map[string]any) {
	a.logExecution(ctx, "info", eventType, message, attributes, true)
}

func (a *Agent) logExecutionWarn(ctx context.Context, eventType, message string, attributes map[string]any) {
	a.logExecution(ctx, "warn", eventType, message, attributes, true)
}

func (a *Agent) logExecutionError(ctx context.Context, eventType, message string, attributes map[string]any) {
	a.logExecution(ctx, "error", eventType, message, attributes, true)
}

func (a *Agent) normalizeExecutionContext(exec ExecutionContext) ExecutionContext {
	if exec.WorkflowID == "" && exec.RunID != "" {
		exec.WorkflowID = exec.RunID
	}
	if exec.RootWorkflowID == "" {
		switch {
		case exec.WorkflowID != "":
			exec.RootWorkflowID = exec.WorkflowID
		case exec.RunID != "":
			exec.RootWorkflowID = exec.RunID
		}
	}
	if exec.AgentNodeID == "" && a != nil {
		exec.AgentNodeID = a.cfg.NodeID
	}
	return exec
}
