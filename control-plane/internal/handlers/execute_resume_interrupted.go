package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

const (
	defaultResumeInterruptedMaxAttempts = 1
	defaultResumeInterruptedWindow      = time.Hour
	defaultResumeInterruptedLimit       = 25
	defaultResumeInterruptedDelay       = 15 * time.Second
	interruptedHandoffWorkTimeout       = 30 * time.Second
)

var (
	resumeInterruptedRuns        atomic.Bool
	resumeInterruptedMaxAttempts atomic.Int64
	resumeInterruptedWindowNanos atomic.Int64
	resumeInterruptedLimit       atomic.Int64
	resumeInterruptedDelayNanos  atomic.Int64
	interruptedHandoffClaims     sync.Map
)

type InterruptedRunResumeDependencies struct {
	Payloads      services.PayloadStore
	Webhooks      services.WebhookDispatcher
	Timeout       time.Duration
	InternalToken string
}

func init() {
	resumeInterruptedMaxAttempts.Store(defaultResumeInterruptedMaxAttempts)
	resumeInterruptedWindowNanos.Store(int64(defaultResumeInterruptedWindow))
	resumeInterruptedLimit.Store(defaultResumeInterruptedLimit)
	resumeInterruptedDelayNanos.Store(int64(defaultResumeInterruptedDelay))
}

func SetResumeInterruptedRuns(enabled bool) { resumeInterruptedRuns.Store(enabled) }
func ResumeInterruptedRuns() bool           { return resumeInterruptedRuns.Load() }
func SetResumeInterruptedMaxAttempts(value int) {
	resumeInterruptedMaxAttempts.Store(int64(value))
}
func ResumeInterruptedMaxAttempts() int { return int(resumeInterruptedMaxAttempts.Load()) }
func SetResumeInterruptedWindow(value time.Duration) {
	resumeInterruptedWindowNanos.Store(int64(value))
}
func ResumeInterruptedWindow() time.Duration {
	return time.Duration(resumeInterruptedWindowNanos.Load())
}
func SetResumeInterruptedLimit(value int) { resumeInterruptedLimit.Store(int64(value)) }
func ResumeInterruptedLimit() int         { return int(resumeInterruptedLimit.Load()) }
func SetResumeInterruptedDelay(value time.Duration) {
	resumeInterruptedDelayNanos.Store(int64(value))
}
func ResumeInterruptedDelay() time.Duration {
	return time.Duration(resumeInterruptedDelayNanos.Load())
}

func isInterruptedStatusReason(reason string) bool {
	reason = strings.TrimSpace(reason)
	return strings.HasPrefix(reason, "agent_restart_orphaned") ||
		reason == string(ErrorCategoryControlPlaneShutdown) ||
		reason == types.ExecutionReasonAgentShutdownCancelled
}

func interruptedRunHandoffReason(interrupted *types.Execution) (string, bool) {
	if !ResumeInterruptedRuns() || interrupted == nil {
		return "", false
	}
	if interrupted.Status != types.ExecutionStatusFailed && interrupted.Status != types.ExecutionStatusCancelled {
		return "", false
	}
	reason := ""
	if interrupted.StatusReason != nil {
		reason = strings.TrimSpace(*interrupted.StatusReason)
	}
	if !isInterruptedStatusReason(reason) {
		return "", false
	}
	if interrupted.ParentExecutionID != nil && strings.TrimSpace(*interrupted.ParentExecutionID) != "" {
		return "", false
	}
	if interrupted.RestartedAsExecutionID != nil && strings.TrimSpace(*interrupted.RestartedAsExecutionID) != "" {
		return "", false
	}
	return reason, true
}

func (c *executionController) handOffInterruptedRun(ctx context.Context, interrupted *types.Execution) (string, bool) {
	if _, ok := interruptedRunHandoffReason(interrupted); !ok {
		return "", false
	}
	if _, claimed := interruptedHandoffClaims.LoadOrStore(interrupted.ExecutionID, struct{}{}); claimed {
		return "", false
	}
	defer interruptedHandoffClaims.Delete(interrupted.ExecutionID)
	return c.handOffClaimedInterruptedRun(ctx, interrupted.ExecutionID)
}

// scheduleInterruptedRunHandoff owns the in-process claim from scheduling
// through the delayed handoff. Its context is intentionally detached from the
// request or reap that found the candidate: both may end before an agent's
// replacement instance is ready to receive the successor.
func (c *executionController) scheduleInterruptedRunHandoff(interrupted *types.Execution) {
	if _, ok := interruptedRunHandoffReason(interrupted); !ok {
		return
	}
	executionID := interrupted.ExecutionID
	if _, claimed := interruptedHandoffClaims.LoadOrStore(executionID, struct{}{}); claimed {
		return
	}

	delay := ResumeInterruptedDelay()
	if delay < 0 {
		delay = 0
	}
	go func() {
		defer interruptedHandoffClaims.Delete(executionID)

		workTimeout := c.timeout
		if workTimeout < interruptedHandoffWorkTimeout {
			workTimeout = interruptedHandoffWorkTimeout
		}
		handoffCtx, cancel := context.WithTimeout(context.Background(), delay+workTimeout+interruptedHandoffWorkTimeout)
		defer cancel()

		if delay > 0 {
			timer := time.NewTimer(delay)
			defer timer.Stop()
			select {
			case <-handoffCtx.Done():
				return
			case <-timer.C:
			}
		}
		c.handOffClaimedInterruptedRun(handoffCtx, executionID)
	}()
}

// handOffClaimedInterruptedRun must only be called while executionID is held
// in interruptedHandoffClaims. It deliberately loads the execution at fire
// time and applies every guard to that fresh record: a late completion, manual
// restart, feature toggle, or competing trigger during the delay must win.
func (c *executionController) handOffClaimedInterruptedRun(ctx context.Context, executionID string) (string, bool) {
	interrupted, err := c.store.GetExecutionRecord(ctx, executionID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to refresh interrupted execution before handoff")
		return "", false
	}
	reason, ok := interruptedRunHandoffReason(interrupted)
	if !ok {
		return "", false
	}

	attempt := c.resumeAttemptForRun(ctx, interrupted.RunID)
	if attempt >= ResumeInterruptedMaxAttempts() {
		return "", false
	}
	response, err := c.startRestart(ctx, interrupted, restartOptions{
		Scope:         "workflow",
		Reuse:         "all-succeeded",
		Kind:          "resume",
		Reason:        fmt.Sprintf("auto-resume after %s", reason),
		ResumeAttempt: attempt + 1,
	})
	if err != nil {
		logger.Logger.Warn().Err(err).
			Str("execution_id", interrupted.ExecutionID).
			Str("run_id", interrupted.RunID).
			Str("reason", reason).
			Msg("failed to hand off interrupted run")
		return "", false
	}
	interrupted.RestartedAsExecutionID = &response.ExecutionID
	logger.Logger.Info().
		Str("execution_id", interrupted.ExecutionID).
		Str("run_id", interrupted.RunID).
		Str("restarted_as", response.ExecutionID).
		Str("reason", reason).
		Msg("handed off interrupted run to restart")
	return response.ExecutionID, true
}

func (c *executionController) resumeAttemptForRun(ctx context.Context, runID string) int {
	reader, ok := c.store.(workflowRunMetadataReader)
	if !ok || strings.TrimSpace(runID) == "" {
		return 0
	}
	run, err := reader.GetWorkflowRun(ctx, runID)
	if err != nil || run == nil || len(run.Metadata) == 0 {
		return 0
	}
	var namespaces map[string]json.RawMessage
	if json.Unmarshal(run.Metadata, &namespaces) != nil {
		return 0
	}
	var lineage struct {
		Kind          string `json:"kind"`
		ResumeAttempt int    `json:"resume_attempt"`
	}
	if json.Unmarshal(namespaces["lineage"], &lineage) != nil || lineage.Kind != "resume" {
		return 0
	}
	return lineage.ResumeAttempt
}

// ResumeInterruptedRunsOnStartup resumes bounded, recent interrupted roots.
// It deliberately returns before touching storage when the feature is off.
func ResumeInterruptedRunsOnStartup(ctx context.Context, store ExecutionStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken string) {
	if !ResumeInterruptedRuns() {
		return
	}
	if delay := ResumeInterruptedDelay(); delay > 0 {
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
	}
	limit := ResumeInterruptedLimit()
	if limit <= 0 {
		return
	}
	cutoff := time.Now().UTC().Add(-ResumeInterruptedWindow())
	executions, err := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		UpdatedAfter:         &cutoff,
		TerminalOnly:         true,
		RootOnly:             true,
		WithoutRestartedAs:   true,
		StatusReasons:        []string{string(ErrorCategoryControlPlaneShutdown), types.ExecutionReasonAgentShutdownCancelled},
		StatusReasonPrefixes: []string{"agent_restart_orphaned"},
		Limit:                limit,
		SortBy:               "updated_at",
		SortDescending:       false,
	})
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to query interrupted runs on startup")
		return
	}
	sort.SliceStable(executions, func(i, j int) bool {
		return executions[i] != nil && executions[j] != nil && executions[i].UpdatedAt.Before(executions[j].UpdatedAt)
	})
	controller := newExecutionController(store, payloads, webhooks, timeout, internalToken)
	candidates, resumed, skipped := 0, 0, 0
	for _, execution := range executions {
		if execution == nil || candidates >= limit {
			continue
		}
		candidates++
		if _, ok := controller.handOffInterruptedRun(ctx, execution); ok {
			resumed++
		} else {
			skipped++
		}
	}
	logger.Logger.Info().Int("candidates", candidates).Int("resumed", resumed).Int("skipped", skipped).Msg("interrupted run startup handoff complete")
}

func (c *executionController) handOffInterruptedAgentExecutions(ctx context.Context, agentNodeID, instanceID string) {
	if !ResumeInterruptedRuns() {
		return
	}
	failed := string(types.ExecutionStatusFailed)
	executions, err := c.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		AgentNodeID: &agentNodeID,
		Status:      &failed,
		RootOnly:    true,
	})
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_node_id", agentNodeID).Msg("failed to load reaped executions for handoff")
		return
	}
	for _, execution := range executions {
		if execution == nil || execution.InstanceID != instanceID || execution.StatusReason == nil || !strings.HasPrefix(*execution.StatusReason, "agent_restart_orphaned") {
			continue
		}
		c.scheduleInterruptedRunHandoff(execution)
	}
}
