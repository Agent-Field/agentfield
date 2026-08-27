package handlers

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// Absorbing agent restarts.
//
// A long-running agent node restarts constantly during development: the user
// saves a file, hits Ctrl-C, or the process crashes on a syntax error. The
// control plane keeps believing the node is healthy for as long as it takes
// the active health checker to notice (check_interval x consecutive_failures,
// ~30s by default) or the heartbeat to go stale (60s). Every call that lands
// inside that window used to create an execution record, fail to dial the
// dead process, and be recorded as a failed execution — charging the user for
// a failure that was really a five-second restart.
//
// Two mechanisms fix that, and they are deliberately different:
//
//   1. ensureAgentDispatchable — when we ALREADY know the node is down, do
//      not create an execution record at all. A request we never dispatched
//      is a rejected request (503 node_unavailable), not a failed execution.
//      This mirrors the check reasoners.go has always had on the legacy
//      proxy route; the /execute path simply never grew one.
//
//   2. dispatchAgentRequest — when we do NOT know the node is down and the
//      dial fails, wait out the restart instead of failing. A dial failure
//      is the one transport error where retrying is unambiguously safe: no
//      bytes reached the agent, so nothing can be executed twice.
//
// When the wait expires we mark the node unreachable, so the NEXT caller
// takes path 1 and fails fast instead of repeating the wait.

const (
	// defaultAgentRestartGrace is how long a dispatch waits for a restarting
	// long-running node before giving up. A Python SDK agent takes ~5-7s to
	// import, register and start serving; 15s covers that with headroom while
	// staying well inside the 90s default agent call timeout.
	defaultAgentRestartGrace = 15 * time.Second

	// agentRestartPoll is how often the node record is re-read while waiting.
	// The SDK heartbeats every 2s and re-registers immediately on boot, so a
	// sub-second poll turns the recovery around as soon as it is visible.
	agentRestartPoll          = 250 * time.Millisecond
	packageUpdateRestartGrace = 10 * time.Minute
)

// agentRestartGraceNanos holds the configured grace window. It follows the
// package-level pattern already used by defaultRedactPayloads: set once at
// server startup, read on every dispatch. Stored as an int64 so tests can
// change it without racing the async worker pool.
var agentRestartGraceNanos atomic.Int64
var updatingAgentNodes = struct {
	sync.RWMutex
	names map[string]int
}{names: make(map[string]int)}

func init() {
	agentRestartGraceNanos.Store(int64(defaultAgentRestartGrace))
}

// SetAgentRestartGrace configures how long a dispatch waits for a restarting
// agent node. A value of zero or less disables the wait entirely, restoring
// the previous fail-on-first-dial-error behaviour. Call once at startup.
func SetAgentRestartGrace(d time.Duration) {
	agentRestartGraceNanos.Store(int64(d))
}

func agentRestartGrace() time.Duration {
	return time.Duration(agentRestartGraceNanos.Load())
}

// SetAgentUpdateInProgress extends dispatch restart tolerance for one node
// while its package job deliberately takes the process down.
func SetAgentUpdateInProgress(nodeID string, active bool) {
	updatingAgentNodes.Lock()
	defer updatingAgentNodes.Unlock()
	if active {
		updatingAgentNodes.names[nodeID]++
		return
	}
	if updatingAgentNodes.names[nodeID] <= 1 {
		delete(updatingAgentNodes.names, nodeID)
		return
	}
	updatingAgentNodes.names[nodeID]--
}

func agentRestartGraceFor(nodeID string) (time.Duration, bool) {
	configured := agentRestartGrace()
	if configured <= 0 {
		return configured, false
	}
	updatingAgentNodes.RLock()
	defer updatingAgentNodes.RUnlock()
	for updating := range updatingAgentNodes.names {
		if packages.NodeIDsEquivalent(updating, nodeID) {
			return packageUpdateRestartGrace, true
		}
	}
	return configured, false
}

// agentHealthMarker is the optional slice of storage used to demote a node we
// could not reach. It is deliberately NOT added to ExecutionStore: every test
// fake in the package implements that interface, and marking health is not a
// capability the execution path may assume. Storage that cannot do it simply
// skips the demotion.
type agentHealthMarker interface {
	UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error
}

var _ agentHealthMarker = (*storage.LocalStorage)(nil)

// ensureAgentDispatchable rejects a request aimed at a node we already know is
// down, before any execution record exists.
//
// Only definitively-down states are rejected. "unknown" means the node has not
// heartbeated yet — common in the seconds after registration — and must be
// allowed through, otherwise the very first call of a session fails. Same for
// "degraded", which is a partial-capacity signal, not an outage.
func ensureAgentDispatchable(agent *types.AgentNode, nodeID string) error {
	if agent == nil {
		return nil
	}
	if _, extended := agentRestartGraceFor(nodeID); extended {
		return nil
	}
	if agent.HealthStatus == types.HealthStatusInactive {
		return &executionPreconditionError{
			code:      http.StatusServiceUnavailable,
			message:   fmt.Sprintf("agent node '%s' is not reachable (health: %s); start the node and retry", agent.ID, agent.HealthStatus),
			category:  ErrorCategoryNodeUnavailable,
			errorCode: "node_unavailable",
		}
	}
	if agent.LifecycleStatus == types.AgentStatusOffline {
		return &executionPreconditionError{
			code:      http.StatusServiceUnavailable,
			message:   fmt.Sprintf("agent node '%s' is offline; start the node and retry", agent.ID),
			category:  ErrorCategoryNodeUnavailable,
			errorCode: "node_unavailable",
		}
	}
	return nil
}

// isDialFailure reports whether err means the connection was never
// established, so no part of the request reached the agent.
//
// This is the safety property the retry depends on. A dial failure is safe to
// repeat; a read/write failure mid-request is not, because the agent may
// already be running the reasoner. net.OpError.Op is the portable signal —
// checking syscall.ECONNREFUSED would not compile the same way across the
// linux/darwin/windows build matrix.
func isDialFailure(err error) bool {
	if err == nil {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return opErr.Op == "dial"
	}
	return false
}

// newAgentRequest builds the POST the control plane sends to an agent node.
//
// Extracted from callAgent so the restart retry can build a second, identical
// request: an *http.Request body is single-use, and the retry may also target
// a different address if the node came back on another port.
func (c *executionController) newAgentRequest(ctx context.Context, plan *preparedExecution, url string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(plan.requestBody))
	if err != nil {
		return nil, fmt.Errorf("create agent request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Run-ID", plan.exec.RunID)
	req.Header.Set("X-Execution-ID", plan.exec.ExecutionID)
	req.Header.Set("X-Workflow-ID", plan.exec.RunID)
	if plan.exec.ParentExecutionID != nil {
		req.Header.Set("X-Parent-Execution-ID", *plan.exec.ParentExecutionID)
	}
	if plan.exec.SessionID != nil {
		req.Header.Set("X-Session-ID", *plan.exec.SessionID)
	}
	if plan.exec.ActorID != nil {
		req.Header.Set("X-Actor-ID", *plan.exec.ActorID)
	}
	if c.internalToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.internalToken)
	}
	if plan.callerDID != "" {
		req.Header.Set("X-Caller-DID", plan.callerDID)
	}
	if plan.targetDID != "" {
		req.Header.Set("X-Target-DID", plan.targetDID)
	}
	if plan.replaySourceRunID != "" {
		req.Header.Set("X-AgentField-Replay-Source-Run-ID", plan.replaySourceRunID)
	}
	if plan.replayBeforeExecutionID != "" {
		req.Header.Set("X-AgentField-Replay-Before-Execution-ID", plan.replayBeforeExecutionID)
	}
	if plan.replayMode != "" {
		req.Header.Set("X-AgentField-Replay-Mode", plan.replayMode)
	}
	return req, nil
}

// dispatchAgentRequest sends the prepared request, absorbing a restart of the
// target node if one is in flight.
//
// The returned response is the caller's to close.
func (c *executionController) dispatchAgentRequest(ctx context.Context, plan *preparedExecution) (*http.Response, error) {
	req, err := c.newAgentRequest(ctx, plan, buildAgentURL(plan.agent, plan.target))
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err == nil {
		return resp, nil
	}
	if !isDialFailure(err) || !c.agentMayRestart(plan) {
		return nil, err
	}
	return c.retryAfterAgentRestart(ctx, plan, err)
}

// agentMayRestart reports whether waiting for this target can ever help.
// Serverless nodes have no resident process to come back, so a dial failure
// against one is a real outage and must surface immediately.
func (c *executionController) agentMayRestart(plan *preparedExecution) bool {
	if plan == nil || plan.agent == nil || plan.target == nil {
		return false
	}
	grace, _ := agentRestartGraceFor(plan.target.NodeID)
	if grace <= 0 {
		return false
	}
	return plan.agent.DeploymentType != "serverless"
}

// retryAfterAgentRestart waits for the target node to come back, then replays
// the dispatch against its current address.
//
// Recovery is detected from the node record rather than by blindly retrying:
// the SDK re-registers on boot with a fresh InstanceID, and it may come back
// on a different port, so re-reading the record is what makes the retry land
// on the right process at the right address.
//
// Note on the orphan reaper: re-registration also triggers
// MarkAgentExecutionsOrphaned, which is why markExecutionAwaitingRestart
// stamps the row before the wait begins. If a re-registration lands in the
// short window between the failed dial and that stamp, the reaper fails the
// row first: the `executions` table is then repaired by completeExecution
// (UpdateExecutionRecord applies no transition validation), but the
// workflow_executions row cannot leave `failed` — its state machine has no
// exits from terminal states — so the DAG UI keeps the reaped failure while
// the execution record shows the retry's real outcome. That residual race is
// narrow (one dial failure followed by a re-registration within milliseconds)
// and strictly better than the pre-grace behaviour, where the same sequence
// failed the run outright.
func (c *executionController) retryAfterAgentRestart(ctx context.Context, plan *preparedExecution, dialErr error) (*http.Response, error) {
	observedInstance := plan.agent.InstanceID
	observedHeartbeat := plan.agent.LastHeartbeat
	startedWaiting := time.Now()
	grace, _ := agentRestartGraceFor(plan.target.NodeID)
	// Both are re-read on every poll below, where the extended flag matters.
	var deadline time.Time
	var updateGrace bool

	logger.Logger.Info().
		Str("execution_id", plan.exec.ExecutionID).
		Str("agent_node_id", plan.target.NodeID).
		Dur("grace", grace).
		Msg("agent node did not accept the connection; waiting for it to come back before failing the execution")

	// Claim the execution before the node re-registers, so the orphan reaper
	// leaves it alone. Without this the restart we are waiting for fails the
	// very execution we are holding, and the caller receives that failure even
	// though the retry went on to succeed.
	c.markExecutionAwaitingRestart(ctx, plan)

	ticker := time.NewTicker(agentRestartPoll)
	defer ticker.Stop()

	lastErr := dialErr
	for {
		select {
		case <-ctx.Done():
			// ctx is dead, so release the hold on a detached context; the
			// caller is about to fail the execution and the reaper exemption
			// must not outlive the wait.
			c.clearExecutionAwaitingRestartDetached(plan)
			return nil, lastErr
		case <-ticker.C:
		}
		// Grace ownership can end while this request is waiting. Re-read it on
		// every poll so a failed restore/update falls back to the configured
		// deadline instead of pinning a worker for the original ten minutes.
		grace, updateGrace = agentRestartGraceFor(plan.target.NodeID)
		deadline = startedWaiting.Add(grace)
		if time.Now().After(deadline) {
			break
		}

		agent, err := c.store.GetAgent(ctx, plan.target.NodeID)
		if err != nil || agent == nil {
			continue
		}
		if !agentCameBack(agent, observedInstance, observedHeartbeat) {
			// While we were waiting, the health checker (or another dispatch
			// whose grace expired first) may have declared the node down.
			// Waiting longer cannot help — it would only serialize behind a
			// verdict that has already been reached — so surface the dial
			// error now instead of burning the rest of the grace.
			if !updateGrace && (agent.HealthStatus == types.HealthStatusInactive || agent.LifecycleStatus == types.AgentStatusOffline) {
				break
			}
			continue
		}

		// The node is back, but the execution may have moved on while we
		// waited: a cancel must win over the replay (mirroring the check
		// callAgent performs before the first dispatch), and a paused
		// execution waits for its resume before any work is handed out.
		if currentExec, execErr := c.store.GetExecutionRecord(ctx, plan.exec.ExecutionID); execErr == nil && currentExec != nil {
			if currentExec.Status == types.ExecutionStatusCancelled {
				return nil, fmt.Errorf("execution cancelled while waiting for agent restart")
			}
			if currentExec.Status == types.ExecutionStatusPaused {
				if resumeErr := c.waitForResume(ctx, plan.exec.ExecutionID); resumeErr != nil {
					return nil, fmt.Errorf("execution paused during agent restart and then cancelled or timed out: %w", resumeErr)
				}
			}
		}

		plan.agent = agent
		req, reqErr := c.newAgentRequest(ctx, plan, buildAgentURL(agent, plan.target))
		if reqErr != nil {
			c.clearExecutionAwaitingRestart(ctx, plan)
			return nil, reqErr
		}
		resp, callErr := c.httpClient.Do(req)
		if callErr == nil {
			c.clearExecutionAwaitingRestart(ctx, plan)
			logger.Logger.Info().
				Str("execution_id", plan.exec.ExecutionID).
				Str("agent_node_id", plan.target.NodeID).
				Msg("agent node came back; execution resumed instead of failing")
			return resp, nil
		}
		if !isDialFailure(callErr) {
			c.clearExecutionAwaitingRestart(ctx, plan)
			return nil, callErr
		}
		// The node announced itself but is not serving yet. Keep waiting
		// against the refreshed record.
		lastErr = callErr
		observedInstance = agent.InstanceID
		observedHeartbeat = agent.LastHeartbeat
	}

	c.clearExecutionAwaitingRestart(ctx, plan)
	if !updateGrace {
		c.markAgentUnreachable(ctx, plan)
	}
	return nil, lastErr
}

// markExecutionAwaitingRestart stamps the execution as held across an agent
// restart. The status stays "running" — only the reason changes — so nothing
// but the orphan reaper's exclusion depends on it.
//
// Best-effort: a failure here costs the exemption, not the execution, and the
// pre-existing behaviour (a reaped row that completeExecution later corrects)
// still applies. There is a sub-millisecond window between the failed dial and
// this write in which a re-registration could still reap the row; the retry
// itself is unaffected, and the terminal write wins.
func (c *executionController) markExecutionAwaitingRestart(ctx context.Context, plan *preparedExecution) {
	c.setExecutionRestartReason(ctx, plan, types.ExecutionReasonAwaitingAgentRestart)
}

// clearExecutionAwaitingRestart releases the claim once the wait is over, so a
// row that goes on to succeed does not carry a stale reason.
func (c *executionController) clearExecutionAwaitingRestart(ctx context.Context, plan *preparedExecution) {
	c.setExecutionRestartReason(ctx, plan, "")
}

// clearExecutionAwaitingRestartDetached releases the claim when the caller's
// context is already cancelled and could not carry the write.
func (c *executionController) clearExecutionAwaitingRestartDetached(plan *preparedExecution) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	c.setExecutionRestartReason(ctx, plan, "")
}

func (c *executionController) setExecutionRestartReason(ctx context.Context, plan *preparedExecution, reason string) {
	if plan == nil || plan.exec == nil {
		return
	}
	_, err := c.store.UpdateExecutionRecord(ctx, plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", plan.exec.ExecutionID)
		}
		// Only ever touch a running row. A terminal or paused execution is
		// somebody else's to describe.
		if current.Status != types.ExecutionStatusRunning {
			return current, nil
		}
		if reason == "" {
			if current.StatusReason == nil || *current.StatusReason != types.ExecutionReasonAwaitingAgentRestart {
				return current, nil
			}
			current.StatusReason = nil
		} else {
			current.StatusReason = &reason
		}
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	if err != nil {
		logger.Logger.Debug().Err(err).
			Str("execution_id", plan.exec.ExecutionID).
			Str("reason", reason).
			Msg("could not record the agent-restart hold on the execution")
	}

	// workflow_executions is the source of truth for the DAG UI and for the
	// dashboard's failure counts, and the reaper writes to it separately. The
	// hold has to be stamped on both rows or the run still shows as failed
	// even though the retry succeeded.
	wfErr := c.store.UpdateWorkflowExecution(ctx, plan.exec.ExecutionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		if current == nil {
			return nil, fmt.Errorf("workflow execution %s not found", plan.exec.ExecutionID)
		}
		if current.Status != types.ExecutionStatusRunning {
			return current, nil
		}
		if reason == "" {
			if current.StatusReason == nil || *current.StatusReason != types.ExecutionReasonAwaitingAgentRestart {
				return current, nil
			}
			current.StatusReason = nil
		} else {
			current.StatusReason = &reason
		}
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	if wfErr != nil {
		logger.Logger.Debug().Err(wfErr).
			Str("execution_id", plan.exec.ExecutionID).
			Str("reason", reason).
			Msg("could not record the agent-restart hold on the workflow execution")
	}
}

// agentCameBack reports whether the node record now describes a live process
// that is not the one we failed to dial.
//
// A changed, non-empty InstanceID is the definitive signal. Older SDKs do not
// report one, so a heartbeat newer than the one we dialled against is accepted
// as the fallback — it can only advance while a process is running.
func agentCameBack(agent *types.AgentNode, observedInstance string, observedHeartbeat time.Time) bool {
	if agent.InstanceID != "" && agent.InstanceID != observedInstance {
		return true
	}
	return agent.LastHeartbeat.After(observedHeartbeat)
}

// markAgentUnreachable demotes a node we waited out, so the next caller is
// rejected by ensureAgentDispatchable instead of repeating the wait.
//
// The update is conditional on the heartbeat we last observed: if the node
// heartbeated while we were waiting, the write is skipped and the health
// checker keeps ownership of the node's status. Failure is logged and
// swallowed — the health checker converges on its own, and losing this
// optimisation must never fail an execution.
func (c *executionController) markAgentUnreachable(ctx context.Context, plan *preparedExecution) {
	marker, ok := c.store.(agentHealthMarker)
	if !ok || plan.agent == nil {
		return
	}
	expected := plan.agent.LastHeartbeat
	err := marker.UpdateAgentHealthAtomic(ctx, plan.target.NodeID, types.HealthStatusInactive, &expected)
	if err != nil {
		logger.Logger.Debug().Err(err).
			Str("agent_node_id", plan.target.NodeID).
			Msg("could not demote unreachable agent node; leaving it to the health checker")
		return
	}
	logger.Logger.Warn().
		Str("agent_node_id", plan.target.NodeID).
		Msg("agent node stayed unreachable for the full restart grace; marked inactive so further calls fail fast")
}
