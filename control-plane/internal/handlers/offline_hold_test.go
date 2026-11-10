package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// The stale window these tests run the sweep with. CreateExecutionRecord
// stamps updated_at itself, so an execution is made quiet by waiting past the
// window rather than by backdating it (as TestLeaseInflightKeepsExecutionsAlive
// does).
const holdStale = 300 * time.Millisecond

func waitPastStale() { time.Sleep(holdStale + 150*time.Millisecond) }

// offlineHoldWorld is a control plane with one node that went to sleep an
// hour ago while it was running exec_<node>.
type offlineHoldWorld struct {
	t     *testing.T
	ctx   context.Context
	store storage.StorageProvider
	sweep *ExecutionCleanupService
	lease *gin.Engine
}

func newOfflineHoldWorld(t *testing.T, ceiling time.Duration) *offlineHoldWorld {
	t.Helper()
	gin.SetMode(gin.TestMode)
	store, ctx := setupTestStorage(t)
	lease := gin.New()
	lease.PATCH("/nodes/:node_id/status", NodeStatusLeaseHandler(store, nil, nil, nil, time.Minute))
	lease.PUT("/executions/:execution_id/status", UpdateExecutionStatusHandler(store, services.NopPayloadStore{}, nil, time.Second))
	sweep := NewExecutionCleanupService(store, config.ExecutionCleanupConfig{
		Enabled: true, StaleExecutionTimeout: holdStale, BatchSize: 100, OfflineHoldCeiling: ceiling,
	})
	return &offlineHoldWorld{t: t, ctx: ctx, store: store, sweep: sweep, lease: lease}
}

// node registers id, opted in to the hold or not, last heard from an hour
// ago, holding one running execution (both tables) named exec_<id>.
func (w *offlineHoldWorld) node(id string, optedIn bool) string {
	w.t.Helper()
	agent := &types.AgentNode{ID: id, Version: "1", LifecycleStatus: types.AgentStatusReady}
	if optedIn {
		agent.Metadata.Custom = map[string]interface{}{types.AgentMetadataHoldsExecutionsOffline: true}
	}
	require.NoError(w.t, w.store.RegisterAgent(w.ctx, agent))
	require.NoError(w.t, w.store.UpdateAgentHeartbeat(w.ctx, id, "1", time.Now().UTC().Add(-time.Hour)))
	exec := "exec_" + id
	now := time.Now().UTC()
	require.NoError(w.t, w.store.CreateExecutionRecord(w.ctx, &types.Execution{
		ExecutionID: exec, RunID: "run_" + id, AgentNodeID: id, NodeID: id, ReasonerID: "intake",
		Status: types.ExecutionStatusRunning, StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}))
	require.NoError(w.t, w.store.StoreWorkflowExecution(w.ctx, &types.WorkflowExecution{
		WorkflowID: "run_" + id, ExecutionID: exec, AgentFieldRequestID: "req_" + id,
		AgentNodeID: id, ReasonerID: "intake", Status: types.ExecutionStatusRunning,
		StartedAt: now, CreatedAt: now, UpdatedAt: now, WorkflowTags: []string{},
		InputData: json.RawMessage("{}"), OutputData: json.RawMessage("{}"),
	}))
	return exec
}

// leaseFrom sends node's lease listing inflight and answers the body.
func (w *offlineHoldWorld) leaseFrom(node string, inflight ...string) map[string]any {
	w.t.Helper()
	if inflight == nil {
		inflight = []string{}
	}
	raw, _ := json.Marshal(map[string]any{"phase": "ready", "version": "1", "inflight": inflight})
	req := httptest.NewRequest(http.MethodPatch, "/nodes/"+node+"/status", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	w.lease.ServeHTTP(rec, req)
	require.Equal(w.t, http.StatusOK, rec.Code, rec.Body.String())
	var body map[string]any
	require.NoError(w.t, json.Unmarshal(rec.Body.Bytes(), &body))
	return body
}

func (w *offlineHoldWorld) exec(id string) *types.Execution {
	w.t.Helper()
	e, err := w.store.GetExecutionRecord(w.ctx, id)
	require.NoError(w.t, err)
	return e
}

func (w *offlineHoldWorld) workflowStatus(id string) string {
	w.t.Helper()
	wf, err := w.store.GetWorkflowExecution(w.ctx, id)
	require.NoError(w.t, err)
	return wf.Status
}

func reasonOf(e *types.Execution) string {
	if e.StatusReason == nil {
		return ""
	}
	return *e.StatusReason
}

func errorOf(e *types.Execution) string {
	if e.ErrorMessage == nil {
		return ""
	}
	return *e.ErrorMessage
}

// A NODE ASLEEP PAST THE STALE TIMEOUT STILL HOLDS ITS WORK. The opted-in
// laptop's execution stays running through any number of sweeps, on both
// tables, marked node_offline; the node that did not opt in is timed out by
// the same sweep exactly as before.
func TestOfflineNodeKeepsItsExecutionPastTheStaleTimeout(t *testing.T) {
	w := newOfflineHoldWorld(t, time.Hour)
	laptop := w.node("laptop", true)
	server := w.node("server", false)
	waitPastStale()

	w.sweep.performCleanup(w.ctx)
	waitPastStale()
	w.sweep.performCleanup(w.ctx) // well past the window since the hold began

	held := w.exec(laptop)
	require.Equal(t, types.ExecutionStatusRunning, held.Status)
	require.Equal(t, types.ExecutionReasonNodeOffline, reasonOf(held))
	require.Equal(t, types.ExecutionStatusRunning, w.workflowStatus(laptop))

	require.Equal(t, types.ExecutionStatusTimeout, w.exec(server).Status)
	require.Equal(t, types.ExecutionStatusTimeout, w.workflowStatus(server))
}

// A NODE THAT COMES BACK WITH THE EXECUTION CARRIES ON. Its first lease lists
// the held execution: the hold ends, the reason clears, and the node's own
// success report lands as on any running execution.
func TestNodeBackWithItsExecutionCarriesOn(t *testing.T) {
	w := newOfflineHoldWorld(t, time.Hour)
	laptop := w.node("laptop", true)
	waitPastStale()
	w.sweep.performCleanup(w.ctx)
	require.Equal(t, types.ExecutionReasonNodeOffline, reasonOf(w.exec(laptop)))

	body := w.leaseFrom("laptop", laptop)
	require.EqualValues(t, 1, body["held_released"])
	require.EqualValues(t, 0, body["held_lost"])
	back := w.exec(laptop)
	require.Equal(t, types.ExecutionStatusRunning, back.Status)
	require.Empty(t, reasonOf(back))

	req := httptest.NewRequest(http.MethodPut, "/executions/"+laptop+"/status", bytes.NewBufferString(`{"status":"succeeded","result":{"ok":true}}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	w.lease.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	require.Equal(t, types.ExecutionStatusSucceeded, w.exec(laptop).Status)
}

// A NODE THAT COMES BACK WITHOUT THE EXECUTION LOST IT. Its first lease does
// not list it, so it fails there and then, saying so — not as a timeout.
func TestNodeBackWithoutItsExecutionFailsIt(t *testing.T) {
	w := newOfflineHoldWorld(t, time.Hour)
	laptop := w.node("laptop", true)
	waitPastStale()
	w.sweep.performCleanup(w.ctx)

	body := w.leaseFrom("laptop")
	require.EqualValues(t, 0, body["held_released"])
	require.EqualValues(t, 1, body["held_lost"])
	lost := w.exec(laptop)
	require.Equal(t, types.ExecutionStatusFailed, lost.Status)
	require.Equal(t, types.ExecutionReasonNodeLostExecution, reasonOf(lost))
	require.Equal(t, "node laptop came back without this execution: it was lost while the node was offline", errorOf(lost))
	require.Equal(t, types.ExecutionStatusFailed, w.workflowStatus(laptop))
}

// A NODE THAT IS NO LONGER REGISTERED HAS NOBODY TO COME BACK. Its held
// execution fails on the next sweep.
func TestRemovedNodeFailsItsHeldExecution(t *testing.T) {
	w := newOfflineHoldWorld(t, time.Hour)
	laptop := w.node("laptop", true)
	waitPastStale()
	w.sweep.performCleanup(w.ctx)
	require.Equal(t, types.ExecutionReasonNodeOffline, reasonOf(w.exec(laptop)))

	require.NoError(t, w.store.DeleteAgentVersion(w.ctx, "laptop", "1"))
	w.sweep.performCleanup(w.ctx)

	gone := w.exec(laptop)
	require.Equal(t, types.ExecutionStatusFailed, gone.Status)
	require.Equal(t, types.ExecutionReasonNodeRemoved, reasonOf(gone))
	require.Equal(t, types.ExecutionStatusFailed, w.workflowStatus(laptop))
}

// NOTHING WAITS FOREVER. A hold that outlives offline_hold_ceiling fails with
// the ceiling in its words.
func TestHeldExecutionFailsPastTheCeiling(t *testing.T) {
	const ceiling = 200 * time.Millisecond
	w := newOfflineHoldWorld(t, ceiling)
	laptop := w.node("laptop", true)
	waitPastStale()
	w.sweep.performCleanup(w.ctx)
	require.Equal(t, types.ExecutionReasonNodeOffline, reasonOf(w.exec(laptop)))

	time.Sleep(ceiling + 100*time.Millisecond)
	w.sweep.performCleanup(w.ctx)

	expired := w.exec(laptop)
	require.Equal(t, types.ExecutionStatusFailed, expired.Status)
	require.Equal(t, types.ExecutionReasonNodeOfflineTooLong, reasonOf(expired))
	require.Contains(t, errorOf(expired), "stayed offline for more than 200ms")
}

// AN ONLINE NODE THAT FORGOT AN EXECUTION IS NOT OFFLINE. An opted-in node
// whose lease is current but does not list the execution gets no hold: the
// stale sweep times it out as it always did.
func TestOnlineOptedInNodeStillHasItsQuietWorkReaped(t *testing.T) {
	w := newOfflineHoldWorld(t, time.Hour)
	laptop := w.node("laptop", true)
	waitPastStale()
	w.leaseFrom("laptop") // alive, and not working on it

	w.sweep.performCleanup(w.ctx)
	require.Equal(t, types.ExecutionStatusTimeout, w.exec(laptop).Status)
}

func TestHoldsExecutionsOfflineReadsTheRegistrationFlag(t *testing.T) {
	for _, tc := range []struct {
		custom map[string]interface{}
		want   bool
	}{
		{nil, false},
		{map[string]interface{}{}, false},
		{map[string]interface{}{"holds_executions_offline": false}, false},
		{map[string]interface{}{"holds_executions_offline": "yes"}, false},
		{map[string]interface{}{"holds_executions_offline": true}, true},
		{map[string]interface{}{"holds_executions_offline": "true"}, true},
	} {
		a := &types.AgentNode{Metadata: types.AgentMetadata{Custom: tc.custom}}
		require.Equal(t, tc.want, a.HoldsExecutionsOffline(), "%v", tc.custom)
	}
	require.False(t, (*types.AgentNode)(nil).HoldsExecutionsOffline())
}
