package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// A LEASE THAT LISTS AN EXECUTION KEEPS IT ALIVE: the stale reaper times out
// a quiet running execution unless the node holding it says, in its lease,
// that it is still working on it. Another node's execution is not kept alive
// by this node's lease, and an unlisted one is reaped as before.
func TestLeaseInflightKeepsExecutionsAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)
	provider, ctx := setupTestStorage(t)
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{ID: "n1", Version: "1", LifecycleStatus: types.AgentStatusReady}))
	old := time.Now().UTC().Add(-time.Hour)
	for id, node := range map[string]string{"exec_listed": "n1", "exec_quiet": "n1", "exec_other": "n2"} {
		require.NoError(t, provider.CreateExecutionRecord(ctx, &types.Execution{
			ExecutionID: id, RunID: "run_" + id, AgentNodeID: node, NodeID: node, ReasonerID: "intake",
			Status: types.ExecutionStatusRunning, StartedAt: old, CreatedAt: old, UpdatedAt: old,
		}))
	}

	// CreateExecutionRecord stamps updated_at itself, so the executions are
	// made quiet by waiting past a short stale window rather than by
	// backdating them.
	const stale = 400 * time.Millisecond
	time.Sleep(stale + 200*time.Millisecond)

	router := gin.New()
	router.PATCH("/nodes/:node_id/status", NodeStatusLeaseHandler(provider, nil, nil, nil, time.Minute))
	req := httptest.NewRequest(http.MethodPatch, "/nodes/n1/status", bytes.NewReader([]byte(`{"phase":"ready","version":"1","inflight":["exec_listed","exec_other","exec_listed"]}`)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	require.Contains(t, rec.Body.String(), `"inflight_touched":1`)

	reaped, err := provider.MarkStaleExecutions(ctx, stale, 100)
	require.NoError(t, err)
	require.Equal(t, 2, reaped)

	for id, want := range map[string]types.ExecutionStatus{"exec_listed": types.ExecutionStatusRunning, "exec_quiet": types.ExecutionStatusTimeout, "exec_other": types.ExecutionStatusTimeout} {
		e, err := provider.GetExecutionRecord(ctx, id)
		require.NoError(t, err)
		require.Equal(t, want, e.Status, id)
	}
}
