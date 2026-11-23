package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// A TRIGGER-STARTED RUN HAS A TREE. The dispatcher files the root under
// run_id = workflow id and must tell the node that run in X-Run-ID: a node
// that is not told mints a fresh run for every child it calls, and the CP then
// files each child as its own one-node run with no parent. Here the node does
// what the Go SDK's Call does with the headers it was dispatched with — sends
// the run it was given as X-Run-ID and its own execution as
// X-Parent-Execution-ID — through the real execute handler.
func TestATriggerRunsChildCallsShareItsRunAndParent(t *testing.T) {
	provider, dispatcher, ctx := setupGenericHMACTestEnv(t)

	child := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"found":"it"}`))
	}))
	defer child.Close()
	var (
		mu     sync.Mutex
		dRun   string
		dExec  string
		dWflow string
	)
	root := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		dRun, dExec, dWflow = r.Header.Get("X-Run-ID"), r.Header.Get("X-Execution-ID"), r.Header.Get("X-Workflow-ID")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer root.Close()

	for _, n := range []*types.AgentNode{
		{ID: "support", BaseURL: root.URL, HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, Reasoners: []types.ReasonerDefinition{{ID: "handle_ticket"}}},
		{ID: "helper", BaseURL: child.URL, HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, Reasoners: []types.ReasonerDefinition{{ID: "search_docs"}}},
	} {
		require.NoError(t, provider.RegisterAgent(ctx, n))
	}
	trig := &types.Trigger{ID: "tr", SourceName: "generic_bearer", TargetNodeID: "support", TargetReasoner: "handle_ticket",
		ManagedBy: types.ManagedByUI, Enabled: true, Config: json.RawMessage(`{}`), CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
	require.NoError(t, provider.CreateTrigger(ctx, trig))
	ev := &types.InboundEvent{ID: "ev", TriggerID: trig.ID, SourceName: trig.SourceName, EventType: "ticket",
		RawPayload: json.RawMessage(`{}`), NormalizedPayload: json.RawMessage(`{}`), IdempotencyKey: "k",
		Status: types.InboundEventStatusReceived, ReceivedAt: time.Now().UTC()}
	require.NoError(t, provider.InsertInboundEvent(ctx, ev))

	dispatcher.DispatchEvent(ctx, trig, ev)
	mu.Lock()
	run, rootExec, wflow := dRun, dExec, dWflow
	mu.Unlock()
	require.NotEmpty(t, rootExec, "the root was never dispatched")
	require.Equal(t, wflow, run, "the node is told the run, and the run is the workflow")
	rootRec, err := provider.GetExecutionRecord(ctx, rootExec)
	require.NoError(t, err)
	require.Equal(t, run, rootRec.RunID)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/api/v1/execute/:target", ExecuteHandler(provider, nil, nil, 10*time.Second, ""))
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/helper.search_docs", bytes.NewBufferString(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Run-ID", run)
		req.Header.Set("X-Parent-Execution-ID", rootExec)
		req.Header.Set("X-Workflow-ID", wflow)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		var out struct {
			ExecutionID string `json:"execution_id"`
			RunID       string `json:"run_id"`
		}
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
		require.Equal(t, run, out.RunID)
		rec, err := provider.GetExecutionRecord(ctx, out.ExecutionID)
		require.NoError(t, err)
		require.Equal(t, run, rec.RunID, "the child is in the trigger's run")
		require.NotNil(t, rec.ParentExecutionID, "the child names its parent")
		require.Equal(t, rootExec, *rec.ParentExecutionID)
	}
}
