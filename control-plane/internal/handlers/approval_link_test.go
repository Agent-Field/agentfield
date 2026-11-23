package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

// ── A PAUSE ON A LINK-ONLY NODE, ASKED AND ANSWERED END TO END ──────────────
//
// A real store, the real hub and pipe, and two linked peers: "worker" is a Go
// SDK node with no address (it takes http deliveries), "workaf" owns the run
// and subscribes. The reasoner pauses with a plan, WorkAF is told by a run
// frame carrying the plan, answers "request changes" with words, the node is
// resumed DOWN ITS LINK with those words, pauses again and is approved.

func withApprovalCallbackBackoff(t *testing.T, b []time.Duration) {
	t.Helper()
	old := approvalCallbackBackoff
	approvalCallbackBackoff = b
	t.Cleanup(func() { approvalCallbackBackoff = old })
}

type linkRig struct {
	t     *testing.T
	store *storage.LocalStorage
	hub   *nodelink.Hub
	pipe  *nodelink.Pipe
	srv   *httptest.Server
	ws    string
}

func newLinkRig(t *testing.T) *linkRig {
	t.Helper()
	gin.SetMode(gin.TestMode)
	dir := t.TempDir()
	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(context.Background(), storage.StorageConfig{Mode: "local",
		Local: storage.LocalStorageConfig{DatabasePath: dir + "/cp.db", KVStorePath: dir + "/cp.bolt"}})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	r := &linkRig{t: t, store: ls}
	r.hub = &nodelink.Hub{Agents: ls, Sweep: 10 * time.Millisecond,
		Router: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })}
	r.pipe = &nodelink.Pipe{Store: ls, Hub: r.hub, Retention: func() time.Duration { return time.Hour }}
	r.hub.Pipe = r.pipe
	agenturl.SetLinkCarrier(r.hub)
	g := gin.New()
	g.GET("/api/v1/nodes/:node_id/link", r.hub.Handler())
	g.POST("/api/v1/executions/:execution_id/request-approval", RequestApprovalHandler(ls))
	g.GET("/api/v1/executions/:execution_id/approval-status", GetApprovalStatusHandler(ls))
	g.POST("/api/v1/executions/:execution_id/approval-response", ResolveApprovalHandler(ls, r.pipe))
	r.srv = httptest.NewServer(g)
	r.ws = "ws" + strings.TrimPrefix(r.srv.URL, "http") + "/api/v1/nodes/"
	ctx := context.Background()
	for _, n := range []string{"worker", "workaf", "other"} {
		require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{ID: n, Version: "1", BaseURL: "http://" + agenturl.LinkHost(n), LastHeartbeat: time.Now()}))
	}
	r.pipe.Start()
	t.Cleanup(func() {
		r.pipe.Stop()
		r.hub.Close()
		r.srv.Close()
		_ = ls.Close(context.Background())
		agenturl.SetLinkCarrier(nil)
	})
	return r
}

// running makes exec a running root execution of run, owned by owner.
func (r *linkRig) running(run, exec, owner string) {
	ctx := context.Background()
	now := time.Now().UTC()
	require.NoError(r.t, r.store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: exec, RunID: run,
		AgentNodeID: "worker", NodeID: "worker", ReasonerID: "plan", Status: types.ExecutionStatusRunning,
		StartedAt: now, CreatedAt: now, UpdatedAt: now}))
	runID := run
	require.NoError(r.t, r.store.StoreWorkflowExecution(ctx, &types.WorkflowExecution{ExecutionID: exec, WorkflowID: run,
		RunID: &runID, AgentNodeID: "worker", ReasonerID: "plan", Status: types.ExecutionStatusRunning, StartedAt: now}))
	require.NoError(r.t, r.pipe.Own(ctx, run, owner))
}

func (r *linkRig) do(method, path, owner string, body any) (int, map[string]any) {
	r.t.Helper()
	var rd *bytes.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		rd = bytes.NewReader(raw)
	} else {
		rd = bytes.NewReader(nil)
	}
	req, _ := http.NewRequest(method, r.srv.URL+path, rd)
	req.Header.Set("Content-Type", "application/json")
	if owner != "" {
		req.Header.Set(RunOwnerHeader, owner)
	}
	resp, err := http.DefaultClient.Do(req)
	require.NoError(r.t, err)
	defer resp.Body.Close()
	out := map[string]any{}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

type linkPeer struct {
	t      *testing.T
	conn   *websocket.Conn
	wmu    sync.Mutex
	frames chan nodelink.Frame
	seq    atomic.Int64
	serve  func(nodelink.DeliverBody) *nodelink.Ack
}

func (r *linkRig) dial(node string, accepts []string, serve func(nodelink.DeliverBody) *nodelink.Ack) *linkPeer {
	r.t.Helper()
	conn, _, err := websocket.DefaultDialer.Dial(r.ws+node+"/link", nil)
	require.NoError(r.t, err)
	p := &linkPeer{t: r.t, conn: conn, frames: make(chan nodelink.Frame, 256), serve: serve}
	r.t.Cleanup(func() { _ = conn.Close() })
	go p.read()
	p.next(nodelink.TWelcome, nil)
	p.send(nodelink.Frame{T: nodelink.THello, Seq: p.seq.Add(1)}, nodelink.Hello{InstanceID: node, Accepts: accepts})
	p.next(nodelink.TOK, nil)
	return p
}

func (p *linkPeer) read() {
	defer close(p.frames)
	for {
		_, raw, err := p.conn.ReadMessage()
		if err != nil {
			return
		}
		f, err := nodelink.Decode(raw)
		if err != nil {
			continue
		}
		if f.T == nodelink.TDeliver && p.serve != nil {
			var d nodelink.DeliverBody
			_ = json.Unmarshal(f.Body, &d)
			if ack := p.serve(d); ack != nil {
				p.send(nodelink.Frame{T: nodelink.TAck, ID: f.ID, Exec: f.Exec}, ack)
			}
			continue
		}
		p.frames <- f
	}
}

func (p *linkPeer) send(f nodelink.Frame, body any) {
	raw, err := nodelink.Encode(f, body)
	require.NoError(p.t, err)
	p.wmu.Lock()
	defer p.wmu.Unlock()
	require.NoError(p.t, p.conn.WriteMessage(websocket.TextMessage, raw))
}

// next is the next frame of type t that match accepts (nil: any).
func (p *linkPeer) next(t string, match func(nodelink.Frame) bool) nodelink.Frame {
	p.t.Helper()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case f, ok := <-p.frames:
			require.True(p.t, ok, "link closed waiting for %s", t)
			if f.T == t && (match == nil || match(f)) {
				return f
			}
		case <-timeout:
			p.t.Fatalf("no %s frame", t)
		}
	}
}

// execState is the next run frame's state for exec whose status is status.
func (p *linkPeer) execState(exec, status string) nodelink.ExecState {
	p.t.Helper()
	var got nodelink.ExecState
	p.next(nodelink.TRun, func(f nodelink.Frame) bool {
		var rs nodelink.RunState
		_ = json.Unmarshal(f.Body, &rs)
		for _, e := range rs.Executions {
			if e.ExecutionID == exec && e.Status == status {
				got = e
				return true
			}
		}
		return false
	})
	return got
}

type callbackBody struct {
	ExecutionID       string `json:"execution_id"`
	Decision          string `json:"decision"`
	Feedback          string `json:"feedback"`
	ApprovalRequestID string `json:"approval_request_id"`
	NewStatus         string `json:"new_status"`
}

func TestPauseOnALinkOnlyNodeIsAskedPushedAndResumedOverTheLink(t *testing.T) {
	withApprovalCallbackBackoff(t, []time.Duration{10 * time.Millisecond})
	r := newLinkRig(t)
	r.running("run_1", "exec_1", "workaf")

	callbacks := make(chan callbackBody, 8)
	r.dial("worker", []string{nodelink.KindHTTP, nodelink.KindSteer, nodelink.KindCancel}, func(d nodelink.DeliverBody) *nodelink.Ack {
		if d.Kind != nodelink.KindHTTP || d.Path != "/webhooks/approval" || d.Method != http.MethodPost {
			return &nodelink.Ack{Outcome: nodelink.OutcomeRejected, Error: "unexpected " + d.Kind + " " + d.Path}
		}
		var cb callbackBody
		_ = json.Unmarshal(d.Body, &cb)
		callbacks <- cb
		res, _ := json.Marshal(nodelink.HTTPResult{Code: 200, Body: []byte(`{"status":"received","resolved":true}`)})
		return &nodelink.Ack{Outcome: nodelink.OutcomeAccepted, Result: res}
	})
	workaf := r.dial("workaf", nil, nil)
	workaf.send(nodelink.Frame{T: nodelink.TSubscribe, Seq: workaf.seq.Add(1)}, nodelink.Subscribe{})
	workaf.execState("exec_1", types.ExecutionStatusRunning) // the snapshot

	plan := "# Plan\n\n1. Add the table\n2. Backfill\n"
	// 1. The reasoner pauses with a plan and no external request id.
	asked := time.Now()
	code, out := r.do(http.MethodPost, "/api/v1/executions/exec_1/request-approval", "", map[string]any{
		"callback_url": "http://worker.nodelink/webhooks/approval",
		"kind":         "plan_review", "question": "Approve the plan for #12?", "body": plan,
	})
	require.Equal(t, 200, code, out)
	id1, _ := out["approval_request_id"].(string)
	require.True(t, strings.HasPrefix(id1, "apr_"), "the control plane mints an id: %v", out)

	// 2. WorkAF is told by a pushed run frame, carrying the whole question.
	st := workaf.execState("exec_1", types.ExecutionStatusWaiting)
	require.Less(t, time.Since(asked), time.Second, "WorkAF learns of the pause within a second")
	require.Equal(t, "waiting_for_approval", st.StatusReason)
	require.NotNil(t, st.Approval)
	require.Equal(t, id1, st.Approval.ID)
	require.Equal(t, "plan_review", st.Approval.Kind)
	require.Equal(t, "Approve the plan for #12?", st.Approval.Question)
	require.Equal(t, plan, st.Approval.Body)
	require.Equal(t, types.DefaultApprovalOptions, st.Approval.Options)
	require.NotEmpty(t, st.Approval.ExpiresAt)

	code, out = r.do(http.MethodGet, "/api/v1/executions/exec_1/approval-status", "", nil)
	require.Equal(t, 200, code)
	require.Equal(t, "pending", out["status"])
	require.Equal(t, id1, out["approval_request_id"])
	require.Equal(t, plan, out["body"])
	require.Equal(t, "plan_review", out["kind"])

	// 3. Only the run's owner may answer.
	code, _ = r.do(http.MethodPost, "/api/v1/executions/exec_1/approval-response", "other", map[string]any{"decision": "approved"})
	require.Equal(t, http.StatusForbidden, code)

	// 4. Ask for changes, with words: the node is resumed down its link with them.
	code, out = r.do(http.MethodPost, "/api/v1/executions/exec_1/approval-response", "workaf", map[string]any{
		"decision": "request_changes", "reason": "backfill in batches of 1000",
	})
	require.Equal(t, 200, code, out)
	cb := <-callbacks
	require.Equal(t, callbackBody{ExecutionID: "exec_1", Decision: "request_changes", Feedback: "backfill in batches of 1000",
		ApprovalRequestID: id1, NewStatus: types.ExecutionStatusRunning}, cb)
	st = workaf.execState("exec_1", types.ExecutionStatusRunning)
	require.Nil(t, st.Approval)
	require.Contains(t, st.StatusReason, "backfill in batches of 1000")

	code, out = r.do(http.MethodGet, "/api/v1/executions/exec_1/approval-status", "", nil)
	require.Equal(t, 200, code)
	require.Equal(t, "request_changes", out["status"])
	require.Equal(t, id1, out["approval_request_id"])
	require.Equal(t, "backfill in batches of 1000", out["feedback"])

	// 5. The reasoner revises and pauses again; this time it is approved.
	code, out = r.do(http.MethodPost, "/api/v1/executions/exec_1/request-approval", "", map[string]any{
		"approval_request_id": "swe-plan-exec_1-r2",
		"callback_url":        "http://worker.nodelink/webhooks/approval",
		"kind":                "plan_review", "question": "Approve the revised plan?", "body": plan + "3. In batches of 1000\n",
	})
	require.Equal(t, 200, code, out)
	st = workaf.execState("exec_1", types.ExecutionStatusWaiting)
	require.Equal(t, "swe-plan-exec_1-r2", st.Approval.ID)
	require.Contains(t, st.Approval.Body, "batches of 1000")

	code, _ = r.do(http.MethodPost, "/api/v1/executions/exec_1/approval-response", "workaf", map[string]any{"decision": "approved"})
	require.Equal(t, 200, code)
	cb = <-callbacks
	require.Equal(t, "approved", cb.Decision)
	require.Equal(t, "swe-plan-exec_1-r2", cb.ApprovalRequestID)

	// 6. A second answer (hax, or a second door) is told it was answered.
	code, out = r.do(http.MethodPost, "/api/v1/executions/exec_1/approval-response", "workaf", map[string]any{"decision": "rejected"})
	require.Equal(t, 200, code)
	require.Equal(t, "already_processed", out["status"])
	require.Equal(t, "approved", out["approval_status"])
	select {
	case extra := <-callbacks:
		t.Fatalf("a second answer resumed the node again: %+v", extra)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestApprovalCallbackRetriesUntilTheNodeTakesIt(t *testing.T) {
	withApprovalCallbackBackoff(t, []time.Duration{5 * time.Millisecond, 5 * time.Millisecond, 5 * time.Millisecond})
	r := newLinkRig(t)
	r.running("run_2", "exec_2", "workaf")
	var hits atomic.Int32
	got := make(chan callbackBody, 1)
	r.dial("worker", []string{nodelink.KindHTTP}, func(d nodelink.DeliverBody) *nodelink.Ack {
		if hits.Add(1) < 3 {
			res, _ := json.Marshal(nodelink.HTTPResult{Code: 503, Body: []byte(`busy`)})
			return &nodelink.Ack{Outcome: nodelink.OutcomeAccepted, Result: res}
		}
		var cb callbackBody
		_ = json.Unmarshal(d.Body, &cb)
		got <- cb
		res, _ := json.Marshal(nodelink.HTTPResult{Code: 200, Body: []byte(`{}`)})
		return &nodelink.Ack{Outcome: nodelink.OutcomeAccepted, Result: res}
	})
	code, out := r.do(http.MethodPost, "/api/v1/executions/exec_2/request-approval", "", map[string]any{
		"callback_url": "http://worker.nodelink/webhooks/approval",
	})
	require.Equal(t, 200, code, out)
	code, _ = r.do(http.MethodPost, "/api/v1/executions/exec_2/approval-response", "", map[string]any{"decision": "approved"})
	require.Equal(t, 200, code)
	select {
	case cb := <-got:
		require.Equal(t, "approved", cb.Decision)
	case <-time.After(5 * time.Second):
		t.Fatal("the callback was never retried into the node")
	}
	require.EqualValues(t, 3, hits.Load())
}

func TestApprovalCallbackThatNeverLandsIsRecorded(t *testing.T) {
	withApprovalCallbackBackoff(t, []time.Duration{time.Millisecond, time.Millisecond})
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()
	oldClient := approvalCallbackClient
	approvalCallbackClient = func(string) *http.Client { return srv.Client() }
	t.Cleanup(func() { approvalCallbackClient = oldClient })

	store := &recordingEventStore{testExecutionStorage: newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})}
	ctrl := &webhookApprovalController{store: store}
	ctrl.notifyApprovalCallback(approvalCallback{url: srv.URL + "/webhooks/approval", executionID: "exec-9",
		decision: "approved", newStatus: "running", approvalRequestID: "req-9"})
	require.EqualValues(t, 3, hits.Load(), "one attempt and two retries")
	require.Len(t, store.events, 1)
	require.Equal(t, "execution.approval_callback_failed", store.events[0].EventType)
	require.Contains(t, string(store.events[0].Payload), `"attempts":3`)
	require.Contains(t, string(store.events[0].Payload), "502")
}

type recordingEventStore struct {
	*testExecutionStorage
	events []*types.WorkflowExecutionEvent
}

func (s *recordingEventStore) StoreWorkflowExecutionEvent(_ context.Context, e *types.WorkflowExecutionEvent) error {
	s.events = append(s.events, e)
	return nil
}

func TestRequestApprovalRefusesABadQuestion(t *testing.T) {
	r := newLinkRig(t)
	r.running("run_3", "exec_3", "workaf")
	code, out := r.do(http.MethodPost, "/api/v1/executions/exec_3/request-approval", "", map[string]any{"kind": "vote"})
	require.Equal(t, 400, code, out)
	code, out = r.do(http.MethodPost, "/api/v1/executions/exec_3/request-approval", "", map[string]any{"body": strings.Repeat("x", maxApprovalBody+1)})
	require.Equal(t, 400, code, out)
	// Still running: a refused question does not pause the run.
	code, out = r.do(http.MethodPost, "/api/v1/executions/exec_3/request-approval", "", map[string]any{
		"kind": "question", "question": "Which database?", "options": []string{" Postgres ", "SQLite", ""},
	})
	require.Equal(t, 200, code, out)
	code, out = r.do(http.MethodGet, "/api/v1/executions/exec_3/approval-status", "", nil)
	require.Equal(t, 200, code)
	require.Equal(t, []any{"Postgres", "SQLite"}, out["options"])
	require.Equal(t, "question", out["kind"])
}
