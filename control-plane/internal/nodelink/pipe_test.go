package nodelink

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

var _ PipeStore = (*storage.LocalStorage)(nil)

// ── A CONTROL PLANE ON A REAL STORE, WITH LINKS ─────────────────────────────

type rig struct {
	t     *testing.T
	store *storage.LocalStorage
	hub   *Hub
	pipe  *Pipe
	srv   *httptest.Server
	base  string
	shift atomic.Int64 // the pipe's clock offset, for the retention ceiling
}

func openStore(t *testing.T, dir string) *storage.LocalStorage {
	t.Helper()
	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(context.Background(), storage.StorageConfig{Mode: "local",
		Local: storage.LocalStorageConfig{DatabasePath: dir + "/cp.db", KVStorePath: dir + "/cp.bolt"}})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	return ls
}

// newRig starts a control plane on the store in dir. Starting a second one
// on the same dir after closing the first is a control plane restart.
func newRig(t *testing.T, dir string) *rig {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := &rig{t: t, store: openStore(t, dir)}
	// The lease, status and log re-entries are the router's; they are not
	// what these tests are about, so the router just answers them.
	r.hub = &Hub{Agents: r.store, Sweep: 10 * time.Millisecond,
		Router: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })}
	r.pipe = &Pipe{Store: r.store, Hub: r.hub, InternalToken: "itok", SteerWait: 2 * time.Second,
		Retention: func() time.Duration { return time.Hour },
		Now:       func() time.Time { return time.Now().Add(time.Duration(r.shift.Load())) }}
	r.hub.Pipe = r.pipe
	agenturl.SetLinkCarrier(r.hub)
	g := gin.New()
	g.GET("/api/v1/nodes/:node_id/link", r.hub.Handler())
	r.srv = httptest.NewServer(g)
	r.base = "ws" + strings.TrimPrefix(r.srv.URL, "http") + "/api/v1/nodes/"
	ctx := context.Background()
	for _, n := range []string{"codeaf-factory", "worker", "other", "steerbox"} {
		if a, _ := r.store.GetAgent(ctx, n); a == nil {
			require.NoError(t, r.store.RegisterAgent(ctx, &types.AgentNode{ID: n, Version: "1", LastHeartbeat: time.Now()}))
		}
	}
	t.Cleanup(r.close)
	return r
}

var closed sync.Map

func (r *rig) close() {
	if _, done := closed.LoadOrStore(r, true); done {
		return
	}
	r.pipe.Stop()
	r.hub.Close()
	r.srv.Close()
	_ = r.store.Close(context.Background())
	agenturl.SetLinkCarrier(nil)
}

// run makes a run owned by owner with executions (id → parent, "" root).
func (r *rig) run(runID, owner string, execs ...[2]string) {
	ctx := context.Background()
	for _, e := range execs {
		x := &types.Execution{ExecutionID: e[0], RunID: runID, AgentNodeID: "worker", NodeID: "worker", ReasonerID: "build",
			Status: types.ExecutionStatusRunning, StartedAt: time.Now().UTC(), CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
		if e[1] != "" {
			p := e[1]
			x.ParentExecutionID = &p
		}
		require.NoError(r.t, r.store.CreateExecutionRecord(ctx, x))
	}
	require.NoError(r.t, r.pipe.Own(ctx, runID, owner))
}

func (r *rig) held(exec string) int64 {
	n, err := r.store.RelayCount(context.Background(), exec)
	require.NoError(r.t, err)
	return n
}

func entries(seqs ...int64) []Entry {
	out := make([]Entry, 0, len(seqs))
	for _, s := range seqs {
		out = append(out, Entry{Seq: s, Kind: "line", Data: json.RawMessage(`{"type":"say","n":` + itoa(s) + `}`)})
	}
	return out
}

func itoa(n int64) string { b, _ := json.Marshal(n); return string(b) }

// ── A LINKED NODE, AS A TEST DRIVES IT ──────────────────────────────────────

type peer struct {
	t      *testing.T
	conn   *websocket.Conn
	wmu    sync.Mutex
	frames chan Frame
	seq    atomic.Int64
	// kept is what call read past while waiting for its ok, in order; the
	// readers take from it before the channel.
	kept []Frame
	// serve answers a deliver frame; nil answer sends no ack.
	serve func(f Frame, d DeliverBody) *Ack
}

func (r *rig) dial(node string, accepts []string, serve func(Frame, DeliverBody) *Ack) *peer {
	r.t.Helper()
	conn, _, err := websocket.DefaultDialer.Dial(r.base+node+"/link", nil)
	require.NoError(r.t, err)
	p := &peer{t: r.t, conn: conn, frames: make(chan Frame, 256), serve: serve}
	r.t.Cleanup(func() { _ = conn.Close() })
	go p.read()
	require.Equal(r.t, TWelcome, p.next(TWelcome).T)
	ok := p.call(Frame{T: THello}, Hello{InstanceID: node, Accepts: accepts})
	require.Equal(r.t, 200, ok.Code)
	return p
}

func (p *peer) read() {
	defer close(p.frames)
	for {
		_, raw, err := p.conn.ReadMessage()
		if err != nil {
			return
		}
		f, err := Decode(raw)
		if err != nil {
			continue
		}
		if f.T == TDeliver && p.serve != nil {
			var d DeliverBody
			_ = json.Unmarshal(f.Body, &d)
			if ack := p.serve(f, d); ack != nil {
				p.send(Frame{T: TAck, ID: f.ID, Exec: f.Exec}, ack)
			}
			continue
		}
		p.frames <- f
	}
}

func (p *peer) send(f Frame, body any) {
	raw, err := Encode(f, body)
	require.NoError(p.t, err)
	p.wmu.Lock()
	defer p.wmu.Unlock()
	require.NoError(p.t, p.conn.WriteMessage(websocket.TextMessage, raw))
}

// next is the next frame of type t, skipping others.
func (p *peer) take() (Frame, bool) {
	if len(p.kept) == 0 {
		return Frame{}, false
	}
	f := p.kept[0]
	p.kept = p.kept[1:]
	return f, true
}

func (p *peer) next(t string) Frame {
	p.t.Helper()
	for f, ok := p.take(); ok; f, ok = p.take() {
		if f.T == t {
			return f
		}
	}
	timeout := time.After(5 * time.Second)
	for {
		select {
		case f, ok := <-p.frames:
			require.True(p.t, ok, "link closed waiting for %s", t)
			if f.T == t {
				return f
			}
		case <-timeout:
			p.t.Fatalf("no %s frame", t)
		}
	}
}

// call sends f with a fresh seq and answers the ok on it.
func (p *peer) call(f Frame, body any) OKBody {
	p.t.Helper()
	f.Seq = p.seq.Add(1)
	p.send(f, body)
	timeout := time.After(5 * time.Second)
	for {
		select {
		case got := <-p.frames:
			if got.T == TOK && got.Seq == f.Seq {
				var ok OKBody
				require.NoError(p.t, json.Unmarshal(got.Body, &ok))
				return ok
			}
			if got.T != TOK {
				p.kept = append(p.kept, got) // not ours: the test reads it next
			}
		case <-timeout:
			p.t.Fatalf("no ok for %s", f.T)
		}
	}
}

// collect reads entries frames until it has seen seq through for exec, then
// keeps reading for quiet; it answers every seq it got, per execution, in
// the order they arrived.
func (p *peer) collect(through map[string]int64, quiet time.Duration) map[string][]int64 {
	p.t.Helper()
	got := map[string][]int64{}
	done := func() bool {
		for exec, want := range through {
			s := got[exec]
			if len(s) == 0 || s[len(s)-1] < want {
				return false
			}
		}
		return true
	}
	for f, ok := p.take(); ok; f, ok = p.take() {
		if f.T == TEntries {
			var d EntriesDown
			require.NoError(p.t, json.Unmarshal(f.Body, &d))
			for _, e := range d.Entries {
				got[f.Exec] = append(got[f.Exec], e.Seq)
			}
		}
	}
	deadline := time.After(5 * time.Second)
	for !done() {
		select {
		case f := <-p.frames:
			if f.T == TEntries {
				var d EntriesDown
				require.NoError(p.t, json.Unmarshal(f.Body, &d))
				for _, e := range d.Entries {
					got[f.Exec] = append(got[f.Exec], e.Seq)
				}
			}
		case <-deadline:
			p.t.Fatalf("entries never arrived: got %v want through %v", got, through)
		}
	}
	p.quietFor(quiet, func(f Frame) {
		if f.T == TEntries {
			var d EntriesDown
			_ = json.Unmarshal(f.Body, &d)
			for _, e := range d.Entries {
				got[f.Exec] = append(got[f.Exec], e.Seq)
			}
		}
	})
	return got
}

func (p *peer) quietFor(d time.Duration, each func(Frame)) {
	for f, ok := p.take(); ok; f, ok = p.take() {
		each(f)
	}
	end := time.After(d)
	for {
		select {
		case f := <-p.frames:
			each(f)
		case <-end:
			return
		}
	}
}

// ── EXECUTE AND DISPATCH REACH A LINK-ONLY NODE ─────────────────────────────

func TestCarrierReachesALinkOnlyNode(t *testing.T) {
	r := newRig(t, t.TempDir())
	// The Go SDK node: it takes the request itself (accepts http).
	r.dial("worker", []string{KindHTTP, KindSteer, KindCancel}, func(f Frame, d DeliverBody) *Ack {
		if d.Kind != KindHTTP || d.Method != http.MethodPost || d.Path != "/reasoners/build" || string(d.Body) != `{"job":62}` {
			return &Ack{Outcome: OutcomeRejected, Error: "unexpected " + d.Kind + " " + d.Path}
		}
		res, _ := json.Marshal(HTTPResult{Code: 200, Body: []byte(`{"built":true}`)})
		return &Ack{Outcome: OutcomeAccepted, Result: res}
	})
	// The codeaf factory node: it speaks only dispatch and cancel.
	r.dial("codeaf-factory", nil, func(f Frame, d DeliverBody) *Ack {
		if d.Kind != KindDispatch || d.Reasoner != "intake" || d.Headers["X-Execution-ID"] != "exec_f" {
			return &Ack{Outcome: OutcomeRejected, Error: "unexpected " + d.Kind}
		}
		return &Ack{Outcome: OutcomeDone, Result: json.RawMessage(`{"filed":7}`)}
	})

	post := func(node, reasoner, execID, body string) (*http.Response, error) {
		u, err := agenturl.ForNode(node, "", "reasoners", reasoner) // no base_url
		require.NoError(t, err)
		require.Equal(t, node+".nodelink", u.Host)
		req, _ := http.NewRequest(http.MethodPost, u.String(), strings.NewReader(body))
		req.Header.Set("X-Execution-ID", execID)
		return agenturl.NewClient(5 * time.Second).Do(req)
	}
	resp, err := post("worker", "build", "exec_w", `{"job":62}`)
	require.NoError(t, err)
	raw, _ := io.ReadAll(resp.Body)
	require.Equal(t, 200, resp.StatusCode)
	require.JSONEq(t, `{"built":true}`, string(raw))

	resp, err = post("codeaf-factory", "intake", "exec_f", `{}`)
	require.NoError(t, err)
	raw, _ = io.ReadAll(resp.Body)
	require.Equal(t, 200, resp.StatusCode)
	require.JSONEq(t, `{"filed":7}`, string(raw))
	require.Zero(t, r.hub.Pending("worker"))
	require.Zero(t, r.hub.Pending("codeaf-factory"))

	// No link at all: the call fails as an unreachable node does, it does not hang.
	old := LinkWait
	LinkWait = 100 * time.Millisecond
	t.Cleanup(func() { LinkWait = old })
	_, err = post("other", "build", "exec_o", `{}`)
	require.ErrorIs(t, err, ErrNoLink)
}

func TestDispatchAckReadsAsTheHTTPAnswer(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "http://n.nodelink/reasoners/x", nil)
	require.Equal(t, 202, dispatchResponse(req, Outcome{Outcome: OutcomeAccepted}).StatusCode)
	require.Equal(t, 202, dispatchResponse(req, Outcome{Outcome: OutcomeDuplicate}).StatusCode)
	require.Equal(t, 422, dispatchResponse(req, Outcome{Outcome: OutcomeRejected, Error: "no"}).StatusCode)
	_, ok := reasonerPath(httptest.NewRequest(http.MethodGet, "http://n.nodelink/reasoners/x", nil))
	require.False(t, ok, "only a POST to a reasoner is a dispatch")
}

// ── STORE AND FORWARD ───────────────────────────────────────────────────────

func TestEntriesDedupeOnExecutionAndSeq(t *testing.T) {
	r := newRig(t, t.TempDir())
	r.run("run_1", "codeaf-factory", [2]string{"exec_1", ""})
	ctx := context.Background()

	stored, through, err := r.pipe.Put(ctx, "exec_1", entries(1, 2, 3))
	require.NoError(t, err)
	require.Equal(t, 3, stored)
	require.EqualValues(t, 3, through)

	// A replay of an unanswered batch, overlapping what is held.
	stored, through, err = r.pipe.Put(ctx, "exec_1", entries(2, 3, 4))
	require.NoError(t, err)
	require.Equal(t, 1, stored)
	require.EqualValues(t, 4, through)

	// The same over the link: entries up-frame, answered with through.
	node := r.dial("worker", []string{KindHTTP}, nil)
	ok := node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: entries(4, 5)})
	require.Equal(t, 200, ok.Code)
	require.EqualValues(t, 5, ok.Through)
	require.EqualValues(t, 5, r.held("exec_1"))

	rows, err := r.store.RelayAfter(ctx, "exec_1", 0, 100)
	require.NoError(t, err)
	for i, e := range rows {
		require.EqualValues(t, i+1, e.Seq)
		require.JSONEq(t, `{"type":"say","n":`+itoa(e.Seq)+`}`, string(e.Data), "the node's bytes pass verbatim")
	}

	bad := node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: []Entry{{Seq: 0, Kind: "line"}}})
	require.Equal(t, 400, bad.Code)
}

func TestAckDeletesTheTranscriptAndKeepsTheControlData(t *testing.T) {
	r := newRig(t, t.TempDir())
	r.run("run_1", "codeaf-factory", [2]string{"exec_1", ""})
	owner := r.dial("codeaf-factory", nil, nil)
	require.Equal(t, 200, owner.call(Frame{T: TSubscribe}, Subscribe{}).Code)
	node := r.dial("worker", []string{KindHTTP}, nil)

	require.Equal(t, 200, node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: entries(1, 2, 3)}).Code)
	got := owner.collect(map[string]int64{"exec_1": 3}, 50*time.Millisecond)
	require.Equal(t, []int64{1, 2, 3}, got["exec_1"])
	require.EqualValues(t, 3, r.held("exec_1"), "held until the owner acks")

	ok := owner.call(Frame{T: TEntriesAck, Exec: "exec_1"}, EntriesAck{Through: 2})
	require.Equal(t, 200, ok.Code)
	require.EqualValues(t, 2, ok.Dropped)
	require.EqualValues(t, 1, r.held("exec_1"))
	ok = owner.call(Frame{T: TEntriesAck, Exec: "exec_1"}, EntriesAck{Through: 3})
	require.EqualValues(t, 1, ok.Dropped)

	// NO TRANSCRIPT ROW REMAINS ONCE ACKED, and a late replay of an acked
	// seq is not held again.
	require.Zero(t, r.held(""))
	require.Equal(t, 200, node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: entries(2, 3)}).Code)
	require.Zero(t, r.held(""))

	// The control data stays.
	_, err := r.store.UpdateExecutionRecord(context.Background(), "exec_1", func(e *types.Execution) (*types.Execution, error) {
		now := time.Now().UTC()
		e.Status, e.CompletedAt, e.ResultPayload = types.ExecutionStatusSucceeded, &now, json.RawMessage(`{"ok":true}`)
		return e, nil
	})
	require.NoError(t, err)
	rec, err := r.store.GetExecutionRecord(context.Background(), "exec_1")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusSucceeded, rec.Status)
	require.JSONEq(t, `{"ok":true}`, string(rec.ResultPayload))
	require.NotNil(t, rec.CompletedAt)
}

func TestRetentionIsTheCeiling(t *testing.T) {
	r := newRig(t, t.TempDir())
	ctx := context.Background()
	r.run("run_1", "codeaf-factory", [2]string{"exec_old", ""}, [2]string{"exec_new", ""})

	// Nobody subscribed: entries wait, inside the ceiling.
	_, _, err := r.pipe.Put(ctx, "exec_old", entries(1, 2))
	require.NoError(t, err)
	_, _, err = r.pipe.Put(ctx, "exec_new", entries(1))
	require.NoError(t, err)
	require.EqualValues(t, 2, r.held("exec_old"))

	// The retention period (1h) passes with nobody acking: the next entry in
	// prunes everything older than the ceiling, acked or not.
	r.shift.Store(int64(61 * time.Minute))
	_, _, err = r.pipe.Put(ctx, "exec_new", entries(2))
	require.NoError(t, err)
	require.Zero(t, r.held("exec_old"))
}

// ── SUBSCRIPTION ────────────────────────────────────────────────────────────

func TestSubscriberResumesAfterReconnect(t *testing.T) {
	r := newRig(t, t.TempDir())
	r.run("run_1", "codeaf-factory", [2]string{"exec_1", ""})
	ctx := context.Background()

	owner := r.dial("codeaf-factory", nil, nil)
	owner.call(Frame{T: TSubscribe}, Subscribe{})
	_, _, err := r.pipe.Put(ctx, "exec_1", entries(1, 2))
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2}, owner.collect(map[string]int64{"exec_1": 2}, 20*time.Millisecond)["exec_1"])
	owner.call(Frame{T: TEntriesAck, Exec: "exec_1"}, EntriesAck{Through: 1})

	// WorkAF goes away; the run keeps working.
	_ = owner.conn.Close()
	waitFor(t, "the owner's link to drop", func() bool { return !r.hub.Live("codeaf-factory") })
	_, _, err = r.pipe.Put(ctx, "exec_1", entries(3))
	require.NoError(t, err)

	// Back: it is sent everything it has not acked — 2 (sent, never acked) and 3.
	owner = r.dial("codeaf-factory", nil, nil)
	owner.call(Frame{T: TSubscribe}, Subscribe{})
	require.Equal(t, []int64{2, 3}, owner.collect(map[string]int64{"exec_1": 3}, 50*time.Millisecond)["exec_1"])

	// Back again, saying it already keeps 3: nothing is resent, new ones flow.
	_ = owner.conn.Close()
	waitFor(t, "the owner's link to drop", func() bool { return !r.hub.Live("codeaf-factory") })
	owner = r.dial("codeaf-factory", nil, nil)
	owner.call(Frame{T: TSubscribe}, Subscribe{After: map[string]int64{"exec_1": 3}})
	_, _, err = r.pipe.Put(ctx, "exec_1", entries(4))
	require.NoError(t, err)
	require.Equal(t, []int64{4}, owner.collect(map[string]int64{"exec_1": 4}, 50*time.Millisecond)["exec_1"])
}

func TestControlPlaneRestartLosesNothingAndDuplicatesNothing(t *testing.T) {
	dir := t.TempDir()
	r := newRig(t, dir)
	r.run("run_1", "codeaf-factory", [2]string{"exec_1", ""})
	owner := r.dial("codeaf-factory", nil, nil)
	owner.call(Frame{T: TSubscribe}, Subscribe{})
	node := r.dial("worker", []string{KindHTTP}, nil)
	require.Equal(t, 200, node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: entries(1, 2, 3)}).Code)
	require.Equal(t, []int64{1, 2, 3}, owner.collect(map[string]int64{"exec_1": 3}, 20*time.Millisecond)["exec_1"])
	owner.call(Frame{T: TEntriesAck, Exec: "exec_1"}, EntriesAck{Through: 1})

	// The control plane dies. The node never heard an answer for 2 and 3 (say),
	// so it replays them with the one it made meanwhile.
	r.close()
	r = newRig(t, dir)
	node = r.dial("worker", []string{KindHTTP}, nil)
	require.Equal(t, 200, node.call(Frame{T: TEntries, Exec: "exec_1"}, EntriesUp{Entries: entries(2, 3, 4)}).Code)
	require.EqualValues(t, 3, r.held("exec_1"), "2 and 3 were held across the restart; 4 is new; none twice")

	owner = r.dial("codeaf-factory", nil, nil)
	owner.call(Frame{T: TSubscribe}, Subscribe{})
	got := owner.collect(map[string]int64{"exec_1": 4}, 100*time.Millisecond)
	require.Equal(t, []int64{2, 3, 4}, got["exec_1"], "each unacked entry exactly once")

	owner.call(Frame{T: TEntriesAck, Exec: "exec_1"}, EntriesAck{Through: 4})
	require.Zero(t, r.held(""))
	rec, err := r.store.GetExecutionRecord(context.Background(), "exec_1")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, rec.Status, "the control data survived the restart")
}

func TestSubscriptionIsScopedToTheRunsTheNodeOwns(t *testing.T) {
	r := newRig(t, t.TempDir())
	ctx := context.Background()
	// WorkAF's run: a root and a nested call (a descendant in the tree).
	r.run("run_mine", "codeaf-factory", [2]string{"exec_root", ""}, [2]string{"exec_child", "exec_root"})
	// Somebody else's run on the same worker.
	r.run("run_theirs", "other", [2]string{"exec_theirs", ""})
	// A run with no owner at all.
	r.run("run_nobody", "", [2]string{"exec_nobody", ""})

	mine := r.dial("codeaf-factory", nil, nil)
	mine.call(Frame{T: TSubscribe}, Subscribe{})
	snap := mine.next(TRun)
	var state RunState
	require.NoError(t, json.Unmarshal(snap.Body, &state))
	require.Equal(t, "run_mine", state.RunID)
	require.True(t, state.Full)
	ids := []string{}
	for _, e := range state.Executions {
		ids = append(ids, e.ExecutionID)
		if e.ExecutionID == "exec_child" {
			require.Equal(t, "exec_root", e.ParentExecutionID)
		}
	}
	require.ElementsMatch(t, []string{"exec_root", "exec_child"}, ids)

	theirs := r.dial("other", nil, nil)
	theirs.call(Frame{T: TSubscribe}, Subscribe{})
	_ = theirs.next(TRun) // its own snapshot

	for _, exec := range []string{"exec_child", "exec_theirs", "exec_nobody"} {
		_, _, err := r.pipe.Put(ctx, exec, entries(1, 2))
		require.NoError(t, err)
	}
	got := mine.collect(map[string]int64{"exec_child": 2}, 150*time.Millisecond)
	require.Equal(t, map[string][]int64{"exec_child": {1, 2}}, got, "WorkAF never receives another run's entries")
	require.Equal(t, map[string][]int64{"exec_theirs": {1, 2}}, theirs.collect(map[string]int64{"exec_theirs": 2}, 50*time.Millisecond))

	// Nor may it read, ack or steer them.
	ok := mine.call(Frame{T: TEntriesAck, Exec: "exec_theirs"}, EntriesAck{Through: 2})
	require.Equal(t, 403, ok.Code)
	require.EqualValues(t, 2, r.held("exec_theirs"))
	_, err := r.pipe.Read(ctx, "codeaf-factory", "exec_theirs", 0, 10)
	require.Equal(t, 403, CodeOf(err))
	ok = mine.call(Frame{T: TSteer, Exec: "exec_theirs"}, SteerUp{Message: "hi"})
	require.Equal(t, 403, ok.Code)

	// Tree and status changes are pushed the same way, only to the owner.
	r.pipe.Start()
	_, err = r.store.UpdateExecutionRecord(ctx, "exec_child", func(e *types.Execution) (*types.Execution, error) {
		now := time.Now().UTC()
		e.Status, e.CompletedAt = types.ExecutionStatusSucceeded, &now
		return e, nil
	})
	require.NoError(t, err)
	events.PublishExecutionCompleted("exec_child", "run_mine", "worker", nil)
	events.PublishExecutionUpdated("exec_theirs", "run_theirs", "worker", "running", nil)
	f := mine.next(TRun)
	state = RunState{}
	require.NoError(t, json.Unmarshal(f.Body, &state))
	require.Equal(t, "run_mine", state.RunID)
	require.False(t, state.Full)
	require.Len(t, state.Executions, 1)
	require.Equal(t, "exec_child", state.Executions[0].ExecutionID)
	require.Equal(t, types.ExecutionStatusSucceeded, state.Executions[0].Status)
	require.NotEmpty(t, state.Executions[0].CompletedAt)
	mine.quietFor(100*time.Millisecond, func(f Frame) {
		if f.T == TRun {
			var s RunState
			_ = json.Unmarshal(f.Body, &s)
			require.Equal(t, "run_mine", s.RunID, "another run's tree reached WorkAF")
		}
	})
}

// ── STEER ───────────────────────────────────────────────────────────────────

func TestSteerIsAnsweredOnlyWhenTheNodeTookIt(t *testing.T) {
	r := newRig(t, t.TempDir())
	ctx := context.Background()
	r.run("run_1", "codeaf-factory", [2]string{"exec_codeaf", ""}, [2]string{"exec_plain", ""}, [2]string{"exec_done", ""})
	_, err := r.store.UpdateExecutionRecord(ctx, "exec_done", func(e *types.Execution) (*types.Execution, error) {
		e.Status = types.ExecutionStatusSucceeded
		return e, nil
	})
	require.NoError(t, err)

	var mu sync.Mutex
	var notes []string
	hold := make(chan struct{})
	node := r.dial("worker", []string{KindHTTP, KindSteer, KindCancel}, func(f Frame, d DeliverBody) *Ack {
		var body struct{ Message string }
		_ = json.Unmarshal(d.Body, &body)
		answer := func(code int, raw string) *Ack {
			res, _ := json.Marshal(HTTPResult{Code: code, Body: []byte(raw)})
			return &Ack{Outcome: OutcomeAccepted, Result: res}
		}
		if d.Headers["Authorization"] != "Bearer itok" {
			return answer(401, `{"error":"unauthorized"}`)
		}
		switch d.Path {
		case "/_internal/executions/exec_codeaf/steer":
			mu.Lock()
			notes = append(notes, body.Message)
			mu.Unlock()
			return answer(200, `{"ok":true}`)
		case "/_internal/executions/exec_plain/steer":
			return answer(409, `{"error":"this execution is not running the codeaf harness, so it cannot be steered"}`)
		case "/_internal/executions/exec_slow/steer":
			<-hold
			return nil
		}
		return answer(404, `{}`)
	})
	_ = node
	owner := r.dial("codeaf-factory", nil, nil)

	ok := owner.call(Frame{T: TSteer, Exec: "exec_codeaf"}, SteerUp{Message: "use tabs"})
	require.Equal(t, 200, ok.Code, ok.Error)
	mu.Lock()
	require.Equal(t, []string{"use tabs"}, notes, "ok means the node's mailbox has it")
	mu.Unlock()

	ok = owner.call(Frame{T: TSteer, Exec: "exec_plain"}, SteerUp{Message: "x"})
	require.Equal(t, 409, ok.Code)
	require.Equal(t, "this execution is not running the codeaf harness, so it cannot be steered", ok.Error)

	ok = owner.call(Frame{T: TSteer, Exec: "exec_done"}, SteerUp{Message: "x"})
	require.Equal(t, 409, ok.Code)
	require.Equal(t, "this step has ended", ok.Error)

	ok = owner.call(Frame{T: TSteer, Exec: "exec_codeaf"}, SteerUp{Message: ""})
	require.Equal(t, 400, ok.Code)

	// A node that never answers: refused in plain words when the wait ends.
	r.run("run_slow", "codeaf-factory", [2]string{"exec_slow", ""})
	r.pipe.SteerWait = 200 * time.Millisecond
	ok = owner.call(Frame{T: TSteer, Exec: "exec_slow"}, SteerUp{Message: "x"})
	close(hold)
	require.Equal(t, 409, ok.Code)
	require.Equal(t, "the node did not take the note: it did not answer in time", ok.Error)
	r.pipe.SteerWait = 2 * time.Second

	// The node is gone (no link, no base_url): refused at once, in plain words.
	_ = node.conn.Close()
	waitFor(t, "the worker's link to drop", func() bool { return !r.hub.Live("worker") })
	start := time.Now()
	ok = owner.call(Frame{T: TSteer, Exec: "exec_codeaf"}, SteerUp{Message: "x"})
	require.Equal(t, 409, ok.Code)
	require.Equal(t, "the node running this step is not connected to the control plane, so the note cannot reach it", ok.Error)
	require.Less(t, time.Since(start), time.Second, "a gone node is refused without waiting")
}

func TestSteerToANodeThatTakesOnlySteerDeliveries(t *testing.T) {
	r := newRig(t, t.TempDir())
	ctx := context.Background()
	x := &types.Execution{ExecutionID: "exec_s", RunID: "run_s", AgentNodeID: "steerbox", NodeID: "steerbox", ReasonerID: "r",
		Status: types.ExecutionStatusRunning, StartedAt: time.Now(), CreatedAt: time.Now(), UpdatedAt: time.Now()}
	require.NoError(t, r.store.CreateExecutionRecord(ctx, x))
	require.NoError(t, r.pipe.Own(ctx, "run_s", "codeaf-factory"))
	var took atomic.Int32
	r.dial("steerbox", []string{KindSteer, KindCancel}, func(f Frame, d DeliverBody) *Ack {
		if d.Kind != KindSteer || f.Exec != "exec_s" {
			return &Ack{Outcome: OutcomeRejected, Error: "unexpected"}
		}
		if took.Add(1) == 1 {
			return &Ack{Outcome: OutcomeAccepted}
		}
		return &Ack{Outcome: OutcomeRejected, Error: "this execution is not running the codeaf harness, so it cannot be steered"}
	})
	via, err := r.pipe.Steer(ctx, "codeaf-factory", "exec_s", "first")
	require.NoError(t, err)
	require.Equal(t, "link", via)
	_, err = r.pipe.Steer(ctx, "codeaf-factory", "exec_s", "second")
	require.Equal(t, 409, CodeOf(err))
	require.Equal(t, "this execution is not running the codeaf harness, so it cannot be steered", err.Error())
	_, err = r.pipe.Steer(ctx, "other", "exec_s", "not yours")
	require.Equal(t, 403, CodeOf(err))
}
