package nodelink

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ── THE PIPE: HARNESS ENTRIES PASS THROUGH, CONTROL DATA STAYS ──────────────
//
// The node that runs an execution sends its harness entries up (an entries
// frame on its link, or POST /executions/:id/relay), numbered per execution
// from 1. The control plane holds them in execution_relay only until the
// run's subscriber — the node that owns the run, WorkAF's codeaf-factory —
// acks them, then deletes them. WORKAF IS THE SYSTEM OF RECORD; THE CONTROL
// PLANE IS A PIPE. The execution-log retention setting is the ceiling: an
// entry nobody acked within it is dropped, so an offline subscriber can catch
// up while it is gone and the table cannot grow without bound.
//
// IDEMPOTENT ON (execution, seq). A resend of a seq already held is ignored;
// a resend of a seq already acked (and so deleted) is ignored too, because
// the pipe remembers each execution's acked high-water mark in memory and
// refuses anything at or below it. After a control plane restart that memory
// is gone, so a replay of an acked seq would be held once more and sent once
// more: the subscriber's own store dedupes on (execution, seq), which is why
// that is safe, and the node replays only what the control plane never
// answered, so in practice nothing is resent.
//
// SCOPE. A subscriber is sent the entries and the tree of the runs it owns
// (storage: run_owners), every descendant execution included, because every
// execution of a run shares its run_id. Nothing of another run ever goes
// down its link.
//
// The tree and the status are pushed the same way: every execution event is
// read back from the store and sent as a run frame to the run's owner, and a
// subscribe is answered with a full snapshot of every run the subscriber
// owns that is open, has entries held, or that it names.

// PipeStore is what the pipe reads and writes; LocalStorage is one.
type PipeStore interface {
	RelayPut(ctx context.Context, executionID string, entries []storage.RelayEntry) (int, error)
	RelayAfter(ctx context.Context, executionID string, after int64, limit int) ([]storage.RelayEntry, error)
	RelayAck(ctx context.Context, executionID string, through int64) (int64, error)
	RelayPrune(ctx context.Context, olderThan time.Time) error
	RelayCount(ctx context.Context, executionID string) (int64, error)
	RelayExecutions(ctx context.Context) ([]string, error)
	SetRunOwner(ctx context.Context, runID, owner string) error
	RunOwner(ctx context.Context, runID string) (string, error)
	RunsOwnedBy(ctx context.Context, owner string, since time.Time) ([]string, error)
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
}

// Pipe is the store-and-forward between nodes and run owners.
type Pipe struct {
	Store PipeStore
	Hub   *Hub
	// Retention is the ceiling an unacked entry is held for; zero is 24h.
	Retention func() time.Duration
	// InternalToken is the bearer a steer carries to the node.
	InternalToken string
	// SteerWait is how long a steer waits for the node to take it (15s).
	SteerWait time.Duration
	// PushBatch is the most entries one down frame carries (200).
	PushBatch int
	// Now is the clock the retention ceiling is read against; tests move it.
	Now func() time.Time

	mu        sync.Mutex
	subs      map[string]*subscriber // owner node → its live subscription
	acked     map[string]int64       // execution → highest seq acked since start
	owner     map[string]string      // execution → run owner, once known
	runOf     map[string]string      // execution → run id
	lastPrune time.Time              // when RelayPrune last ran on this pipe
	stop      chan struct{}
	once      sync.Once
}

type subscriber struct {
	node string
	l    *link
	mu   sync.Mutex       // one push at a time, so entries go down in order
	sent map[string]int64 // execution → highest seq sent down this link
}

// PipeError is a refusal with the status its HTTP door answers and the words
// a person reads.
type PipeError struct {
	Code  int
	Words string
}

func (e *PipeError) Error() string { return e.Words }

func refuse(code int, words string) error { return &PipeError{Code: code, Words: words} }

// CodeOf is err's HTTP status: a PipeError's own, else 500.
func CodeOf(err error) int {
	var pe *PipeError
	if errors.As(err, &pe) {
		return pe.Code
	}
	return http.StatusInternalServerError
}

func (p *Pipe) init() {
	p.once.Do(func() {
		p.subs = map[string]*subscriber{}
		p.acked = map[string]int64{}
		p.owner = map[string]string{}
		p.runOf = map[string]string{}
		p.stop = make(chan struct{})
		if p.SteerWait <= 0 {
			p.SteerWait = 15 * time.Second
		}
		if p.PushBatch <= 0 {
			p.PushBatch = 200
		}
	})
}

func (p *Pipe) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}

func (p *Pipe) retention() time.Duration {
	if p.Retention != nil {
		if d := p.Retention(); d > 0 {
			return d
		}
	}
	return 24 * time.Hour
}

// pruneIfDue prunes the relay to the retention ceiling at most once per minute
// on p's clock, so a burst of Puts does not sweep the table on every batch.
func (p *Pipe) pruneIfDue(ctx context.Context) {
	now := p.now()
	p.mu.Lock()
	if !p.lastPrune.IsZero() && now.Sub(p.lastPrune) < time.Minute {
		p.mu.Unlock()
		return
	}
	p.lastPrune = now
	p.mu.Unlock()
	_ = p.Store.RelayPrune(ctx, now.Add(-p.retention()))
}

// Start follows the execution event bus, so tree and status changes reach
// owners as they happen, and prunes the relay to the retention ceiling once a
// minute. Stop ends both.
func (p *Pipe) Start() {
	p.init()
	id := "nodelink-pipe-" + newID("")
	ch := events.GlobalExecutionEventBus.Subscribe(id)
	go func() {
		defer events.GlobalExecutionEventBus.Unsubscribe(id)
		for {
			select {
			case <-p.stop:
				return
			case ev, ok := <-ch:
				if !ok {
					return
				}
				if ev.Type == events.ExecutionLogEntry || ev.ExecutionID == "" {
					continue
				}
				p.Changed(context.Background(), ev.ExecutionID)
			}
		}
	}()
	go func() {
		t := time.NewTicker(time.Minute)
		defer t.Stop()
		for {
			select {
			case <-p.stop:
				return
			case <-t.C:
				p.pruneIfDue(context.Background())
			}
		}
	}()
}

// Stop ends what Start began.
func (p *Pipe) Stop() {
	p.init()
	p.mu.Lock()
	defer p.mu.Unlock()
	select {
	case <-p.stop:
	default:
		close(p.stop)
	}
}

// ── IN: ENTRIES FROM THE NODE ───────────────────────────────────────────────

// Put holds entries for exec and pushes them on to the run's owner. It
// answers how many were new and the highest seq the node may now forget; when
// exec's run has no owner nobody will ever ack them, so nothing is stored and
// through = max seq lets the node drop the batch without retrying.
func (p *Pipe) Put(ctx context.Context, exec string, entries []Entry) (int, int64, error) {
	p.init()
	if strings.TrimSpace(exec) == "" {
		return 0, 0, refuse(http.StatusBadRequest, "an entries frame names its execution")
	}
	if len(entries) == 0 {
		return 0, 0, refuse(http.StatusBadRequest, "entries: [{seq, kind, data}] is required")
	}
	p.mu.Lock()
	floor := p.acked[exec]
	p.mu.Unlock()
	var through int64
	rows := make([]storage.RelayEntry, 0, len(entries))
	for _, e := range entries {
		if e.Seq <= 0 || strings.TrimSpace(e.Kind) == "" {
			return 0, 0, refuse(http.StatusBadRequest, "every entry has a seq from 1 and a kind")
		}
		if e.Seq > through {
			through = e.Seq
		}
		if e.Seq <= floor {
			continue // already acked by the owner and deleted: a replay
		}
		rows = append(rows, storage.RelayEntry{Seq: e.Seq, Kind: e.Kind, Data: e.Data})
	}
	_, owner, err := p.ownerOf(ctx, exec)
	if err != nil {
		return 0, 0, err
	}
	if owner == "" {
		p.pruneIfDue(ctx)
		return 0, through, nil
	}
	stored := 0
	if len(rows) > 0 {
		n, err := p.Store.RelayPut(ctx, exec, rows)
		if err != nil {
			return 0, 0, err
		}
		stored = n
	}
	p.pruneIfDue(ctx)
	if stored > 0 {
		go p.forward(context.WithoutCancel(ctx), exec)
	}
	return stored, through, nil
}

// ── OWNERSHIP ───────────────────────────────────────────────────────────────

// Own records owner as runID's owner unless it has one.
func (p *Pipe) Own(ctx context.Context, runID, owner string) error {
	return p.Store.SetRunOwner(ctx, runID, owner)
}

// ownerOf is exec's run and that run's owner ("" for none yet).
func (p *Pipe) ownerOf(ctx context.Context, exec string) (string, string, error) {
	p.init()
	p.mu.Lock()
	run, o := p.runOf[exec], p.owner[exec]
	p.mu.Unlock()
	if o != "" {
		return run, o, nil
	}
	if run == "" {
		rec, err := p.Store.GetExecutionRecord(ctx, exec)
		if err != nil {
			return "", "", err
		}
		if rec == nil {
			return "", "", nil
		}
		run = rec.RunID
	}
	o, err := p.Store.RunOwner(ctx, run)
	if err != nil {
		return run, "", err
	}
	p.mu.Lock()
	p.runOf[exec] = run
	if o != "" {
		p.owner[exec] = o // an owner never moves, so it is safe to keep
	}
	p.mu.Unlock()
	return run, o, nil
}

// ── OUT: TO THE OWNER'S LINK ────────────────────────────────────────────────

func (p *Pipe) sub(node string) *subscriber {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.subs[node]
}

// forward sends exec's held entries the owner has not been sent yet.
func (p *Pipe) forward(ctx context.Context, exec string) {
	run, owner, err := p.ownerOf(ctx, exec)
	if err != nil || owner == "" {
		return // held until an owner subscribes, or the ceiling drops it
	}
	s := p.sub(owner)
	if s == nil {
		return
	}
	p.push(ctx, s, run, exec)
}

func (p *Pipe) push(ctx context.Context, s *subscriber, run, exec string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if p.sub(s.node) != s {
		return // replaced by a newer subscription
	}
	for {
		after := s.sent[exec]
		rows, err := p.Store.RelayAfter(ctx, exec, after, p.PushBatch)
		if err != nil || len(rows) == 0 {
			return
		}
		down := EntriesDown{RunID: run, Entries: make([]Entry, 0, len(rows))}
		for _, r := range rows {
			down.Entries = append(down.Entries, Entry{Seq: r.Seq, Kind: r.Kind, Data: r.Data, At: r.At.UTC().Format(time.RFC3339Nano)})
		}
		if err := s.l.send(Frame{T: TEntries, Exec: exec, Seq: rows[len(rows)-1].Seq}, down); err != nil {
			logger.Logger.Debug().Err(err).Str("node_id", s.node).Str("execution_id", exec).Msg("nodelink: entries push failed; resent on resubscribe")
			return
		}
		s.sent[exec] = rows[len(rows)-1].Seq
		if len(rows) < p.PushBatch {
			return
		}
	}
}

// Changed pushes exec's current control data to its run's owner.
func (p *Pipe) Changed(ctx context.Context, exec string) {
	run, owner, err := p.ownerOf(ctx, exec)
	if err != nil || owner == "" {
		return
	}
	s := p.sub(owner)
	if s == nil {
		return
	}
	rec, err := p.Store.GetExecutionRecord(ctx, exec)
	if err != nil || rec == nil {
		return
	}
	_ = s.l.send(Frame{T: TRun, Exec: exec}, RunState{RunID: run, Owner: owner, Executions: []ExecState{p.stateWithApproval(ctx, rec)}})
}

// maxResult is the largest result a run frame carries; a larger one is
// flagged result_omitted and read from GET /executions/:id.
const maxResult = 256 << 10

// fitApprovalBodies drops approval bodies from a snapshot that would not fit
// one frame, flagging each so the owner reads it from approval-status. One
// execution's body is capped at ask time, so a single-execution frame always
// fits; only a run with several waiting at once can need this.
func fitApprovalBodies(state *RunState) {
	raw, err := json.Marshal(state)
	if err == nil && len(raw) < MaxFrame-(64<<10) {
		return
	}
	for i := range state.Executions {
		if a := state.Executions[i].Approval; a != nil && a.Body != "" {
			a.Body, a.BodyOmitted = "", true
		}
	}
}

// approvalStore is the part of a store that can say what a waiting
// execution is asking. A store without it still pushes status and reason.
type approvalStore interface {
	GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error)
	GetApprovalQuestion(ctx context.Context, executionID, approvalRequestID string) (*types.ApprovalQuestion, error)
}

// stateWithApproval is stateOf plus the pending approval's question, so the
// run's owner learns from the one frame that a person is needed and what
// they are asked.
func (p *Pipe) stateWithApproval(ctx context.Context, e *types.Execution) ExecState {
	s := stateOf(e)
	if s.Status != types.ExecutionStatusWaiting {
		return s
	}
	as, ok := p.Store.(approvalStore)
	if !ok {
		return s
	}
	wf, err := as.GetWorkflowExecution(ctx, e.ExecutionID)
	if err != nil || wf == nil || wf.ApprovalRequestID == nil || *wf.ApprovalRequestID == "" {
		return s
	}
	if wf.ApprovalStatus == nil || *wf.ApprovalStatus != "pending" {
		return s
	}
	a := &ApprovalState{ID: *wf.ApprovalRequestID}
	if wf.ApprovalExpiresAt != nil {
		a.ExpiresAt = wf.ApprovalExpiresAt.UTC().Format(time.RFC3339)
	}
	if q, err := as.GetApprovalQuestion(ctx, e.ExecutionID, a.ID); err == nil && q != nil {
		a.Kind, a.Question, a.Body, a.Options = q.Kind, q.Question, q.Body, q.Options
	}
	s.Approval = a
	return s
}

func stateOf(e *types.Execution) ExecState {
	s := ExecState{ExecutionID: e.ExecutionID, AgentNodeID: e.AgentNodeID, ReasonerID: e.ReasonerID, Status: e.Status}
	if e.StatusReason != nil {
		s.StatusReason = *e.StatusReason
	}
	if e.ParentExecutionID != nil {
		s.ParentExecutionID = *e.ParentExecutionID
	}
	if !e.StartedAt.IsZero() {
		s.StartedAt = e.StartedAt.UTC().Format(time.RFC3339Nano)
	}
	if e.CompletedAt != nil {
		s.CompletedAt = e.CompletedAt.UTC().Format(time.RFC3339Nano)
	}
	if e.DurationMS != nil {
		s.DurationMS = *e.DurationMS
	}
	if e.ErrorMessage != nil {
		s.Error = *e.ErrorMessage
	}
	if len(e.ResultPayload) > 0 && json.Valid(e.ResultPayload) {
		if len(e.ResultPayload) > maxResult {
			s.ResultOmitted = true
		} else {
			s.Result = e.ResultPayload
		}
	}
	return s
}

// ── THE SUBSCRIBER'S SIDE ───────────────────────────────────────────────────

// Subscribe makes l's node a subscriber, replacing any older subscription it
// held, then sends it a snapshot of its runs and every entry held for them.
func (p *Pipe) Subscribe(ctx context.Context, l *link, req Subscribe) {
	p.init()
	s := &subscriber{node: l.node, l: l, sent: map[string]int64{}}
	for exec, seq := range req.After {
		s.sent[exec] = seq
	}
	p.mu.Lock()
	p.subs[l.node] = s
	p.mu.Unlock()
	go p.catchUp(context.WithoutCancel(ctx), s, req.Runs)
}

// unsubscribe drops l's subscription when its link goes.
func (p *Pipe) unsubscribe(l *link) {
	p.init()
	p.mu.Lock()
	defer p.mu.Unlock()
	if s := p.subs[l.node]; s != nil && s.l == l {
		delete(p.subs, l.node)
	}
}

func (p *Pipe) catchUp(ctx context.Context, s *subscriber, named []string) {
	held, _ := p.Store.RelayExecutions(ctx)
	heldRun := map[string][]string{}
	for _, exec := range held {
		run, owner, err := p.ownerOf(ctx, exec)
		if err == nil && owner == s.node {
			heldRun[run] = append(heldRun[run], exec)
		}
	}
	want := map[string]bool{}
	for _, r := range named {
		want[r] = true
	}
	owned, _ := p.Store.RunsOwnedBy(ctx, s.node, p.now().Add(-p.retention()))
	for _, run := range owned {
		execs, err := p.Store.QueryExecutionRecords(ctx, types.ExecutionFilter{RunID: &run})
		if err != nil || len(execs) == 0 {
			continue
		}
		open := false
		for _, e := range execs {
			if !types.IsTerminalExecutionStatus(e.Status) {
				open = true
			}
		}
		if !open && !want[run] && len(heldRun[run]) == 0 {
			continue
		}
		state := RunState{RunID: run, Owner: s.node, Full: true}
		for _, e := range execs {
			state.Executions = append(state.Executions, p.stateWithApproval(ctx, e))
		}
		fitApprovalBodies(&state)
		_ = s.l.send(Frame{T: TRun}, state)
	}
	for run, execs := range heldRun {
		for _, exec := range execs {
			p.push(ctx, s, run, exec)
		}
	}
}

// Ack deletes exec's held entries through seq, for its run's owner only.
func (p *Pipe) Ack(ctx context.Context, caller, exec string, through int64) (int64, error) {
	p.init()
	if through <= 0 {
		return 0, refuse(http.StatusBadRequest, "through: the last seq kept is required")
	}
	if err := p.scope(ctx, caller, exec); err != nil {
		return 0, err
	}
	n, err := p.Store.RelayAck(ctx, exec, through)
	if err != nil {
		return 0, err
	}
	p.mu.Lock()
	if through > p.acked[exec] {
		p.acked[exec] = through
	}
	p.mu.Unlock()
	return n, nil
}

// Read is the HTTP fallback's read of exec's held entries after seq.
func (p *Pipe) Read(ctx context.Context, caller, exec string, after int64, limit int) ([]storage.RelayEntry, error) {
	if err := p.scope(ctx, caller, exec); err != nil {
		return nil, err
	}
	return p.Store.RelayAfter(ctx, exec, after, limit)
}

// Scope is scope for another door: the approval answer route holds a
// caller to the runs it owns the same way.
func (p *Pipe) Scope(ctx context.Context, caller, exec string) error {
	p.init()
	return p.scope(ctx, caller, exec)
}

// scope refuses a caller that does not own exec's run. An empty caller is
// the API key holder speaking for itself (the HTTP fallback with no node
// named), which the control plane already trusts with every execution.
func (p *Pipe) scope(ctx context.Context, caller, exec string) error {
	if caller == "" {
		return nil
	}
	_, owner, err := p.ownerOf(ctx, exec)
	if err != nil {
		return err
	}
	if owner != caller {
		return refuse(http.StatusForbidden, "this step belongs to a run you do not own")
	}
	return nil
}

// ── STEER: OWNER → CONTROL PLANE → THE NODE RUNNING THE STEP ────────────────

// Steer hands a person's note to the node running exec and answers only
// when that node took it. It goes down the node's link when the node holds
// one (as an http delivery, or a steer delivery for a node that takes only
// that), over HTTP when the node has an address, and is refused in plain
// words when the node has neither.
func (p *Pipe) Steer(ctx context.Context, caller, exec, message string) (string, error) {
	p.init()
	message = strings.TrimSpace(message)
	if message == "" || len(message) > 4000 {
		return "", refuse(http.StatusBadRequest, "message is 1 to 4000 bytes")
	}
	rec, err := p.Store.GetExecutionRecord(ctx, exec)
	if err != nil || rec == nil {
		return "", refuse(http.StatusNotFound, "no such execution")
	}
	if err := p.scope(ctx, caller, exec); err != nil {
		return "", err
	}
	if types.IsTerminalExecutionStatus(rec.Status) {
		return "", refuse(http.StatusConflict, "this step has ended")
	}
	agent, err := p.Store.GetAgent(ctx, rec.AgentNodeID)
	if err != nil || agent == nil {
		return "", refuse(http.StatusConflict, "the node that runs this step is not registered")
	}
	ctx, cancel := context.WithTimeout(ctx, p.SteerWait)
	defer cancel()
	switch {
	case p.Hub != nil && p.Hub.LiveFor(agent.ID, KindHTTP):
		return "link", p.steerHTTP(ctx, agent.ID, "http://"+agenturl.LinkHost(agent.ID), exec, message)
	case p.Hub != nil && p.Hub.LiveFor(agent.ID, KindSteer):
		return "link", p.steerDelivery(ctx, agent.ID, exec, message)
	case agent.BaseURL != "":
		if _, linkOnly := agenturl.LinkNode(hostOf(agent.BaseURL)); !linkOnly {
			return "http", p.steerHTTP(ctx, agent.ID, agent.BaseURL, exec, message)
		}
	}
	return "", refuse(http.StatusConflict, "the node running this step is not connected to the control plane, so the note cannot reach it")
}

func hostOf(raw string) string {
	if i := strings.Index(raw, "://"); i >= 0 {
		raw = raw[i+3:]
	}
	if i := strings.IndexAny(raw, "/?#"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

func (p *Pipe) steerDelivery(ctx context.Context, node, exec, message string) error {
	done := make(chan Outcome, 1)
	id := p.Hub.Deliver(node, Delivery{Kind: KindSteer, ExecutionID: exec, Message: message, TTL: p.SteerWait,
		OnSettle: func(o Outcome) { done <- o }})
	select {
	case o := <-done:
		switch o.Outcome {
		case OutcomeAccepted, OutcomeDone, OutcomeDuplicate:
			return nil
		case OutcomeExpired:
			return refuse(http.StatusConflict, "the node did not take the note: it did not answer")
		default:
			return refuse(http.StatusConflict, nodeWords(o.Error))
		}
	case <-ctx.Done():
		p.Hub.settle(id, Outcome{Outcome: OutcomeExpired, Error: "steer timed out"})
		return refuse(http.StatusConflict, "the node did not take the note: it did not answer in time")
	}
}

func (p *Pipe) steerHTTP(ctx context.Context, node, base, exec, message string) error {
	target, err := agenturl.ForNode(node, base, "_internal", "executions", exec, "steer")
	if err != nil {
		return refuse(http.StatusConflict, "the node has no address the note can reach")
	}
	raw, _ := json.Marshal(map[string]string{"message": message})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", exec)
	if p.InternalToken != "" {
		req.Header.Set("Authorization", "Bearer "+p.InternalToken)
	}
	resp, err := agenturl.NewClient(0).Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return refuse(http.StatusConflict, "the node did not take the note: it did not answer in time")
		}
		return refuse(http.StatusConflict, "the node did not take the note: "+err.Error())
	}
	defer resp.Body.Close()
	answer, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode < 300 {
		return nil
	}
	if resp.StatusCode == http.StatusNotFound {
		return refuse(http.StatusConflict, "the node does not take notes for this step")
	}
	var refusal struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal(answer, &refusal)
	if refusal.Error == "" {
		refusal.Error = strings.TrimSpace(string(answer))
	}
	return refuse(http.StatusConflict, nodeWords(refusal.Error))
}

// nodeWords is a node's refusal as a person reads it.
func nodeWords(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "the node did not take the note"
	}
	return s
}

// ── THE HUB'S SIDE: FRAMES FROM A LINK ──────────────────────────────────────

// frame serves one pipe frame from l, answering it with an ok on its seq.
// It reports false for a frame type that is not the pipe's.
func (p *Pipe) frame(l *link, f Frame) bool {
	ctx := context.Background()
	answer := func(code int, err error, through, dropped int64) {
		body := OKBody{Code: code, Through: through, Dropped: dropped}
		if err != nil {
			body.Code, body.Error = CodeOf(err), err.Error()
		}
		_ = l.send(Frame{T: TOK, Seq: f.Seq, Exec: f.Exec}, body)
	}
	switch f.T {
	case TEntries:
		var up EntriesUp
		if err := json.Unmarshal(f.Body, &up); err != nil {
			answer(0, refuse(http.StatusBadRequest, "entries: not JSON"), 0, 0)
			return true
		}
		_, through, err := p.Put(ctx, f.Exec, up.Entries)
		answer(http.StatusOK, err, through, 0)
	case TSubscribe:
		var req Subscribe
		_ = json.Unmarshal(f.Body, &req)
		answer(http.StatusOK, nil, 0, 0)
		p.Subscribe(ctx, l, req)
	case TEntriesAck:
		var req EntriesAck
		_ = json.Unmarshal(f.Body, &req)
		n, err := p.Ack(ctx, l.node, f.Exec, req.Through)
		answer(http.StatusOK, err, 0, n)
	case TSteer:
		var req SteerUp
		_ = json.Unmarshal(f.Body, &req)
		// A steer waits on another node; it must not hold up this link's
		// reading, so it is answered from its own goroutine.
		go func() {
			_, err := p.Steer(ctx, l.node, f.Exec, req.Message)
			answer(http.StatusOK, err, 0, 0)
		}()
	default:
		return false
	}
	return true
}
