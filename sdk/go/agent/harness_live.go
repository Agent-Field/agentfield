package agent

// Live runs relay their output to the control plane for a watcher to read.
// A steerable run opens a mailbox for notes from either node-side carrier.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// ErrNotSteerable means the execution has no open mailbox for notes.
var ErrNotSteerable = errors.New("this execution is not running anything that takes notes right now, so it cannot be steered")

type steerBoxes struct {
	mu    sync.Mutex
	boxes map[string]chan string
}

func (s *steerBoxes) open(id string, box chan string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.boxes == nil {
		s.boxes = map[string]chan string{}
	}
	s.boxes[id] = box
}

func (s *steerBoxes) close(id string, box chan string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.boxes[id] == box {
		delete(s.boxes, id)
	}
}

func (s *steerBoxes) get(id string) chan string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.boxes[id]
}

// Steer hands a person's note to a running execution that accepts notes. It
// never waits for the execution to read a note from a full mailbox.
func (a *Agent) Steer(executionID, text string) error {
	return a.queueSteerNote(executionID, text, false)
}

func (a *Agent) queueSteerNote(executionID, text string, relayNote bool) error {
	executionID, text = strings.TrimSpace(executionID), strings.TrimSpace(text)
	if text == "" {
		return errors.New("a steer note has words")
	}
	a.steer.mu.Lock()
	defer a.steer.mu.Unlock()
	box := a.steer.boxes[executionID]
	if box == nil {
		return ErrNotSteerable
	}
	select {
	case box <- text:
		if relayNote {
			// Hold the mailbox open until the receipt is part of the transcript.
			raw, _ := json.Marshal(map[string]string{"text": text})
			a.relayFor(executionID).send(RelayPerson, raw)
		}
		return nil
	default:
		return errors.New("the run has notes it has not read yet; send this again in a moment")
	}
}

// LiveRun is the live watch and steer channel of a watched execution.
type LiveRun struct {
	relay      *execRelay
	notes      chan string
	deferFlush bool
}

// LiveRun returns the channel a watched reasoner can use to report its
// progress and receive notes. It returns false outside a watched execution.
func (a *Agent) LiveRun(ctx context.Context) (*LiveRun, bool) {
	ec := executionContextFrom(ctx)
	id := strings.TrimSpace(ec.ExecutionID)
	if a == nil || id == "" || !ec.RunWatched {
		return nil, false
	}
	return &LiveRun{relay: a.relayFor(id), notes: make(chan string, 16)}, true
}

// Begin announces the run and opens its mailbox if it takes notes.
func (l *LiveRun) Begin(info harness.LiveInfo) {
	if info.Steerable {
		l.relay.agent.steer.open(l.relay.id, l.notes)
	}
	raw, _ := json.Marshal(info)
	l.relay.send(RelayStarted, raw)
}

// Emit relays one line with the data supplied by the caller.
func (l *LiveRun) Emit(line json.RawMessage) {
	l.relay.send(RelayLine, line)
}

// Notes receives steer notes while a steerable run is active.
func (l *LiveRun) Notes() <-chan string {
	return l.notes
}

// End stops accepting notes, relays the end of the run, and waits up to 30
// seconds for the relay to flush.
func (l *LiveRun) End() {
	l.relay.agent.steer.close(l.relay.id, l.notes)
	l.relay.send(RelayEnded, json.RawMessage(`{}`))
	if !l.deferFlush {
		l.flush()
	}
}

func (l *LiveRun) flush() {
	l.relay.flush(30 * time.Second)
}

// liveHarness wires the agent's live channel to a watched harness run.
func (a *Agent) liveHarness(ctx context.Context) (*harness.Live, func()) {
	live, ok := a.LiveRun(ctx)
	if !ok {
		return nil, func() {}
	}
	// Providers can finish an attempt before the runner retries or falls back.
	// Flush once when the full harness invocation returns.
	live.deferFlush = true
	return &harness.Live{Begin: live.Begin, Event: live.Emit, Steer: live.Notes(), End: live.End}, live.flush
}

// handleInternalExecution serves the control plane's two worker
// notifications: POST /_internal/executions/:id/cancel and .../steer.
func (a *Agent) handleInternalExecution(w http.ResponseWriter, r *http.Request) {
	if strings.HasSuffix(r.URL.Path, "/steer") {
		a.handleInternalSteer(w, r)
		return
	}
	a.handleInternalCancel(w, r)
}

// handleInternalSteer is POST /_internal/executions/:id/steer {message}.
// It reports 200 when the execution took the note or 409 with the refusal.
func (a *Agent) handleInternalSteer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	id := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/_internal/executions/"), "/steer"))
	if id == "" || strings.Contains(id, "/") {
		http.NotFound(w, r)
		return
	}
	var body struct {
		Message string `json:"message"`
	}
	_ = json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&body)
	w.Header().Set("Content-Type", "application/json")
	if err := a.steerFromControlPlane(id, body.Message); err != nil {
		w.WriteHeader(http.StatusConflict)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error(), "execution_id": id})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "execution_id": id})
}

// steerFromControlPlane accepts a note from either carrier and records its
// receipt for the watcher.
func (a *Agent) steerFromControlPlane(id, text string) error {
	return a.queueSteerNote(id, text, true)
}
