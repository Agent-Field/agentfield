package agent

// ── A CODEAF HARNESS RUN, WATCHED AND STEERED WITH NOTHING WRITTEN FOR IT ──
//
// Inside a watched run (X-AgentField-Run-Watched: 1), a reasoner that calls
// a.Harness(...) on the codeaf provider gets, for free:
//
//   - WATCH: every line codeaf streams while it works — one line of the run's
//     own conversation record, carried VERBATIM — is relayed to the control
//     plane under THIS execution (relay.go), numbered from 1 by this node and
//     replayed until the control plane answers, so a control plane restart
//     mid-run loses no line and doubles none. The control plane only holds a
//     line until WorkAF, which keeps the transcript, has acked it.
//   - STEER: while the run works the execution has a mailbox. A person's note
//     reaches it from the control plane — POST /_internal/executions/:id/steer,
//     which travels down the node link for a node with no address — and the
//     harness writes it to the run's steer file, where codeaf reads it at its
//     next turn boundary.
//
// A relayed "started" entry ({harness: codeaf, steerable: true}) opens the run
// and "ended" closes it, so a reader can tell whether a step is steerable now.
//
// STEERING EXISTS ONLY WHILE A CODEAF RUN IS WORKING. The mailbox opens in
// harness.Live.Begin, which only the codeaf provider calls, and closes in End.
// A steer for any other execution — a plain reasoner, another harness, a run
// that has finished — is refused with ErrNotSteerable, and the control plane
// answers the person with that refusal.

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

// ErrNotSteerable is a steer for an execution that is not running the codeaf
// harness right now.
var ErrNotSteerable = errors.New("this execution is not running the codeaf harness, so it cannot be steered")

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

// Steer hands a person's note to the codeaf run working in executionID. It
// never blocks: a mailbox that is full refuses rather than holding the caller.
func (a *Agent) Steer(executionID, text string) error {
	executionID, text = strings.TrimSpace(executionID), strings.TrimSpace(text)
	if text == "" {
		return errors.New("a steer note has words")
	}
	box := a.steer.get(executionID)
	if box == nil {
		return ErrNotSteerable
	}
	select {
	case box <- text:
	default:
		return errors.New("the run has notes it has not read yet; send this again in a moment")
	}
	return nil
}

// liveHarness arms harness.Live for the execution in ctx when the control
// plane marked its run as watched. An unwatched run, or a harness call outside
// a reasoner, gets nil so codeaf runs its plain non-streaming exec path.
func (a *Agent) liveHarness(ctx context.Context) (*harness.Live, func()) {
	ec := executionContextFrom(ctx)
	id := strings.TrimSpace(ec.ExecutionID)
	if a == nil || id == "" || !ec.RunWatched {
		return nil, func() {}
	}
	relay := a.relayFor(id)
	box := make(chan string, 16)
	live := &harness.Live{
		Steer: box,
		Begin: func() {
			a.steer.open(id, box)
			relay.send(RelayStarted, json.RawMessage(`{"harness":"codeaf","steerable":true}`))
		},
		End: func() {
			a.steer.close(id, box)
			relay.send(RelayEnded, json.RawMessage(`{}`))
		},
		Event: func(line json.RawMessage) { relay.send(RelayLine, line) },
	}
	return live, func() { relay.flush(30 * time.Second) }
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
// 200 when the note reached a codeaf run's mailbox; 409 with the refusal's
// words when the execution is not one.
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

// steerFromControlPlane is a steer that came from a person through the
// control plane, by either carrier. It also relays that a person sent it, so
// the record shows the note the moment it arrives, before codeaf reads it.
func (a *Agent) steerFromControlPlane(id, text string) error {
	if err := a.Steer(id, text); err != nil {
		return err
	}
	raw, _ := json.Marshal(map[string]string{"text": strings.TrimSpace(text)})
	a.relayFor(id).send(RelayPerson, raw)
	return nil
}
