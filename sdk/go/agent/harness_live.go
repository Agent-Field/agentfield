package agent

// ── A CODEAF HARNESS RUN, WATCHED AND STEERED WITH NOTHING WRITTEN FOR IT ──
//
// A reasoner that calls a.Harness(...) on the codeaf provider gets, for free:
//
//   - WATCH: every line codeaf streams while it works becomes one structured
//     execution log on THIS execution (event_type "codeaf.event", source
//     "sdk.harness.codeaf"), with codeaf's own line carried verbatim in
//     attributes.codeaf. They are posted in order by one sender per run, so
//     the control plane's seq is codeaf's order.
//   - STEER: while the run works the execution has a mailbox. A person's note
//     reaches it from the control plane — down the node link (link.go) or
//     POST /_internal/executions/:id/steer — and the harness writes it to the
//     run's steer file, where codeaf reads it at its next turn boundary.
//
// Two system logs bracket the run, "codeaf.harness.started" with
// attributes {harness: "codeaf", steerable: true} and "codeaf.harness.ended",
// so a reader can tell from the logs alone whether a step is steerable now.
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

const (
	harnessLogSource   = "sdk.harness.codeaf"
	harnessEventType   = "codeaf.event"
	harnessStartedType = "codeaf.harness.started"
	harnessEndedType   = "codeaf.harness.ended"
	harnessPersonType  = "codeaf.person"
)

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

// liveHarness arms harness.Live for the execution in ctx, or returns nil when
// there is none (a harness call outside a reasoner).
func (a *Agent) liveHarness(ctx context.Context) (*harness.Live, func()) {
	execCtx := executionContextFrom(ctx)
	id := strings.TrimSpace(execCtx.ExecutionID)
	if a == nil || id == "" {
		return nil, func() {}
	}
	logger := a.executionLogger(ctx, harnessLogSource)
	sender := newOrderedLogSender(a, id)
	box := make(chan string, 16)
	live := &harness.Live{
		Steer: box,
		Begin: func() {
			a.steer.open(id, box)
			sender.send(logger.entry("info", harnessStartedType, "codeaf harness started", map[string]any{"harness": "codeaf", "steerable": true}, true))
		},
		End: func() {
			a.steer.close(id, box)
			sender.send(logger.entry("info", harnessEndedType, "codeaf harness ended", map[string]any{"harness": "codeaf", "steerable": false}, true))
		},
		Event: func(line json.RawMessage) {
			sender.send(logger.entry("info", harnessEventType, codeafEventSummary(line), map[string]any{"codeaf": line}, false))
		},
	}
	return live, sender.close
}

// codeafEventSummary is the log line's message: the event's own words when it
// has them, else its kind. The full event is in attributes.codeaf.
func codeafEventSummary(line json.RawMessage) string {
	var ev struct {
		Kind string `json:"kind"`
		Text string `json:"text"`
		Name string `json:"name"`
	}
	_ = json.Unmarshal(line, &ev)
	words := strings.TrimSpace(firstNonEmptyString(ev.Text, ev.Name, ev.Kind, "codeaf event"))
	if len(words) > 300 {
		words = words[:300] + "…"
	}
	return words
}

func firstNonEmptyString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// orderedLogSender posts one execution's logs in the order they were made,
// batching whatever queued while the last post was in flight.
type orderedLogSender struct {
	agent *Agent
	id    string
	queue chan ExecutionLogEntry
	done  chan struct{}
	once  sync.Once
}

func newOrderedLogSender(a *Agent, id string) *orderedLogSender {
	s := &orderedLogSender{agent: a, id: id, queue: make(chan ExecutionLogEntry, 4096), done: make(chan struct{})}
	go s.run()
	return s
}

func (s *orderedLogSender) send(entry ExecutionLogEntry) {
	writeStructuredExecutionLog(entry)
	select {
	case s.queue <- entry:
	default:
		// A control plane that cannot keep up loses transcript lines, never
		// the run: the queue is bounded and the harness never waits on it.
	}
}

func (s *orderedLogSender) run() {
	defer close(s.done)
	for first := range s.queue {
		batch := []ExecutionLogEntry{first}
	more:
		for len(batch) < 200 {
			select {
			case e, ok := <-s.queue:
				if !ok {
					break more
				}
				batch = append(batch, e)
			default:
				break more
			}
		}
		if s.agent.client == nil {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := s.agent.client.PostExecutionLogs(ctx, s.id, map[string]any{"entries": batch}); err != nil {
			s.agent.logger.Printf("warn: codeaf transcript send failed: %v", err)
		}
		cancel()
	}
}

// close sends what is queued and waits for it, at most ten seconds.
func (s *orderedLogSender) close() {
	s.once.Do(func() {
		close(s.queue)
		select {
		case <-s.done:
		case <-time.After(10 * time.Second):
		}
	})
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
// control plane, by either carrier. It also leaves a log saying a person
// sent it, so the transcript shows the note the moment it arrives, before
// codeaf reads it at its next turn boundary.
func (a *Agent) steerFromControlPlane(id, text string) error {
	if err := a.Steer(id, text); err != nil {
		return err
	}
	if a.client != nil {
		entry := ExecutionLogEntry{V: executionLogVersion, TS: time.Now().UTC().Format(time.RFC3339Nano), ExecutionID: id, AgentNodeID: a.cfg.NodeID,
			Level: "info", Source: harnessLogSource, EventType: harnessPersonType, Message: strings.TrimSpace(text), Attributes: map[string]any{"harness": "codeaf"}}
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = a.client.PostExecutionLogs(ctx, id, entry)
		}()
	}
	return nil
}
