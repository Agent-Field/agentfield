package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ── THE RELAY, FROM THE NODE'S SIDE ─────────────────────────────────────────
//
// A codeaf harness run's lines go to the control plane's relay — an
// `entries` frame on the node's link, or POST /api/v1/executions/:id/relay
// when no link is up — which holds them only until WorkAF —
// the system of record for the transcript — has acked them.
//
// NOTHING IS LOST AND NOTHING IS DOUBLED ACROSS A CONTROL PLANE OUTAGE. This
// node numbers each execution's entries from 1 and keeps every entry the
// control plane has not answered; a post that fails is posted again, from the
// oldest, until one succeeds. (execution_id, seq) is the control plane's key,
// so a batch it stored before dying unanswered is stored once when replayed.

// Relay entry kinds.
const (
	RelayStarted = "started" // {harness: "codeaf", steerable: true}
	RelayLine    = "line"    // one codeaf conversation-record line, verbatim
	RelayPerson  = "person"  // {text}: a person's steer as this node received it
	RelayEnded   = "ended"   // {}
)

// RelayEntry is one entry on the wire.
type RelayEntry struct {
	Seq  int64           `json:"seq"`
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data,omitempty"`
}

// relayBatch is the most entries one post carries.
const relayBatch = 200

// relayMaxBackoff is the longest a failed post waits before the next try.
var relayMaxBackoff = 5 * time.Second

type execRelay struct {
	agent   *Agent
	id      string
	mu      sync.Mutex
	next    int64
	pending []RelayEntry
	wake    chan struct{}
	empty   *sync.Cond
	running bool
}

type relays struct {
	mu sync.Mutex
	m  map[string]*execRelay
}

// relayFor is execution id's relay, made on first use.
func (a *Agent) relayFor(id string) *execRelay {
	a.relays.mu.Lock()
	defer a.relays.mu.Unlock()
	if a.relays.m == nil {
		a.relays.m = map[string]*execRelay{}
	}
	r := a.relays.m[id]
	if r == nil {
		r = &execRelay{agent: a, id: id, wake: make(chan struct{}, 1)}
		r.empty = sync.NewCond(&r.mu)
		a.relays.m[id] = r
	}
	return r
}

// send queues one entry and wakes the sender. It never blocks on the network.
func (r *execRelay) send(kind string, data json.RawMessage) {
	if r.agent.client == nil {
		return
	}
	r.mu.Lock()
	r.next++
	r.pending = append(r.pending, RelayEntry{Seq: r.next, Kind: kind, Data: append(json.RawMessage(nil), data...)})
	if !r.running {
		r.running = true
		go r.loop()
	}
	r.mu.Unlock()
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

func (r *execRelay) loop() {
	backoff := 100 * time.Millisecond
	for {
		r.mu.Lock()
		if len(r.pending) == 0 {
			r.running = false
			r.empty.Broadcast()
			r.mu.Unlock()
			return
		}
		batch := append([]RelayEntry(nil), r.pending[:min(len(r.pending), relayBatch)]...)
		r.mu.Unlock()

		through, err := r.deliver(batch)
		if err != nil {
			if !strings.Contains(err.Error(), "connection refused") {
				r.agent.logger.Printf("warn: relay for %s waiting on the control plane: %v", r.id, err)
			}
			select {
			case <-time.After(backoff):
			case <-r.wake:
			}
			if backoff *= 2; backoff > relayMaxBackoff {
				backoff = relayMaxBackoff
			}
			continue
		}
		backoff = 100 * time.Millisecond
		// What the control plane holds is what it says it holds; an older one
		// that answers no "through" held the whole batch.
		if through == 0 {
			through = batch[len(batch)-1].Seq
		}
		r.mu.Lock()
		keep := r.pending[:0]
		for _, e := range r.pending {
			if e.Seq > through {
				keep = append(keep, e)
			}
		}
		r.pending = keep
		r.mu.Unlock()
	}
}

// deliver hands one batch to the control plane and says through which seq it
// now holds them. THE LINK FIRST: an `entries` frame on the link this node
// already holds, its data seqs in the body and the frame's own seq the reply
// key. With no link up, or no answer on it, the same entries go by HTTP
// (POST /api/v1/executions/:id/relay), which the control plane serves by the
// same code. Either way the control plane holds them only until WorkAF acks.
func (r *execRelay) deliver(batch []RelayEntry) (int64, error) {
	ok, err := r.agent.linkAsk("entries", r.id, map[string]any{"entries": batch}, 10*time.Second)
	if err == nil {
		switch ok.Code {
		case 200:
			return ok.Through, nil
		case 400:
			// An entry the control plane can never take would hold every
			// line after it forever: it is said and let go.
			r.agent.logger.Printf("warn: relay for %s: the control plane refused seqs %d–%d: %s", r.id, batch[0].Seq, batch[len(batch)-1].Seq, ok.Error)
			return batch[len(batch)-1].Seq, nil
		default:
			return 0, fmt.Errorf("the control plane answered %d: %s", ok.Code, ok.Error)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return r.agent.client.PostExecutionRelay(ctx, r.id, batch)
}

// flush waits until every entry was answered, at most timeout. Entries still
// pending afterwards keep being sent in the background.
func (r *execRelay) flush(timeout time.Duration) {
	done := make(chan struct{})
	go func() {
		r.mu.Lock()
		for r.running && len(r.pending) > 0 {
			r.empty.Wait()
		}
		r.mu.Unlock()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(timeout):
	}
}

// relayPending is how many entries wait for the control plane, for tests.
func (r *execRelay) relayPending() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.pending)
}
