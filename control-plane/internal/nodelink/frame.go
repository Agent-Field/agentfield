// Package nodelink is the node-initiated link: one WebSocket a node opens to
// the control plane (GET /api/v1/nodes/:node_id/link) and keeps open, so the
// control plane can reach a node that has no address it can dial — a laptop
// behind NAT, a container with no ingress.
//
// THE LINK IS A CARRIER, NOT A SECOND IMPLEMENTATION. Down-frames carry the
// exact body and headers today's HTTP dispatch and cancel send. Up-frames are
// re-entered through the control plane's own router as the HTTP calls they
// replace (POST /executions/:id/status, POST /executions/:id/logs, PATCH
// /nodes/:id/status), so execution state has one implementation whichever
// carrier brought it.
//
// PROTOTYPE SCOPE: one replica, the outbox is in memory. A delivery waits in
// the hub until a link acks it and is redelivered until then; a control plane
// restart loses deliveries not yet acked (their executions stay running and
// the stale reaper ends them, as it ends a lost HTTP dispatch today).
package nodelink

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Version is the frame envelope's version.
const Version = 1

// MaxFrame is the largest frame either side accepts.
const MaxFrame = 1 << 20

// Frame types. Down is control plane → node, up is node → control plane.
const (
	// Down.
	TWelcome = "welcome" // {link_id, heartbeat_s, max_frame}
	TDeliver = "deliver" // id = delivery id; body = DeliverBody
	TOK      = "ok"      // seq = the up-frame answered; body = OKBody
	// Up.
	THello     = "hello"     // body = Hello
	TAck       = "ack"       // id = delivery id; body = Ack
	TStatus    = "status"    // exec; body = the POST /executions/:id/status body
	TProgress  = "progress"  // exec; body = a non-terminal status body
	TLog       = "log"       // exec; body = the POST /executions/:id/logs body
	THeartbeat = "heartbeat" // body = Heartbeat
)

var known = map[string]bool{TWelcome: true, TDeliver: true, TOK: true, THello: true, TAck: true, TStatus: true, TProgress: true, TLog: true, THeartbeat: true}

// Frame is the one envelope every message travels in.
type Frame struct {
	V    int             `json:"v"`
	T    string          `json:"t"`
	ID   string          `json:"id,omitempty"`
	Exec string          `json:"exec,omitempty"`
	Seq  int64           `json:"seq,omitempty"`
	TS   string          `json:"ts,omitempty"`
	Body json.RawMessage `json:"body,omitempty"`
}

// Delivery kinds.
const (
	KindDispatch = "dispatch"
	KindCancel   = "cancel"
)

// DeliverBody is a deliver frame's body.
type DeliverBody struct {
	Kind string `json:"kind"`
	// Dispatch.
	Reasoner       string            `json:"reasoner,omitempty"`
	Input          json.RawMessage   `json:"input,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`
	IdempotencyKey string            `json:"idempotency_key,omitempty"`
	// Cancel.
	Reason string `json:"reason,omitempty"`
}

// Ack outcomes.
const (
	OutcomeAccepted  = "accepted"  // the 202 path: the execution stays running
	OutcomeDone      = "done"      // the sync 2xx path: result completes it
	OutcomeRejected  = "rejected"  // the 4xx/5xx path
	OutcomeDuplicate = "duplicate" // already seen; treated as accepted
	OutcomeExpired   = "expired"   // hub-side only: no link took it in time
)

// Ack is an ack frame's body.
type Ack struct {
	Outcome string          `json:"outcome"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// Hello is the node's first frame.
type Hello struct {
	InstanceID string   `json:"instance_id"`
	SDK        string   `json:"sdk,omitempty"`
	Version    string   `json:"version,omitempty"`
	Inflight   []string `json:"inflight,omitempty"`
}

// Heartbeat renews the lease and keeps inflight executions alive.
type Heartbeat struct {
	Inflight    []string `json:"inflight,omitempty"`
	HealthScore *int     `json:"health_score,omitempty"`
}

// Welcome is the control plane's first frame.
type Welcome struct {
	LinkID     string `json:"link_id"`
	HeartbeatS int    `json:"heartbeat_s"`
	MaxFrame   int    `json:"max_frame"`
}

// OKBody answers one up-frame with the HTTP status its re-entered handler
// gave, so the node reads it exactly as it would read the HTTP answer.
type OKBody struct {
	Code  int    `json:"code"`
	Error string `json:"error,omitempty"`
}

// Encode builds a frame of type t with body b (nil for none).
func Encode(f Frame, body any) ([]byte, error) {
	f.V = Version
	if f.TS == "" {
		f.TS = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		f.Body = raw
	}
	if !known[f.T] {
		return nil, fmt.Errorf("nodelink: unknown frame type %q", f.T)
	}
	raw, err := json.Marshal(f)
	if err != nil {
		return nil, err
	}
	if len(raw) > MaxFrame {
		return nil, errors.New("nodelink: frame over 1 MiB")
	}
	return raw, nil
}

// Decode reads one frame, refusing an unknown version or type and anything
// over MaxFrame.
func Decode(raw []byte) (Frame, error) {
	if len(raw) > MaxFrame {
		return Frame{}, errors.New("nodelink: frame over 1 MiB")
	}
	var f Frame
	if err := json.Unmarshal(raw, &f); err != nil {
		return Frame{}, fmt.Errorf("nodelink: not a frame: %w", err)
	}
	if f.V != Version {
		return Frame{}, fmt.Errorf("nodelink: frame version %d, want %d", f.V, Version)
	}
	if !known[f.T] {
		return Frame{}, fmt.Errorf("nodelink: unknown frame type %q", f.T)
	}
	return f, nil
}
