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

// RunWatchedHeader is set to "1" on execution deliveries whose run has an
// owner watching its live harness transcript; X-AgentField-Run-Owner itself is
// never forwarded to nodes because on inbound requests it scopes ownership.
const RunWatchedHeader = "X-AgentField-Run-Watched"

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

	// The pipe (pipe.go). TEntries goes both ways: up from the node that ran
	// an execution (body = EntriesUp), and down to the run's subscriber (body
	// = EntriesDown). TRun is down only. TSubscribe, TEntriesAck and TSteer are
	// up from the subscriber; each is answered by an ok frame on its seq.
	TEntries    = "entries"     // exec; up EntriesUp, down EntriesDown
	TEntriesAck = "entries_ack" // exec; body = EntriesAck
	TSubscribe  = "subscribe"   // body = Subscribe
	TRun        = "run"         // down; body = RunState
	TSteer      = "steer"       // exec; body = SteerUp
)

var known = map[string]bool{TWelcome: true, TDeliver: true, TOK: true, THello: true, TAck: true, TStatus: true, TProgress: true, TLog: true, THeartbeat: true,
	TEntries: true, TEntriesAck: true, TSubscribe: true, TRun: true, TSteer: true}

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
	// KindHTTP carries one HTTP request the control plane would have dialed
	// (agenturl's link carrier): the node serves it on its own handler and
	// acks with the response. A node takes it only if its hello accepts it.
	KindHTTP = "http"
	// KindSteer is a person's note for a running codeaf harness execution, for
	// a node that takes it as its own delivery rather than over KindHTTP.
	KindSteer = "steer"
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
	// Steer (a node whose hello accepts "steer" but not "http").
	Message string `json:"message,omitempty"`
	// HTTP.
	Method string `json:"method,omitempty"`
	Path   string `json:"path,omitempty"`
	Body   []byte `json:"body,omitempty"`
}

// HTTPResult is an http delivery's ack result: the node's response.
type HTTPResult struct {
	Code    int               `json:"code"`
	Headers map[string]string `json:"headers,omitempty"`
	Body    []byte            `json:"body,omitempty"`
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
	// Accepts is the delivery kinds this node serves. Empty is the original
	// set (dispatch, cancel), which is what the codeaf factory's node speaks.
	Accepts []string `json:"accepts,omitempty"`
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
	// Through answers an entries frame: the highest seq the control plane
	// now holds or has already passed on, so the node may forget every entry
	// up to it.
	Through int64 `json:"through,omitempty"`
	// Dropped answers an entries_ack: how many held entries were deleted.
	Dropped int64 `json:"dropped,omitempty"`
}

// Entry is one harness entry in flight. Data is the node's bytes, verbatim.
type Entry struct {
	Seq  int64           `json:"seq"`
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data,omitempty"`
	At   string          `json:"at,omitempty"`
}

// EntriesUp is an up entries frame's body.
type EntriesUp struct {
	Entries []Entry `json:"entries"`
}

// EntriesDown is a down entries frame's body: entries of one execution,
// oldest first, every seq above the subscriber's last ack.
type EntriesDown struct {
	RunID   string  `json:"run_id"`
	Entries []Entry `json:"entries"`
}

// Subscribe is the subscriber's subscribe body. The link's own node is the
// subscriber; it is sent only the runs it owns. After skips entries the
// subscriber already keeps but has not acked yet (exec → seq). Runs names
// runs it wants a fresh tree for even if they ended while it was away.
type Subscribe struct {
	After map[string]int64 `json:"after,omitempty"`
	Runs  []string         `json:"runs,omitempty"`
}

// EntriesAck is an entries_ack body: the subscriber keeps every entry of
// the frame's execution through Through, so the control plane drops them.
// (The frame's own seq is the up-frame's, answered by the ok.)
type EntriesAck struct {
	Through int64 `json:"through"`
}

// SteerUp is a steer frame's body.
type SteerUp struct {
	Message string `json:"message"`
}

// RunState is a run frame's body: some executions of one run as the control
// plane holds them now. Full is true for a whole-tree snapshot (sent on
// subscribe); otherwise it carries the executions that just changed.
type RunState struct {
	RunID      string      `json:"run_id"`
	Owner      string      `json:"owner"`
	Full       bool        `json:"full,omitempty"`
	Executions []ExecState `json:"executions"`
}

// ExecState is one execution's control data, the only part of it the
// control plane keeps for good.
type ExecState struct {
	ExecutionID       string          `json:"execution_id"`
	ParentExecutionID string          `json:"parent_execution_id,omitempty"`
	AgentNodeID       string          `json:"agent_node_id"`
	ReasonerID        string          `json:"reasoner_id"`
	Status            string          `json:"status"`
	StatusReason      string          `json:"status_reason,omitempty"`
	StartedAt         string          `json:"started_at,omitempty"`
	CompletedAt       string          `json:"completed_at,omitempty"`
	DurationMS        int64           `json:"duration_ms,omitempty"`
	Error             string          `json:"error,omitempty"`
	Result            json.RawMessage `json:"result,omitempty"`
	ResultOmitted     bool            `json:"result_omitted,omitempty"`
	// Approval is the question a waiting execution asks a person; it is
	// present only while that approval is pending.
	Approval *ApprovalState `json:"approval,omitempty"`
}

// ApprovalState is a pending approval as the run's owner shows it: the
// request's id (what an answer resolves), its kind, the one-line question,
// the full markdown body and the option labels.
type ApprovalState struct {
	ID       string `json:"id"`
	Kind     string `json:"kind,omitempty"`
	Question string `json:"question,omitempty"`
	Body     string `json:"body,omitempty"`
	// BodyOmitted is set when a whole-run snapshot was too large to carry
	// every body; the body is then read from GET …/approval-status.
	BodyOmitted bool     `json:"body_omitted,omitempty"`
	Options     []string `json:"options,omitempty"`
	ExpiresAt   string   `json:"expires_at,omitempty"`
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
