package agent

// ── THE NODE LINK, FROM THE SDK'S SIDE ──────────────────────────────────────
//
// The control plane's node link (control-plane/internal/nodelink) is one
// WebSocket a node opens and keeps open, so the control plane can reach a
// node it cannot dial. This SDK opens it after registering and serves:
//
//   - http: one request the control plane would have dialed (a dispatch, a
//     nested call, a cancel, a steer, a probe), run on this agent's own
//     handler, with the response as the ack. A node in LINK-ONLY mode
//     (AGENTFIELD_NODE_LINK_ONLY=1) registers base_url http://<id>.nodelink and
//     listens on loopback only, so it has no inbound port at all and every
//     request reaches it this way;
//   - steer and cancel, the two kinds that came before the http carrier.
//
// A REDELIVERY IS NEVER A SECOND RUN. The control plane redelivers a delivery
// until it is acked, on a timer and on every reconnect. A delivery id this
// node is still serving is ignored; one it has served is answered with the
// ack it gave, on whichever link is up when it is asked again.
//
// It is on by default and costs nothing when the control plane has no link:
// a control plane that refuses the upgrade is asked again only every few
// minutes. AGENTFIELD_NODE_LINK=off turns it off.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const linkVersion = 1

type linkFrame struct {
	V    int             `json:"v"`
	T    string          `json:"t"`
	ID   string          `json:"id,omitempty"`
	Exec string          `json:"exec,omitempty"`
	Seq  int64           `json:"seq,omitempty"`
	TS   string          `json:"ts,omitempty"`
	Body json.RawMessage `json:"body,omitempty"`
}

type linkDeliver struct {
	Kind    string            `json:"kind"`
	Message string            `json:"message,omitempty"`
	Reason  string            `json:"reason,omitempty"`
	Method  string            `json:"method,omitempty"`
	Path    string            `json:"path,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	Body    []byte            `json:"body,omitempty"`
}

type linkHTTPResult struct {
	Code    int               `json:"code"`
	Headers map[string]string `json:"headers,omitempty"`
	Body    []byte            `json:"body,omitempty"`
}

// linkState is the link's live connection and what it has served.
type linkState struct {
	mu     sync.Mutex
	conn   *websocket.Conn
	wmu    sync.Mutex
	served map[string]*servedDelivery
	// seq is this node's own up-frame counter (hello is 1); waits holds the
	// asker of each frame still waiting for its ok, by that counter.
	seq   int64
	waits map[int64]chan linkFrame
}

// errNoLink says no link is up to carry a frame; the caller uses HTTP.
var errNoLink = errors.New("no node link is up")

// linkOK is the body of the control plane's ok to an up frame.
type linkOK struct {
	Code    int    `json:"code"`
	Error   string `json:"error,omitempty"`
	Through int64  `json:"through,omitempty"`
	Dropped int64  `json:"dropped,omitempty"`
}

// linkAsk sends one up frame and waits for its ok. THE FRAME'S seq IS THIS
// NODE'S OWN COUNTER AND NOTHING ELSE: the control plane answers it by that
// number, so a data seq never goes there (it travels in the body).
func (a *Agent) linkAsk(t, exec string, body any, timeout time.Duration) (linkOK, error) {
	reply := make(chan linkFrame, 1)
	a.link.mu.Lock()
	if a.link.conn == nil {
		a.link.mu.Unlock()
		return linkOK{}, errNoLink
	}
	if a.link.seq < 1 {
		a.link.seq = 1
	}
	a.link.seq++
	seq := a.link.seq
	if a.link.waits == nil {
		a.link.waits = map[int64]chan linkFrame{}
	}
	a.link.waits[seq] = reply
	a.link.mu.Unlock()
	defer func() {
		a.link.mu.Lock()
		delete(a.link.waits, seq)
		a.link.mu.Unlock()
	}()
	if !a.linkWrite(linkFrame{T: t, Exec: exec, Seq: seq}, body) {
		return linkOK{}, errNoLink
	}
	select {
	case f := <-reply:
		var ok linkOK
		if err := json.Unmarshal(f.Body, &ok); err != nil {
			return linkOK{}, fmt.Errorf("the control plane's answer to %s is not an ok: %w", t, err)
		}
		return ok, nil
	case <-time.After(timeout):
		return linkOK{}, fmt.Errorf("the control plane did not answer %s within %s", t, timeout)
	}
}

// linkAnswered hands an ok frame to the frame that is waiting for it.
func (a *Agent) linkAnswered(f linkFrame) {
	a.link.mu.Lock()
	reply := a.link.waits[f.Seq]
	a.link.mu.Unlock()
	if reply != nil {
		select {
		case reply <- f:
		default:
		}
	}
}

type servedDelivery struct {
	ack  []byte // nil while still being served
	exec string
	at   time.Time
}

// linkKeep is how long a served delivery's ack is kept for a redelivery.
const linkKeep = 30 * time.Minute

func linkEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENTFIELD_NODE_LINK"))) {
	case "0", "off", "false", "no":
		return false
	}
	return true
}

// LinkOnly says this node has no inbound port and is reached only down its
// link (AGENTFIELD_NODE_LINK_ONLY=1).
func LinkOnly() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENTFIELD_NODE_LINK_ONLY"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// linkOnlyURL is the base_url a link-only node registers.
func linkOnlyURL(nodeID string) string { return "http://" + nodeID + ".nodelink" }

// linkURL is the control plane's link address for this node.
func (a *Agent) linkURL() (string, bool) {
	base, err := url.Parse(strings.TrimSpace(a.cfg.AgentFieldURL))
	if err != nil || base.Host == "" {
		return "", false
	}
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	default:
		base.Scheme = "ws"
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/api/v1/nodes/" + url.PathEscape(a.cfg.NodeID) + "/link"
	return base.String(), true
}

func (a *Agent) startLink() {
	if !linkEnabled() || a.cfg.NodeID == "" {
		return
	}
	target, ok := a.linkURL()
	if !ok {
		return
	}
	go a.linkLoop(target)
}

func (a *Agent) linkLoop(target string) {
	backoff := time.Second
	for {
		up, status := a.linkOnce(target)
		if up {
			backoff = time.Second
		}
		wait := backoff
		if !LinkOnly() && (status == http.StatusNotFound || status == http.StatusUnauthorized || status == http.StatusForbidden) {
			// No link on this control plane (or not for us): ask again rarely.
			wait = 5 * time.Minute
		}
		select {
		case <-a.stopLease:
			return
		case <-time.After(wait):
		}
		if backoff < 10*time.Second {
			backoff *= 2
		}
	}
}

// linkWrite sends one frame on whichever link is up now; false when none is.
func (a *Agent) linkWrite(f linkFrame, body any) bool {
	f.V = linkVersion
	f.TS = time.Now().UTC().Format(time.RFC3339Nano)
	if body != nil {
		raw, _ := json.Marshal(body)
		f.Body = raw
	}
	raw, _ := json.Marshal(f)
	a.link.mu.Lock()
	conn := a.link.conn
	a.link.mu.Unlock()
	if conn == nil {
		return false
	}
	a.link.wmu.Lock()
	defer a.link.wmu.Unlock()
	_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return conn.WriteMessage(websocket.TextMessage, raw) == nil
}

// linkOnce holds one link until it drops. It says whether the link came up,
// and the HTTP status a refused upgrade answered.
func (a *Agent) linkOnce(target string) (bool, int) {
	header := http.Header{}
	if a.cfg.APIKey != "" {
		header.Set("X-API-Key", a.cfg.APIKey)
	} else if a.cfg.Token != "" {
		header.Set("Authorization", "Bearer "+a.cfg.Token)
	}
	header.Set("X-Agent-Instance-ID", a.cfg.NodeID)
	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, resp, err := dialer.Dial(target, header)
	if err != nil {
		if resp != nil {
			return false, resp.StatusCode
		}
		return false, 0
	}
	defer conn.Close()
	a.link.mu.Lock()
	a.link.conn = conn
	if a.link.served == nil {
		a.link.served = map[string]*servedDelivery{}
	}
	a.link.mu.Unlock()
	defer func() {
		a.link.mu.Lock()
		if a.link.conn == conn {
			a.link.conn = nil
		}
		a.link.mu.Unlock()
	}()
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-a.stopLease:
			_ = conn.Close()
		case <-stop:
		}
	}()
	if !a.linkWrite(linkFrame{T: "hello", Seq: 1}, map[string]any{"instance_id": a.cfg.NodeID, "sdk": "go", "version": a.cfg.Version, "accepts": []string{"http", "steer", "cancel"}}) {
		return false, 0
	}
	a.logger.Printf("node link up (%s)", target)
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			a.logger.Printf("node link down: %v", err)
			return true, 0
		}
		var f linkFrame
		if json.Unmarshal(raw, &f) != nil {
			continue
		}
		switch f.T {
		case "deliver":
			a.linkServe(f)
		case "ok":
			a.linkAnswered(f)
		}
	}
}

// linkServe answers one delivery, once.
func (a *Agent) linkServe(f linkFrame) {
	a.link.mu.Lock()
	if a.link.served == nil {
		a.link.served = map[string]*servedDelivery{}
	}
	for id, s := range a.link.served {
		if s.ack != nil && time.Since(s.at) > linkKeep {
			delete(a.link.served, id)
		}
	}
	seen := a.link.served[f.ID]
	if seen == nil {
		a.link.served[f.ID] = &servedDelivery{exec: f.Exec, at: time.Now()}
	}
	var cached []byte
	if seen != nil {
		cached = seen.ack
	}
	a.link.mu.Unlock()
	if seen != nil {
		if cached != nil {
			a.linkWrite(linkFrame{T: "ack", ID: f.ID, Exec: f.Exec}, json.RawMessage(cached))
		}
		return
	}
	var d linkDeliver
	_ = json.Unmarshal(f.Body, &d)
	answer := func(ack map[string]any) {
		raw, _ := json.Marshal(ack)
		a.link.mu.Lock()
		if s := a.link.served[f.ID]; s != nil {
			s.ack, s.at = raw, time.Now()
		}
		a.link.mu.Unlock()
		a.linkWrite(linkFrame{T: "ack", ID: f.ID, Exec: f.Exec}, json.RawMessage(raw))
	}
	switch d.Kind {
	case "http":
		// A request can run for minutes (a synchronous reasoner), so it is
		// served beside the read loop, never in it.
		go func() { answer(map[string]any{"outcome": "accepted", "result": a.serveLinkHTTP(d)}) }()
	case "steer":
		if err := a.steerFromControlPlane(f.Exec, d.Message); err != nil {
			answer(map[string]any{"outcome": "rejected", "error": err.Error()})
		} else {
			a.logger.Printf("steer for %s arrived over the link", f.Exec)
			answer(map[string]any{"outcome": "accepted"})
		}
	case "cancel":
		a.CancelExecution(f.Exec)
		answer(map[string]any{"outcome": "accepted"})
	default:
		answer(map[string]any{"outcome": "rejected", "error": "this node does not take " + d.Kind + " over the link"})
	}
}

// serveLinkHTTP runs one carried request on this agent's own handler.
func (a *Agent) serveLinkHTTP(d linkDeliver) linkHTTPResult {
	method := d.Method
	if method == "" {
		method = http.MethodPost
	}
	req := httptest.NewRequest(method, d.Path, bytes.NewReader(d.Body))
	for k, v := range d.Headers {
		req.Header.Set(k, v)
	}
	req.RemoteAddr = "127.0.0.1:0"
	if strings.HasSuffix(d.Path, "/steer") {
		a.logger.Printf("steer for %s arrived over the link", req.Header.Get("X-Execution-ID"))
	}
	rec := httptest.NewRecorder()
	a.handler().ServeHTTP(rec, req)
	out := linkHTTPResult{Code: rec.Code, Body: rec.Body.Bytes(), Headers: map[string]string{}}
	for k, v := range rec.Header() {
		if len(v) > 0 {
			out.Headers[k] = v[0]
		}
	}
	return out
}
