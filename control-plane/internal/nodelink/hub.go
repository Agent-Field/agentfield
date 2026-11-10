package nodelink

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Dispatch is a dispatch delivery's payload: the HTTP dispatch's body and
// headers, unchanged.
type Dispatch struct {
	Reasoner string
	Input    json.RawMessage
	Headers  map[string]string
}

// HTTPRequest is an http delivery's request: method, path with query,
// headers and body.
type HTTPRequest struct {
	Method  string
	Path    string
	Headers map[string]string
	Body    []byte
}

// Outcome is how a delivery ended: the node's ack, or expiry.
type Outcome struct {
	Outcome string
	Result  json.RawMessage
	Error   string
}

// Delivery is one thing the control plane sends down a node's link.
type Delivery struct {
	Kind        string
	ExecutionID string
	// Dedupe is the trigger's idempotency key, carried to the node so it can
	// dedupe on it as well as on the execution id.
	Dedupe   string
	Dispatch *Dispatch
	Reason   string
	// Message is a steer delivery's note.
	Message string
	// HTTP is an http delivery's request.
	HTTP *HTTPRequest
	// TTL is how long the delivery waits for an ack; zero is the kind's
	// default (dispatch 10 minutes, cancel 1 hour).
	TTL time.Duration
	// OnSettle runs once, on the first ack or on expiry. nil is fine.
	OnSettle func(Outcome)
}

type pending struct {
	id      string
	node    string
	d       Delivery
	sentAt  time.Time
	tries   int
	expires time.Time
	settled bool
}

type agentGetter interface {
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
}

// Hub holds every live link and every delivery waiting on one.
type Hub struct {
	// Router is the control plane's own router; up-frames re-enter it.
	Router http.Handler
	// APIKey is the control plane's API key: a node may authenticate with it,
	// and re-entered up-frames carry it.
	APIKey string
	// InternalToken is the control plane's internal token; a node may
	// authenticate with it too.
	InternalToken string
	// Agents is where a link's node must already be registered.
	Agents agentGetter
	// Pipe serves the harness-entry frames (pipe.go). nil refuses them.
	Pipe *Pipe

	// RedeliverAfter is how long a sent delivery waits for its ack before it
	// is sent again (default 30s). PingEvery and PongWait are the WebSocket
	// keepalive (default 20s and 45s). Sweep is the outbox's tick (1s).
	RedeliverAfter time.Duration
	PingEvery      time.Duration
	PongWait       time.Duration
	Sweep          time.Duration

	mu     sync.Mutex
	links  map[string]*link
	queue  map[string][]*pending
	byID   map[string]*pending
	once   sync.Once
	stopCh chan struct{}
}

type link struct {
	id       string
	node     string
	instance string
	version  string
	accepts  map[string]bool
	conn     *websocket.Conn
	wmu      sync.Mutex
	done     chan struct{}
}

func (h *Hub) init() {
	h.once.Do(func() {
		h.links = map[string]*link{}
		h.queue = map[string][]*pending{}
		h.byID = map[string]*pending{}
		h.stopCh = make(chan struct{})
		if h.RedeliverAfter <= 0 {
			h.RedeliverAfter = 30 * time.Second
		}
		if h.PingEvery <= 0 {
			h.PingEvery = 20 * time.Second
		}
		if h.PongWait <= 0 {
			h.PongWait = 45 * time.Second
		}
		if h.Sweep <= 0 {
			h.Sweep = time.Second
		}
		go h.sweep()
	})
}

// Close stops the sweeper and closes every link with 1012 (service restart),
// so nodes reconnect at once.
func (h *Hub) Close() {
	h.init()
	h.mu.Lock()
	select {
	case <-h.stopCh:
	default:
		close(h.stopCh)
	}
	links := make([]*link, 0, len(h.links))
	for _, l := range h.links {
		links = append(links, l)
	}
	h.mu.Unlock()
	for _, l := range links {
		l.write(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseServiceRestart, "restart"))
		_ = l.conn.Close()
	}
}

// Live says node holds a link now.
func (h *Hub) Live(node string) bool {
	h.init()
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.links[node] != nil
}

// LiveFor says node holds a link now that serves deliveries of kind. A link
// whose hello named no kinds serves the original two, dispatch and cancel.
func (h *Hub) LiveFor(node, kind string) bool {
	h.init()
	h.mu.Lock()
	defer h.mu.Unlock()
	l := h.links[node]
	if l == nil {
		return false
	}
	if l.accepts == nil {
		return kind == KindDispatch || kind == KindCancel
	}
	return l.accepts[kind]
}

// Links is every node with a live link and its instance, for status pages.
func (h *Hub) Links() map[string]string {
	h.init()
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make(map[string]string, len(h.links))
	for n, l := range h.links {
		out[n] = l.instance
	}
	return out
}

// Deliver queues d for node and pushes it at once when the node is linked.
// It returns the delivery id. A delivery is pushed again every
// RedeliverAfter, and on every reconnect, until an ack settles it or it
// expires.
func (h *Hub) Deliver(node string, d Delivery) string {
	h.init()
	ttl := d.TTL
	if ttl <= 0 {
		ttl = 10 * time.Minute
		if d.Kind == KindCancel {
			ttl = time.Hour
		}
	}
	p := &pending{id: newID("dlv_"), node: node, d: d, expires: time.Now().Add(ttl)}
	h.mu.Lock()
	h.queue[node] = append(h.queue[node], p)
	h.byID[p.id] = p
	l := h.links[node]
	h.mu.Unlock()
	if l != nil {
		h.push(l, p)
	}
	return p.id
}

// Pending is how many deliveries wait for node, for tests and status.
func (h *Hub) Pending(node string) int {
	h.init()
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.queue[node])
}

func (h *Hub) push(l *link, p *pending) {
	h.mu.Lock()
	if p.settled {
		h.mu.Unlock()
		return
	}
	p.sentAt = time.Now()
	p.tries++
	tries := p.tries
	h.mu.Unlock()
	body := DeliverBody{Kind: p.d.Kind, IdempotencyKey: p.d.Dedupe, Reason: p.d.Reason, Message: p.d.Message}
	if p.d.Dispatch != nil {
		body.Reasoner, body.Input, body.Headers = p.d.Dispatch.Reasoner, p.d.Dispatch.Input, p.d.Dispatch.Headers
	}
	if p.d.HTTP != nil {
		body.Method, body.Path, body.Headers, body.Body = p.d.HTTP.Method, p.d.HTTP.Path, p.d.HTTP.Headers, p.d.HTTP.Body
	}
	raw, err := Encode(Frame{T: TDeliver, ID: p.id, Exec: p.d.ExecutionID}, body)
	if err != nil {
		h.settle(p.id, Outcome{Outcome: OutcomeRejected, Error: err.Error()})
		return
	}
	if err := l.write(websocket.TextMessage, raw); err != nil {
		return
	}
	logger.Logger.Info().Str("node_id", l.node).Str("link_id", l.id).Str("delivery_id", p.id).Str("kind", p.d.Kind).Str("execution_id", p.d.ExecutionID).Int("attempt", tries).Msg("nodelink: delivered over link")
}

// settle ends a delivery once. A second ack for it (a node that got a
// redelivery and acked both) changes nothing.
func (h *Hub) settle(id string, o Outcome) bool {
	h.mu.Lock()
	p := h.byID[id]
	if p == nil || p.settled {
		h.mu.Unlock()
		return false
	}
	p.settled = true
	delete(h.byID, id)
	q := h.queue[p.node]
	for i, x := range q {
		if x == p {
			h.queue[p.node] = append(q[:i:i], q[i+1:]...)
			break
		}
	}
	if len(h.queue[p.node]) == 0 {
		delete(h.queue, p.node)
	}
	h.mu.Unlock()
	if p.d.OnSettle != nil {
		p.d.OnSettle(o)
	}
	return true
}

// sweep redelivers what is unacked and expires what is too old.
func (h *Hub) sweep() {
	t := time.NewTicker(h.Sweep)
	defer t.Stop()
	for {
		select {
		case <-h.stopCh:
			return
		case <-t.C:
		}
		now := time.Now()
		var expired []string
		type send struct {
			l *link
			p *pending
		}
		var sends []send
		h.mu.Lock()
		for node, q := range h.queue {
			l := h.links[node]
			for _, p := range q {
				switch {
				case now.After(p.expires):
					expired = append(expired, p.id)
				// An http delivery already on this link is not sent again on a
				// timer: a synchronous reasoner may work for minutes, and the
				// node dedupes on the delivery id anyway. It goes again only
				// on a new link (adopt clears sentAt).
				case l != nil && p.d.Kind == KindHTTP && !p.sentAt.IsZero():
				case l != nil && (p.sentAt.IsZero() || now.Sub(p.sentAt) >= h.RedeliverAfter):
					sends = append(sends, send{l, p})
				}
			}
		}
		h.mu.Unlock()
		for _, id := range expired {
			h.settle(id, Outcome{Outcome: OutcomeExpired})
		}
		for _, s := range sends {
			h.push(s.l, s.p)
		}
	}
}

func (l *link) write(kind int, raw []byte) error {
	l.wmu.Lock()
	defer l.wmu.Unlock()
	_ = l.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return l.conn.WriteMessage(kind, raw)
}

func (l *link) send(f Frame, body any) error {
	raw, err := Encode(f, body)
	if err != nil {
		return err
	}
	return l.write(websocket.TextMessage, raw)
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  32 << 10,
	WriteBufferSize: 32 << 10,
	// A node is not a browser. Any request carrying an Origin is refused
	// before the upgrade (see Handler), so the check here can pass.
	CheckOrigin: func(*http.Request) bool { return true },
}

// authorized says the upgrade carries the API key or the internal token, in
// Authorization: Bearer or X-API-Key. NEVER the query string. A control plane
// with neither configured lets every node link, as it lets every call in.
func (h *Hub) authorized(r *http.Request) bool {
	if h.APIKey == "" && h.InternalToken == "" {
		return true
	}
	got := strings.TrimSpace(r.Header.Get("X-API-Key"))
	if bearer, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok {
		got = strings.TrimSpace(bearer)
	}
	if got == "" {
		return false
	}
	for _, want := range []string{h.APIKey, h.InternalToken} {
		if want != "" && subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1 {
			return true
		}
	}
	return false
}

// Handler is GET /api/v1/nodes/:node_id/link.
func (h *Hub) Handler() gin.HandlerFunc {
	h.init()
	return func(c *gin.Context) {
		node := c.Param("node_id")
		if c.GetHeader("Origin") != "" {
			c.JSON(http.StatusForbidden, gin.H{"error": "a node link is not opened from a browser"})
			return
		}
		if !h.authorized(c.Request) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}
		if h.Agents != nil {
			if a, err := h.Agents.GetAgent(c.Request.Context(), node); err != nil || a == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "node not registered; register before opening a link"})
				return
			}
		}
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			return
		}
		conn.SetReadLimit(MaxFrame)
		h.serve(node, c.GetHeader("X-Agent-Instance-ID"), conn)
	}
}

func (h *Hub) serve(node, instance string, conn *websocket.Conn) {
	l := &link{id: newID("lnk_"), node: node, instance: instance, conn: conn, done: make(chan struct{})}
	defer close(l.done)
	if err := l.send(Frame{T: TWelcome}, Welcome{LinkID: l.id, HeartbeatS: int(h.PingEvery.Seconds()), MaxFrame: MaxFrame}); err != nil {
		_ = conn.Close()
		return
	}
	_ = conn.SetReadDeadline(time.Now().Add(h.PongWait))
	conn.SetPongHandler(func(string) error { return conn.SetReadDeadline(time.Now().Add(h.PongWait)) })
	go func() {
		t := time.NewTicker(h.PingEvery)
		defer t.Stop()
		for {
			select {
			case <-l.done:
				return
			case <-t.C:
				if err := l.write(websocket.PingMessage, nil); err != nil {
					return
				}
			}
		}
	}()
	helloed := false
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			break
		}
		_ = conn.SetReadDeadline(time.Now().Add(h.PongWait))
		f, err := Decode(raw)
		if err != nil {
			logger.Logger.Debug().Err(err).Str("node_id", node).Msg("nodelink: bad frame")
			continue
		}
		if !helloed && f.T != THello {
			continue
		}
		switch f.T {
		case THello:
			var hello Hello
			_ = json.Unmarshal(f.Body, &hello)
			if hello.InstanceID != "" {
				l.instance = hello.InstanceID
			}
			l.version = hello.Version
			if len(hello.Accepts) > 0 {
				l.accepts = map[string]bool{}
				for _, k := range hello.Accepts {
					l.accepts[k] = true
				}
			}
			h.adopt(l)
			helloed = true
			logger.Logger.Info().Str("node_id", node).Str("instance_id", l.instance).Str("link_id", l.id).Int("inflight", len(hello.Inflight)).Msg("nodelink: link up")
			code, msg := h.lease(l, hello.Inflight, nil)
			_ = l.send(Frame{T: TOK, Seq: f.Seq}, OKBody{Code: code, Error: msg})
			h.flush(l)
		case TAck:
			var ack Ack
			_ = json.Unmarshal(f.Body, &ack)
			if ack.Outcome == "" {
				ack.Outcome = OutcomeAccepted
			}
			h.settle(f.ID, Outcome(ack))
		case THeartbeat:
			var hb Heartbeat
			_ = json.Unmarshal(f.Body, &hb)
			code, msg := h.lease(l, hb.Inflight, hb.HealthScore)
			_ = l.send(Frame{T: TOK, Seq: f.Seq}, OKBody{Code: code, Error: msg})
		case TStatus, TProgress:
			body := f.Body
			if f.T == TProgress {
				body = asRunning(body)
			}
			code, msg := h.reenter(http.MethodPost, "/api/v1/executions/"+url.PathEscape(f.Exec)+"/status", body)
			_ = l.send(Frame{T: TOK, Seq: f.Seq, Exec: f.Exec}, OKBody{Code: code, Error: msg})
		case TLog:
			code, msg := h.reenter(http.MethodPost, "/api/v1/executions/"+url.PathEscape(f.Exec)+"/logs", f.Body)
			_ = l.send(Frame{T: TOK, Seq: f.Seq, Exec: f.Exec}, OKBody{Code: code, Error: msg})
		default:
			if h.Pipe == nil || !h.Pipe.frame(l, f) {
				_ = l.send(Frame{T: TOK, Seq: f.Seq, Exec: f.Exec}, OKBody{Code: http.StatusNotImplemented, Error: "this control plane does not take " + f.T + " frames"})
			}
		}
	}
	h.drop(l)
	_ = conn.Close()
	logger.Logger.Info().Str("node_id", node).Str("link_id", l.id).Msg("nodelink: link down")
}

// adopt makes l the node's link. PROTOTYPE: one link per node; a newer link
// replaces an older one (a reconnect that beat the old socket's timeout).
func (h *Hub) adopt(l *link) {
	h.mu.Lock()
	old := h.links[l.node]
	h.links[l.node] = l
	// Everything sent on an older link and not acked goes again now.
	for _, p := range h.queue[l.node] {
		p.sentAt = time.Time{}
	}
	h.mu.Unlock()
	if old != nil && old != l {
		_ = old.conn.Close()
	}
}

func (h *Hub) drop(l *link) {
	h.mu.Lock()
	if h.links[l.node] == l {
		delete(h.links, l.node)
	}
	h.mu.Unlock()
	if h.Pipe != nil {
		h.Pipe.unsubscribe(l)
	}
}

// flush pushes everything waiting for l's node, oldest first.
func (h *Hub) flush(l *link) {
	h.mu.Lock()
	q := append([]*pending(nil), h.queue[l.node]...)
	h.mu.Unlock()
	for _, p := range q {
		h.push(l, p)
	}
}

// lease is a heartbeat re-entered as the lease PATCH, with inflight.
func (h *Hub) lease(l *link, inflight []string, score *int) (int, string) {
	body := map[string]any{"phase": "ready", "inflight": inflight}
	if l.version != "" {
		body["version"] = l.version
	}
	if score != nil {
		body["health_score"] = *score
	}
	raw, _ := json.Marshal(body)
	return h.reenter(http.MethodPatch, "/api/v1/nodes/"+url.PathEscape(l.node)+"/status", raw)
}

// reenter runs one up-frame through the control plane's own router as the
// HTTP call it replaces, and answers that call's status and error.
func (h *Hub) reenter(method, path string, body []byte) (int, string) {
	if h.Router == nil {
		return http.StatusServiceUnavailable, "no router"
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-AgentField-Source", "nodelink")
	if h.APIKey != "" {
		req.Header.Set("X-API-Key", h.APIKey)
	}
	req.RemoteAddr = "127.0.0.1:0"
	rec := httptest.NewRecorder()
	h.Router.ServeHTTP(rec, req)
	if rec.Code >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(rec.Body, 600))
		return rec.Code, strings.TrimSpace(string(raw))
	}
	return rec.Code, ""
}

// asRunning makes a progress body a non-terminal status body.
func asRunning(body json.RawMessage) json.RawMessage {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil || m == nil {
		m = map[string]any{}
	}
	m["status"] = "running"
	raw, _ := json.Marshal(m)
	return raw
}

func newID(prefix string) string {
	b := make([]byte, 10)
	_, _ = rand.Read(b)
	return prefix + hex.EncodeToString(b)
}
