package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

// Memory change actions, as the control plane reports them in MemoryChangeEvent.Action.
// The control plane does not distinguish a first write from an overwrite: both
// are MemoryActionSet. A set whose PreviousData is empty had no prior value.
const (
	MemoryActionSet    = "set"
	MemoryActionDelete = "delete"
)

// ErrMemoryEventsUnsupported is returned by Memory.Watch, Memory.OnChange and
// Memory.History when the memory backend cannot stream change events. Only
// ControlPlaneMemoryBackend can; the default in-memory backend cannot.
var ErrMemoryEventsUnsupported = errors.New("memory backend does not support change events; configure a ControlPlaneMemoryBackend")

// MemoryChangeEvent describes one change to a memory key. It mirrors the
// control plane's wire format, except that the "actor" scope is reported as
// ScopeUser so it compares equal to the SDK's own scope constants.
type MemoryChangeEvent struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Scope     MemoryScope `json:"scope"`
	ScopeID   string      `json:"scope_id"`
	Key       string      `json:"key"`
	// Action is MemoryActionSet or MemoryActionDelete.
	Action string `json:"action"`
	// Data is the new value as JSON. It is empty for deletes.
	Data json.RawMessage `json:"data,omitempty"`
	// PreviousData is the value before the change as JSON, if there was one.
	PreviousData json.RawMessage     `json:"previous_data,omitempty"`
	Metadata     MemoryEventMetadata `json:"metadata"`
}

// UnmarshalJSON decodes the control plane's wire format, mapping its "actor"
// scope to ScopeUser.
func (e *MemoryChangeEvent) UnmarshalJSON(data []byte) error {
	type wire MemoryChangeEvent
	if err := json.Unmarshal(data, (*wire)(e)); err != nil {
		return err
	}
	if e.Scope == "actor" {
		e.Scope = ScopeUser
	}
	return nil
}

// MemoryEventMetadata identifies who made a memory change.
type MemoryEventMetadata struct {
	AgentID    string `json:"agent_id,omitempty"`
	ActorID    string `json:"actor_id,omitempty"`
	WorkflowID string `json:"workflow_id,omitempty"`
}

// Value decodes Data into a generic Go value. It returns nil for deletes.
func (e MemoryChangeEvent) Value() (any, error) {
	if len(e.Data) == 0 {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal(e.Data, &v); err != nil {
		return nil, err
	}
	return v, nil
}

// DecodeData unmarshals Data into dest.
func (e MemoryChangeEvent) DecodeData(dest any) error {
	if len(e.Data) == 0 {
		return errors.New("memory event has no data")
	}
	return json.Unmarshal(e.Data, dest)
}

// WatchOptions selects which memory changes a subscription receives.
//
// Patterns use the control plane's glob syntax (Go's path.Match):
// "*" matches any run of characters except "/", "?" matches one character,
// and "[...]" matches a character class. Every other character, including
// ".", is literal, so "config.*" matches "config.theme" but not "configX".
// No patterns means every key.
type WatchOptions struct {
	Patterns []string
	// Scope limits events to one scope. Empty means every scope.
	Scope MemoryScope
	// ScopeID limits events to one scope ID. Empty means every ID.
	ScopeID string
	// BufferSize is the capacity of the Events channel. Default 64.
	BufferSize int
	// MaxReconnectAttempts bounds consecutive reconnects after the connection
	// drops. Zero means the default of 5; a negative value disables reconnects.
	// A connection only resets the count once it proves healthy by delivering
	// a message or staying up for 30s, so a server that accepts and then
	// immediately drops the connection still exhausts the attempts.
	MaxReconnectAttempts int
	// OnReconnect, if set, is called after the subscription reconnects. Changes
	// made between the drop and the reconnect were not delivered; backfill them
	// with History, setting Since to the last event's Timestamp. It runs on the
	// subscription's goroutine, so no events are delivered until it returns.
	// Do not call the subscription's Close from inside it: Close waits for
	// that goroutine and would deadlock. Cancel the Watch context instead.
	OnReconnect func()
}

// MemoryHistoryOptions selects past memory changes for Memory.History.
type MemoryHistoryOptions struct {
	Patterns []string
	Scope    MemoryScope
	ScopeID  string
	// Since returns only events strictly after this time, so passing the last
	// event already seen does not return it again. Zero means no lower bound.
	Since time.Time
	// Limit caps the number of events, keeping the most recent. Zero means the
	// control plane default.
	Limit int
}

// MemorySubscription is a live stream of memory change events.
type MemorySubscription interface {
	// Events delivers changes in the order the control plane published them.
	// Changes made concurrently by different writers have no defined order. It
	// is closed when the subscription ends.
	Events() <-chan MemoryChangeEvent
	// Err reports why the subscription ended: nil after Close or context
	// cancellation, otherwise the error that stopped reconnection.
	Err() error
	// Close ends the subscription and waits for it to stop. It is safe to call
	// more than once.
	Close() error
}

// MemoryEventBackend is implemented by memory backends that can stream and
// replay change events.
type MemoryEventBackend interface {
	WatchMemory(ctx context.Context, opts WatchOptions) (MemorySubscription, error)
	MemoryEventHistory(ctx context.Context, opts MemoryHistoryOptions) ([]MemoryChangeEvent, error)
}

// Watch subscribes to memory changes. The subscription lasts until Close is
// called or ctx is cancelled.
//
// The initial connection is made before Watch returns, so authorization and
// connection errors surface here. If the connection later drops, the
// subscription reconnects with exponential backoff (1s doubling to 30s). The
// connection is kept alive with pings, and one that goes silent for 60s is
// treated as dropped. Changes made while disconnected are not replayed; set
// WatchOptions.OnReconnect to learn about gaps and use History with Since set
// to the last event's Timestamp to backfill them.
//
// Delivery waits while the Events channel is full, but the control plane only
// buffers about 100 events per subscriber. A consumer that falls further
// behind is disconnected, which reconnects the subscription and calls
// OnReconnect, so keep consumers fast or hand work to another goroutine.
func (m *Memory) Watch(ctx context.Context, opts WatchOptions) (MemorySubscription, error) {
	eb, ok := m.backend.(MemoryEventBackend)
	if !ok {
		return nil, ErrMemoryEventsUnsupported
	}
	return eb.WatchMemory(ctx, opts)
}

// OnChange calls handler for every change to a key matching pattern, in any
// scope. Handlers run one at a time on a dedicated goroutine, in the order
// Events would deliver them. Use Watch for scope filtering, or to learn about
// reconnect gaps through WatchOptions.OnReconnect, which OnChange cannot report.
//
// Close the returned subscription to stop. Once Close returns, the handler is
// not running and will not be called again. Because Close waits for a running
// handler, do not call it from inside the handler; cancel ctx instead. Do not
// read the subscription's Events channel yourself.
func (m *Memory) OnChange(ctx context.Context, pattern string, handler func(MemoryChangeEvent)) (MemorySubscription, error) {
	if handler == nil {
		return nil, errors.New("memory OnChange: handler is nil")
	}
	ctx, cancel := context.WithCancel(ctx)
	sub, err := m.Watch(ctx, WatchOptions{Patterns: []string{pattern}})
	if err != nil {
		cancel()
		return nil, err
	}
	hs := &handlerSubscription{
		MemorySubscription: sub,
		cancel:             cancel,
		done:               make(chan struct{}),
	}
	go hs.run(ctx, handler)
	return hs, nil
}

// handlerSubscription runs OnChange's handler and stops it on Close.
type handlerSubscription struct {
	MemorySubscription
	cancel context.CancelFunc
	done   chan struct{}
}

func (h *handlerSubscription) run(ctx context.Context, handler func(MemoryChangeEvent)) {
	defer close(h.done)
	defer h.cancel()
	events := h.Events()
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-events:
			if !ok {
				return
			}
			// Buffered events may still be pending after Close or cancel;
			// they must not reach the handler.
			if ctx.Err() != nil {
				return
			}
			handler(ev)
		}
	}
}

// Close stops the subscription and waits for any running handler to return.
func (h *handlerSubscription) Close() error {
	h.cancel()
	err := h.MemorySubscription.Close()
	<-h.done
	return err
}

// History returns past memory changes recorded by the control plane.
func (m *Memory) History(ctx context.Context, opts MemoryHistoryOptions) ([]MemoryChangeEvent, error) {
	eb, ok := m.backend.(MemoryEventBackend)
	if !ok {
		return nil, ErrMemoryEventsUnsupported
	}
	return eb.MemoryEventHistory(ctx, opts)
}

const (
	defaultMemoryEventBuffer          = 64
	defaultMemoryEventReconnects      = 5
	memoryEventsHandshakeTimeout      = 10 * time.Second
	memoryEventsHandshakeBodyMaxBytes = 4096
	// memoryEventsMaxBackoff caps the reconnect delay.
	memoryEventsMaxBackoff = 30 * time.Second
	// memoryEventsStableAfter is how long a connection that delivers nothing
	// must stay up before it resets the reconnect count.
	memoryEventsStableAfter = 30 * time.Second
)

// memoryEventTimings holds the subscription's timing knobs. Zero fields take
// the defaults; tests shorten them per backend.
type memoryEventTimings struct {
	// reconnectBase is the first reconnect delay; it doubles up to
	// memoryEventsMaxBackoff.
	reconnectBase time.Duration
	// pingInterval is how often the client pings; pongWait is how long the
	// connection may stay silent before it is treated as dead.
	pingInterval time.Duration
	pongWait     time.Duration
}

func (t memoryEventTimings) withDefaults() memoryEventTimings {
	if t.reconnectBase <= 0 {
		t.reconnectBase = time.Second
	}
	if t.pingInterval <= 0 {
		t.pingInterval = 25 * time.Second
	}
	if t.pongWait <= 0 {
		t.pongWait = 60 * time.Second
	}
	return t
}

func (t memoryEventTimings) reconnectDelay(attempt int) time.Duration {
	delay := t.reconnectBase
	for i := 1; i < attempt && delay < memoryEventsMaxBackoff; i++ {
		delay *= 2
	}
	return min(delay, memoryEventsMaxBackoff)
}

// memoryEventsHandshakeError is a WebSocket upgrade the control plane refused.
type memoryEventsHandshakeError struct {
	StatusCode int
	Body       string
}

func (e *memoryEventsHandshakeError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("memory events subscribe failed: status=%d", e.StatusCode)
	}
	return fmt.Sprintf("memory events subscribe failed: status=%d body=%s", e.StatusCode, e.Body)
}

// permanent reports whether retrying cannot help, e.g. bad credentials or a
// forbidden scope.
func (e *memoryEventsHandshakeError) permanent() bool {
	return e.StatusCode >= 400 && !isTransientPollStatus(e.StatusCode)
}

// WatchMemory implements MemoryEventBackend over the control plane's
// /api/v1/memory/events/ws endpoint.
func (b *ControlPlaneMemoryBackend) WatchMemory(ctx context.Context, opts WatchOptions) (MemorySubscription, error) {
	query, err := b.eventQuery(opts.Patterns, opts.Scope, opts.ScopeID)
	if err != nil {
		return nil, err
	}
	target, err := url.Parse(b.baseURL)
	if err != nil {
		return nil, err
	}
	switch target.Scheme {
	case "https":
		target.Scheme = "wss"
	default:
		target.Scheme = "ws"
	}
	target = target.JoinPath("/api/v1/memory/events/ws")
	target.RawQuery = query.Encode()

	header := http.Header{}
	b.applyIdentityHeaders(header, opts.Scope, opts.ScopeID)

	bufferSize := opts.BufferSize
	if bufferSize <= 0 {
		bufferSize = defaultMemoryEventBuffer
	}
	maxReconnects := opts.MaxReconnectAttempts
	if maxReconnects == 0 {
		maxReconnects = defaultMemoryEventReconnects
	}

	sub := &wsMemorySubscription{
		target:        target.String(),
		header:        header,
		dialer:        b.wsDialer(),
		timings:       b.eventTimings.withDefaults(),
		maxReconnects: maxReconnects,
		onReconnect:   opts.OnReconnect,
		events:        make(chan MemoryChangeEvent, bufferSize),
		done:          make(chan struct{}),
	}
	conn, err := sub.dial(ctx)
	if err != nil {
		return nil, err
	}
	sub.ctx, sub.cancel = context.WithCancel(ctx)
	go sub.run(conn)
	return sub, nil
}

// MemoryEventHistory implements MemoryEventBackend over the control plane's
// /api/v1/memory/events/history endpoint.
func (b *ControlPlaneMemoryBackend) MemoryEventHistory(ctx context.Context, opts MemoryHistoryOptions) ([]MemoryChangeEvent, error) {
	query, err := b.eventQuery(opts.Patterns, opts.Scope, opts.ScopeID)
	if err != nil {
		return nil, err
	}
	if !opts.Since.IsZero() {
		query.Set("since", opts.Since.UTC().Format(time.RFC3339Nano))
	}
	if opts.Limit > 0 {
		limit := opts.Limit
		if !opts.Since.IsZero() {
			// Older control planes include an event at exactly Since; ask for
			// one more so dropping it below still leaves Limit events.
			limit++
		}
		query.Set("limit", strconv.Itoa(limit))
	}
	endpoint, err := url.JoinPath(b.baseURL, "/api/v1/memory/events/history")
	if err != nil {
		return nil, err
	}
	if encoded := query.Encode(); encoded != "" {
		endpoint += "?" + encoded
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	b.applyHeaders(req, opts.Scope, opts.ScopeID)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("memory event history failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var events []MemoryChangeEvent
	if err := json.NewDecoder(resp.Body).Decode(&events); err != nil {
		return nil, err
	}
	kept := events[:0]
	for _, ev := range events {
		if !opts.Since.IsZero() && !ev.Timestamp.After(opts.Since) {
			continue
		}
		kept = append(kept, ev)
	}
	if opts.Limit > 0 && len(kept) > opts.Limit {
		kept = kept[len(kept)-opts.Limit:]
	}
	return kept, nil
}

// wsDialer builds a WebSocket dialer that connects the way the backend's
// HTTP client does: same proxy, TLS settings, dialers and cookie jar. That is
// only possible when the client's Transport is a *http.Transport; a wrapping
// RoundTripper cannot be inspected, so its settings and any headers it
// injects do not apply to the WebSocket handshake.
func (b *ControlPlaneMemoryBackend) wsDialer() *websocket.Dialer {
	d := &websocket.Dialer{
		Proxy:            http.ProxyFromEnvironment,
		HandshakeTimeout: memoryEventsHandshakeTimeout,
	}
	rt := http.DefaultTransport
	if b.httpClient != nil {
		if b.httpClient.Transport != nil {
			rt = b.httpClient.Transport
		}
		d.Jar = b.httpClient.Jar
	}
	if t, ok := rt.(*http.Transport); ok {
		d.Proxy = t.Proxy
		d.NetDialContext = t.DialContext
		d.NetDialTLSContext = t.DialTLSContext
		if t.TLSClientConfig != nil {
			d.TLSClientConfig = t.TLSClientConfig.Clone()
		}
	}
	return d
}

// eventQuery builds the patterns/scope/scope_id filter the control plane
// applies server-side. Patterns are validated here so a typo fails fast
// instead of silently matching nothing.
func (b *ControlPlaneMemoryBackend) eventQuery(patterns []string, scope MemoryScope, scopeID string) (url.Values, error) {
	query := url.Values{}
	cleaned := make([]string, 0, len(patterns))
	for _, p := range patterns {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if strings.Contains(p, ",") {
			return nil, fmt.Errorf("memory event pattern %q: commas are not allowed", p)
		}
		if _, err := path.Match(p, ""); err != nil {
			return nil, fmt.Errorf("memory event pattern %q: %w", p, err)
		}
		cleaned = append(cleaned, p)
	}
	if len(cleaned) > 0 {
		query.Set("patterns", strings.Join(cleaned, ","))
	}
	if scope != "" {
		query.Set("scope", b.apiScope(scope))
	}
	if scopeID != "" {
		query.Set("scope_id", scopeID)
	}
	return query, nil
}

type wsMemorySubscription struct {
	target        string
	header        http.Header
	dialer        *websocket.Dialer
	timings       memoryEventTimings
	maxReconnects int
	onReconnect   func()

	ctx    context.Context
	cancel context.CancelFunc
	events chan MemoryChangeEvent
	done   chan struct{}
	err    error // written by run before done closes
}

func (s *wsMemorySubscription) Events() <-chan MemoryChangeEvent { return s.events }

func (s *wsMemorySubscription) Err() error {
	select {
	case <-s.done:
		return s.err
	default:
		return nil
	}
}

func (s *wsMemorySubscription) Close() error {
	s.cancel()
	<-s.done
	return nil
}

func (s *wsMemorySubscription) dial(ctx context.Context) (*websocket.Conn, error) {
	conn, resp, err := s.dialer.DialContext(ctx, s.target, s.header)
	if err != nil {
		if resp != nil {
			defer resp.Body.Close()
			body, _ := io.ReadAll(io.LimitReader(resp.Body, memoryEventsHandshakeBodyMaxBytes))
			return nil, &memoryEventsHandshakeError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(body))}
		}
		return nil, err
	}
	return conn, nil
}

func (s *wsMemorySubscription) run(conn *websocket.Conn) {
	defer close(s.done)
	defer close(s.events)

	// attempts counts reconnects since the last healthy connection, so a
	// server that accepts and then drops every connection still runs out.
	attempts := 0
	for {
		healthy, readErr := s.pump(conn)
		if s.ctx.Err() != nil {
			return
		}
		if healthy {
			attempts = 0
		}

		for conn = nil; conn == nil; {
			if attempts >= s.maxReconnects {
				s.err = fmt.Errorf("memory events connection lost after %d reconnect attempts: %w", attempts, readErr)
				return
			}
			attempts++
			if !sleepCtx(s.ctx, s.timings.reconnectDelay(attempts)) {
				return
			}

			var err error
			conn, err = s.dial(s.ctx)
			if err == nil {
				break
			}
			if s.ctx.Err() != nil {
				return
			}
			var hsErr *memoryEventsHandshakeError
			if errors.As(err, &hsErr) && hsErr.permanent() {
				s.err = err
				return
			}
			readErr = err
		}
		if s.onReconnect != nil {
			s.onReconnect()
		}
	}
}

// pump forwards events from one connection until it fails or the
// subscription is cancelled. It always closes conn. It reports the connection
// healthy if it delivered any message or stayed up for memoryEventsStableAfter.
func (s *wsMemorySubscription) pump(conn *websocket.Conn) (healthy bool, _ error) {
	started := time.Now()
	received := false

	// Any frame from the server proves the connection is alive. Without a
	// read deadline a half-open connection would block ReadMessage forever.
	extend := func() { _ = conn.SetReadDeadline(time.Now().Add(s.timings.pongWait)) }
	extend()
	conn.SetPongHandler(func(string) error { extend(); return nil })
	conn.SetPingHandler(func(data string) error {
		extend()
		err := conn.WriteControl(websocket.PongMessage, []byte(data), time.Now().Add(s.timings.pongWait))
		if errors.Is(err, websocket.ErrCloseSent) {
			return nil
		}
		return err
	})

	stop := make(chan struct{})
	defer close(stop)
	go func() {
		ping := time.NewTicker(s.timings.pingInterval)
		defer ping.Stop()
		for {
			select {
			case <-s.ctx.Done():
				_ = conn.Close()
				return
			case <-stop:
				return
			case <-ping.C:
				// A failed ping surfaces as a read error or deadline.
				_ = conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(s.timings.pongWait))
			}
		}
	}()
	defer func() { _ = conn.Close() }()

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			return received || time.Since(started) >= memoryEventsStableAfter, err
		}
		received = true
		extend()
		var ev MemoryChangeEvent
		if err := json.Unmarshal(msg, &ev); err != nil {
			// A frame we cannot decode is skipped rather than ending the stream.
			continue
		}
		select {
		case s.events <- ev:
			// Delivery may have waited on a slow consumer; that must not
			// count as silence from the server.
			extend()
		case <-s.ctx.Done():
			return false, s.ctx.Err()
		}
	}
}
