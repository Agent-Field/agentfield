package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
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
// Patterns use the control plane's glob syntax (Go's path/filepath.Match):
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
	MaxReconnectAttempts int
}

// MemoryHistoryOptions selects past memory changes for Memory.History.
type MemoryHistoryOptions struct {
	Patterns []string
	Scope    MemoryScope
	ScopeID  string
	// Since returns only events after this time. Zero means no lower bound.
	Since time.Time
	// Limit caps the number of events. Zero means the control plane default.
	Limit int
}

// MemorySubscription is a live stream of memory change events.
type MemorySubscription interface {
	// Events delivers changes in the order the control plane sent them. It is
	// closed when the subscription ends.
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
// subscription reconnects with exponential backoff (1s doubling to 30s).
// Changes made while disconnected are not replayed; use History with Since
// set to the last event's Timestamp to backfill them.
//
// Delivery blocks while the Events channel is full, so a slow consumer slows
// the stream rather than losing events.
func (m *Memory) Watch(ctx context.Context, opts WatchOptions) (MemorySubscription, error) {
	eb, ok := m.backend.(MemoryEventBackend)
	if !ok {
		return nil, ErrMemoryEventsUnsupported
	}
	return eb.WatchMemory(ctx, opts)
}

// OnChange calls handler for every change to a key matching pattern, in any
// scope. Handlers run one at a time on a dedicated goroutine. Close the
// returned subscription to stop. Use Watch for scope filtering.
func (m *Memory) OnChange(ctx context.Context, pattern string, handler func(MemoryChangeEvent)) (MemorySubscription, error) {
	if handler == nil {
		return nil, errors.New("memory OnChange: handler is nil")
	}
	sub, err := m.Watch(ctx, WatchOptions{Patterns: []string{pattern}})
	if err != nil {
		return nil, err
	}
	go func() {
		for ev := range sub.Events() {
			handler(ev)
		}
	}()
	return sub, nil
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
	memoryEventsMaxReconnectBackoff   = 30 * time.Second
	memoryEventsHandshakeTimeout      = 10 * time.Second
	memoryEventsHandshakeBodyMaxBytes = 4096
)

// memoryEventsReconnectBase is the first reconnect delay; tests shorten it.
var memoryEventsReconnectBase = time.Second

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
	return e.StatusCode >= 400 && e.StatusCode < 500 &&
		e.StatusCode != http.StatusRequestTimeout && e.StatusCode != http.StatusTooManyRequests
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
		maxReconnects: maxReconnects,
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
		query.Set("limit", strconv.Itoa(opts.Limit))
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
	for i := range events {
		events[i].Scope = sdkScope(events[i].Scope)
	}
	return events, nil
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
		if _, err := filepath.Match(p, ""); err != nil {
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

// sdkScope maps the control plane's "actor" scope back to ScopeUser.
func sdkScope(scope MemoryScope) MemoryScope {
	if scope == "actor" {
		return ScopeUser
	}
	return scope
}

type wsMemorySubscription struct {
	target        string
	header        http.Header
	maxReconnects int

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
	dialer := websocket.Dialer{
		Proxy:            http.ProxyFromEnvironment,
		HandshakeTimeout: memoryEventsHandshakeTimeout,
	}
	conn, resp, err := dialer.DialContext(ctx, s.target, s.header)
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

	for {
		readErr := s.pump(conn)
		if s.ctx.Err() != nil {
			return
		}

		attempts := 0
		for conn = nil; conn == nil; {
			if attempts >= s.maxReconnects {
				s.err = fmt.Errorf("memory events connection lost after %d reconnect attempts: %w", attempts, readErr)
				return
			}
			attempts++
			timer := time.NewTimer(memoryEventsReconnectDelay(attempts))
			select {
			case <-s.ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
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
	}
}

// pump forwards events from one connection until it fails or the
// subscription is cancelled. It always closes conn.
func (s *wsMemorySubscription) pump(conn *websocket.Conn) error {
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-s.ctx.Done():
			_ = conn.Close()
		case <-stop:
		}
	}()
	defer func() { _ = conn.Close() }()

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			return err
		}
		var ev MemoryChangeEvent
		if err := json.Unmarshal(msg, &ev); err != nil {
			// A frame we cannot decode is skipped rather than ending the stream.
			continue
		}
		ev.Scope = sdkScope(ev.Scope)
		select {
		case s.events <- ev:
		case <-s.ctx.Done():
			return s.ctx.Err()
		}
	}
}

func memoryEventsReconnectDelay(attempt int) time.Duration {
	delay := memoryEventsReconnectBase
	for i := 1; i < attempt && delay < memoryEventsMaxReconnectBackoff; i++ {
		delay *= 2
	}
	if delay > memoryEventsMaxReconnectBackoff {
		delay = memoryEventsMaxReconnectBackoff
	}
	return delay
}
