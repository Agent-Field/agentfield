package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeMemoryEventsServer serves /api/v1/memory/events/ws and hands each
// accepted connection to the test.
type fakeMemoryEventsServer struct {
	*httptest.Server
	conns    chan *websocket.Conn
	requests chan *http.Request
	// reject, when non-zero, refuses the upgrade with that status.
	reject atomic.Int32
}

func newFakeMemoryEventsServer(t *testing.T) *fakeMemoryEventsServer {
	t.Helper()
	f := &fakeMemoryEventsServer{
		conns:    make(chan *websocket.Conn, 8),
		requests: make(chan *http.Request, 8),
	}
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/memory/events/ws" {
			http.NotFound(w, r)
			return
		}
		f.requests <- r
		if status := f.reject.Load(); status != 0 {
			http.Error(w, `{"error":"denied"}`, int(status))
			return
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		f.conns <- conn
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeMemoryEventsServer) nextConn(t *testing.T) *websocket.Conn {
	t.Helper()
	select {
	case c := <-f.conns:
		return c
	case <-time.After(5 * time.Second):
		t.Fatal("no websocket connection")
		return nil
	}
}

func (f *fakeMemoryEventsServer) nextRequest(t *testing.T) *http.Request {
	t.Helper()
	select {
	case r := <-f.requests:
		return r
	case <-time.After(5 * time.Second):
		t.Fatal("no websocket request")
		return nil
	}
}

func recvEvent(t *testing.T, sub MemorySubscription) MemoryChangeEvent {
	t.Helper()
	select {
	case ev, ok := <-sub.Events():
		require.True(t, ok, "events channel closed")
		return ev
	case <-time.After(5 * time.Second):
		t.Fatal("no event")
		return MemoryChangeEvent{}
	}
}

func waitClosed(t *testing.T, sub MemorySubscription) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-sub.Events():
			if !ok {
				return
			}
		case <-deadline:
			t.Fatal("events channel not closed")
		}
	}
}

func shortReconnects(t *testing.T) {
	t.Helper()
	prev := memoryEventsReconnectBase
	memoryEventsReconnectBase = 10 * time.Millisecond
	t.Cleanup(func() { memoryEventsReconnectBase = prev })
}

func TestMemoryWatch_SendsFiltersAndIdentity(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "tok", "node-1"))

	sub, err := mem.Watch(context.Background(), WatchOptions{
		Patterns: []string{"user_*", " config.* ", ""},
		Scope:    ScopeUser,
		ScopeID:  "actor-9",
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = sub.Close() })

	r := srv.nextRequest(t)
	q := r.URL.Query()
	assert.Equal(t, "user_*,config.*", q.Get("patterns"))
	assert.Equal(t, "actor", q.Get("scope"))
	assert.Equal(t, "actor-9", q.Get("scope_id"))
	assert.Equal(t, "Bearer tok", r.Header.Get("Authorization"))
	assert.Equal(t, "node-1", r.Header.Get("X-Agent-Node-ID"))
	assert.Equal(t, "actor-9", r.Header.Get("X-Actor-ID"))
}

func TestMemoryWatch_NoFiltersSendsNoQuery(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{})
	require.NoError(t, err)
	t.Cleanup(func() { _ = sub.Close() })

	assert.Empty(t, srv.nextRequest(t).URL.RawQuery)
}

func TestMemoryWatch_DecodesEvents(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{})
	require.NoError(t, err)
	t.Cleanup(func() { _ = sub.Close() })

	conn := srv.nextConn(t)
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{
		"id":"e1","type":"memory_change","timestamp":"2026-01-02T03:04:05Z",
		"scope":"actor","scope_id":"a1","key":"user_1","action":"set",
		"data":{"theme":"dark"},"previous_data":{"theme":"light"},
		"metadata":{"agent_id":"node-1","actor_id":"a1","workflow_id":"wf"}}`)))
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`not json`)))
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{
		"id":"e2","scope":"session","scope_id":"s1","key":"user_1","action":"delete",
		"previous_data":{"theme":"dark"}}`)))

	ev := recvEvent(t, sub)
	assert.Equal(t, "e1", ev.ID)
	assert.Equal(t, ScopeUser, ev.Scope)
	assert.Equal(t, "a1", ev.ScopeID)
	assert.Equal(t, MemoryActionSet, ev.Action)
	assert.Equal(t, "a1", ev.Metadata.ActorID)
	assert.Equal(t, time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), ev.Timestamp.UTC())
	val, err := ev.Value()
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"theme": "dark"}, val)
	var typed struct{ Theme string }
	require.NoError(t, ev.DecodeData(&typed))
	assert.Equal(t, "dark", typed.Theme)

	// The malformed frame is skipped.
	ev = recvEvent(t, sub)
	assert.Equal(t, "e2", ev.ID)
	assert.Equal(t, MemoryActionDelete, ev.Action)
	val, err = ev.Value()
	require.NoError(t, err)
	assert.Nil(t, val)
	assert.Error(t, ev.DecodeData(&typed))
}

func TestMemoryWatch_HandshakeRejectedReturnsError(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	srv.reject.Store(http.StatusForbidden)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{Scope: ScopeSession, ScopeID: "s"})
	require.Error(t, err)
	assert.Nil(t, sub)
	assert.Contains(t, err.Error(), "status=403")
	assert.Contains(t, err.Error(), "denied")
}

func TestMemoryWatch_InvalidPattern(t *testing.T) {
	mem := NewMemory(NewControlPlaneMemoryBackend("http://127.0.0.1:1", "", ""))

	_, err := mem.Watch(context.Background(), WatchOptions{Patterns: []string{"user_["}})
	assert.ErrorContains(t, err, "user_[")

	_, err = mem.Watch(context.Background(), WatchOptions{Patterns: []string{"a,b"}})
	assert.ErrorContains(t, err, "commas")
}

func TestMemoryWatch_UnsupportedBackend(t *testing.T) {
	mem := NewMemory(nil)

	_, err := mem.Watch(context.Background(), WatchOptions{})
	assert.ErrorIs(t, err, ErrMemoryEventsUnsupported)
	_, err = mem.OnChange(context.Background(), "*", func(MemoryChangeEvent) {})
	assert.ErrorIs(t, err, ErrMemoryEventsUnsupported)
	_, err = mem.History(context.Background(), MemoryHistoryOptions{})
	assert.ErrorIs(t, err, ErrMemoryEventsUnsupported)
}

func TestMemoryWatch_CloseEndsSubscription(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{})
	require.NoError(t, err)
	srv.nextConn(t)

	require.NoError(t, sub.Close())
	require.NoError(t, sub.Close())
	waitClosed(t, sub)
	assert.NoError(t, sub.Err())
}

func TestMemoryWatch_ContextCancelEndsSubscription(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	ctx, cancel := context.WithCancel(context.Background())
	sub, err := mem.Watch(ctx, WatchOptions{})
	require.NoError(t, err)
	srv.nextConn(t)

	cancel()
	waitClosed(t, sub)
	assert.NoError(t, sub.Err())
}

func TestMemoryWatch_CloseUnblocksPendingDelivery(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{BufferSize: 1})
	require.NoError(t, err)
	conn := srv.nextConn(t)
	for i := 0; i < 3; i++ {
		require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{"key":"k","action":"set"}`)))
	}
	// Nobody reads: the subscription blocks on a full buffer until Close.
	time.Sleep(50 * time.Millisecond)

	done := make(chan struct{})
	go func() { _ = sub.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Close blocked on a full buffer")
	}
}

func TestMemoryWatch_ReconnectsAfterDrop(t *testing.T) {
	shortReconnects(t)
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{Patterns: []string{"k*"}, Scope: ScopeSession})
	require.NoError(t, err)
	t.Cleanup(func() { _ = sub.Close() })

	srv.nextRequest(t)
	_ = srv.nextConn(t).Close()

	// The reconnect keeps the original filters.
	r := srv.nextRequest(t)
	assert.Equal(t, "k*", r.URL.Query().Get("patterns"))
	assert.Equal(t, "session", r.URL.Query().Get("scope"))

	conn := srv.nextConn(t)
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{"id":"after","key":"k1","action":"set"}`)))
	assert.Equal(t, "after", recvEvent(t, sub).ID)
	assert.NoError(t, sub.Err())
}

func TestMemoryWatch_GivesUpAfterMaxReconnects(t *testing.T) {
	shortReconnects(t)
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{MaxReconnectAttempts: 2})
	require.NoError(t, err)
	srv.nextRequest(t)

	srv.reject.Store(http.StatusServiceUnavailable)
	_ = srv.nextConn(t).Close()

	waitClosed(t, sub)
	require.Error(t, sub.Err())
	assert.Contains(t, sub.Err().Error(), "after 2 reconnect attempts")
	// One initial connection plus exactly two retries.
	srv.nextRequest(t)
	srv.nextRequest(t)
	select {
	case <-srv.requests:
		t.Fatal("reconnected more than MaxReconnectAttempts times")
	default:
	}
}

func TestMemoryWatch_StopsOnPermanentRejection(t *testing.T) {
	shortReconnects(t)
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{MaxReconnectAttempts: 10})
	require.NoError(t, err)
	srv.nextRequest(t)

	srv.reject.Store(http.StatusUnauthorized)
	_ = srv.nextConn(t).Close()

	waitClosed(t, sub)
	require.Error(t, sub.Err())
	assert.Contains(t, sub.Err().Error(), "status=401")
	srv.nextRequest(t)
	select {
	case <-srv.requests:
		t.Fatal("retried after a permanent rejection")
	default:
	}
}

func TestMemoryWatch_NegativeMaxReconnectsDisablesReconnect(t *testing.T) {
	shortReconnects(t)
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	sub, err := mem.Watch(context.Background(), WatchOptions{MaxReconnectAttempts: -1})
	require.NoError(t, err)
	_ = srv.nextConn(t).Close()

	waitClosed(t, sub)
	assert.ErrorContains(t, sub.Err(), "after 0 reconnect attempts")
}

func TestMemoryOnChange_DeliversToHandler(t *testing.T) {
	srv := newFakeMemoryEventsServer(t)
	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", ""))

	var mu sync.Mutex
	var keys []string
	got := make(chan struct{}, 2)
	sub, err := mem.OnChange(context.Background(), "user_*.preferences", func(ev MemoryChangeEvent) {
		mu.Lock()
		keys = append(keys, ev.Key)
		mu.Unlock()
		got <- struct{}{}
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = sub.Close() })

	assert.Equal(t, "user_*.preferences", srv.nextRequest(t).URL.Query().Get("patterns"))
	conn := srv.nextConn(t)
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{"key":"user_1.preferences","action":"set"}`)))
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(`{"key":"user_2.preferences","action":"delete"}`)))
	for i := 0; i < 2; i++ {
		select {
		case <-got:
		case <-time.After(5 * time.Second):
			t.Fatal("handler not called")
		}
	}
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"user_1.preferences", "user_2.preferences"}, keys)
}

func TestMemoryOnChange_NilHandler(t *testing.T) {
	mem := NewMemory(NewControlPlaneMemoryBackend("http://127.0.0.1:1", "", ""))
	_, err := mem.OnChange(context.Background(), "*", nil)
	assert.Error(t, err)
}

func TestMemoryWatch_HTTPSBaseURLUsesWSS(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusForbidden)
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "", "")
	// The TLS handshake fails against the self-signed cert, which proves the
	// client dialled wss:// rather than ws:// (which would get a plain 403).
	_, err := b.WatchMemory(context.Background(), WatchOptions{})
	require.Error(t, err)
	assert.NotContains(t, err.Error(), "status=403")
	assert.Contains(t, err.Error(), "certificate")
}

func TestMemoryHistory_SendsFiltersAndDecodes(t *testing.T) {
	var got *url.URL
	var gotHeader http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL
		gotHeader = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"id": "h1", "scope": "actor", "scope_id": "a1", "key": "user_1", "action": "set", "data": 5},
		})
	}))
	defer srv.Close()

	mem := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "tok", "node-1"))
	since := time.Date(2026, 5, 6, 7, 8, 9, 500, time.FixedZone("x", 3600))
	events, err := mem.History(context.Background(), MemoryHistoryOptions{
		Patterns: []string{"user_*"},
		Scope:    ScopeUser,
		ScopeID:  "a1",
		Since:    since,
		Limit:    10,
	})
	require.NoError(t, err)

	assert.Equal(t, "/api/v1/memory/events/history", got.Path)
	q := got.Query()
	assert.Equal(t, "user_*", q.Get("patterns"))
	assert.Equal(t, "actor", q.Get("scope"))
	assert.Equal(t, "a1", q.Get("scope_id"))
	assert.Equal(t, "10", q.Get("limit"))
	parsed, err := time.Parse(time.RFC3339, q.Get("since"))
	require.NoError(t, err)
	assert.True(t, parsed.Equal(since))
	assert.Equal(t, "Bearer tok", gotHeader.Get("Authorization"))

	require.Len(t, events, 1)
	assert.Equal(t, ScopeUser, events[0].Scope)
	val, err := events[0].Value()
	require.NoError(t, err)
	assert.Equal(t, float64(5), val)
}

func TestMemoryHistory_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := NewMemory(NewControlPlaneMemoryBackend(srv.URL, "", "")).History(context.Background(), MemoryHistoryOptions{})
	assert.ErrorContains(t, err, "status=500")
}

func TestMemoryEventsReconnectDelay(t *testing.T) {
	prev := memoryEventsReconnectBase
	memoryEventsReconnectBase = time.Second
	defer func() { memoryEventsReconnectBase = prev }()

	assert.Equal(t, time.Second, memoryEventsReconnectDelay(1))
	assert.Equal(t, 2*time.Second, memoryEventsReconnectDelay(2))
	assert.Equal(t, 16*time.Second, memoryEventsReconnectDelay(5))
	assert.Equal(t, 30*time.Second, memoryEventsReconnectDelay(6))
	assert.Equal(t, 30*time.Second, memoryEventsReconnectDelay(50))
}
