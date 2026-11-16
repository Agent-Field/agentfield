package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

type memoryEventSubscription struct {
	scope   string
	scopeID string
	ch      chan types.MemoryChangeEvent
}

type memoryEventStorageStub struct {
	storage.StorageProvider
	mu           sync.Mutex
	subs         map[int]*memoryEventSubscription
	nextID       int
	subscribeErr error
}

func newMemoryEventStorageStub() *memoryEventStorageStub {
	return &memoryEventStorageStub{
		subs: make(map[int]*memoryEventSubscription),
	}
}

func (s *memoryEventStorageStub) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	if s.subscribeErr != nil {
		return nil, s.subscribeErr
	}

	sub := &memoryEventSubscription{
		scope:   scope,
		scopeID: scopeID,
		ch:      make(chan types.MemoryChangeEvent, 16),
	}

	s.mu.Lock()
	id := s.nextID
	s.nextID++
	s.subs[id] = sub
	s.mu.Unlock()

	go func() {
		<-ctx.Done()
		s.mu.Lock()
		if existing, ok := s.subs[id]; ok {
			delete(s.subs, id)
			close(existing.ch)
		}
		s.mu.Unlock()
	}()

	return sub.ch, nil
}

func (s *memoryEventStorageStub) publish(event types.MemoryChangeEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, sub := range s.subs {
		if sub.scope != "" && sub.scope != event.Scope {
			continue
		}
		if sub.scopeID != "" && sub.scopeID != event.ScopeID {
			continue
		}

		select {
		case sub.ch <- event:
		default:
		}
	}
}

// dropAll closes every subscriber channel while its context is still live,
// as storage does for a subscriber that falls behind.
func (s *memoryEventStorageStub) dropAll() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, sub := range s.subs {
		delete(s.subs, id)
		close(sub.ch)
	}
}

func (s *memoryEventStorageStub) ActiveSubscriptions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.subs)
}

func (s *memoryEventStorageStub) GetExecutionEventBus() *events.ExecutionEventBus {
	return events.NewExecutionEventBus()
}

func (s *memoryEventStorageStub) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return events.NewEventBus[*types.WorkflowExecutionEvent]()
}

func (s *memoryEventStorageStub) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return events.NewEventBus[*types.ExecutionLogEntry]()
}

func newMemoryEvent(scope, scopeID, key string) types.MemoryChangeEvent {
	return types.MemoryChangeEvent{
		ID:        fmt.Sprintf("%s:%s:%s", scope, scopeID, key),
		Type:      "memory.changed",
		Timestamp: time.Now().UTC(),
		Scope:     scope,
		ScopeID:   scopeID,
		Key:       key,
		Action:    "set",
		Data:      json.RawMessage(`{"ok":true}`),
	}
}

func newMemoryEventsRouter(store storage.StorageProvider) *gin.Engine {
	handler := NewMemoryEventsHandler(store)
	router := gin.New()
	router.GET("/ws", handler.WebSocketHandler)
	router.GET("/sse", handler.SSEHandler)
	return router
}

func newMemoryEventsTestServer(t *testing.T) (*memoryEventStorageStub, *httptest.Server) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	store := newMemoryEventStorageStub()
	server := httptest.NewServer(newMemoryEventsRouter(store))
	t.Cleanup(server.Close)
	return store, server
}

// dialMemoryEventsWS connects to the /ws endpoint and waits until the
// handler's subscription is active, so events published next are delivered.
func dialMemoryEventsWS(t *testing.T, store *memoryEventStorageStub, server *httptest.Server, query string) *websocket.Conn {
	t.Helper()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/ws"+query, nil)
	require.NoError(t, err)
	waitForCondition(t, time.Second, func() bool {
		return store.ActiveSubscriptions() == 1
	}, "websocket subscription should become active")
	return conn
}

func waitForCondition(t *testing.T, timeout time.Duration, fn func() bool, msg string) {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("condition not met: %s", msg)
}

func startSSERead(body io.Reader) (<-chan string, <-chan error) {
	dataCh := make(chan string, 1)
	errCh := make(chan error, 1)

	go func() {
		reader := bufio.NewReader(body)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				errCh <- err
				return
			}
			if strings.HasPrefix(line, "data:") {
				dataCh <- strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				return
			}
		}
	}()

	return dataCh, errCh
}

func TestMemoryEventsHandler_WebSocketUpgradeFailureReturnsBadRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newMemoryEventStorageStub()
	router := newMemoryEventsRouter(store)

	req := httptest.NewRequest(http.MethodGet, "/ws", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestMemoryEventsHandler_WebSocketUpgradeAndPatternFilter(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	conn := dialMemoryEventsWS(t, store, server, "?patterns=user.*")
	defer conn.Close()

	store.publish(newMemoryEvent("session", "s1", "system.name"))
	store.publish(newMemoryEvent("session", "s1", "user.name"))

	require.NoError(t, conn.SetReadDeadline(time.Now().Add(time.Second)))

	var event types.MemoryChangeEvent
	require.NoError(t, conn.ReadJSON(&event))
	require.Equal(t, "user.name", event.Key)
	require.Equal(t, "session", event.Scope)
	require.Equal(t, "s1", event.ScopeID)
}

func TestMemoryEventsHandler_SSEHappyPathHonorsScopeFilter(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+"/sse?scope=session&scope_id=s1", nil)
	require.NoError(t, err)

	// Run the request in a goroutine; headers are flushed immediately but we
	// still need to wait for the subscription to be active before publishing.
	type doResult struct {
		resp *http.Response
		err  error
	}
	doCh := make(chan doResult, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		doCh <- doResult{resp: resp, err: err}
	}()

	waitForCondition(t, 2*time.Second, func() bool {
		return store.ActiveSubscriptions() == 1
	}, "sse subscription should become active")

	store.publish(newMemoryEvent("session", "s2", "user.blocked"))
	store.publish(newMemoryEvent("session", "s1", "user.allowed"))

	var resp *http.Response
	select {
	case res := <-doCh:
		require.NoError(t, res.err)
		resp = res.resp
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for SSE response headers")
	}
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Contains(t, resp.Header.Get("Content-Type"), "text/event-stream")

	dataCh, errCh := startSSERead(resp.Body)

	select {
	case line := <-dataCh:
		require.Contains(t, line, "user.allowed")
		require.NotContains(t, line, "user.blocked")
	case err := <-errCh:
		t.Fatalf("unexpected SSE read error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for SSE event")
	}

	cancel()
	waitForCondition(t, 2*time.Second, func() bool {
		return store.ActiveSubscriptions() == 0
	}, "sse subscription should be released after cancel")
}

func TestMemoryEventsHandler_InvalidPatternIsRejected(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	resp, err := http.Get(server.URL + "/sse?patterns=[invalid")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusBadRequest, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	require.Contains(t, string(body), "[invalid")

	_, wsResp, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/ws?patterns=ok,[invalid", nil)
	require.Error(t, err)
	require.NotNil(t, wsResp)
	require.Equal(t, http.StatusBadRequest, wsResp.StatusCode)

	require.Equal(t, 0, store.ActiveSubscriptions())
}

func TestMemoryEventsHandler_WebSocketDisconnectReleasesSubscriptionWithoutEvents(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	conn := dialMemoryEventsWS(t, store, server, "")

	// No event is published: the handler must notice the disconnect itself.
	require.NoError(t, conn.Close())
	waitForCondition(t, 2*time.Second, func() bool {
		return store.ActiveSubscriptions() == 0
	}, "closing websocket should release the subscription")
}

func TestMemoryEventsHandler_WebSocketDroppedSubscriberGetsTryAgainLater(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	conn := dialMemoryEventsWS(t, store, server, "")
	defer conn.Close()

	// Storage closes the channel of a subscriber that fell behind.
	store.dropAll()

	require.NoError(t, conn.SetReadDeadline(time.Now().Add(2*time.Second)))
	_, _, err := conn.ReadMessage()
	require.True(t, websocket.IsCloseError(err, websocket.CloseTryAgainLater), "got %v", err)
}

func TestMemoryEventsHandler_WebSocketAnswersPings(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	conn := dialMemoryEventsWS(t, store, server, "")
	defer conn.Close()

	pong := make(chan string, 1)
	conn.SetPongHandler(func(data string) error { pong <- data; return nil })
	go func() {
		for {
			if _, _, err := conn.NextReader(); err != nil {
				return
			}
		}
	}()
	require.NoError(t, conn.WriteControl(websocket.PingMessage, []byte("hi"), time.Now().Add(time.Second)))
	select {
	case data := <-pong:
		require.Equal(t, "hi", data)
	case <-time.After(2 * time.Second):
		t.Fatal("no pong")
	}
}

func TestMemoryEventsHandler_WebSocketBackpressureDisconnectsCleanly(t *testing.T) {
	store, server := newMemoryEventsTestServer(t)

	baseline := runtime.NumGoroutine()

	conn := dialMemoryEventsWS(t, store, server, "")

	for i := 0; i < 100; i++ {
		store.publish(newMemoryEvent("session", "s1", fmt.Sprintf("user.%d", i)))
	}

	require.NoError(t, conn.Close())
	server.CloseClientConnections()
	// Give goroutines a moment to observe the closed connection before
	// publishing another event that triggers cleanup.
	time.Sleep(50 * time.Millisecond)
	store.publish(newMemoryEvent("session", "s1", "user.after-close"))

	// Use a generous timeout — goroutine cleanup after a burst of 100 events
	// through a closed WebSocket can take a few scheduling cycles, especially
	// under CI load.
	waitForCondition(t, 5*time.Second, func() bool {
		return store.ActiveSubscriptions() == 0 && runtime.NumGoroutine() <= baseline+10
	}, "closing websocket should release subscription after burst publish")
}
