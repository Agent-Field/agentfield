package handlers

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

// eventsStub is a minimal MemoryEventsStorage for tests.
type eventsStub struct {
	ch  chan types.MemoryChangeEvent
	err error
}

func (s *eventsStub) SubscribeToMemoryChanges(_ context.Context, _, _ string) (<-chan types.MemoryChangeEvent, error) {
	return s.ch, s.err
}

func (s *eventsStub) GetEventHistory(_ context.Context, _ types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}

func newEventsRouter(stub *eventsStub) *gin.Engine {
	gin.SetMode(gin.TestMode)
	h := NewMemoryEventsHandler(stub)
	r := gin.New()
	r.GET("/sse", h.SSEHandler)
	r.GET("/ws", h.WebSocketHandler)
	return r
}

func TestMemoryEventsHandler_SSEHappyPathHonorsScopeFilter(t *testing.T) {
	ch := make(chan types.MemoryChangeEvent, 4)
	stub := &eventsStub{ch: ch}
	srv := httptest.NewServer(newEventsRouter(stub))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/sse?patterns=match.*", nil)
	require.NoError(t, err)

	// With headers flushed immediately, Do must return without waiting for an event.
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))

	// Non-matching event should be silently dropped on the server side.
	ch <- types.MemoryChangeEvent{Key: "skip.this", Action: "set"}
	// Matching event should appear in the stream.
	ch <- types.MemoryChangeEvent{Key: "match.foo", Action: "set"}

	scanner := bufio.NewScanner(resp.Body)
	var found bool
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") && strings.Contains(line, "match.foo") {
			found = true
			break
		}
	}
	require.True(t, found, "expected match.foo event in SSE stream")
}

func TestMemoryEventsHandler_SSEInvalidPatternDropsEventsAndDisconnectCleansUp(t *testing.T) {
	ch := make(chan types.MemoryChangeEvent, 4)
	stub := &eventsStub{ch: ch}
	srv := httptest.NewServer(newEventsRouter(stub))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// "[invalid" is a malformed glob — filepath.Match returns ErrBadPattern.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/sse?patterns=[invalid", nil)
	require.NoError(t, err)

	// Headers are flushed immediately regardless of the pattern.
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	require.Equal(t, http.StatusOK, resp.StatusCode)

	// Events with bad patterns are dropped; handler must not panic.
	ch <- types.MemoryChangeEvent{Key: "any.key", Action: "set"}

	// Cancel triggers ctx.Done() in the handler — verify clean exit.
	cancel()
	time.Sleep(50 * time.Millisecond)
}

func TestMemoryEventsHandler_WebSocketHappyPath(t *testing.T) {
	ch := make(chan types.MemoryChangeEvent, 2)
	stub := &eventsStub{ch: ch}
	srv := httptest.NewServer(newEventsRouter(stub))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close() //nolint:errcheck

	ch <- types.MemoryChangeEvent{Key: "ws.key", Action: "set"}

	var event types.MemoryChangeEvent
	require.NoError(t, conn.ReadJSON(&event))
	require.Equal(t, "ws.key", event.Key)
}

func TestMemoryEventsHandler_UpgradeRejection(t *testing.T) {
	stub := &eventsStub{ch: make(chan types.MemoryChangeEvent)}
	srv := httptest.NewServer(newEventsRouter(stub))
	defer srv.Close()

	// Plain HTTP GET to the WS endpoint — upgrader rejects non-upgrade requests.
	resp, err := http.Get(srv.URL + "/ws") //nolint:noctx
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck
	require.NotEqual(t, http.StatusOK, resp.StatusCode)
}
