package middleware_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

type concreteMemoryStreamStore struct {
	storage.StorageProvider
	mu            sync.Mutex
	nodes         []*types.AgentNode
	events        chan types.MemoryChangeEvent
	participation atomic.Int32
}

func (s *concreteMemoryStreamStore) GetAgent(context.Context, string) (*types.AgentNode, error) {
	return nil, nil
}

func (s *concreteMemoryStreamStore) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*types.AgentNode(nil), s.nodes...), nil
}

func (s *concreteMemoryStreamStore) SubscribeToMemoryChanges(context.Context, string, string) (<-chan types.MemoryChangeEvent, error) {
	return s.events, nil
}

func (s *concreteMemoryStreamStore) QueryExecutionRecords(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
	s.participation.Add(1)
	return nil, nil
}

// SPEC-7: real concrete WS/SSE subscriptions must stop delivering an actor's
// events when it registers, after snapshot expiry. No sleeps or participation.
func TestMemoryConcreteEventStreamsReauthorize(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, transport := range []string{"ws", "sse"} {
		t.Run(transport, func(t *testing.T) {
			store := &concreteMemoryStreamStore{events: make(chan types.MemoryChangeEvent, 2)}
			var nanos atomic.Int64
			nanos.Store(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano())
			r := gin.New()
			r.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
			r.Use(middleware.MemoryPermissionMiddleware(nil, store, nil, middleware.MemoryPermissionConfig{
				Enabled: true, EnforceScopeOwnership: true, ExecutionReader: store,
			}))
			r.Use(middleware.MemoryRegistryClockForTest(func() time.Time { return time.Unix(0, nanos.Load()) }))
			h := handlers.NewMemoryEventsHandler(store)
			r.GET("/ws", h.WebSocketHandler)
			r.GET("/sse", h.SSEHandler)
			server := httptest.NewServer(r)
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			headers := http.Header{"X-API-Key": {"shared-key"}, "X-Agent-Node-ID": {"beta"}}
			path := "/" + transport + "?scope=actor&scope_id=alpha"
			var conn *websocket.Conn
			var reader *bufio.Reader
			if transport == "ws" {
				var err error
				conn, _, err = websocket.DefaultDialer.DialContext(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+path, headers)
				require.NoError(t, err)
				defer conn.Close()
				require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
			} else {
				req, err := http.NewRequestWithContext(ctx, "GET", server.URL+path, nil)
				require.NoError(t, err)
				req.Header = headers
				resp, err := server.Client().Do(req)
				require.NoError(t, err)
				defer resp.Body.Close()
				require.Equal(t, http.StatusOK, resp.StatusCode)
				reader = bufio.NewReader(resp.Body)
			}
			store.events <- types.MemoryChangeEvent{Scope: "actor", ScopeID: "alpha", Key: "before-registration"}
			var first types.MemoryChangeEvent
			if conn != nil {
				require.NoError(t, conn.ReadJSON(&first))
			} else {
				for {
					line, err := reader.ReadString('\n')
					require.NoError(t, err)
					if strings.HasPrefix(line, "data:") {
						require.NoError(t, json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &first))
						break
					}
				}
			}
			require.Equal(t, "before-registration", first.Key)
			store.mu.Lock()
			store.nodes = []*types.AgentNode{{ID: "alpha"}}
			store.mu.Unlock()
			nanos.Add(int64(3 * time.Second))
			store.events <- types.MemoryChangeEvent{Scope: "actor", ScopeID: "alpha", Key: "PRIVATE_AFTER_REGISTRATION"}
			close(store.events)
			if conn != nil {
				var next types.MemoryChangeEvent
				require.Error(t, conn.ReadJSON(&next), "stream must close without delivering the revoked event")
			} else {
				rest, err := io.ReadAll(reader)
				require.NoError(t, err)
				require.NotContains(t, string(rest), "PRIVATE_AFTER_REGISTRATION")
			}
			require.Zero(t, store.participation.Load(), "concrete selectors retain bearer semantics")
		})
	}
}
