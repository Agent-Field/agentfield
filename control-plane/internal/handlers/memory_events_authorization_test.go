package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// Reuse the live subscription fake, with mutable stored executions rather than
// precomputed visibility. Its mutex also guards membership and query counts.
type participatingMemoryEventStore struct {
	*memoryEventStorageStub
	executions []*types.Execution
	queries    int
	history    []*types.MemoryChangeEvent
}

func (s *participatingMemoryEventStore) QueryExecutionRecords(_ context.Context, f types.ExecutionFilter) ([]*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.queries++
	for _, execution := range s.executions {
		if f.AgentNodeID != nil && execution.AgentNodeID != *f.AgentNodeID {
			continue
		}
		if f.RunID != nil && execution.RunID != *f.RunID {
			continue
		}
		if f.SessionID != nil && (execution.SessionID == nil || *execution.SessionID != *f.SessionID) {
			continue
		}
		if f.ActorID != nil && (execution.ActorID == nil || *execution.ActorID != *f.ActorID) {
			continue
		}
		return []*types.Execution{execution}, nil
	}
	return nil, nil
}

func (s *participatingMemoryEventStore) addExecution(execution *types.Execution) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.executions = append(s.executions, execution)
}

func (s *participatingMemoryEventStore) queryCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queries
}

func (s *participatingMemoryEventStore) GetEventHistory(context.Context, types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return s.history, nil
}

func participatingMemoryEventsRouter(store *participatingMemoryEventStore) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
	r.Use(middleware.MemoryPermissionMiddleware(nil, nil, nil, middleware.MemoryPermissionConfig{
		Enabled: true, EnforceScopeOwnership: true, ExecutionReader: store,
	}))
	r.GET("/sse", NewMemoryEventsHandler(store).SSEHandler)
	r.GET("/history", GetEventHistoryHandler(store))
	return r
}

// C6 / SPEC-5: real SSE + auth middleware reproduce write, call beta, write.
// A global barrier proves the first workflow events were processed and dropped
// before beta's execution is inserted, without sleeps or a 30-second wait.
func TestMemoryEventsSSEImmediatelySeesNewParticipation(t *testing.T) {
	store := &participatingMemoryEventStore{memoryEventStorageStub: newMemoryEventStorageStub()}
	server := httptest.NewServer(participatingMemoryEventsRouter(store))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+"/sse", nil)
	require.NoError(t, err)
	req.Header.Set("X-API-Key", "shared-key")
	req.Header.Set("X-Agent-Node-ID", "agent-beta")
	resp, err := server.Client().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Contains(t, resp.Header.Get("Content-Type"), "text/event-stream")

	for _, key := range []string{"before-1", "before-2", "before-3"} {
		store.publish(newMemoryEvent("workflow", "run-1", key))
	}
	store.publish(newMemoryEvent("global", "global", "barrier"))
	readNext := func(wantKey string) {
		t.Helper()
		data, errors := startSSERead(resp.Body)
		select {
		case line := <-data:
			var event types.MemoryChangeEvent
			require.NoError(t, json.Unmarshal([]byte(line), &event))
			require.Equal(t, wantKey, event.Key)
		case err := <-errors:
			t.Fatalf("SSE read failed: %v", err)
		case <-ctx.Done():
			t.Fatal("timed out waiting for SSE event")
		}
	}
	readNext("barrier")
	require.Equal(t, 3, store.queryCount())
	store.addExecution(&types.Execution{AgentNodeID: "agent-beta", RunID: "run-1"})
	store.publish(newMemoryEvent("workflow", "run-1", "after-call"))
	readNext("after-call")
	require.Equal(t, 4, store.queryCount())
	store.publish(newMemoryEvent("workflow", "run-1", "cached-positive"))
	readNext("cached-positive")
	require.Equal(t, 4, store.queryCount())
}

// C6 / SPEC-5: the real history handler reuses a negative within one response,
// then makes a fresh decision on the next authenticated request.
func TestMemoryEventHistoryParticipationCacheIsPerRequest(t *testing.T) {
	first := newMemoryEvent("workflow", "run-1", "first")
	second := newMemoryEvent("workflow", "run-1", "second")
	store := &participatingMemoryEventStore{
		memoryEventStorageStub: newMemoryEventStorageStub(),
		history:                []*types.MemoryChangeEvent{&first, &second},
	}
	router := participatingMemoryEventsRouter(store)
	request := func() []*types.MemoryChangeEvent {
		t.Helper()
		req := httptest.NewRequest(http.MethodGet, "/history", nil)
		req.Header.Set("X-API-Key", "shared-key")
		req.Header.Set("X-Agent-Node-ID", "agent-beta")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, req)
		require.Equal(t, http.StatusOK, response.Code)
		var events []*types.MemoryChangeEvent
		require.NoError(t, json.Unmarshal(response.Body.Bytes(), &events))
		return events
	}
	require.Empty(t, request())
	require.Equal(t, 1, store.queryCount(), "repeated scope uses the cached negative")
	store.addExecution(&types.Execution{AgentNodeID: "agent-beta", RunID: "run-1"})
	require.Equal(t, store.history, request())
	require.Equal(t, 2, store.queryCount(), "new request queries once, then caches the positive")
}
