package services

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// fakeAgentStore implements just enough of storage.StorageProvider to
// satisfy CancelDispatcher's lookups. Anything else panics so accidental
// extra dependencies surface in tests.
type fakeAgentStore struct {
	storage.StorageProvider

	mu     sync.Mutex
	agents map[string]*types.AgentNode
}

func newFakeAgentStore() *fakeAgentStore {
	return &fakeAgentStore{agents: make(map[string]*types.AgentNode)}
}

func (s *fakeAgentStore) seed(id, baseURL string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.agents[id] = &types.AgentNode{ID: id, BaseURL: baseURL}
}

func (s *fakeAgentStore) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	a, ok := s.agents[id]
	if !ok {
		return nil, nil
	}
	clone := *a
	return &clone, nil
}

func TestCancelDispatcher_DeliversCallback(t *testing.T) {
	bus := events.NewExecutionEventBus()

	var (
		calls       int32
		gotPath     string
		gotBody     map[string]any
		gotHeaders  http.Header
		callDoneCh  = make(chan struct{}, 1)
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		gotPath = r.URL.Path
		gotHeaders = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.WriteHeader(http.StatusOK)
		select {
		case callDoneCh <- struct{}{}:
		default:
		}
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-1", srv.URL)

	d := NewCancelDispatcher(store, CancelDispatcherConfig{
		Bus:        bus,
		HTTPClient: &http.Client{Timeout: 2 * time.Second},
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()

	// Give the goroutine a beat to subscribe before we publish — the bus
	// drops events when no subscriber is registered. Polling for the
	// subscriber count avoids a flaky time.Sleep.
	require.Eventually(t, func() bool {
		return bus.GetSubscriberCount() >= 1
	}, time.Second, 5*time.Millisecond, "dispatcher did not subscribe")

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-42",
		WorkflowID:  "wf-1",
		AgentNodeID: "agent-1",
		Status:      "cancelled",
		Timestamp:   time.Now().UTC(),
		Data:        map[string]interface{}{"reason": "user clicked cancel"},
	})

	select {
	case <-callDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("worker did not receive cancel callback")
	}

	require.EqualValues(t, 1, atomic.LoadInt32(&calls))
	require.Equal(t, "/_internal/executions/exec-42/cancel", gotPath)
	require.Equal(t, "exec-42", gotHeaders.Get("X-Execution-ID"))
	require.Equal(t, "wf-1", gotHeaders.Get("X-Workflow-ID"))
	require.Equal(t, "cancel-dispatcher", gotHeaders.Get("X-AgentField-Source"))
	require.Equal(t, "exec-42", gotBody["execution_id"])
	require.Equal(t, "user clicked cancel", gotBody["reason"])
}

func TestCancelDispatcher_IgnoresNonCancelEvents(t *testing.T) {
	bus := events.NewExecutionEventBus()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-1", srv.URL)

	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()

	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCompleted,
		ExecutionID: "exec-99",
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
	})

	// Briefly wait — there's no positive signal because we expect no call.
	// 100ms is plenty for the dispatcher to drop the event on the floor.
	time.Sleep(100 * time.Millisecond)
	require.Zero(t, atomic.LoadInt32(&calls), "non-cancel event should not trigger callback")
}

func TestCancelDispatcher_HandlesUnregisteredAgent(t *testing.T) {
	bus := events.NewExecutionEventBus()
	store := newFakeAgentStore()
	// No agent seeded — lookup returns (nil, nil).

	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()

	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	// Should not panic, should not block the dispatcher.
	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-orphan",
		AgentNodeID: "agent-missing",
		Timestamp:   time.Now().UTC(),
	})

	// Followup event still gets through.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	store.seed("agent-1", srv.URL)

	delivered := make(chan struct{}, 1)
	srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		delivered <- struct{}{}
	})

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-43",
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case <-delivered:
	case <-time.After(2 * time.Second):
		t.Fatal("dispatcher stopped processing after orphan event")
	}
}

func TestCancelDispatcher_StopIsIdempotent(t *testing.T) {
	bus := events.NewExecutionEventBus()
	d := NewCancelDispatcher(newFakeAgentStore(), CancelDispatcherConfig{Bus: bus})
	d.Start(context.Background())
	d.Stop()
	d.Stop() // second call must be a no-op
}
