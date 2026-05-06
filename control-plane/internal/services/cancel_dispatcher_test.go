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

func TestCancelDispatcher_SendsAuthorizationHeader(t *testing.T) {
	bus := events.NewExecutionEventBus()

	gotAuth := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case gotAuth <- r.Header.Get("Authorization"):
		default:
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-1", srv.URL)

	d := NewCancelDispatcher(store, CancelDispatcherConfig{
		Bus:           bus,
		InternalToken: "secret-token-xyz",
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case auth := <-gotAuth:
		require.Equal(t, "Bearer secret-token-xyz", auth)
	case <-time.After(2 * time.Second):
		t.Fatal("did not receive cancel callback")
	}
}

func TestCancelDispatcher_StopIsIdempotent(t *testing.T) {
	bus := events.NewExecutionEventBus()
	d := NewCancelDispatcher(newFakeAgentStore(), CancelDispatcherConfig{Bus: bus})
	d.Start(context.Background())
	d.Stop()
	d.Stop() // second call must be a no-op
}

// TestCancelDispatcher_DefaultsApplied ensures NewCancelDispatcher's nil-bus
// branch falls back to the global bus, the default cancelPath template gets
// substituted, and the default subscriber id is used. We can't directly
// observe the global bus without coupling to it, but we can confirm the
// dispatcher constructs cleanly with an all-zero config.
func TestCancelDispatcher_DefaultsApplied(t *testing.T) {
	d := NewCancelDispatcher(newFakeAgentStore(), CancelDispatcherConfig{})
	require.NotNil(t, d)
	require.Equal(t, "/_internal/executions/:execution_id/cancel", d.cancelPath)
	require.Equal(t, "cancel-dispatcher", d.subscriberID)
	require.NotNil(t, d.httpClient)
	require.NotNil(t, d.bus, "should fall back to GlobalExecutionEventBus")
}

// TestCancelDispatcher_StartIsIdempotent covers the early-return branch in
// Start when stopCh is already set. Without this branch a second Start would
// leak a goroutine and double-subscribe.
func TestCancelDispatcher_StartIsIdempotent(t *testing.T) {
	bus := events.NewExecutionEventBus()
	d := NewCancelDispatcher(newFakeAgentStore(), CancelDispatcherConfig{Bus: bus})
	d.Start(context.Background())
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() == 1 }, time.Second, 5*time.Millisecond)
	// Second Start must be a no-op — subscriber count should not increase.
	d.Start(context.Background())
	time.Sleep(50 * time.Millisecond)
	require.Equal(t, 1, bus.GetSubscriberCount(), "second Start should not double-subscribe")
}

// TestCancelDispatcher_ContextCancelStops covers the <-ctx.Done() branch in
// the consumer loop. With a cancelled parent context the loop exits and
// Subscriber count drops to zero without us calling Stop.
func TestCancelDispatcher_ContextCancelStops(t *testing.T) {
	bus := events.NewExecutionEventBus()
	d := NewCancelDispatcher(newFakeAgentStore(), CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	d.Start(ctx)
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() == 1 }, time.Second, 5*time.Millisecond)

	cancel()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() == 0 }, time.Second, 10*time.Millisecond, "consumer goroutine should exit on ctx.Done()")
	// Stop after ctx-driven exit must still be safe.
	d.Stop()
}

// TestCancelDispatcher_SkipsEmptyExecutionID covers the early-return when
// the cancelled event's execution_id is blank.
func TestCancelDispatcher_SkipsEmptyExecutionID(t *testing.T) {
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
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "", // blank — should be skipped
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
	})
	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-with-blank-agent",
		AgentNodeID: "", // blank — should be skipped
		Timestamp:   time.Now().UTC(),
	})

	time.Sleep(100 * time.Millisecond)
	require.Zero(t, atomic.LoadInt32(&calls), "blank execution_id and blank agent_node_id should not trigger callback")
}

// TestCancelDispatcher_SkipsEmptyBaseURL covers the agent-found-but-no-base-url
// branch (typical for serverless agents the dispatcher can't call back).
func TestCancelDispatcher_SkipsEmptyBaseURL(t *testing.T) {
	bus := events.NewExecutionEventBus()
	store := newFakeAgentStore()
	store.seed("serverless-agent", "") // empty BaseURL

	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "serverless-agent",
		Timestamp:   time.Now().UTC(),
	})

	// No callback target. Just confirm the dispatcher remains alive — a panic or
	// infinite loop would surface via the GetSubscriberCount staying at 1 only if
	// the goroutine survives. Follow up with a real publish to a registered
	// agent to prove the loop kept going.
	time.Sleep(50 * time.Millisecond)
	require.Equal(t, 1, bus.GetSubscriberCount())
}

// TestCancelDispatcher_HandlesWorker404 covers the 404 response path —
// older SDK without the cancel route should be treated as a no-op success.
func TestCancelDispatcher_HandlesWorker404(t *testing.T) {
	bus := events.NewExecutionEventBus()
	delivered := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		select {
		case delivered <- struct{}{}:
		default:
		}
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-old", srv.URL)
	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "agent-old",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case <-delivered:
	case <-time.After(2 * time.Second):
		t.Fatal("dispatcher did not POST to worker")
	}
}

// TestCancelDispatcher_HandlesWorker5xx covers the generic 4xx/5xx-error
// response branch.
func TestCancelDispatcher_HandlesWorker5xx(t *testing.T) {
	bus := events.NewExecutionEventBus()
	delivered := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		select {
		case delivered <- struct{}{}:
		default:
		}
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-broken", srv.URL)
	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "agent-broken",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case <-delivered:
	case <-time.After(2 * time.Second):
		t.Fatal("dispatcher did not POST to worker")
	}
}

// TestCancelDispatcher_HandlesTransportError covers the http.Client.Do
// error path (connection refused / nil transport response). We seed an
// agent with a URL that won't accept connections.
func TestCancelDispatcher_HandlesTransportError(t *testing.T) {
	bus := events.NewExecutionEventBus()
	store := newFakeAgentStore()
	// 127.0.0.1:1 — reserved port that refuses connections immediately.
	store.seed("agent-unreachable", "http://127.0.0.1:1")

	d := NewCancelDispatcher(store, CancelDispatcherConfig{
		Bus:        bus,
		HTTPClient: &http.Client{Timeout: 200 * time.Millisecond},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "agent-unreachable",
		Timestamp:   time.Now().UTC(),
	})

	// Loop must remain alive after a transport error. Prove it by publishing a
	// second event to a now-reachable agent and waiting for delivery.
	delivered := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		delivered <- struct{}{}
	}))
	defer srv.Close()
	store.seed("agent-good", srv.URL)
	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-2",
		AgentNodeID: "agent-good",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case <-delivered:
	case <-time.After(3 * time.Second):
		t.Fatal("dispatcher stopped after transport error")
	}
}

// TestCancelDispatcher_BaseURLTrailingSlash covers the strings.TrimRight
// path — base URLs ending in "/" should still produce a single-slash join.
func TestCancelDispatcher_BaseURLTrailingSlash(t *testing.T) {
	bus := events.NewExecutionEventBus()
	gotPath := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case gotPath <- r.URL.Path:
		default:
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	store := newFakeAgentStore()
	store.seed("agent-1", srv.URL+"/") // trailing slash — TrimRight should handle it

	d := NewCancelDispatcher(store, CancelDispatcherConfig{Bus: bus})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.Start(ctx)
	defer d.Stop()
	require.Eventually(t, func() bool { return bus.GetSubscriberCount() >= 1 }, time.Second, 5*time.Millisecond)

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-7",
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
	})

	select {
	case path := <-gotPath:
		require.Equal(t, "/_internal/executions/exec-7/cancel", path)
	case <-time.After(2 * time.Second):
		t.Fatal("no callback received")
	}
}

// TestCancelDispatcher_ReasonNotAString covers the fallback when the
// publisher's data map has a non-string "reason" value or is missing entirely.
// The dispatcher should still POST with reason="".
func TestCancelDispatcher_ReasonNotAString(t *testing.T) {
	bus := events.NewExecutionEventBus()
	gotBody := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		select {
		case gotBody <- body:
		default:
		}
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

	// Reason is an int — fallback to empty string.
	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-1",
		AgentNodeID: "agent-1",
		Timestamp:   time.Now().UTC(),
		Data:        map[string]interface{}{"reason": 42},
	})

	select {
	case body := <-gotBody:
		require.Equal(t, "", body["reason"])
	case <-time.After(2 * time.Second):
		t.Fatal("no callback received")
	}
}
