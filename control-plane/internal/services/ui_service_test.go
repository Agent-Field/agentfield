package services

import (
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func newTestUIService() *UIService {
	return &UIService{
		clients:        sync.Map{},
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}
}

func collectNodeEvents(client <-chan NodeEvent) (<-chan NodeEvent, <-chan struct{}) {
	out := make(chan NodeEvent, 16)
	done := make(chan struct{})

	go func() {
		defer close(done)
		for event := range client {
			out <- event
		}
	}()

	return out, done
}

func requireNodeEvent(t *testing.T, events <-chan NodeEvent, within time.Duration) NodeEvent {
	t.Helper()

	select {
	case event := <-events:
		return event
	case <-time.After(within):
		t.Fatalf("timed out waiting for node event within %s", within)
		return NodeEvent{}
	}
}

func requireNoNodeEvent(t *testing.T, events <-chan NodeEvent, within time.Duration) {
	t.Helper()

	select {
	case event := <-events:
		t.Fatalf("unexpected node event received: %#v", event)
	case <-time.After(within):
	}
}

func TestUIServiceBroadcastToMultipleSubscribersAndClientClose(t *testing.T) {
	service := newTestUIService()

	clientA := service.RegisterClient()
	clientB := service.RegisterClient()

	eventsA, doneA := collectNodeEvents(clientA)
	eventsB, doneB := collectNodeEvents(clientB)

	first := AgentNodeSummaryForUI{
		ID:              "node-1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
	service.BroadcastEvent("node_registered", first)

	require.Equal(t, "node_registered", requireNodeEvent(t, eventsA, 200*time.Millisecond).Type)
	require.Equal(t, "node_registered", requireNodeEvent(t, eventsB, 200*time.Millisecond).Type)

	service.DeregisterClient(clientA)
	<-doneA

	second := AgentNodeSummaryForUI{
		ID:              "node-2",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
	service.BroadcastEvent("node_registered", second)

	eventB := requireNodeEvent(t, eventsB, 200*time.Millisecond)
	require.Equal(t, "node_registered", eventB.Type)
	requireNoNodeEvent(t, eventsA, 50*time.Millisecond)

	service.DeregisterClient(clientB)
	<-doneB
}

func TestUIServiceStopHeartbeatStopsFurtherEvents(t *testing.T) {
	service := newTestUIService()
	service.startHeartbeat()
	service.heartbeatTicker.Reset(10 * time.Millisecond)

	client := service.RegisterClient()
	events, done := collectNodeEvents(client)

	heartbeat := requireNodeEvent(t, events, 200*time.Millisecond)
	require.Equal(t, "heartbeat", heartbeat.Type)

	service.StopHeartbeat()
	requireNoNodeEvent(t, events, 40*time.Millisecond)

	service.DeregisterClient(client)
	<-done
}

func TestUIServiceDeduplicatesIdenticalNodeStatusEvents(t *testing.T) {
	service := newTestUIService()

	client := service.RegisterClient()
	events, done := collectNodeEvents(client)

	initial := AgentNodeSummaryForUI{
		ID:              "node-1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
	service.BroadcastEvent("node_status_changed", initial)
	require.Equal(t, "node_status_changed", requireNodeEvent(t, events, 200*time.Millisecond).Type)

	service.BroadcastEvent("node_status_changed", initial)
	requireNoNodeEvent(t, events, 50*time.Millisecond)

	changed := initial
	changed.LifecycleStatus = types.AgentStatusDegraded
	service.BroadcastEvent("node_status_changed", changed)

	event := requireNodeEvent(t, events, 200*time.Millisecond)
	summary, ok := event.Node.(AgentNodeSummaryForUI)
	require.True(t, ok)
	require.Equal(t, types.AgentStatusDegraded, summary.LifecycleStatus)

	service.DeregisterClient(client)
	<-done
}

func TestUIServiceConcurrentRegisterAndClose(t *testing.T) {
	service := newTestUIService()
	baseline := runtime.NumGoroutine()

	const workers = 200

	var wg sync.WaitGroup
	start := make(chan struct{})

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start

			client := service.RegisterClient()
			service.DeregisterClient(client)
		}()
	}

	close(start)
	wg.Wait()

	require.Equal(t, 0, service.countClients())

	runtime.GC()
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= baseline+2 {
			break
		}
		runtime.Gosched()
	}

	require.LessOrEqual(t, runtime.NumGoroutine(), baseline+4)
}
