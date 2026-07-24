package observability

import (
	"context"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestTracer(t *testing.T) (*Tracer, func()) {
	t.Helper()
	cfg := TracerConfig{
		Enabled:     true,
		Endpoint:    "localhost:4318",
		ServiceName: "test-agentfield",
		Insecure:    true,
	}
	tracer, shutdown, err := InitTracer(context.Background(), cfg)
	require.NoError(t, err)
	require.NotNil(t, tracer)
	return tracer, func() { shutdown(context.Background()) } //nolint:errcheck
}

// waitForSubscription gives goroutines time to subscribe to event buses.
func waitForSubscription() {
	time.Sleep(50 * time.Millisecond)
}

// waitForEvent gives the subscriber goroutine time to process an event.
func waitForEvent() {
	time.Sleep(50 * time.Millisecond)
}

func TestExecutionTracer_StartStop(t *testing.T) {
	tracer, cleanup := newTestTracer(t)
	defer cleanup()

	et := NewExecutionTracer(tracer)
	require.NotNil(t, et)

	et.Start(context.Background())
	waitForSubscription()

	// Verify it subscribed to event buses
	assert.Greater(t, events.GlobalExecutionEventBus.GetSubscriberCount(), 0)
	assert.Greater(t, events.GlobalReasonerEventBus.GetSubscriberCount(), 0)

	et.Stop()
}

func TestExecutionTracer_HandlesExecutionLifecycle(t *testing.T) {
	tracer, cleanup := newTestTracer(t)
	defer cleanup()

	et := NewExecutionTracer(tracer)
	et.Start(context.Background())
	waitForSubscription()
	defer et.Stop()

	// Simulate execution lifecycle via event bus
	events.PublishExecutionCreated("exec-1", "wf-1", "node-1", nil)
	waitForEvent()

	// Verify span is tracked
	et.mu.Lock()
	_, exists := et.spans["exec-1"]
	et.mu.Unlock()
	assert.True(t, exists, "execution span should be tracked after creation")

	// Complete the execution
	events.PublishExecutionCompleted("exec-1", "wf-1", "node-1", nil)
	waitForEvent()

	// Verify span is ended and removed
	et.mu.Lock()
	_, exists = et.spans["exec-1"]
	et.mu.Unlock()
	assert.False(t, exists, "execution span should be removed after completion")
}

func TestExecutionTracer_HandlesFailedExecution(t *testing.T) {
	tracer, cleanup := newTestTracer(t)
	defer cleanup()

	et := NewExecutionTracer(tracer)
	et.Start(context.Background())
	waitForSubscription()
	defer et.Stop()

	events.PublishExecutionStarted("exec-fail", "wf-1", "node-1", nil)
	waitForEvent()

	et.mu.Lock()
	_, exists := et.spans["exec-fail"]
	et.mu.Unlock()
	assert.True(t, exists)

	events.PublishExecutionFailed("exec-fail", "wf-1", "node-1", map[string]interface{}{"error": "timeout"})
	waitForEvent()

	et.mu.Lock()
	_, exists = et.spans["exec-fail"]
	et.mu.Unlock()
	assert.False(t, exists, "failed execution span should be cleaned up")
}

func TestExecutionTracer_DuplicateCreatedIgnored(t *testing.T) {
	tracer, cleanup := newTestTracer(t)
	defer cleanup()

	et := NewExecutionTracer(tracer)
	et.Start(context.Background())
	waitForSubscription()
	defer et.Stop()

	events.PublishExecutionCreated("exec-dup", "wf-1", "node-1", nil)
	waitForEvent()

	et.mu.Lock()
	span1 := et.spans["exec-dup"]
	et.mu.Unlock()
	require.NotNil(t, span1, "span should exist after first creation event")

	// Publish another created event for the same execution
	events.PublishExecutionCreated("exec-dup", "wf-1", "node-1", nil)
	waitForEvent()

	et.mu.Lock()
	span2 := et.spans["exec-dup"]
	et.mu.Unlock()

	assert.Equal(t, span1, span2, "duplicate creation should not replace existing span")

	// Cleanup
	events.PublishExecutionCompleted("exec-dup", "wf-1", "node-1", nil)
	waitForEvent()
}

func TestExecutionTracer_StopEndsOpenSpans(t *testing.T) {
	tracer, cleanup := newTestTracer(t)
	defer cleanup()

	et := NewExecutionTracer(tracer)
	et.Start(context.Background())
	waitForSubscription()

	events.PublishExecutionCreated("exec-open", "wf-1", "node-1", nil)
	waitForEvent()

	et.mu.Lock()
	_, exists := et.spans["exec-open"]
	et.mu.Unlock()
	assert.True(t, exists)

	// Stop should end all open spans
	et.Stop()

	et.mu.Lock()
	count := len(et.spans)
	et.mu.Unlock()
	assert.Equal(t, 0, count, "all spans should be cleaned up after Stop")
}
