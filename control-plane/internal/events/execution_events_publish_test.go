package events

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExecutionPublishersSetCanonicalStatus asserts each execution publish
// helper emits the expected event Type *and* the canonical Status string. The
// existing TestExecutionPublishers checks Type only; this covers the status
// contract (e.g. completed -> succeeded) the handlers depend on.
func TestExecutionPublishersSetCanonicalStatus(t *testing.T) {
	subID := "test-execution-status-sub"
	ch := GlobalExecutionEventBus.Subscribe(subID)
	defer GlobalExecutionEventBus.Unsubscribe(subID)

	tests := []struct {
		name       string
		publish    func()
		wantType   ExecutionEventType
		wantStatus string
	}{
		{"created", func() { PublishExecutionCreated("e", "w", "n", nil) }, ExecutionCreated, "created"},
		{"started", func() { PublishExecutionStarted("e", "w", "n", nil) }, ExecutionStarted, "running"},
		{"updated preserves caller status", func() { PublishExecutionUpdated("e", "w", "n", "custom", nil) }, ExecutionUpdated, "custom"},
		{"completed uses succeeded", func() { PublishExecutionCompleted("e", "w", "n", nil) }, ExecutionCompleted, string(types.ExecutionStatusSucceeded)},
		{"failed", func() { PublishExecutionFailed("e", "w", "n", nil) }, ExecutionFailed, "failed"},
		{"waiting", func() { PublishExecutionWaiting("e", "w", "n", nil) }, ExecutionWaiting, string(types.ExecutionStatusWaiting)},
		{"paused", func() { PublishExecutionPaused("e", "w", "n", nil) }, ExecutionPaused, string(types.ExecutionStatusPaused)},
		{"resumed uses running", func() { PublishExecutionResumed("e", "w", "n", nil) }, ExecutionResumed, string(types.ExecutionStatusRunning)},
		{"cancelled", func() { PublishExecutionCancelled("e", "w", "n", nil) }, ExecutionCancelledEvent, string(types.ExecutionStatusCancelled)},
		{"approval resolved forwards new status", func() { PublishExecutionApprovalResolved("e", "w", "n", "approved", nil) }, ExecutionApprovalResolved, "approved"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.publish()
			select {
			case ev := <-ch:
				assert.Equal(t, tt.wantType, ev.Type)
				assert.Equal(t, tt.wantStatus, ev.Status)
			case <-time.After(5 * time.Second):
				t.Fatalf("timed out waiting for %s", tt.wantType)
			}
		})
	}
}

// TestExecutionPublisherForwardsDataPayload confirms the arbitrary data payload
// is forwarded to subscribers unchanged.
func TestExecutionPublisherForwardsDataPayload(t *testing.T) {
	subID := "test-execution-data-sub"
	ch := GlobalExecutionEventBus.Subscribe(subID)
	defer GlobalExecutionEventBus.Unsubscribe(subID)

	payload := map[string]interface{}{"tokens": 42, "note": "hello"}
	PublishExecutionUpdated("e", "w", "n", "running", payload)

	select {
	case ev := <-ch:
		require.Equal(t, ExecutionUpdated, ev.Type)
		data, ok := ev.Data.(map[string]interface{})
		require.True(t, ok, "data payload should be forwarded unchanged")
		assert.Equal(t, 42, data["tokens"])
		assert.Equal(t, "hello", data["note"])
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for ExecutionUpdated")
	}
}
