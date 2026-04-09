package events

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestExecutionPublishers(t *testing.T) {
	subID := "test-execution-sub"
	ch := GlobalExecutionEventBus.Subscribe(subID)
	defer GlobalExecutionEventBus.Unsubscribe(subID)

	tests := []struct {
		name     string
		publish  func()
		expected ExecutionEventType
	}{
		{
			name: "PublishExecutionCreated",
			publish: func() {
				PublishExecutionCreated("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionCreated,
		},
		{
			name: "PublishExecutionStarted",
			publish: func() {
				PublishExecutionStarted("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionStarted,
		},
		{
			name: "PublishExecutionUpdated",
			publish: func() {
				PublishExecutionUpdated("exec1", "wf1", "node1", "running", nil)
			},
			expected: ExecutionUpdated,
		},
		{
			name: "PublishExecutionCompleted",
			publish: func() {
				PublishExecutionCompleted("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionCompleted,
		},
		{
			name: "PublishExecutionFailed",
			publish: func() {
				PublishExecutionFailed("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionFailed,
		},
		{
			name: "PublishExecutionWaiting",
			publish: func() {
				PublishExecutionWaiting("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionWaiting,
		},
		{
			name: "PublishExecutionPaused",
			publish: func() {
				PublishExecutionPaused("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionPaused,
		},
		{
			name: "PublishExecutionResumed",
			publish: func() {
				PublishExecutionResumed("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionResumed,
		},
		{
			name: "PublishExecutionCancelled",
			publish: func() {
				PublishExecutionCancelled("exec1", "wf1", "node1", nil)
			},
			expected: ExecutionCancelledEvent,
		},
		{
			name: "PublishExecutionApprovalResolved",
			publish: func() {
				PublishExecutionApprovalResolved("exec1", "wf1", "node1", "approved", nil)
			},
			expected: ExecutionApprovalResolved,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.publish()
			select {
			case event := <-ch:
				require.Equal(t, tt.expected, event.Type)
				require.Equal(t, "exec1", event.ExecutionID)
			case <-time.After(5 * time.Second):
				t.Errorf("Timeout waiting for event %s", tt.expected)
			}
		})
	}
}

func TestNodePublishers(t *testing.T) {
	subID := "test-node-sub"
	ch := GlobalNodeEventBus.Subscribe(subID)
	defer GlobalNodeEventBus.Unsubscribe(subID)

	tests := []struct {
		name     string
		publish  func()
		expected NodeEventType
	}{
		{"PublishNodeOnline", func() { PublishNodeOnline("node1", nil) }, NodeOnline},
		{"PublishNodeOffline", func() { PublishNodeOffline("node1", nil) }, NodeOffline},
		{"PublishNodeRegistered", func() { PublishNodeRegistered("node1", nil) }, NodeRegistered},
		{"PublishNodeStatusUpdated", func() { PublishNodeStatusUpdated("node1", "active", nil) }, NodeStatusUpdated},
		{"PublishNodeHealthChanged", func() { PublishNodeHealthChanged("node1", "healthy", nil) }, NodeHealthChanged},
		{"PublishNodeRemoved", func() { PublishNodeRemoved("node1", nil) }, NodeRemoved},
		{"PublishNodesRefresh", func() { PublishNodesRefresh(nil) }, NodesRefresh},
		{"PublishNodeHeartbeat", func() { PublishNodeHeartbeat() }, NodeHeartbeat},
		{"PublishNodeUnifiedStatusChanged", func() { PublishNodeUnifiedStatusChanged("node1", nil, nil, "src", "reason") }, NodeUnifiedStatusChanged},
		{"PublishNodeStateTransition", func() { PublishNodeStateTransition("node1", "off", "on", "reason") }, NodeStateTransition},
		{"PublishNodeStatusRefreshed", func() { PublishNodeStatusRefreshed("node1", nil) }, NodeStatusRefreshed},
		{"PublishBulkStatusUpdate", func() { PublishBulkStatusUpdate(10, 8, 2, []string{"err"}) }, BulkStatusUpdate},
		{"PublishNodeStatusUpdatedEnhanced", func() { PublishNodeStatusUpdatedEnhanced("node1", nil, nil, "src", "reason") }, NodeStatusUpdated},
		{"PublishNodeHealthChangedEnhanced", func() { PublishNodeHealthChangedEnhanced("node1", "old", "new", nil, "src", "reason") }, NodeHealthChanged},
		{"PublishSystemStateSnapshot", func() { PublishSystemStateSnapshot(nil) }, SystemStateSnapshot},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.publish()
			select {
			case event := <-ch:
				require.Equal(t, tt.expected, event.Type)
			case <-time.After(5 * time.Second):
				t.Errorf("Timeout waiting for event %s", tt.expected)
			}
		})
	}
}

func TestReasonerPublishers(t *testing.T) {
	subID := "test-reasoner-sub"
	ch := GlobalReasonerEventBus.Subscribe(subID)
	defer GlobalReasonerEventBus.Unsubscribe(subID)

	tests := []struct {
		name     string
		publish  func()
		expected ReasonerEventType
	}{
		{"PublishReasonerOnline", func() { PublishReasonerOnline("r1", "node1", nil) }, ReasonerOnline},
		{"PublishReasonerOffline", func() { PublishReasonerOffline("r1", "node1", nil) }, ReasonerOffline},
		{"PublishReasonerUpdated", func() { PublishReasonerUpdated("r1", "node1", "active", nil) }, ReasonerUpdated},
		{"PublishNodeStatusChanged", func() { PublishNodeStatusChanged("node1", "active", nil) }, NodeStatusChanged},
		{"PublishReasonersRefresh", func() { PublishReasonersRefresh(nil) }, ReasonersRefresh},
		{"PublishHeartbeat", func() { PublishHeartbeat() }, Heartbeat},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.publish()
			select {
			case event := <-ch:
				require.Equal(t, tt.expected, event.Type)
			case <-time.After(5 * time.Second):
				t.Errorf("Timeout waiting for event %s", tt.expected)
			}
		})
	}
}
