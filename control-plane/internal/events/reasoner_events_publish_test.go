package events

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestStartHeartbeatEmitsOnlyWithSubscribers(t *testing.T) {
	// StartHeartbeat only publishes while at least one subscriber is attached.
	// Start the ticker first, then subscribe, and confirm a heartbeat arrives.
	StartHeartbeat(10 * time.Millisecond)

	subID := "test-reasoner-hb-start"
	ch := GlobalReasonerEventBus.Subscribe(subID)
	defer GlobalReasonerEventBus.Unsubscribe(subID)

	select {
	case ev := <-ch:
		assert.Equal(t, Heartbeat, ev.Type)
	case <-time.After(2 * time.Second):
		t.Fatal("expected a heartbeat once a subscriber was present")
	}
}

func TestReasonerPublishHelpersSetTypeAndStatus(t *testing.T) {
	subID := "test-reasoner-helpers"
	ch := GlobalReasonerEventBus.Subscribe(subID)
	defer GlobalReasonerEventBus.Unsubscribe(subID)

	tests := []struct {
		name       string
		publish    func()
		wantType   ReasonerEventType
		wantStatus string
	}{
		{"online sets online status", func() { PublishReasonerOnline("r1", "n1", nil) }, ReasonerOnline, "online"},
		{"offline sets offline status", func() { PublishReasonerOffline("r1", "n1", nil) }, ReasonerOffline, "offline"},
		{"updated preserves caller status", func() { PublishReasonerUpdated("r1", "n1", "degraded", nil) }, ReasonerUpdated, "degraded"},
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

func TestPublishReasonersRefreshEmitsEmptyID(t *testing.T) {
	subID := "test-reasoners-refresh"
	ch := GlobalReasonerEventBus.Subscribe(subID)
	defer GlobalReasonerEventBus.Unsubscribe(subID)

	PublishReasonersRefresh(map[string]interface{}{"count": 3})
	select {
	case ev := <-ch:
		assert.Equal(t, ReasonersRefresh, ev.Type)
		assert.Empty(t, ev.ReasonerID)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for ReasonersRefresh")
	}
}
