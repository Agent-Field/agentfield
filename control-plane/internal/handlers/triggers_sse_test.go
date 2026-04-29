package handlers

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestStreamTriggerEvents_DeliversPublishedEvent confirms the SSE handler
// subscribes to the global trigger event bus and writes frames whose
// trigger_id matches the URL parameter — and filters out events for other
// triggers. End-to-end with a real LocalStorage so the handler's existence
// check (GetTrigger) passes.
func TestStreamTriggerEvents_DeliversPublishedEvent(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Real Trigger row so the handler's GetTrigger lookup succeeds.
	trig := &types.Trigger{
		ID:             "trg_sse_test",
		SourceName:     "stripe",
		TargetNodeID:   "node-x",
		TargetReasoner: "handle_x",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	dispatcher := services.NewTriggerDispatcher(provider, nil)
	h := NewTriggerHandlers(provider, dispatcher, nil)

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/api/v1/triggers/:trigger_id/events/stream", h.StreamTriggerEvents())

	streamCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := httptest.NewRequest("GET", "/api/v1/triggers/trg_sse_test/events/stream", nil).WithContext(streamCtx)
	w := httptest.NewRecorder()

	// Run the SSE handler in a goroutine — it blocks on the bus channel.
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(w, req)
		close(done)
	}()

	// Give the handler a beat to subscribe and emit the "connected" frame.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if events.GlobalTriggerEventBus.SubscriberCount() > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	require.Greater(t, events.GlobalTriggerEventBus.SubscriberCount(), 0, "SSE handler should have subscribed")

	// Publish an event for our trigger; the handler should write it.
	events.GlobalTriggerEventBus.Publish(events.TriggerEvent{
		Type:       events.TriggerEventTypeReceived,
		TriggerID:  "trg_sse_test",
		EventID:    "evt_sse_001",
		SourceName: "stripe",
		EventType:  "payment_intent.succeeded",
		Status:     types.InboundEventStatusReceived,
		Timestamp:  time.Now().UTC(),
	})

	// Publish an event for a DIFFERENT trigger to confirm filtering.
	events.GlobalTriggerEventBus.Publish(events.TriggerEvent{
		Type:      events.TriggerEventTypeReceived,
		TriggerID: "some_other_trigger",
		EventID:   "evt_other_001",
	})

	// Wait briefly for frames to be written.
	deadline = time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(w.Body.String(), "evt_sse_001") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	body := w.Body.String()
	assert.Contains(t, body, "connected", "initial connected frame")
	assert.Contains(t, body, "evt_sse_001", "subscribed trigger's event delivered")
	assert.NotContains(t, body, "evt_other_001", "other trigger's event filtered out")

	cancel()
	select {
	case <-done:
	case <-time.After(1 * time.Second):
		t.Fatal("SSE handler did not exit within 1s of context cancel")
	}
}

// TestStreamTriggerEvents_TriggerNotFound returns 404 when the trigger ID
// doesn't exist, so a typo URL doesn't open a perpetual stream.
func TestStreamTriggerEvents_TriggerNotFound(t *testing.T) {
	provider, _ := setupTestStorage(t)
	h := NewTriggerHandlers(provider, services.NewTriggerDispatcher(provider, nil), nil)

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/api/v1/triggers/:trigger_id/events/stream", h.StreamTriggerEvents())

	req := httptest.NewRequest("GET", "/api/v1/triggers/nonexistent/events/stream", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, 404, w.Code)
	assert.Contains(t, w.Body.String(), "not found")
}
