package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"

	"github.com/gin-gonic/gin"
)

// StreamTriggerEvents handles GET /api/v1/triggers/:trigger_id/events/stream.
//
// Subscribes to the global trigger event bus and writes SSE frames for every
// event whose trigger_id matches the URL parameter. Sends an immediate
// "connected" frame so the browser's EventSource confirms the stream opened,
// then a heartbeat every 15 seconds to keep proxies from idling out the
// connection. Closes cleanly on client disconnect or context cancel.
//
// The bus drops events when this subscriber's channel is full (slow client) —
// the UI is expected to do a one-shot refetch via GET /events on reconnect to
// catch up, so dropped frames don't break correctness.
func (h *TriggerHandlers) StreamTriggerEvents() gin.HandlerFunc {
	return func(c *gin.Context) {
		triggerID := c.Param("trigger_id")
		if triggerID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "trigger_id is required"})
			return
		}

		// Verify the trigger exists so we don't open a stream for a typo URL.
		// (The 404 here gives the UI a meaningful response shape.)
		if _, err := h.storage.GetTrigger(c.Request.Context(), triggerID); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}

		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no") // disable nginx buffering

		subscriberID := fmt.Sprintf("trigger_sse_%s_%d", triggerID, time.Now().UnixNano())
		ch := events.GlobalTriggerEventBus.Subscribe(subscriberID)
		defer events.GlobalTriggerEventBus.Unsubscribe(subscriberID)

		// Initial connected frame — lets the EventSource client confirm open.
		connected := map[string]any{
			"type":       "connected",
			"trigger_id": triggerID,
			"timestamp":  time.Now().UTC().Format(time.RFC3339),
		}
		if payload, err := json.Marshal(connected); err == nil {
			if !writeTriggerSSE(c, payload) {
				return
			}
		}

		ctx := c.Request.Context()
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				ping := map[string]any{
					"type":      "ping",
					"timestamp": time.Now().UTC().Format(time.RFC3339),
				}
				if payload, err := json.Marshal(ping); err == nil {
					if !writeTriggerSSE(c, payload) {
						return
					}
				}
			case ev, ok := <-ch:
				if !ok {
					// Bus unsubscribed externally — close cleanly.
					return
				}
				if ev.TriggerID != triggerID {
					continue
				}
				if payload, err := json.Marshal(ev); err == nil {
					if !writeTriggerSSE(c, payload) {
						return
					}
				}
			}
		}
	}
}

// writeTriggerSSE writes a single SSE data frame and flushes. Returns false on
// any write error so the caller can stop the loop. Mirrors the writeSSE helper
// in handlers/ui/executions.go (kept local here so handlers/ has no import
// cycle on the ui package).
func writeTriggerSSE(c *gin.Context, payload []byte) bool {
	if _, err := fmt.Fprintf(c.Writer, "data: %s\n\n", payload); err != nil {
		return false
	}
	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}
	return true
}
