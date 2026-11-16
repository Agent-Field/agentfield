package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// MemoryEventsHandler handles real-time memory event subscriptions.
type MemoryEventsHandler struct {
	storage  storage.StorageProvider
	upgrader websocket.Upgrader
}

// NewMemoryEventsHandler creates a new MemoryEventsHandler.
// Origin checking is not needed because auth middleware already validates API keys
// before requests reach this handler.
func NewMemoryEventsHandler(storage storage.StorageProvider) *MemoryEventsHandler {
	return &MemoryEventsHandler{
		storage: storage,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}
}

const (
	// memoryEventsPongWait is how long the WebSocket may go without any frame
	// from the client, pongs included, before it is treated as dead.
	memoryEventsPongWait = 60 * time.Second
	// memoryEventsPingInterval must be shorter than memoryEventsPongWait.
	memoryEventsPingInterval = 25 * time.Second
	memoryEventsWriteWait    = 10 * time.Second
)

func normalizePatterns(raw string) []string {
	if raw == "" {
		return nil
	}

	parts := strings.Split(raw, ",")
	patterns := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			patterns = append(patterns, trimmed)
		}
	}
	return patterns
}

// parsePatterns reads the patterns query parameter and rejects malformed
// globs, which would otherwise silently match nothing.
func parsePatterns(c *gin.Context) ([]string, bool) {
	patterns := normalizePatterns(c.Query("patterns"))
	for _, p := range patterns {
		if _, err := path.Match(p, ""); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid pattern %q: %v", p, err)})
			return nil, false
		}
	}
	return patterns, true
}

// WebSocketHandler handles WebSocket connections for memory events.
//
// Events are forwarded in the order they were published. The server pings the
// client and drops the connection when nothing arrives within
// memoryEventsPongWait. If the client falls too far behind, the connection is
// closed with CloseTryAgainLater so the client knows events were missed.
func (h *MemoryEventsHandler) WebSocketHandler(c *gin.Context) {
	eventFilter, allowed := authorizeMemoryEvents(c)
	if !allowed {
		return
	}
	patterns, ok := parsePatterns(c)
	if !ok {
		return
	}
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		// upgrader.Upgrade automatically sends an error response, so just return
		return
	}
	defer conn.Close()

	// The subscription lives exactly as long as the connection.
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	writeClose := func(code int, text string) {
		msg := websocket.FormatCloseMessage(code, text)
		if err := conn.WriteControl(websocket.CloseMessage, msg, time.Now().Add(memoryEventsWriteWait)); err != nil {
			logger.Logger.Debug().Err(err).Msg("failed to send websocket close message")
		}
	}

	// Parse query parameters for filtering
	scope := c.Query("scope")
	scopeID := c.Query("scope_id")

	eventChan, err := h.storage.SubscribeToMemoryChanges(ctx, scope, scopeID)
	if err != nil {
		writeClose(websocket.CloseInternalServerErr, "failed to subscribe to events")
		return
	}

	// Read from the client so control frames are processed. Any frame,
	// including a pong or ping, proves the client is alive.
	extend := func() { _ = conn.SetReadDeadline(time.Now().Add(memoryEventsPongWait)) }
	extend()
	conn.SetPongHandler(func(string) error { extend(); return nil })
	conn.SetPingHandler(func(data string) error {
		extend()
		err := conn.WriteControl(websocket.PongMessage, []byte(data), time.Now().Add(memoryEventsWriteWait))
		if err == websocket.ErrCloseSent {
			return nil
		}
		return err
	})
	go func() {
		defer cancel()
		for {
			if _, _, err := conn.NextReader(); err != nil {
				return
			}
			extend()
		}
	}()

	ping := time.NewTicker(memoryEventsPingInterval)
	defer ping.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ping.C:
			if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(memoryEventsWriteWait)); err != nil {
				return
			}
		case event, ok := <-eventChan:
			if !ok {
				if ctx.Err() == nil {
					writeClose(websocket.CloseTryAgainLater, "subscriber fell behind; events were dropped")
				}
				return
			}
			if eventFilter != nil && !eventFilter(ctx, &event) {
				continue
			}
			if !types.MatchKeyPatterns(patterns, event.Key) {
				continue
			}
			_ = conn.SetWriteDeadline(time.Now().Add(memoryEventsWriteWait))
			if err := conn.WriteJSON(event); err != nil {
				return // Client disconnected
			}
		}
	}
}

// SSEHandler handles Server-Sent Events connections for memory events. If the
// client falls too far behind, it is sent an "error" event and the stream ends.
func (h *MemoryEventsHandler) SSEHandler(c *gin.Context) {
	ctx := c.Request.Context()
	eventFilter, allowed := authorizeMemoryEvents(c)
	if !allowed {
		return
	}

	patterns, ok := parsePatterns(c)
	if !ok {
		return
	}

	// Parse query parameters for filtering
	scope := c.Query("scope")
	scopeID := c.Query("scope_id")

	// Subscribe to memory changes before flushing headers so we can still
	// return a proper HTTP error status on failure.
	eventChan, err := h.storage.SubscribeToMemoryChanges(ctx, scope, scopeID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to subscribe to events"})
		return
	}

	// Set headers for SSE and flush immediately so the client receives the
	// response and can begin reading the body without waiting for the first event.
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	// Send an initial comment frame as a keepalive marker so clients know
	// the connection is established.
	_, _ = c.Writer.WriteString(": connected\n\n")
	c.Writer.Flush()

	for {
		select {
		case <-ctx.Done():
			// Client disconnected
			return
		case event, ok := <-eventChan:
			if !ok {
				if ctx.Err() == nil {
					// Storage dropped this subscriber for falling behind;
					// tell the client its stream has a gap.
					c.SSEvent("error", "subscriber fell behind; events were dropped")
					c.Writer.Flush()
				}
				return
			}

			if eventFilter != nil && !eventFilter(ctx, &event) {
				continue
			}
			if !types.MatchKeyPatterns(patterns, event.Key) {
				continue
			}

			// Marshal event to JSON
			eventJSON, err := json.Marshal(event)
			if err != nil {
				continue // Skip events that can't be marshaled
			}

			// Send event to client
			c.SSEvent("message", string(eventJSON))
			c.Writer.Flush()
		}
	}
}

// GetEventHistoryHandler handles requests for historical memory events.
func GetEventHistoryHandler(storage storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		eventFilter, allowed := authorizeMemoryEventHistory(c)
		if !allowed {
			return
		}
		var filter types.EventFilter

		// Parse query parameters
		if scope := c.Query("scope"); scope != "" && scope != "*" {
			filter.Scope = &scope
		}
		if scopeID := c.Query("scope_id"); scopeID != "" && scopeID != "*" {
			filter.ScopeID = &scopeID
		}
		patterns, ok := parsePatterns(c)
		if !ok {
			return
		}
		filter.Patterns = patterns
		if since := c.Query("since"); since != "" {
			if sinceTime, err := time.Parse(time.RFC3339, since); err == nil {
				filter.Since = &sinceTime
			}
		}
		if limit := c.Query("limit"); limit != "" {
			if limitVal, err := strconv.Atoi(limit); err == nil {
				filter.Limit = limitVal
			}
		}

		events, err := storage.GetEventHistory(ctx, filter)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get event history"})
			return
		}

		if eventFilter != nil {
			// Storage applies limit before visibility filtering, so wildcard
			// history may return fewer than limit events for this caller.
			visible := make([]*types.MemoryChangeEvent, 0, len(events))
			for _, event := range events {
				if eventFilter(ctx, event) {
					visible = append(visible, event)
				}
			}
			events = visible
		}
		c.JSON(http.StatusOK, events)
	}
}

// Concrete selectors use the same rules as explicit reads. A wildcard presents
// no selector, so each event must establish ownership or execution participation.
func authorizeMemoryEvents(c *gin.Context) (func(context.Context, *types.MemoryChangeEvent) bool, bool) {
	return authorizeMemoryEventSelector(c, false)
}

func authorizeMemoryEventHistory(c *gin.Context) (func(context.Context, *types.MemoryChangeEvent) bool, bool) {
	return authorizeMemoryEventSelector(c, true)
}

func authorizeMemoryEventSelector(c *gin.Context, snapshot bool) (func(context.Context, *types.MemoryChangeEvent) bool, bool) {
	scope, scopeID := c.Query("scope"), c.Query("scope_id")
	if scope != "" && scope != "*" && ((scopeID != "" && scopeID != "*") ||
		(scope != "global" && scope != "actor" && scope != "session" && scope != "workflow")) {
		if !middleware.RequireMemoryScope(c, scope, scopeID) {
			return nil, false
		}
		if !snapshot {
			return middleware.MemoryEventScopeFilter(c), true
		}
		return nil, true
	}
	if snapshot {
		return middleware.MemoryEventSnapshotFilter(c), true
	}
	return middleware.MemoryEventFilter(c), true
}
