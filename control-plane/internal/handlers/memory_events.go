package handlers

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
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

// WebSocketHandler handles WebSocket connections for memory events.
func (h *MemoryEventsHandler) WebSocketHandler(c *gin.Context) {
	ctx := c.Request.Context()
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		// upgrader.Upgrade automatically sends an error response, so just return
		return
	}
	defer conn.Close()

	// Parse query parameters for filtering
	scope := c.Query("scope")
	scopeID := c.Query("scope_id")
	patterns := normalizePatterns(c.Query("patterns"))

	// Subscribe to memory changes
	eventChan, err := h.storage.SubscribeToMemoryChanges(ctx, scope, scopeID)
	if err != nil {
		if writeErr := conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseInternalServerErr, "failed to subscribe to events")); writeErr != nil {
			logger.Logger.Warn().Err(writeErr).Msg("failed to send websocket close message")
		}
		return
	}

	// Goroutine to read messages from the client (e.g., for ping/pong)
	go func() {
		for {
			if _, _, err := conn.NextReader(); err != nil {
				if closeErr := conn.Close(); closeErr != nil {
					logger.Logger.Debug().Err(closeErr).Msg("websocket close returned error")
				}
				break
			}
		}
	}()

	// Forward events to the client
	for event := range eventChan {
		// Apply pattern matching
		if len(patterns) > 0 {
			match := false
			for _, pattern := range patterns {
				if matched, _ := filepath.Match(pattern, event.Key); matched {
					match = true
					break
				}
			}
			if !match {
				continue
			}
		}

		if err := conn.WriteJSON(event); err != nil {
			break // Client disconnected
		}
	}
}

// SSEHandler handles Server-Sent Events connections for memory events.
func (h *MemoryEventsHandler) SSEHandler(c *gin.Context) {
	ctx := c.Request.Context()
	// Set headers for SSE
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")

	// Parse query parameters for filtering
	scope := c.Query("scope")
	scopeID := c.Query("scope_id")
	patterns := normalizePatterns(c.Query("patterns"))

	// Subscribe to memory changes
	eventChan, err := h.storage.SubscribeToMemoryChanges(ctx, scope, scopeID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to subscribe to events"})
		return
	}

	// Create a channel to notify when the client disconnects
	clientClosed := c.Writer.CloseNotify()

	for {
		select {
		case <-clientClosed:
			// Client disconnected
			return
		case event := <-eventChan:
			// Apply pattern matching
			if len(patterns) > 0 {
				match := false
				for _, pattern := range patterns {
					if matched, _ := filepath.Match(pattern, event.Key); matched {
						match = true
						break
					}
				}
				if !match {
					continue
				}
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
		var filter types.EventFilter

		// Parse query parameters
		if scope := c.Query("scope"); scope != "" {
			filter.Scope = &scope
		}
		if scopeID := c.Query("scope_id"); scopeID != "" {
			filter.ScopeID = &scopeID
		}
		if patterns := c.Query("patterns"); patterns != "" {
			filter.Patterns = strings.Split(patterns, ",")
		}
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

		c.JSON(http.StatusOK, events)
	}
}
