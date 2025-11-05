package events

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// NodeEventType represents the type of node event
type NodeEventType string

const (
	NodeOnline           NodeEventType = "node_online"
	NodeOffline          NodeEventType = "node_offline"
	NodeRegistered       NodeEventType = "node_registered"
	NodeStatusUpdated    NodeEventType = "node_status_changed"
	NodeRemoved          NodeEventType = "node_removed"
	NodeHealthChanged    NodeEventType = "node_health_changed"
	NodeMCPHealthChanged NodeEventType = "mcp_health_changed"
	NodesRefresh         NodeEventType = "nodes_refresh"
	NodeHeartbeat        NodeEventType = "node_heartbeat"

	// New unified status events
	NodeUnifiedStatusChanged NodeEventType = "node_unified_status_changed"
	NodeStateTransition      NodeEventType = "node_state_transition"
	NodeStatusRefreshed      NodeEventType = "node_status_refreshed"
	BulkStatusUpdate         NodeEventType = "bulk_status_update"
)

// NodeEvent represents a node state change event
type NodeEvent struct {
	Type      NodeEventType `json:"type"`
	NodeID    string        `json:"node_id,omitempty"`
	Status    string        `json:"status,omitempty"`
	Timestamp time.Time     `json:"timestamp"`
	Data      interface{}   `json:"data,omitempty"`

	// Enhanced fields for unified status system
	OldStatus interface{} `json:"old_status,omitempty"` // Previous status for comparison
	NewStatus interface{} `json:"new_status,omitempty"` // New status for comparison
	Source    string      `json:"source,omitempty"`     // Source of the status change
	Reason    string      `json:"reason,omitempty"`     // Reason for the change
}

// NodeEventBus manages node event broadcasting
type NodeEventBus struct {
	subscribers map[string]chan NodeEvent
	mutex       sync.RWMutex
}

// NewNodeEventBus creates a new node event bus
func NewNodeEventBus() *NodeEventBus {
	return &NodeEventBus{
		subscribers: make(map[string]chan NodeEvent),
	}
}

// Subscribe adds a new subscriber to the event bus
func (bus *NodeEventBus) Subscribe(subscriberID string) chan NodeEvent {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	ch := make(chan NodeEvent, 100) // Buffer to prevent blocking
	bus.subscribers[subscriberID] = ch

	logger.Logger.Debug().Msgf("[NodeEventBus] Subscriber %s added, total subscribers: %d", subscriberID, len(bus.subscribers))
	return ch
}

// Unsubscribe removes a subscriber from the event bus
func (bus *NodeEventBus) Unsubscribe(subscriberID string) {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	if ch, exists := bus.subscribers[subscriberID]; exists {
		close(ch)
		delete(bus.subscribers, subscriberID)
		logger.Logger.Debug().Msgf("[NodeEventBus] Subscriber %s removed, total subscribers: %d", subscriberID, len(bus.subscribers))
	}
}

// Publish broadcasts an event to all subscribers with improved error handling
func (bus *NodeEventBus) Publish(event NodeEvent) {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()

	// Add event filtering to prevent spam
	if bus.shouldFilterEvent(event) {
		logger.Logger.Debug().Msgf("[NodeEventBus] Filtering duplicate event: %s for node %s", event.Type, event.NodeID)
		return
	}

	successCount := 0
	for subscriberID, ch := range bus.subscribers {
		select {
		case ch <- event:
			// Event sent successfully
			successCount++
		default:
			// Channel is full, skip this subscriber to prevent blocking
			logger.Logger.Warn().Msgf("[NodeEventBus] Warning: Channel full for subscriber %s, skipping event", subscriberID)
		}
	}

	if successCount > 0 {
		logger.Logger.Debug().Msgf("[NodeEventBus] Published %s event to %d/%d subscribers", event.Type, successCount, len(bus.subscribers))
	}
}

// GetSubscriberCount returns the number of active subscribers
func (bus *NodeEventBus) GetSubscriberCount() int {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()
	return len(bus.subscribers)
}

// ToJSON converts a node event to JSON string
func (event *NodeEvent) ToJSON() (string, error) {
	data, err := json.Marshal(event)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Global event bus instance
var GlobalNodeEventBus = NewNodeEventBus()

// Helper functions for common event types

// PublishNodeOnline publishes a node online event
func PublishNodeOnline(nodeID string, data interface{}) {
	event := NodeEvent{
		Type:      NodeOnline,
		NodeID:    nodeID,
		Status:    "online",
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeOffline publishes a node offline event
func PublishNodeOffline(nodeID string, data interface{}) {
	event := NodeEvent{
		Type:      NodeOffline,
		NodeID:    nodeID,
		Status:    "offline",
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeRegistered publishes a node registered event
func PublishNodeRegistered(nodeID string, data interface{}) {
	event := NodeEvent{
		Type:      NodeRegistered,
		NodeID:    nodeID,
		Status:    "registered",
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeStatusUpdated publishes a node status change event
func PublishNodeStatusUpdated(nodeID, status string, data interface{}) {
	event := NodeEvent{
		Type:      NodeStatusUpdated,
		NodeID:    nodeID,
		Status:    status,
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeHealthChanged publishes a node health change event
func PublishNodeHealthChanged(nodeID, healthStatus string, data interface{}) {
	event := NodeEvent{
		Type:      NodeHealthChanged,
		NodeID:    nodeID,
		Status:    healthStatus,
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeMCPHealthChanged publishes an MCP health change event
func PublishNodeMCPHealthChanged(nodeID string, data interface{}) {
	event := NodeEvent{
		Type:      NodeMCPHealthChanged,
		NodeID:    nodeID,
		Timestamp: time.Now(),
		Data:      data,
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing NodeMCPHealthChanged event - NodeID: %s", nodeID)

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeRemoved publishes a node removed event
func PublishNodeRemoved(nodeID string, data interface{}) {
	event := NodeEvent{
		Type:      NodeRemoved,
		NodeID:    nodeID,
		Status:    "removed",
		Timestamp: time.Now(),
		Data:      data,
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing NodeRemoved event - NodeID: %s", nodeID)

	GlobalNodeEventBus.Publish(event)
}

// PublishNodesRefresh publishes a general refresh event
func PublishNodesRefresh(data interface{}) {
	event := NodeEvent{
		Type:      NodesRefresh,
		Timestamp: time.Now(),
		Data:      data,
	}
	GlobalNodeEventBus.Publish(event)
}

// PublishNodeHeartbeat publishes a heartbeat event to keep connections alive
func PublishNodeHeartbeat() {
	event := NodeEvent{
		Type:      NodeHeartbeat,
		Timestamp: time.Now(),
	}
	GlobalNodeEventBus.Publish(event)
}

// StartNodeHeartbeat starts a goroutine that sends periodic heartbeat events
func StartNodeHeartbeat(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			if GlobalNodeEventBus.GetSubscriberCount() > 0 {
				PublishNodeHeartbeat()
			}
		}
	}()
}

// shouldFilterEvent determines if an event should be filtered to prevent spam
func (bus *NodeEventBus) shouldFilterEvent(event NodeEvent) bool {
	// Filter heartbeat events if there are no subscribers
	if event.Type == NodeHeartbeat && len(bus.subscribers) == 0 {
		return true
	}

	// FIXED: Add deduplication for status events to prevent spam
	// Check for duplicate events within a short time window (1 second)
	if bus.isDuplicateStatusEvent(event) {
		logger.Logger.Debug().Msgf("[NodeEventBus] Filtering duplicate status event: %s for node %s", event.Type, event.NodeID)
		return true
	}

	return false
}

// lastEventCache stores recent events for deduplication
var lastEventCache = make(map[string]NodeEvent)
var lastEventCacheMutex sync.RWMutex

// isDuplicateStatusEvent checks if a status event is a duplicate
func (bus *NodeEventBus) isDuplicateStatusEvent(event NodeEvent) bool {
	// Only check status-related events
	statusEventTypes := map[NodeEventType]bool{
		NodeUnifiedStatusChanged: true,
		NodeStatusUpdated:        true,
		NodeHealthChanged:        true,
		NodeOnline:               true,
		NodeOffline:              true,
		NodeStateTransition:      true,
	}

	if !statusEventTypes[event.Type] {
		return false
	}

	// Create cache key
	cacheKey := fmt.Sprintf("%s:%s", event.Type, event.NodeID)

	lastEventCacheMutex.Lock()
	defer lastEventCacheMutex.Unlock()

	// Check if we have a recent event
	if lastEvent, exists := lastEventCache[cacheKey]; exists {
		// Check if events are too close in time (within 1 second)
		if time.Since(lastEvent.Timestamp) < 1*time.Second {
			// For status events, also check if the actual status changed
			if event.Type == NodeUnifiedStatusChanged || event.Type == NodeStatusUpdated || event.Type == NodeHealthChanged {
				return bus.compareStatusEventData(lastEvent, event)
			}
			return true // Other events are considered duplicates if within 1 second
		}
	}

	// Cache this event
	lastEventCache[cacheKey] = event

	// Clean up old cache entries (keep only last 50 per event type)
	if len(lastEventCache) > 200 {
		bus.cleanupEventCache()
	}

	return false
}

// compareStatusEventData compares two status events to see if they represent the same status
func (bus *NodeEventBus) compareStatusEventData(lastEvent, newEvent NodeEvent) bool {
	// Compare the status field
	if lastEvent.Status != newEvent.Status {
		return false
	}

	// For unified status events, compare old/new status
	if newEvent.Type == NodeUnifiedStatusChanged {
		if lastEvent.OldStatus != newEvent.OldStatus || lastEvent.NewStatus != newEvent.NewStatus {
			return false
		}
	}

	// Events are the same
	return true
}

// cleanupEventCache removes old entries from the event cache
func (bus *NodeEventBus) cleanupEventCache() {
	// Remove entries older than 5 minutes
	cutoff := time.Now().Add(-5 * time.Minute)
	for key, event := range lastEventCache {
		if event.Timestamp.Before(cutoff) {
			delete(lastEventCache, key)
		}
	}
}

// PublishNodeUnifiedStatusChanged publishes a unified status change event
func PublishNodeUnifiedStatusChanged(nodeID string, oldStatus, newStatus interface{}, source, reason string) {
	event := NodeEvent{
		Type:      NodeUnifiedStatusChanged,
		NodeID:    nodeID,
		Timestamp: time.Now(),
		OldStatus: oldStatus,
		NewStatus: newStatus,
		Source:    source,
		Reason:    reason,
		Data: map[string]interface{}{
			"old_status": oldStatus,
			"new_status": newStatus,
			"source":     source,
			"reason":     reason,
		},
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing NodeUnifiedStatusChanged event - NodeID: %s, Source: %s", nodeID, source)

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeStateTransition publishes a state transition event
func PublishNodeStateTransition(nodeID string, fromState, toState, reason string) {
	event := NodeEvent{
		Type:      NodeStateTransition,
		NodeID:    nodeID,
		Status:    toState,
		Timestamp: time.Now(),
		Source:    "state_transition",
		Reason:    reason,
		Data: map[string]interface{}{
			"from_state": fromState,
			"to_state":   toState,
			"reason":     reason,
		},
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing NodeStateTransition event - NodeID: %s, %s -> %s", nodeID, fromState, toState)

	GlobalNodeEventBus.Publish(event)
}

// PublishNodeStatusRefreshed publishes a status refresh event
func PublishNodeStatusRefreshed(nodeID string, status interface{}) {
	event := NodeEvent{
		Type:      NodeStatusRefreshed,
		NodeID:    nodeID,
		Status:    "refreshed",
		Timestamp: time.Now(),
		Data:      status,
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing NodeStatusRefreshed event - NodeID: %s", nodeID)

	GlobalNodeEventBus.Publish(event)
}

// PublishBulkStatusUpdate publishes a bulk status update event
func PublishBulkStatusUpdate(nodeCount int, successful int, failed int, errors []string) {
	event := NodeEvent{
		Type:      BulkStatusUpdate,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"total_nodes": nodeCount,
			"successful":  successful,
			"failed":      failed,
			"errors":      errors,
		},
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing BulkStatusUpdate event - Total: %d, Success: %d, Failed: %d", nodeCount, successful, failed)

	GlobalNodeEventBus.Publish(event)
}

// Enhanced helper functions with backward compatibility

// PublishNodeStatusUpdatedEnhanced publishes an enhanced node status change event with old/new status
func PublishNodeStatusUpdatedEnhanced(nodeID string, oldStatus, newStatus interface{}, source, reason string) {
	// Publish new unified event
	PublishNodeUnifiedStatusChanged(nodeID, oldStatus, newStatus, source, reason)

	// Maintain backward compatibility with legacy event
	statusStr := "unknown"
	if newStatus != nil {
		if statusMap, ok := newStatus.(map[string]interface{}); ok {
			if state, exists := statusMap["state"]; exists {
				statusStr = fmt.Sprintf("%v", state)
			}
		}
	}

	PublishNodeStatusUpdated(nodeID, statusStr, newStatus)
}

// PublishNodeHealthChangedEnhanced publishes an enhanced health change event
func PublishNodeHealthChangedEnhanced(nodeID string, oldHealth, newHealth string, data interface{}, source, reason string) {
	event := NodeEvent{
		Type:      NodeHealthChanged,
		NodeID:    nodeID,
		Status:    newHealth,
		Timestamp: time.Now(),
		OldStatus: oldHealth,
		NewStatus: newHealth,
		Source:    source,
		Reason:    reason,
		Data:      data,
	}

	logger.Logger.Debug().Msgf("🔍 NODE_EVENT_DEBUG: Publishing Enhanced NodeHealthChanged event - NodeID: %s, %s -> %s", nodeID, oldHealth, newHealth)

	GlobalNodeEventBus.Publish(event)
}
