package events

import (
	"encoding/json"
	"sync"
	"time"
)

// ReasonerEventType represents the type of reasoner event
type ReasonerEventType string

const (
	ReasonerOnline    ReasonerEventType = "reasoner_online"
	ReasonerOffline   ReasonerEventType = "reasoner_offline"
	ReasonerUpdated   ReasonerEventType = "reasoner_updated"
	NodeStatusChanged ReasonerEventType = "node_status_changed"
	ReasonersRefresh  ReasonerEventType = "reasoners_refresh"
	Heartbeat         ReasonerEventType = "heartbeat"
)

// ReasonerEvent represents a reasoner state change event
type ReasonerEvent struct {
	Type       ReasonerEventType `json:"type"`
	ReasonerID string            `json:"reasoner_id,omitempty"`
	NodeID     string            `json:"node_id,omitempty"`
	Status     string            `json:"status,omitempty"`
	Timestamp  time.Time         `json:"timestamp"`
	Data       interface{}       `json:"data,omitempty"`
}

// ReasonerEventBus manages reasoner event broadcasting
type ReasonerEventBus struct {
	subscribers map[string]chan ReasonerEvent
	mutex       sync.RWMutex
}

// NewReasonerEventBus creates a new reasoner event bus
func NewReasonerEventBus() *ReasonerEventBus {
	return &ReasonerEventBus{
		subscribers: make(map[string]chan ReasonerEvent),
	}
}

// Subscribe adds a new subscriber to the event bus
func (bus *ReasonerEventBus) Subscribe(subscriberID string) chan ReasonerEvent {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	ch := make(chan ReasonerEvent, 100) // Buffer to prevent blocking
	bus.subscribers[subscriberID] = ch

	return ch
}

// Unsubscribe removes a subscriber from the event bus
func (bus *ReasonerEventBus) Unsubscribe(subscriberID string) {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	if ch, exists := bus.subscribers[subscriberID]; exists {
		close(ch)
		delete(bus.subscribers, subscriberID)
	}
}

// Publish broadcasts an event to all subscribers
func (bus *ReasonerEventBus) Publish(event ReasonerEvent) {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()

	for _, ch := range bus.subscribers {
		select {
		case ch <- event:
			// Event sent successfully
		default:
			// Channel is full, skip this subscriber to prevent blocking
		}
	}
}

// GetSubscriberCount returns the number of active subscribers
func (bus *ReasonerEventBus) GetSubscriberCount() int {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()
	return len(bus.subscribers)
}

// ToJSON converts a reasoner event to JSON string
func (event *ReasonerEvent) ToJSON() (string, error) {
	data, err := json.Marshal(event)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Global event bus instance
var GlobalReasonerEventBus = NewReasonerEventBus()

// Helper functions for common event types

// PublishReasonerOnline publishes a reasoner online event
func PublishReasonerOnline(reasonerID, nodeID string, data interface{}) {
	event := ReasonerEvent{
		Type:       ReasonerOnline,
		ReasonerID: reasonerID,
		NodeID:     nodeID,
		Status:     "online",
		Timestamp:  time.Now(),
		Data:       data,
	}

	GlobalReasonerEventBus.Publish(event)
}

// PublishReasonerOffline publishes a reasoner offline event
func PublishReasonerOffline(reasonerID, nodeID string, data interface{}) {
	event := ReasonerEvent{
		Type:       ReasonerOffline,
		ReasonerID: reasonerID,
		NodeID:     nodeID,
		Status:     "offline",
		Timestamp:  time.Now(),
		Data:       data,
	}

	GlobalReasonerEventBus.Publish(event)
}

// PublishReasonerUpdated publishes a reasoner updated event
func PublishReasonerUpdated(reasonerID, nodeID, status string, data interface{}) {
	event := ReasonerEvent{
		Type:       ReasonerUpdated,
		ReasonerID: reasonerID,
		NodeID:     nodeID,
		Status:     status,
		Timestamp:  time.Now(),
		Data:       data,
	}
	GlobalReasonerEventBus.Publish(event)
}

// PublishNodeStatusChanged publishes a node status change event
func PublishNodeStatusChanged(nodeID, status string, data interface{}) {
	event := ReasonerEvent{
		Type:      NodeStatusChanged,
		NodeID:    nodeID,
		Status:    status,
		Timestamp: time.Now(),
		Data:      data,
	}

	GlobalReasonerEventBus.Publish(event)
}

// PublishReasonersRefresh publishes a general refresh event
func PublishReasonersRefresh(data interface{}) {
	event := ReasonerEvent{
		Type:      ReasonersRefresh,
		Timestamp: time.Now(),
		Data:      data,
	}
	GlobalReasonerEventBus.Publish(event)
}

// PublishHeartbeat publishes a heartbeat event to keep connections alive
func PublishHeartbeat() {
	event := ReasonerEvent{
		Type:      Heartbeat,
		Timestamp: time.Now(),
	}
	GlobalReasonerEventBus.Publish(event)
}

// StartHeartbeat starts a goroutine that sends periodic heartbeat events
func StartHeartbeat(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			if GlobalReasonerEventBus.GetSubscriberCount() > 0 {
				PublishHeartbeat()
			}
		}
	}()
}
