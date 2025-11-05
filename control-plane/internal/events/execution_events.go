package events

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/pkg/types"
)

// ExecutionEventType represents the type of execution event
type ExecutionEventType string

const (
	ExecutionCreated   ExecutionEventType = "execution_created"
	ExecutionStarted   ExecutionEventType = "execution_started"
	ExecutionUpdated   ExecutionEventType = "execution_updated"
	ExecutionCompleted ExecutionEventType = "execution_completed"
	ExecutionFailed    ExecutionEventType = "execution_failed"
)

// ExecutionEvent represents an execution state change event
type ExecutionEvent struct {
	Type        ExecutionEventType `json:"type"`
	ExecutionID string             `json:"execution_id"`
	WorkflowID  string             `json:"workflow_id"`
	AgentNodeID string             `json:"agent_node_id"`
	Status      string             `json:"status"`
	Timestamp   time.Time          `json:"timestamp"`
	Data        interface{}        `json:"data,omitempty"`
}

// ExecutionEventBus manages execution event broadcasting
type ExecutionEventBus struct {
	subscribers map[string]chan ExecutionEvent
	mutex       sync.RWMutex
}

// NewExecutionEventBus creates a new execution event bus
func NewExecutionEventBus() *ExecutionEventBus {
	return &ExecutionEventBus{
		subscribers: make(map[string]chan ExecutionEvent),
	}
}

// Subscribe adds a new subscriber to the event bus
func (bus *ExecutionEventBus) Subscribe(subscriberID string) chan ExecutionEvent {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	ch := make(chan ExecutionEvent, 100) // Buffer to prevent blocking
	bus.subscribers[subscriberID] = ch

	logger.Logger.Debug().Msgf("[ExecutionEventBus] Subscriber %s added, total subscribers: %d", subscriberID, len(bus.subscribers))
	return ch
}

// Unsubscribe removes a subscriber from the event bus
func (bus *ExecutionEventBus) Unsubscribe(subscriberID string) {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	if ch, exists := bus.subscribers[subscriberID]; exists {
		close(ch)
		delete(bus.subscribers, subscriberID)
		logger.Logger.Debug().Msgf("[ExecutionEventBus] Subscriber %s removed, total subscribers: %d", subscriberID, len(bus.subscribers))
	}
}

// Publish broadcasts an event to all subscribers
func (bus *ExecutionEventBus) Publish(event ExecutionEvent) {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()

	logger.Logger.Debug().Msgf("[ExecutionEventBus] Publishing event: %s for execution %s to %d subscribers",
		event.Type, event.ExecutionID, len(bus.subscribers))

	for subscriberID, ch := range bus.subscribers {
		select {
		case ch <- event:
			// Event sent successfully
		default:
			// Channel is full, skip this subscriber to prevent blocking
			logger.Logger.Warn().Msgf("[ExecutionEventBus] Warning: Channel full for subscriber %s, skipping event", subscriberID)
		}
	}
}

// GetSubscriberCount returns the number of active subscribers
func (bus *ExecutionEventBus) GetSubscriberCount() int {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()
	return len(bus.subscribers)
}

// ToJSON converts an execution event to JSON string
func (event *ExecutionEvent) ToJSON() (string, error) {
	data, err := json.Marshal(event)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Global event bus instance
var GlobalExecutionEventBus = NewExecutionEventBus()

// Helper functions for common event types

// PublishExecutionCreated publishes an execution created event
func PublishExecutionCreated(executionID, workflowID, agentNodeID string, data interface{}) {
	event := ExecutionEvent{
		Type:        ExecutionCreated,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      "created",
		Timestamp:   time.Now(),
		Data:        data,
	}
	GlobalExecutionEventBus.Publish(event)
}

// PublishExecutionStarted publishes an execution started event
func PublishExecutionStarted(executionID, workflowID, agentNodeID string, data interface{}) {
	event := ExecutionEvent{
		Type:        ExecutionStarted,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      "running",
		Timestamp:   time.Now(),
		Data:        data,
	}
	GlobalExecutionEventBus.Publish(event)
}

// PublishExecutionUpdated publishes an execution updated event
func PublishExecutionUpdated(executionID, workflowID, agentNodeID, status string, data interface{}) {
	event := ExecutionEvent{
		Type:        ExecutionUpdated,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      status,
		Timestamp:   time.Now(),
		Data:        data,
	}
	GlobalExecutionEventBus.Publish(event)
}

// PublishExecutionCompleted publishes an execution completed event
func PublishExecutionCompleted(executionID, workflowID, agentNodeID string, data interface{}) {
	event := ExecutionEvent{
		Type:        ExecutionCompleted,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      string(types.ExecutionStatusSucceeded),
		Timestamp:   time.Now(),
		Data:        data,
	}
	GlobalExecutionEventBus.Publish(event)
}

// PublishExecutionFailed publishes an execution failed event
func PublishExecutionFailed(executionID, workflowID, agentNodeID string, data interface{}) {
	event := ExecutionEvent{
		Type:        ExecutionFailed,
		ExecutionID: executionID,
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		Status:      "failed",
		Timestamp:   time.Now(),
		Data:        data,
	}
	GlobalExecutionEventBus.Publish(event)
}
