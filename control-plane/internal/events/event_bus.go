package events

import "sync"

// EventBus provides a generic pub/sub channel for real-time updates.
type EventBus[T any] struct {
	subscribers map[string]chan T
	mutex       sync.RWMutex
	bufferSize  int
}

// NewEventBus constructs an EventBus with a default buffer for subscriber channels.
func NewEventBus[T any]() *EventBus[T] {
	return &EventBus[T]{
		subscribers: make(map[string]chan T),
		bufferSize:  100,
	}
}

// Subscribe registers a subscriber and returns a channel to receive events.
func (bus *EventBus[T]) Subscribe(subscriberID string) chan T {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	ch := make(chan T, bus.bufferSize)
	bus.subscribers[subscriberID] = ch
	return ch
}

// Unsubscribe removes the subscriber and closes the channel.
func (bus *EventBus[T]) Unsubscribe(subscriberID string) {
	bus.mutex.Lock()
	defer bus.mutex.Unlock()

	if ch, ok := bus.subscribers[subscriberID]; ok {
		close(ch)
		delete(bus.subscribers, subscriberID)
	}
}

// Publish delivers an event to all subscribers without blocking.
func (bus *EventBus[T]) Publish(event T) {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()

	for id, ch := range bus.subscribers {
		select {
		case ch <- event:
		default:
			// drop event for slow subscriber to avoid blocking
			_ = id
		}
	}
}

// SubscriberCount returns number of active subscribers.
func (bus *EventBus[T]) SubscriberCount() int {
	bus.mutex.RLock()
	defer bus.mutex.RUnlock()
	return len(bus.subscribers)
}
