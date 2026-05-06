package events

import "time"

// TriggerEvent is a real-time pub/sub message about an inbound event's
// lifecycle. SSE streams in the trigger UI subscribe to a global instance of
// this bus to push live updates without polling.
//
// Type values mirror the InboundEvent.Status transitions plus a "received"
// initial-state event so subscribers don't need to wait for dispatch.
type TriggerEvent struct {
	Type           string    `json:"type"` // "event.received" | "event.dispatched" | "event.failed"
	TriggerID      string    `json:"trigger_id"`
	EventID        string    `json:"event_id"`
	SourceName     string    `json:"source_name"`
	EventType      string    `json:"event_type,omitempty"`
	Status         string    `json:"status,omitempty"` // current InboundEvent.Status
	IdempotencyKey string    `json:"idempotency_key,omitempty"`
	VCID           string    `json:"vc_id,omitempty"`
	ErrorMessage   string    `json:"error_message,omitempty"`
	Timestamp      time.Time `json:"timestamp"`
}

// TriggerEvent type constants — using strings (not iota) so wire JSON stays
// readable on the SSE client side.
const (
	TriggerEventTypeReceived   = "event.received"
	TriggerEventTypeDispatched = "event.dispatched"
	TriggerEventTypeFailed     = "event.failed"
)

// GlobalTriggerEventBus is the process-wide bus used by storage write paths
// (publish) and SSE handlers (subscribe). Initialized at package load so it's
// ready before the storage layer wires through.
var GlobalTriggerEventBus = NewEventBus[TriggerEvent]()
