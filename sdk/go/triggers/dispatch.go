// Package triggers dispatch support: envelope detection, unwrapping, and
// context propagation.
//
// When the control plane dispatches a webhook event to a reasoner it wraps
// the payload in an envelope: {"event": <payload>, "_meta": <metadata>}.
// This file provides the detection/unwrap logic plus context helpers so the
// handler can retrieve the trigger metadata via FromContext(ctx).
//
// Direct calls (no envelope) pass through unchanged and FromContext returns
// nil, so existing reasoners are unaffected.

package triggers

import (
	"context"
	"encoding/json"
	"time"
)

// contextKey is the private key type for storing *Context in a context.Context.
type contextKey struct{}

// NewContext returns a copy of parent carrying the trigger context tc.
// Called by the SDK dispatch path; user code normally uses FromContext.
func NewContext(parent context.Context, tc *Context) context.Context {
	if tc == nil {
		return parent
	}
	return context.WithValue(parent, contextKey{}, tc)
}

// FromContext returns the *Context carried by ctx, or nil when the reasoner
// was invoked directly rather than dispatched by an inbound trigger.
//
// Usage in a reasoner handler:
//
//	func handlePayment(ctx context.Context, in map[string]any) (any, error) {
//	    if tc := triggers.FromContext(ctx); tc != nil {
//	        // dispatched via a trigger — tc.Source, tc.EventID, etc.
//	    }
//	    return nil, nil
//	}
func FromContext(ctx context.Context) *Context {
	if ctx == nil {
		return nil
	}
	tc, _ := ctx.Value(contextKey{}).(*Context)
	return tc
}

// envelopeMeta mirrors the dispatcher's _meta payload shape.
type envelopeMeta struct {
	TriggerID      string `json:"trigger_id"`
	Source         string `json:"source"`
	EventType      string `json:"event_type"`
	EventID        string `json:"event_id"`
	IdempotencyKey string `json:"idempotency_key"`
	ReceivedAt     string `json:"received_at"`
	VCID           string `json:"vc_id"`
}

// IsEnvelope reports whether the decoded request body is a dispatcher trigger
// envelope: an object carrying both "event" and "_meta" keys, where "_meta"
// contains a "trigger_id".
func IsEnvelope(body map[string]any) bool {
	if body == nil {
		return false
	}
	if _, ok := body["event"]; !ok {
		return false
	}
	rawMeta, ok := body["_meta"]
	if !ok {
		return false
	}
	meta, ok := rawMeta.(map[string]any)
	if !ok {
		return false
	}
	_, hasTriggerID := meta["trigger_id"]
	return hasTriggerID
}

// Unwrap detects and unwraps a dispatcher trigger envelope.
//
// For a trigger dispatch it returns the inner event payload and a populated
// *Context. For a direct call (not an envelope) it returns the body unchanged
// and a nil *Context, so callers can treat both uniformly.
func Unwrap(body map[string]any) (map[string]any, *Context) {
	if !IsEnvelope(body) {
		return body, nil
	}

	event, _ := body["event"].(map[string]any)
	if event == nil {
		// "event" present but not an object (e.g. a list or scalar). Preserve
		// it under a conventional key so the handler still sees the payload.
		event = map[string]any{"event": body["event"]}
	}

	rawMeta, _ := json.Marshal(body["_meta"])
	var meta envelopeMeta
	_ = json.Unmarshal(rawMeta, &meta)

	receivedAt := parseReceivedAt(meta.ReceivedAt)

	tc := &Context{
		TriggerID:      meta.TriggerID,
		Source:         meta.Source,
		EventType:      meta.EventType,
		EventID:        meta.EventID,
		IdempotencyKey: meta.IdempotencyKey,
		ReceivedAt:     receivedAt,
		VCID:           meta.VCID,
	}
	return event, tc
}

// parseReceivedAt parses the dispatcher's received_at timestamp, falling back
// to time.Now() when the value is missing or unparseable so downstream code
// never sees a zero time.
func parseReceivedAt(value string) time.Time {
	if value == "" {
		return time.Now().UTC()
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05Z",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.Parse(layout, value); err == nil {
			return t
		}
	}
	return time.Now().UTC()
}

// ApplyTransform picks the best-matching binding for tc and runs its Transform
// against input, returning the transformed value.
//
// Matching rules (identical to the Python and TypeScript SDKs):
//  1. binding.Source must equal tc.Source
//  2. when the binding declares EventTypes, tc.EventType must match one of
//     them exactly or by prefix ("pull_request" matches "pull_request.opened")
//  3. a binding with explicit EventTypes wins over a catch-all binding
//  4. if the winning binding has no Transform, input is returned unchanged
//
// A panicking Transform is recovered and the raw input returned, so a buggy
// transform degrades to pass-through instead of failing the dispatch.
func ApplyTransform(tc *Context, bindings []Binding, input map[string]any) any {
	if tc == nil || len(bindings) == 0 {
		return input
	}

	var best *Binding
	bestSpecificity := -1

	for i := range bindings {
		b := &bindings[i]
		if b.Kind != EventBinding {
			continue
		}
		if b.Source != tc.Source {
			continue
		}
		if len(b.EventTypes) > 0 {
			if !matchesEventType(b.EventTypes, tc.EventType) {
				continue
			}
			if bestSpecificity < 1 {
				best, bestSpecificity = b, 1
			}
			continue
		}
		if bestSpecificity < 0 {
			best, bestSpecificity = b, 0
		}
	}

	if best == nil || best.TransformFn == nil {
		return input
	}

	result := func() (out any) {
		defer func() {
			if recover() != nil {
				out = input
			}
		}()
		return best.TransformFn(input)
	}()
	return result
}

// matchesEventType reports whether eventType matches any filter exactly or as
// a dotted prefix.
func matchesEventType(filters []string, eventType string) bool {
	for _, f := range filters {
		if f == "" || f == eventType {
			return true
		}
		if len(eventType) > len(f)+1 && eventType[:len(f)] == f && eventType[len(f)] == '.' {
			return true
		}
	}
	return false
}
