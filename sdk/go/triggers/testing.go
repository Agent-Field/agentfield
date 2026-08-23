package triggers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"testing"
	"time"
)

// HandlerFunc mirrors the Go SDK's reasoner handler signature. Declared here
// rather than imported from the agent package so the triggers package stays
// dependency-free (and importable from agent without a cycle).
type HandlerFunc func(ctx context.Context, input map[string]any) (any, error)

// SimulatedContextFrom returns the *Context attached by SimulateEvent or
// SimulateSchedule, or nil when ctx carries none.
//
// This is an alias for FromContext: the helpers attach the context through the
// same mechanism the live dispatch path uses, so a handler reading
// FromContext(ctx) in production sees the simulated context unchanged under
// test. Prefer FromContext in handler code; this name is kept for symmetry
// with the Simulate* helpers.
func SimulatedContextFrom(ctx context.Context) *Context {
	return FromContext(ctx)
}

// SimulateEventOpts configures a simulated event dispatch.
//
// Only Source is required; every identifier defaults to a fresh random value
// so repeated simulations are independently dedup-safe.
type SimulateEventOpts struct {
	// Source is the provider name ("stripe", "github", ...). Required.
	Source string
	// Body is the inbound event payload, typically from LoadFixture.
	Body map[string]any
	// EventType is the provider's event type. Defaults to "".
	EventType string
	// Bindings are the reasoner's declared bindings. When supplied, the
	// matching binding's Transform runs before the handler, exactly as the
	// live dispatch path does.
	Bindings []Binding
	// The remaining fields override the auto-generated values.
	TriggerID      string
	EventID        string
	IdempotencyKey string
	ReceivedAt     time.Time
	VCID           string
	// Ctx is the parent context. Defaults to context.Background().
	Ctx context.Context
}

// SimulateScheduleOpts configures a simulated cron dispatch.
type SimulateScheduleOpts struct {
	// Cron is the expression recorded on the synthetic body for test
	// introspection. It does NOT schedule anything — this is a one-shot call.
	Cron string
	// Bindings are the reasoner's declared bindings.
	Bindings []Binding
	// ReceivedAt overrides the auto-generated timestamp.
	ReceivedAt time.Time
	// Ctx is the parent context. Defaults to context.Background().
	Ctx context.Context
}

// SimulateEvent runs handler as if a trigger of opts.Source had fired with
// opts.Body, without a control plane, HTTP server, or real provider.
//
// It builds the *Context the runtime would have produced, applies the matching
// binding's Transform when Bindings are supplied, and invokes the handler with
// the trigger context attached — the same shape the live dispatch path
// delivers.
//
//	func TestHandlePayment(t *testing.T) {
//	    result, err := triggers.SimulateEvent(t, handlePayment, triggers.SimulateEventOpts{
//	        Source:    "stripe",
//	        EventType: "payment_intent.succeeded",
//	        Body:      triggers.LoadFixture(t, "stripe"),
//	    })
//	    ...
//	}
func SimulateEvent(t *testing.T, handler HandlerFunc, opts SimulateEventOpts) (any, error) {
	t.Helper()

	if handler == nil {
		t.Fatal("SimulateEvent: handler must not be nil")
	}
	if opts.Source == "" {
		t.Fatal("SimulateEvent: opts.Source is required")
	}

	body := opts.Body
	if body == nil {
		body = map[string]any{}
	}

	tc := &Context{
		TriggerID:      orRandomID(opts.TriggerID, "trg_sim_"),
		Source:         opts.Source,
		EventType:      opts.EventType,
		EventID:        orRandomID(opts.EventID, "evt_sim_"),
		IdempotencyKey: orRandomID(opts.IdempotencyKey, "idem_sim_"),
		ReceivedAt:     orNow(opts.ReceivedAt),
		VCID:           opts.VCID,
	}

	// Apply the matching binding's Transform, mirroring the dispatch path.
	input := body
	if len(opts.Bindings) > 0 {
		transformed := ApplyTransform(tc, opts.Bindings, body)
		if resolved, ok := transformed.(map[string]any); ok {
			input = resolved
		} else {
			input = map[string]any{"input": transformed}
		}
	}

	parent := opts.Ctx
	if parent == nil {
		parent = context.Background()
	}
	return handler(NewContext(parent, tc), input)
}

// SimulateSchedule runs a cron-triggered handler with a synthetic tick event.
//
// Convenience wrapper around SimulateEvent with Source "cron" and EventType
// "tick". The cron expression is recorded on the body for test introspection
// but nothing is actually scheduled — this is a single invocation.
func SimulateSchedule(t *testing.T, handler HandlerFunc, opts SimulateScheduleOpts) (any, error) {
	t.Helper()

	body := map[string]any{}
	if opts.Cron != "" {
		body["cron"] = opts.Cron
	}

	return SimulateEvent(t, handler, SimulateEventOpts{
		Source:     "cron",
		EventType:  "tick",
		Body:       body,
		Bindings:   opts.Bindings,
		ReceivedAt: opts.ReceivedAt,
		Ctx:        opts.Ctx,
	})
}

// orRandomID returns value when non-empty, else prefix plus 12 random hex
// characters. Falls back to a timestamp-derived suffix if the entropy source
// is unavailable, so a helper can never fail a test for this reason.
func orRandomID(value, prefix string) string {
	if value != "" {
		return value
	}
	buf := make([]byte, 6)
	if _, err := rand.Read(buf); err != nil {
		return prefix + hex.EncodeToString([]byte(time.Now().Format("150405")))
	}
	return prefix + hex.EncodeToString(buf)
}

// orNow returns value when set, else the current UTC time.
func orNow(value time.Time) time.Time {
	if value.IsZero() {
		return time.Now().UTC()
	}
	return value
}
