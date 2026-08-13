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

// simContextKey is the context key used by the test helpers to attach the
// synthetic *Context.
//
// Note: this deliberately mirrors what the dispatch path does. Once the
// dispatch-side helpers land (#514) this will be replaced by the shared
// NewContext/FromContext pair so the simulated and live paths use one
// implementation.
type simContextKey struct{}

// withSimContext attaches tc to ctx for retrieval by SimulatedContextFrom.
func withSimContext(parent context.Context, tc *Context) context.Context {
	if tc == nil {
		return parent
	}
	return context.WithValue(parent, simContextKey{}, tc)
}

// SimulatedContextFrom returns the *Context attached by SimulateEvent or
// SimulateSchedule, or nil when ctx carries none.
//
// Handlers under test that read the trigger context should use this in tests;
// production code uses the dispatch-side accessor.
func SimulatedContextFrom(ctx context.Context) *Context {
	if ctx == nil {
		return nil
	}
	tc, _ := ctx.Value(simContextKey{}).(*Context)
	return tc
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
		transformed := applyMatchingTransform(tc, opts.Bindings, body)
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
	return handler(withSimContext(parent, tc), input)
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

// applyMatchingTransform picks the best-matching binding for tc and runs its
// Transform against input.
//
// Matching rules (identical to the Python and TypeScript SDKs):
//  1. binding.Source must equal tc.Source
//  2. when the binding declares EventTypes, tc.EventType must match one of
//     them exactly or by dotted prefix ("pull_request" matches
//     "pull_request.opened")
//  3. a binding with explicit EventTypes wins over a catch-all binding
//  4. if the winning binding has no Transform, input is returned unchanged
//
// A panicking Transform is recovered and the raw input returned, so a buggy
// transform degrades to pass-through instead of failing the test run with an
// unhelpful stack.
func applyMatchingTransform(tc *Context, bindings []Binding, input map[string]any) any {
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
			if !eventTypeMatches(b.EventTypes, tc.EventType) {
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

	return func() (out any) {
		defer func() {
			if recover() != nil {
				out = input
			}
		}()
		return best.TransformFn(input)
	}()
}

// eventTypeMatches reports whether eventType matches any filter exactly or as
// a dotted prefix.
func eventTypeMatches(filters []string, eventType string) bool {
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
