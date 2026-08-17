package triggers

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// LoadFixture
// ---------------------------------------------------------------------------

func TestLoadFixtureAllSources(t *testing.T) {
	for _, name := range FixtureNames() {
		t.Run(name, func(t *testing.T) {
			fixture := LoadFixture(t, name)
			if len(fixture) == 0 {
				t.Fatalf("fixture %q decoded to an empty object", name)
			}
		})
	}
}

func TestLoadFixtureAcceptsExplicitJSONSuffix(t *testing.T) {
	withSuffix := LoadFixture(t, "stripe.json")
	withoutSuffix := LoadFixture(t, "stripe")

	if withSuffix["id"] != withoutSuffix["id"] {
		t.Fatal("expected stripe.json and stripe to resolve to the same fixture")
	}
}

func TestLoadFixtureKnownFieldValues(t *testing.T) {
	tests := []struct {
		name  string
		check func(t *testing.T, f map[string]any)
	}{
		{"stripe", func(t *testing.T, f map[string]any) {
			if f["type"] != "payment_intent.succeeded" {
				t.Fatalf("type = %v, want payment_intent.succeeded", f["type"])
			}
			data, ok := f["data"].(map[string]any)
			if !ok {
				t.Fatal("expected data object")
			}
			obj, ok := data["object"].(map[string]any)
			if !ok {
				t.Fatal("expected data.object")
			}
			if obj["amount"] != float64(5000) {
				t.Fatalf("amount = %v, want 5000", obj["amount"])
			}
		}},
		{"github", func(t *testing.T, f map[string]any) {
			if f["action"] != "opened" {
				t.Fatalf("action = %v, want opened", f["action"])
			}
			pr, ok := f["pull_request"].(map[string]any)
			if !ok {
				t.Fatal("expected pull_request object")
			}
			if pr["number"] != float64(42) {
				t.Fatalf("pull_request.number = %v, want 42", pr["number"])
			}
		}},
		{"slack", func(t *testing.T, f map[string]any) {
			if f["type"] != "event_callback" {
				t.Fatalf("type = %v, want event_callback", f["type"])
			}
			ev, ok := f["event"].(map[string]any)
			if !ok {
				t.Fatal("expected event object")
			}
			if ev["type"] != "app_mention" {
				t.Fatalf("event.type = %v, want app_mention", ev["type"])
			}
		}},
		{"cron", func(t *testing.T, f map[string]any) {
			if f["cron"] != "0 9 * * *" {
				t.Fatalf("cron = %v, want 0 9 * * *", f["cron"])
			}
			if f["fired_at"] == nil {
				t.Fatal("expected fired_at to be present")
			}
		}},
		{"generic_hmac", func(t *testing.T, f map[string]any) {
			if f["event"] != "order.created" {
				t.Fatalf("event = %v, want order.created", f["event"])
			}
			if f["order_id"] != "ord_demo_42" {
				t.Fatalf("order_id = %v, want ord_demo_42", f["order_id"])
			}
		}},
		{"generic_bearer", func(t *testing.T, f map[string]any) {
			if f["kind"] != "internal.notification" {
				t.Fatalf("kind = %v, want internal.notification", f["kind"])
			}
			if f["notification_id"] != "notif_demo_77" {
				t.Fatalf("notification_id = %v, want notif_demo_77", f["notification_id"])
			}
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.check(t, LoadFixture(t, tc.name))
		})
	}
}

// TestFixturesAreByteIdenticalToPythonSDK is the parity guard required by the
// issue: the Go fixtures must be byte-for-byte copies of the Python SDK's, so
// a drift in one SDK's fixture cannot silently diverge behaviour between them.
//
// Skips when the Python SDK tree is not present (e.g. a Go-only checkout).
func TestFixturesAreByteIdenticalToPythonSDK(t *testing.T) {
	pythonDir := filepath.Join("..", "..", "python", "agentfield", "fixtures", "triggers")
	if _, err := os.Stat(pythonDir); err != nil {
		t.Skipf("Python SDK fixtures not available at %s: %v", pythonDir, err)
	}

	for _, name := range FixtureNames() {
		t.Run(name, func(t *testing.T) {
			pythonBytes, err := os.ReadFile(filepath.Join(pythonDir, name+".json"))
			if err != nil {
				t.Fatalf("reading Python fixture: %v", err)
			}
			goBytes := RawFixture(t, name)

			if string(goBytes) != string(pythonBytes) {
				t.Fatalf("fixture %q differs from the Python SDK copy\n"+
					"Go:     %d bytes\nPython: %d bytes\n"+
					"re-copy with: cp sdk/python/agentfield/fixtures/triggers/*.json sdk/go/triggers/testdata/",
					name, len(goBytes), len(pythonBytes))
			}
		})
	}
}

func TestFixtureNamesCoversAllEmbeddedFiles(t *testing.T) {
	entries, err := fixtureFS.ReadDir("testdata")
	if err != nil {
		t.Fatalf("reading embedded testdata: %v", err)
	}

	embedded := map[string]bool{}
	for _, e := range entries {
		embedded[e.Name()] = true
	}

	for _, name := range FixtureNames() {
		if !embedded[name+".json"] {
			t.Fatalf("FixtureNames() lists %q but testdata/%s.json is not embedded", name, name)
		}
	}
	if len(embedded) != len(FixtureNames()) {
		t.Fatalf("embedded fixture count (%d) does not match FixtureNames() (%d) — "+
			"a fixture was added without updating FixtureNames()",
			len(embedded), len(FixtureNames()))
	}
}

// ---------------------------------------------------------------------------
// SimulateEvent
// ---------------------------------------------------------------------------

func TestSimulateEventBuildsContextAndPassesBody(t *testing.T) {
	var gotInput map[string]any
	var gotTrigger *Context

	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		gotTrigger = SimulatedContextFrom(ctx)
		return map[string]any{"kind": "payment"}, nil
	}

	result, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:    "stripe",
		EventType: "payment_intent.succeeded",
		Body:      LoadFixture(t, "stripe"),
	})
	if err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if got := result.(map[string]any)["kind"]; got != "payment" {
		t.Fatalf("result kind = %v, want payment", got)
	}
	if gotTrigger == nil {
		t.Fatal("expected a non-nil trigger context")
	}
	if gotTrigger.Source != "stripe" {
		t.Fatalf("Source = %q, want stripe", gotTrigger.Source)
	}
	if gotTrigger.EventType != "payment_intent.succeeded" {
		t.Fatalf("EventType = %q, want payment_intent.succeeded", gotTrigger.EventType)
	}
	// Auto-generated identifiers must be populated.
	if gotTrigger.TriggerID == "" || gotTrigger.EventID == "" || gotTrigger.IdempotencyKey == "" {
		t.Fatalf("expected auto-generated IDs, got %+v", gotTrigger)
	}
	if gotTrigger.ReceivedAt.IsZero() {
		t.Fatal("expected ReceivedAt to be set")
	}
	// The raw fixture reaches the handler when no bindings are supplied.
	if gotInput["type"] != "payment_intent.succeeded" {
		t.Fatalf("expected the raw fixture body, got %v", gotInput["type"])
	}
}

func TestSimulateEventAllSixFixtures(t *testing.T) {
	// The acceptance criterion: every fixture drives the handler and yields a
	// correctly shaped *Context.
	for _, name := range FixtureNames() {
		t.Run(name, func(t *testing.T) {
			var gotTrigger *Context

			handler := func(ctx context.Context, input map[string]any) (any, error) {
				gotTrigger = SimulatedContextFrom(ctx)
				if len(input) == 0 {
					t.Fatal("handler received an empty input")
				}
				return "ok", nil
			}

			eventType := name + ".event"
			if name == "cron" {
				eventType = "tick"
			}

			result, err := SimulateEvent(t, handler, SimulateEventOpts{
				Source:    name,
				EventType: eventType,
				Body:      LoadFixture(t, name),
			})
			if err != nil {
				t.Fatalf("SimulateEvent returned error: %v", err)
			}
			if result != "ok" {
				t.Fatalf("result = %v, want ok", result)
			}
			if gotTrigger == nil {
				t.Fatal("expected a non-nil trigger context")
			}
			if gotTrigger.Source != name {
				t.Fatalf("Source = %q, want %q", gotTrigger.Source, name)
			}
			if gotTrigger.EventType != eventType {
				t.Fatalf("EventType = %q, want %q", gotTrigger.EventType, eventType)
			}
		})
	}
}

func TestSimulateEventAppliesBindingTransform(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source: "stripe",
			Types:  []string{"payment_intent.succeeded"},
			Transform: func(evt map[string]any) any {
				data, _ := evt["data"].(map[string]any)
				return data["object"]
			},
		}),
	}

	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:    "stripe",
		EventType: "payment_intent.succeeded",
		Body:      LoadFixture(t, "stripe"),
		Bindings:  bindings,
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	// The transform peels data.object, so the handler sees the payment intent.
	if gotInput["id"] != "pi_3NYExample1234" {
		t.Fatalf("id = %v, want pi_3NYExample1234 (transform should have run)", gotInput["id"])
	}
	if gotInput["amount"] != float64(5000) {
		t.Fatalf("amount = %v, want 5000", gotInput["amount"])
	}
}

func TestSimulateEventTransformNotAppliedWhenSourceDiffers(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "github",
			Transform: func(map[string]any) any { return map[string]any{"transformed": true} },
		}),
	}

	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:   "stripe",
		Body:     LoadFixture(t, "stripe"),
		Bindings: bindings,
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if _, transformed := gotInput["transformed"]; transformed {
		t.Fatal("a github binding must not transform a stripe event")
	}
}

func TestSimulateEventOverridesIdentifiers(t *testing.T) {
	fixed := time.Date(2026, 1, 15, 10, 0, 0, 0, time.UTC)

	var gotTrigger *Context
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotTrigger = SimulatedContextFrom(ctx)
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:         "stripe",
		TriggerID:      "my_trigger",
		EventID:        "my_event",
		IdempotencyKey: "my_key",
		ReceivedAt:     fixed,
		VCID:           "vc_test",
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if gotTrigger.TriggerID != "my_trigger" {
		t.Fatalf("TriggerID = %q, want my_trigger", gotTrigger.TriggerID)
	}
	if gotTrigger.EventID != "my_event" {
		t.Fatalf("EventID = %q, want my_event", gotTrigger.EventID)
	}
	if gotTrigger.IdempotencyKey != "my_key" {
		t.Fatalf("IdempotencyKey = %q, want my_key", gotTrigger.IdempotencyKey)
	}
	if !gotTrigger.ReceivedAt.Equal(fixed) {
		t.Fatalf("ReceivedAt = %v, want %v", gotTrigger.ReceivedAt, fixed)
	}
	if gotTrigger.VCID != "vc_test" {
		t.Fatalf("VCID = %q, want vc_test", gotTrigger.VCID)
	}
}

func TestSimulateEventGeneratesUniqueIDsPerCall(t *testing.T) {
	var ids []string
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		ids = append(ids, SimulatedContextFrom(ctx).EventID)
		return nil, nil
	}

	for i := 0; i < 2; i++ {
		if _, err := SimulateEvent(t, handler, SimulateEventOpts{Source: "github"}); err != nil {
			t.Fatalf("SimulateEvent returned error: %v", err)
		}
	}

	if ids[0] == ids[1] {
		t.Fatalf("expected unique event IDs across calls, both were %q", ids[0])
	}
}

func TestSimulateEventDefaultsBodyToEmptyMap(t *testing.T) {
	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{Source: "cron"}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}
	if gotInput == nil {
		t.Fatal("expected a non-nil empty map, got nil")
	}
	if len(gotInput) != 0 {
		t.Fatalf("expected an empty body, got %v", gotInput)
	}
}

func TestSimulateEventPropagatesHandlerError(t *testing.T) {
	sentinel := context.DeadlineExceeded
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		return nil, sentinel
	}

	_, err := SimulateEvent(t, handler, SimulateEventOpts{Source: "stripe"})
	if err != sentinel {
		t.Fatalf("err = %v, want %v", err, sentinel)
	}
}

func TestSimulateEventHonoursParentContext(t *testing.T) {
	type parentKey struct{}
	parent := context.WithValue(context.Background(), parentKey{}, "parent-value")

	var gotValue any
	var gotTrigger *Context
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotValue = ctx.Value(parentKey{})
		gotTrigger = SimulatedContextFrom(ctx)
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source: "stripe",
		Ctx:    parent,
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if gotValue != "parent-value" {
		t.Fatalf("parent context value = %v, want parent-value", gotValue)
	}
	if gotTrigger == nil {
		t.Fatal("expected the trigger context to be attached alongside parent values")
	}
}

func TestSimulateEventWrapsNonObjectTransformResult(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "generic_hmac",
			Transform: func(map[string]any) any { return []any{"a", "b"} },
		}),
	}

	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:   "generic_hmac",
		Body:     LoadFixture(t, "generic_hmac"),
		Bindings: bindings,
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	list, ok := gotInput["input"].([]any)
	if !ok {
		t.Fatalf("expected the slice wrapped under \"input\", got %#v", gotInput)
	}
	if len(list) != 2 {
		t.Fatalf("expected 2 items, got %d", len(list))
	}
}

func TestSimulateEventRecoversPanickingTransform(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "stripe",
			Transform: func(map[string]any) any { panic("boom") },
		}),
	}

	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:   "stripe",
		Body:     map[string]any{"id": "pi_raw"},
		Bindings: bindings,
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if gotInput["id"] != "pi_raw" {
		t.Fatalf("expected raw input after panic recovery, got %v", gotInput)
	}
}

// ---------------------------------------------------------------------------
// SimulateSchedule
// ---------------------------------------------------------------------------

func TestSimulateScheduleUsesCronSourceAndTickType(t *testing.T) {
	var gotTrigger *Context
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotTrigger = SimulatedContextFrom(ctx)
		return map[string]any{"count": 1}, nil
	}

	result, err := SimulateSchedule(t, handler, SimulateScheduleOpts{})
	if err != nil {
		t.Fatalf("SimulateSchedule returned error: %v", err)
	}

	if got := result.(map[string]any)["count"]; got != 1 {
		t.Fatalf("count = %v, want 1", got)
	}
	if gotTrigger.Source != "cron" {
		t.Fatalf("Source = %q, want cron", gotTrigger.Source)
	}
	if gotTrigger.EventType != "tick" {
		t.Fatalf("EventType = %q, want tick", gotTrigger.EventType)
	}
}

func TestSimulateScheduleRecordsCronExpressionInBody(t *testing.T) {
	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateSchedule(t, handler, SimulateScheduleOpts{Cron: "0 9 * * *"}); err != nil {
		t.Fatalf("SimulateSchedule returned error: %v", err)
	}

	if gotInput["cron"] != "0 9 * * *" {
		t.Fatalf("cron = %v, want 0 9 * * *", gotInput["cron"])
	}
}

func TestSimulateScheduleEmptyBodyWithoutCron(t *testing.T) {
	var gotInput map[string]any
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	}

	if _, err := SimulateSchedule(t, handler, SimulateScheduleOpts{}); err != nil {
		t.Fatalf("SimulateSchedule returned error: %v", err)
	}

	if len(gotInput) != 0 {
		t.Fatalf("expected an empty body without Cron, got %v", gotInput)
	}
}

func TestSimulateScheduleHonoursReceivedAt(t *testing.T) {
	fixed := time.Date(2026, 6, 1, 9, 0, 0, 0, time.UTC)

	var gotTrigger *Context
	handler := func(ctx context.Context, input map[string]any) (any, error) {
		gotTrigger = SimulatedContextFrom(ctx)
		return nil, nil
	}

	if _, err := SimulateSchedule(t, handler, SimulateScheduleOpts{ReceivedAt: fixed}); err != nil {
		t.Fatalf("SimulateSchedule returned error: %v", err)
	}

	if !gotTrigger.ReceivedAt.Equal(fixed) {
		t.Fatalf("ReceivedAt = %v, want %v", gotTrigger.ReceivedAt, fixed)
	}
}

func TestSimulatedContextFromNilCases(t *testing.T) {
	if SimulatedContextFrom(context.Background()) != nil {
		t.Fatal("expected nil for a plain context")
	}
	//nolint:staticcheck // deliberately passing nil to prove it is handled
	if SimulatedContextFrom(nil) != nil {
		t.Fatal("expected nil for a nil context")
	}
}

// TestSimulatedContextIsReadableByFromContext is the guard against the
// simulated and live dispatch paths drifting apart: a handler written against
// FromContext (as production code is) must see the context the Simulate*
// helpers attach. Before these shared, the helpers used a private key and
// FromContext returned nil under test while working in production.
func TestSimulatedContextIsReadableByFromContext(t *testing.T) {
	var viaFromContext *Context

	handler := func(ctx context.Context, input map[string]any) (any, error) {
		viaFromContext = FromContext(ctx)
		return nil, nil
	}

	if _, err := SimulateEvent(t, handler, SimulateEventOpts{
		Source:    "stripe",
		EventType: "payment_intent.succeeded",
		EventID:   "evt_shared",
	}); err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if viaFromContext == nil {
		t.Fatal("FromContext must see the context attached by SimulateEvent")
	}
	if viaFromContext.Source != "stripe" {
		t.Fatalf("Source = %q, want stripe", viaFromContext.Source)
	}
	if viaFromContext.EventID != "evt_shared" {
		t.Fatalf("EventID = %q, want evt_shared", viaFromContext.EventID)
	}
}
