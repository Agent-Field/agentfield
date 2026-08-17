package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"os"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/triggers"
)

// The demo doubles as a working example of testing trigger reasoners with the
// helpers from the triggers package: no control plane, no HTTP, no provider.
// These tests pin the record shapes and memory keys the UI surfaces depend on,
// and prove the Transform and direct-call contracts hold.

func TestMain(m *testing.M) {
	// The reasoners log every record; keep test output readable.
	log.SetOutput(io.Discard)
	os.Exit(m.Run())
}

// newDemoAgent builds an agent with no control plane URL, so Memory() uses the
// in-memory backend and the reasoners run fully offline.
func newDemoAgent(t *testing.T) *agent.Agent {
	t.Helper()
	app, err := agent.New(agent.Config{
		NodeID:  "triggers-demo-go-test",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	return app
}

// eventReasonerByName looks up a declared webhook reasoner.
func eventReasonerByName(t *testing.T, name string) eventReasoner {
	t.Helper()
	for _, r := range eventReasoners() {
		if r.Name == name {
			return r
		}
	}
	t.Fatalf("no event reasoner named %q", name)
	return eventReasoner{}
}

// bindingFor builds the binding list the dispatch path would have registered,
// so SimulateEvent applies the same Transform the live agent would.
func bindingFor(r eventReasoner) []triggers.Binding {
	return []triggers.Binding{triggers.Event(r.Opts)}
}

// readMemory fetches a key from the agent's in-memory backend using the same
// context shape the handlers write under.
func readMemory(t *testing.T, app *agent.Agent, key string) any {
	t.Helper()
	value, err := app.Memory().Get(context.Background(), key)
	if err != nil {
		t.Fatalf("reading memory key %q: %v", key, err)
	}
	return value
}

// ---------------------------------------------------------------------------
// stripeToPayment
// ---------------------------------------------------------------------------

func TestStripeToPaymentFlattensFixture(t *testing.T) {
	fixture := triggers.LoadFixture(t, "stripe")

	out, ok := stripeToPayment(fixture).(map[string]any)
	if !ok {
		t.Fatalf("expected a map, got %T", stripeToPayment(fixture))
	}

	if out["id"] != "pi_3NYExample1234" {
		t.Fatalf("id = %v, want pi_3NYExample1234", out["id"])
	}
	if out["amount"] != float64(5000) {
		t.Fatalf("amount = %v, want 5000", out["amount"])
	}
	if out["currency"] != "usd" {
		t.Fatalf("currency = %v, want usd", out["currency"])
	}
	if out["customer"] != "cus_example1234" {
		t.Fatalf("customer = %v, want cus_example1234", out["customer"])
	}
	if out["status"] != "succeeded" {
		t.Fatalf("status = %v, want succeeded", out["status"])
	}
}

func TestStripeToPaymentDefaultsCurrencyAndMetadata(t *testing.T) {
	// A payment_intent with neither currency nor metadata must still produce
	// both fields, matching the Python demo's .get(..., default) behaviour.
	event := map[string]any{
		"data": map[string]any{
			"object": map[string]any{"id": "pi_min", "amount": float64(100)},
		},
	}

	out := stripeToPayment(event).(map[string]any)
	if out["currency"] != "usd" {
		t.Fatalf("currency = %v, want the usd default", out["currency"])
	}
	if _, ok := out["metadata"].(map[string]any); !ok {
		t.Fatalf("metadata = %#v, want an empty map default", out["metadata"])
	}
}

func TestStripeToPaymentToleratesMissingNesting(t *testing.T) {
	// A payload with no data.object must not panic; every field is simply nil.
	// This is what makes the demo safe against a malformed provider payload.
	out := stripeToPayment(map[string]any{}).(map[string]any)
	if out["id"] != nil {
		t.Fatalf("id = %v, want nil for an empty payload", out["id"])
	}
}

// ---------------------------------------------------------------------------
// handle_payment
// ---------------------------------------------------------------------------

func TestHandlePaymentRecordAndMemoryKey(t *testing.T) {
	app := newDemoAgent(t)
	r := eventReasonerByName(t, "handle_payment")

	result, err := triggers.SimulateEvent(t, triggers.HandlerFunc(r.Handler(app)), triggers.SimulateEventOpts{
		Source:    "stripe",
		EventType: "payment_intent.succeeded",
		Body:      triggers.LoadFixture(t, "stripe"),
		Bindings:  bindingFor(r),
		EventID:   "evt_fixed",
	})
	if err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	record := result.(map[string]any)
	if record["kind"] != "payment" {
		t.Fatalf("kind = %v, want payment", record["kind"])
	}
	// The binding's Transform must have run: the handler saw data.object.
	if record["stripe_id"] != "pi_3NYExample1234" {
		t.Fatalf("stripe_id = %v, want pi_3NYExample1234 (transform should have run)", record["stripe_id"])
	}
	if record["amount_cents"] != float64(5000) {
		t.Fatalf("amount_cents = %v, want 5000", record["amount_cents"])
	}
	if record["received_via"] != "stripe" {
		t.Fatalf("received_via = %v, want stripe", record["received_via"])
	}
	if record["trigger_event_id"] != "evt_fixed" {
		t.Fatalf("trigger_event_id = %v, want evt_fixed", record["trigger_event_id"])
	}

	// The UI reads this key, so pin it.
	stored := readMemory(t, app, "payment:pi_3NYExample1234")
	if stored == nil {
		t.Fatal("expected a write to payment:pi_3NYExample1234")
	}
	if stored.(map[string]any)["kind"] != "payment" {
		t.Fatalf("stored record = %#v", stored)
	}
}

func TestHandlePaymentDirectCallRecordsDirectCall(t *testing.T) {
	app := newDemoAgent(t)
	r := eventReasonerByName(t, "handle_payment")

	// A direct invocation: no envelope, so no trigger context and no Transform.
	// The record must fall back to "direct_call" rather than panicking on nil.
	result, err := r.Handler(app)(context.Background(), map[string]any{
		"id":     "pi_direct",
		"amount": float64(4200),
	})
	if err != nil {
		t.Fatalf("handler returned error: %v", err)
	}

	record := result.(map[string]any)
	if record["received_via"] != "direct_call" {
		t.Fatalf("received_via = %v, want direct_call", record["received_via"])
	}
	if record["trigger_event_id"] != nil {
		t.Fatalf("trigger_event_id = %v, want nil on a direct call", record["trigger_event_id"])
	}
	if record["stripe_id"] != "pi_direct" {
		t.Fatalf("stripe_id = %v, want pi_direct (transform must be skipped)", record["stripe_id"])
	}
}

// ---------------------------------------------------------------------------
// handle_pr
// ---------------------------------------------------------------------------

func TestHandlePullRequestRecordAndMemoryKey(t *testing.T) {
	app := newDemoAgent(t)
	r := eventReasonerByName(t, "handle_pr")

	result, err := triggers.SimulateEvent(t, triggers.HandlerFunc(r.Handler(app)), triggers.SimulateEventOpts{
		Source:         "github",
		EventType:      "pull_request.opened",
		Body:           triggers.LoadFixture(t, "github"),
		Bindings:       bindingFor(r),
		IdempotencyKey: "delivery_fixed",
	})
	if err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	record := result.(map[string]any)
	if record["kind"] != "pull_request" {
		t.Fatalf("kind = %v, want pull_request", record["kind"])
	}
	if record["action"] != "opened" {
		t.Fatalf("action = %v, want opened", record["action"])
	}
	if record["number"] != float64(42) {
		t.Fatalf("number = %v, want 42", record["number"])
	}
	if record["repo"] != "Agent-Field/agentfield" {
		t.Fatalf("repo = %v, want Agent-Field/agentfield", record["repo"])
	}
	if record["user"] != "demo-user" {
		t.Fatalf("user = %v, want demo-user", record["user"])
	}
	if record["received_via"] != "github" {
		t.Fatalf("received_via = %v, want github", record["received_via"])
	}
	if record["delivery_id"] != "delivery_fixed" {
		t.Fatalf("delivery_id = %v, want delivery_fixed", record["delivery_id"])
	}

	if stored := readMemory(t, app, "pr:Agent-Field/agentfield#42"); stored == nil {
		t.Fatal("expected a write to pr:Agent-Field/agentfield#42")
	}
}

func TestHandlePullRequestSkipsMemoryWriteWhenKeyIncomplete(t *testing.T) {
	app := newDemoAgent(t)
	r := eventReasonerByName(t, "handle_pr")

	// No repository and no number: the demo must return a record but write
	// nothing, rather than persisting a "pr:<nil>#<nil>" key.
	result, err := triggers.SimulateEvent(t, triggers.HandlerFunc(r.Handler(app)), triggers.SimulateEventOpts{
		Source:    "github",
		EventType: "pull_request.opened",
		Body:      map[string]any{"action": "opened"},
		Bindings:  bindingFor(r),
	})
	if err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if result.(map[string]any)["kind"] != "pull_request" {
		t.Fatal("expected a record to still be returned")
	}
	if stored := readMemory(t, app, "pr:<nil>#<nil>"); stored != nil {
		t.Fatalf("expected no memory write for an incomplete key, got %#v", stored)
	}
}

func TestHandlePullRequestFallsBackToNestedNumber(t *testing.T) {
	app := newDemoAgent(t)
	r := eventReasonerByName(t, "handle_pr")

	// Some GitHub event types omit the top-level "number" and only carry it
	// on the nested pull_request object.
	result, err := triggers.SimulateEvent(t, triggers.HandlerFunc(r.Handler(app)), triggers.SimulateEventOpts{
		Source:    "github",
		EventType: "pull_request.synchronize",
		Body: map[string]any{
			"action":       "synchronize",
			"pull_request": map[string]any{"number": float64(7), "title": "Nested only"},
			"repository":   map[string]any{"full_name": "demo/repo"},
		},
		Bindings: bindingFor(r),
	})
	if err != nil {
		t.Fatalf("SimulateEvent returned error: %v", err)
	}

	if got := result.(map[string]any)["number"]; got != float64(7) {
		t.Fatalf("number = %v, want 7 from the nested object", got)
	}
	if stored := readMemory(t, app, "pr:demo/repo#7"); stored == nil {
		t.Fatal("expected a write to pr:demo/repo#7")
	}
}

// ---------------------------------------------------------------------------
// handle_tick
// ---------------------------------------------------------------------------

func TestHandleTickIncrementsCounter(t *testing.T) {
	app := newDemoAgent(t)
	r := tickReasoner()
	handler := triggers.HandlerFunc(r.Handler(app))
	fixed := time.Date(2026, 4, 28, 9, 0, 0, 0, time.UTC)

	for want := 1; want <= 3; want++ {
		result, err := triggers.SimulateSchedule(t, handler, triggers.SimulateScheduleOpts{
			Cron:       r.Cron,
			ReceivedAt: fixed,
		})
		if err != nil {
			t.Fatalf("tick %d returned error: %v", want, err)
		}

		record := result.(map[string]any)
		if record["count"] != want {
			t.Fatalf("count = %v, want %d on tick %d", record["count"], want, want)
		}
		if record["received_via"] != "cron" {
			t.Fatalf("received_via = %v, want cron", record["received_via"])
		}
		if record["last_fired_at"] != fixed.Format(time.RFC3339) {
			t.Fatalf("last_fired_at = %v, want %v", record["last_fired_at"], fixed.Format(time.RFC3339))
		}
	}

	stored := readMemory(t, app, tickCounterKey).(map[string]any)
	if stored["count"] != 3 {
		t.Fatalf("persisted count = %v, want 3", stored["count"])
	}
}

func TestHandleTickDirectCallRecordsDirectCall(t *testing.T) {
	app := newDemoAgent(t)

	result, err := tickReasoner().Handler(app)(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("handler returned error: %v", err)
	}

	record := result.(map[string]any)
	if record["received_via"] != "direct_call" {
		t.Fatalf("received_via = %v, want direct_call", record["received_via"])
	}
	if record["last_fired_at"] != nil {
		t.Fatalf("last_fired_at = %v, want nil on a direct call", record["last_fired_at"])
	}
	if record["count"] != 1 {
		t.Fatalf("count = %v, want 1", record["count"])
	}
}

// TestPreviousCountHandlesJSONWidening is the regression guard for the counter
// resetting to zero after a control-plane round trip: an int written today
// comes back as a float64, and reading only `int` would silently restart the
// count on every agent restart.
func TestPreviousCountHandlesJSONWidening(t *testing.T) {
	cases := []struct {
		name   string
		stored any
		want   int
	}{
		{"missing key", nil, 0},
		{"int", map[string]any{"count": 5}, 5},
		{"int64", map[string]any{"count": int64(6)}, 6},
		{"float64 from JSON", map[string]any{"count": float64(7)}, 7},
		{"absent count field", map[string]any{}, 0},
		{"non-numeric count", map[string]any{"count": "eight"}, 0},
		{"not a record", "garbage", 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := previousCount(tc.stored); got != tc.want {
				t.Fatalf("previousCount(%#v) = %d, want %d", tc.stored, got, tc.want)
			}
		})
	}
}

// TestTickCounterSurvivesJSONRoundTrip proves the widening guard end to end:
// persist a record, marshal and unmarshal it the way the control-plane memory
// backend does, and confirm the next tick continues rather than resetting.
func TestTickCounterSurvivesJSONRoundTrip(t *testing.T) {
	app := newDemoAgent(t)
	handler := tickReasoner().Handler(app)

	if _, err := handler(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("first tick: %v", err)
	}

	// Simulate the round trip: count is an int in memory, a float64 after JSON.
	stored := readMemory(t, app, tickCounterKey)
	raw, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("marshalling stored record: %v", err)
	}
	var roundTripped map[string]any
	if err := json.Unmarshal(raw, &roundTripped); err != nil {
		t.Fatalf("unmarshalling stored record: %v", err)
	}
	if _, isFloat := roundTripped["count"].(float64); !isFloat {
		t.Fatalf("expected count to widen to float64 after JSON, got %T", roundTripped["count"])
	}
	if err := app.Memory().Set(context.Background(), tickCounterKey, roundTripped); err != nil {
		t.Fatalf("re-storing round-tripped record: %v", err)
	}

	result, err := handler(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("second tick: %v", err)
	}
	if got := result.(map[string]any)["count"]; got != 2 {
		t.Fatalf("count = %v after a JSON round trip, want 2 (counter reset)", got)
	}
}

// ---------------------------------------------------------------------------
// Registration shape
// ---------------------------------------------------------------------------

// TestRegisteredTriggerShape pins what the control plane receives, since the
// UI's trigger sheet and signature verification depend on these exact values.
// Asserted on the declared options, which are what OnEvent forwards.
func TestRegisteredTriggerShape(t *testing.T) {
	cases := []struct {
		reasoner  string
		source    string
		secretEnv string
		eventType string
	}{
		{"handle_payment", "stripe", "STRIPE_DEMO_SECRET", "payment_intent.succeeded"},
		{"handle_pr", "github", "GITHUB_DEMO_SECRET", "pull_request"},
	}

	for _, tc := range cases {
		t.Run(tc.reasoner, func(t *testing.T) {
			r := eventReasonerByName(t, tc.reasoner)
			if r.Opts.Source != tc.source {
				t.Fatalf("source = %q, want %q", r.Opts.Source, tc.source)
			}
			if r.Opts.SecretEnv != tc.secretEnv {
				t.Fatalf("secret_env = %q, want %q", r.Opts.SecretEnv, tc.secretEnv)
			}
			if len(r.Opts.Types) != 1 || r.Opts.Types[0] != tc.eventType {
				t.Fatalf("types = %v, want [%s]", r.Opts.Types, tc.eventType)
			}
		})
	}
}

// TestScheduleTriggerConfigCarriesExpression guards the schedule wiring: the
// CP needs the expression in config to run the loop source at all.
func TestScheduleTriggerConfigCarriesExpression(t *testing.T) {
	binding := triggers.Schedule(triggers.ScheduleOpts{Cron: tickReasoner().Cron})

	if binding.Source != "cron" {
		t.Fatalf("source = %q, want cron", binding.Source)
	}

	var cfg map[string]any
	if err := json.Unmarshal(binding.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["expression"] != "* * * * *" {
		t.Fatalf("expression = %v, want * * * * *", cfg["expression"])
	}
	if cfg["timezone"] != "UTC" {
		t.Fatalf("timezone = %v, want UTC", cfg["timezone"])
	}
}

// TestAllReasonersRegisterOnAnAgent is the smoke test that the declarations
// main() iterates actually register without panicking, and that the wire
// payload carries a code_origin for the UI's drift card.
func TestAllReasonersRegisterOnAnAgent(t *testing.T) {
	app := newDemoAgent(t)

	for _, r := range eventReasoners() {
		app.OnEvent(r.Opts, r.Name, r.Handler(app))
	}
	for _, r := range scheduleReasoners() {
		app.OnSchedule(r.Cron, r.Name, r.Handler(app))
	}

	// Three reasoners, each reachable through the agent's own dispatch path.
	for _, name := range []string{"handle_payment", "handle_pr", "handle_tick"} {
		if _, err := app.Execute(context.Background(), name, map[string]any{}); err != nil {
			t.Fatalf("Execute(%q) returned error: %v", name, err)
		}
	}
}
