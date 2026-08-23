package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/triggers"
)

func newTriggerTestAgent(t *testing.T, nodeID string) *Agent {
	t.Helper()
	a, err := New(Config{
		NodeID:        nodeID,
		Version:       "1.0.0",
		AgentFieldURL: "http://localhost:8080",
		Logger:        log.New(io.Discard, "", 0),
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	return a
}

func TestOnEvent_RegistersReasonerWithTrigger(t *testing.T) {
	a := newTriggerTestAgent(t, "test-on-event")

	handler := func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	}

	a.OnEvent(triggers.EventOpts{
		Source:    "stripe",
		Types:     []string{"payment_intent.succeeded"},
		SecretEnv: "STRIPE_SECRET",
	}, "handle_payment", handler)

	r, ok := a.reasoners["handle_payment"]
	if !ok {
		t.Fatal("expected reasoner to be registered")
	}
	if len(r.Triggers) != 1 {
		t.Fatalf("expected 1 trigger, got %d", len(r.Triggers))
	}
	if r.Triggers[0].Source != "stripe" {
		t.Fatalf("expected source stripe, got %s", r.Triggers[0].Source)
	}
	if len(r.Triggers[0].EventTypes) != 1 || r.Triggers[0].EventTypes[0] != "payment_intent.succeeded" {
		t.Fatalf("expected event_types [payment_intent.succeeded], got %v", r.Triggers[0].EventTypes)
	}
	if r.Triggers[0].SecretEnvVar != "STRIPE_SECRET" {
		t.Fatalf("expected secret_env_var STRIPE_SECRET, got %s", r.Triggers[0].SecretEnvVar)
	}
	// Auto-set accepts_webhook
	if r.AcceptsWebhook == nil || *r.AcceptsWebhook != "true" {
		t.Fatal("expected accepts_webhook to be auto-set to 'true'")
	}
	// Code origin should be populated
	if r.Triggers[0].CodeOrigin == "" {
		t.Fatal("expected code_origin to be populated")
	}
}

func TestOnEvent_WithTransform(t *testing.T) {
	a := newTriggerTestAgent(t, "test-on-event-transform")

	transform := func(evt map[string]any) any {
		data, _ := evt["data"].(map[string]any)
		return data["object"]
	}

	a.OnEvent(triggers.EventOpts{
		Source:    "stripe",
		Types:     []string{"payment_intent.succeeded"},
		Transform: transform,
	}, "handle_payment", func(ctx context.Context, input map[string]any) (any, error) {
		return input, nil
	})

	r := a.reasoners["handle_payment"]
	if len(r.triggerBindings) != 1 {
		t.Fatalf("expected 1 trigger binding, got %d", len(r.triggerBindings))
	}
	if r.triggerBindings[0].TransformFn == nil {
		t.Fatal("expected transform to be stored on triggerBindings")
	}
}

func TestOnSchedule_RegistersReasonerWithCronTrigger(t *testing.T) {
	a := newTriggerTestAgent(t, "test-on-schedule")

	a.OnSchedule("* * * * *", "handle_tick", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"tick": true}, nil
	})

	r, ok := a.reasoners["handle_tick"]
	if !ok {
		t.Fatal("expected reasoner to be registered")
	}
	if len(r.Triggers) != 1 {
		t.Fatalf("expected 1 trigger, got %d", len(r.Triggers))
	}
	if r.Triggers[0].Source != "cron" {
		t.Fatalf("expected source cron, got %s", r.Triggers[0].Source)
	}

	var cfg map[string]any
	if err := json.Unmarshal(r.Triggers[0].Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["expression"] != "* * * * *" {
		t.Fatalf("expected expression '* * * * *', got %v", cfg["expression"])
	}
	if cfg["timezone"] != "UTC" {
		t.Fatalf("expected default timezone UTC, got %v", cfg["timezone"])
	}
	// Auto-set accepts_webhook
	if r.AcceptsWebhook == nil || *r.AcceptsWebhook != "true" {
		t.Fatal("expected accepts_webhook to be auto-set to 'true'")
	}
}

func TestOnSchedule_WithTimezone(t *testing.T) {
	a := newTriggerTestAgent(t, "test-on-schedule-tz")

	a.OnSchedule("0 9 * * 1-5", "daily_standup", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	}, WithTimezone("America/New_York"))

	r := a.reasoners["daily_standup"]
	var cfg map[string]any
	if err := json.Unmarshal(r.Triggers[0].Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["timezone"] != "America/New_York" {
		t.Fatalf("expected timezone America/New_York, got %v", cfg["timezone"])
	}
}

func TestOnEvent_ProducesSamePayloadAsWithTriggers(t *testing.T) {
	a1 := newTriggerTestAgent(t, "test-parity-1")
	a1.OnEvent(triggers.EventOpts{
		Source:    "github",
		Types:     []string{"pull_request"},
		SecretEnv: "GH_SECRET",
	}, "handle_pr", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	})

	a2 := newTriggerTestAgent(t, "test-parity-2")
	a2.RegisterReasoner("handle_pr", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	}, WithTriggers(EventTrigger{
		Source:    "github",
		Types:     []string{"pull_request"},
		SecretEnv: "GH_SECRET",
	}))

	r1 := a1.reasoners["handle_pr"]
	r2 := a2.reasoners["handle_pr"]

	if r1.Triggers[0].Source != r2.Triggers[0].Source {
		t.Fatalf("source mismatch: %s != %s", r1.Triggers[0].Source, r2.Triggers[0].Source)
	}
	if len(r1.Triggers[0].EventTypes) != len(r2.Triggers[0].EventTypes) {
		t.Fatal("event_types length mismatch")
	}
	if r1.Triggers[0].EventTypes[0] != r2.Triggers[0].EventTypes[0] {
		t.Fatal("event_types[0] mismatch")
	}
	if r1.Triggers[0].SecretEnvVar != r2.Triggers[0].SecretEnvVar {
		t.Fatal("secret_env_var mismatch")
	}
	if *r1.AcceptsWebhook != *r2.AcceptsWebhook {
		t.Fatal("accepts_webhook mismatch")
	}
}

func TestWithTriggersBinding_AcceptsTriggersPackageBinding(t *testing.T) {
	a := newTriggerTestAgent(t, "test-binding-compat")

	a.RegisterReasoner("handle_event", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	}, WithTriggers(triggers.Event(triggers.EventOpts{
		Source: "slack",
		Types:  []string{"app_mention"},
	})))

	r := a.reasoners["handle_event"]
	if len(r.Triggers) != 1 {
		t.Fatalf("expected 1 trigger, got %d", len(r.Triggers))
	}
	if r.Triggers[0].Source != "slack" {
		t.Fatalf("expected source slack, got %s", r.Triggers[0].Source)
	}
}

func TestWithTriggers_PreservesTransformFromTriggersBinding(t *testing.T) {
	a := newTriggerTestAgent(t, "test-transform-via-withtriggers")

	transform := func(evt map[string]any) any {
		return evt["data"]
	}

	a.RegisterReasoner("handle_event", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	}, WithTriggers(triggers.Event(triggers.EventOpts{
		Source:    "stripe",
		Types:     []string{"payment_intent.succeeded"},
		Transform: transform,
	})))

	r := a.reasoners["handle_event"]
	if len(r.triggerBindings) != 1 {
		t.Fatalf("expected 1 trigger binding with transform, got %d", len(r.triggerBindings))
	}
	if r.triggerBindings[0].TransformFn == nil {
		t.Fatal("expected Transform to be preserved when using WithTriggers(triggers.Event(...))")
	}
	// Wire binding should also be populated
	if len(r.Triggers) != 1 {
		t.Fatalf("expected 1 wire trigger, got %d", len(r.Triggers))
	}
	if r.Triggers[0].Source != "stripe" {
		t.Fatalf("expected source stripe, got %s", r.Triggers[0].Source)
	}
}
