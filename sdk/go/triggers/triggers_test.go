package triggers

import (
	"encoding/json"
	"testing"
	"time"
)

func TestEventCreatesBinding(t *testing.T) {
	b := Event(EventOpts{
		Source:    "stripe",
		Types:     []string{"payment_intent.succeeded", "charge.failed"},
		SecretEnv: "STRIPE_WEBHOOK_SECRET",
	})

	if b.Kind != EventBinding {
		t.Fatalf("expected EventBinding, got %v", b.Kind)
	}
	if b.Source != "stripe" {
		t.Fatalf("expected source stripe, got %s", b.Source)
	}
	if len(b.EventTypes) != 2 {
		t.Fatalf("expected 2 event types, got %d", len(b.EventTypes))
	}
	if b.SecretEnv != "STRIPE_WEBHOOK_SECRET" {
		t.Fatalf("expected secret env STRIPE_WEBHOOK_SECRET, got %s", b.SecretEnv)
	}
	if b.TransformFn != nil {
		t.Fatal("expected nil transform")
	}
}

func TestEventWithTransform(t *testing.T) {
	transform := func(evt map[string]any) any {
		data, _ := evt["data"].(map[string]any)
		return data["object"]
	}

	b := Event(EventOpts{
		Source:    "stripe",
		Types:     []string{"payment_intent.succeeded"},
		Transform: transform,
	})

	if b.TransformFn == nil {
		t.Fatal("expected non-nil transform")
	}

	// Verify the transform works
	input := map[string]any{
		"data": map[string]any{
			"object": map[string]any{"id": "pi_123", "amount": 5000},
		},
	}
	result := b.TransformFn(input)
	obj, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T", result)
	}
	if obj["id"] != "pi_123" {
		t.Fatalf("expected id pi_123, got %v", obj["id"])
	}
}

func TestEventWithConfig(t *testing.T) {
	cfg, _ := json.Marshal(map[string]any{"tolerance": 300})
	b := Event(EventOpts{
		Source: "generic_hmac",
		Config: cfg,
	})

	if b.Config == nil {
		t.Fatal("expected non-nil config")
	}
	var parsed map[string]any
	if err := json.Unmarshal(b.Config, &parsed); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if parsed["tolerance"] != float64(300) {
		t.Fatalf("expected tolerance 300, got %v", parsed["tolerance"])
	}
}

func TestEventMinimalSpec(t *testing.T) {
	b := Event(EventOpts{Source: "github"})

	if b.Source != "github" {
		t.Fatalf("expected source github, got %s", b.Source)
	}
	if len(b.EventTypes) != 0 {
		t.Fatalf("expected no event types, got %d", len(b.EventTypes))
	}
	if b.SecretEnv != "" {
		t.Fatalf("expected empty secret env, got %s", b.SecretEnv)
	}
}

func TestScheduleCreatesBinding(t *testing.T) {
	b := Schedule(ScheduleOpts{
		Cron:     "0 9 * * 1-5",
		Timezone: "America/New_York",
	})

	if b.Kind != ScheduleBinding {
		t.Fatalf("expected ScheduleBinding, got %v", b.Kind)
	}
	if b.Source != "cron" {
		t.Fatalf("expected source cron, got %s", b.Source)
	}

	var cfg map[string]any
	if err := json.Unmarshal(b.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["expression"] != "0 9 * * 1-5" {
		t.Fatalf("expected expression '0 9 * * 1-5', got %v", cfg["expression"])
	}
	if cfg["timezone"] != "America/New_York" {
		t.Fatalf("expected timezone America/New_York, got %v", cfg["timezone"])
	}
}

func TestScheduleDefaultsToUTC(t *testing.T) {
	b := Schedule(ScheduleOpts{Cron: "* * * * *"})

	var cfg map[string]any
	if err := json.Unmarshal(b.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["timezone"] != "UTC" {
		t.Fatalf("expected timezone UTC, got %v", cfg["timezone"])
	}
}

func TestScheduleWithCustomConfig(t *testing.T) {
	customCfg, _ := json.Marshal(map[string]any{
		"custom": true,
	})
	b := Schedule(ScheduleOpts{
		Cron:     "*/5 * * * *",
		Timezone: "Europe/London",
		Config:   customCfg,
	})

	// Custom config fields are merged with expression/timezone
	var cfg map[string]any
	if err := json.Unmarshal(b.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["custom"] != true {
		t.Fatalf("expected custom config to be merged, got %v", cfg)
	}
	if cfg["expression"] != "*/5 * * * *" {
		t.Fatalf("expected expression to be preserved, got %v", cfg["expression"])
	}
	if cfg["timezone"] != "Europe/London" {
		t.Fatalf("expected timezone to be preserved, got %v", cfg["timezone"])
	}
}

func TestScheduleCronOverridesCustomExpression(t *testing.T) {
	customCfg, _ := json.Marshal(map[string]any{
		"expression": "0 0 * * 0",
		"custom":     true,
	})
	b := Schedule(ScheduleOpts{
		Cron:   "*/5 * * * *",
		Config: customCfg,
	})

	var cfg map[string]any
	if err := json.Unmarshal(b.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["expression"] != "*/5 * * * *" {
		t.Fatalf("expected Cron to win over custom expression, got %v", cfg["expression"])
	}
	if cfg["custom"] != true {
		t.Fatalf("expected custom config to be merged, got %v", cfg)
	}
}

func TestScheduleIgnoresUnusableConfig(t *testing.T) {
	cases := map[string]json.RawMessage{
		"malformed": json.RawMessage("{not json"),
		"array":     json.RawMessage(`["a","b"]`),
		"scalar":    json.RawMessage(`42`),
	}

	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			b := Schedule(ScheduleOpts{
				Cron:     "0 9 * * 1-5",
				Timezone: "America/New_York",
				Config:   raw,
			})

			var cfg map[string]any
			if err := json.Unmarshal(b.Config, &cfg); err != nil {
				t.Fatalf("config is not valid JSON: %v", err)
			}
			if cfg["expression"] != "0 9 * * 1-5" {
				t.Fatalf("expected expression '0 9 * * 1-5', got %v", cfg["expression"])
			}
			if cfg["timezone"] != "America/New_York" {
				t.Fatalf("expected timezone America/New_York, got %v", cfg["timezone"])
			}
		})
	}
}

func TestSchedulePreservesCustomTimezoneWhenUnset(t *testing.T) {
	customCfg, _ := json.Marshal(map[string]any{
		"timezone": "Asia/Tokyo",
	})
	b := Schedule(ScheduleOpts{
		Cron:   "* * * * *",
		Config: customCfg,
	})

	var cfg map[string]any
	if err := json.Unmarshal(b.Config, &cfg); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if cfg["timezone"] != "Asia/Tokyo" {
		t.Fatalf("expected custom timezone Asia/Tokyo to be preserved, got %v", cfg["timezone"])
	}
}

func TestContextFields(t *testing.T) {
	now := time.Now()
	ctx := &Context{
		TriggerID:      "tr_abc",
		Source:         "stripe",
		EventType:      "payment_intent.succeeded",
		EventID:        "evt_123",
		IdempotencyKey: "idem_456",
		ReceivedAt:     now,
		VCID:           "vc_789",
	}

	if ctx.TriggerID != "tr_abc" {
		t.Fatalf("expected trigger_id tr_abc, got %s", ctx.TriggerID)
	}
	if ctx.Source != "stripe" {
		t.Fatalf("expected source stripe, got %s", ctx.Source)
	}
	if ctx.EventType != "payment_intent.succeeded" {
		t.Fatalf("expected event_type payment_intent.succeeded, got %s", ctx.EventType)
	}
	if ctx.ReceivedAt != now {
		t.Fatal("expected matching ReceivedAt")
	}
	if ctx.VCID != "vc_789" {
		t.Fatalf("expected vc_id vc_789, got %s", ctx.VCID)
	}
}

func TestContextNilVCID(t *testing.T) {
	ctx := &Context{
		TriggerID: "tr_1",
		Source:    "cron",
	}
	if ctx.VCID != "" {
		t.Fatalf("expected empty VCID, got %s", ctx.VCID)
	}
}

func TestBindingKindConstants(t *testing.T) {
	if EventBinding == ScheduleBinding {
		t.Fatal("EventBinding and ScheduleBinding should be distinct")
	}
}
