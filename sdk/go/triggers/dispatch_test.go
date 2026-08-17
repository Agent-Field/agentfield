package triggers

import (
	"context"
	"testing"
	"time"
)

func TestIsEnvelope(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
		want bool
	}{
		{
			name: "valid envelope",
			body: map[string]any{
				"event": map[string]any{"id": "pi_x"},
				"_meta": map[string]any{"trigger_id": "tr_abc"},
			},
			want: true,
		},
		{
			name: "flat input is not an envelope",
			body: map[string]any{"id": "pi_x", "amount": 4200},
			want: false,
		},
		{
			name: "nil body",
			body: nil,
			want: false,
		},
		{
			name: "missing _meta",
			body: map[string]any{"event": map[string]any{"id": "x"}},
			want: false,
		},
		{
			name: "missing event",
			body: map[string]any{"_meta": map[string]any{"trigger_id": "tr"}},
			want: false,
		},
		{
			name: "_meta without trigger_id",
			body: map[string]any{
				"event": map[string]any{"id": "x"},
				"_meta": map[string]any{"source": "stripe"},
			},
			want: false,
		},
		{
			name: "_meta is not an object",
			body: map[string]any{
				"event": map[string]any{"id": "x"},
				"_meta": "not-an-object",
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsEnvelope(tc.body); got != tc.want {
				t.Fatalf("IsEnvelope() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestUnwrapEnvelope(t *testing.T) {
	body := map[string]any{
		"event": map[string]any{"id": "pi_x", "amount": float64(4200)},
		"_meta": map[string]any{
			"trigger_id":      "tr_abc",
			"source":          "stripe",
			"event_type":      "payment_intent.succeeded",
			"event_id":        "evt_x",
			"idempotency_key": "idem_1",
			"received_at":     "2026-04-28T22:29:54Z",
			"vc_id":           "vc_123",
		},
	}

	input, tc := Unwrap(body)

	if tc == nil {
		t.Fatal("expected non-nil trigger context")
	}
	if tc.TriggerID != "tr_abc" {
		t.Fatalf("TriggerID = %q, want tr_abc", tc.TriggerID)
	}
	if tc.Source != "stripe" {
		t.Fatalf("Source = %q, want stripe", tc.Source)
	}
	if tc.EventType != "payment_intent.succeeded" {
		t.Fatalf("EventType = %q, want payment_intent.succeeded", tc.EventType)
	}
	if tc.EventID != "evt_x" {
		t.Fatalf("EventID = %q, want evt_x", tc.EventID)
	}
	if tc.IdempotencyKey != "idem_1" {
		t.Fatalf("IdempotencyKey = %q, want idem_1", tc.IdempotencyKey)
	}
	if tc.VCID != "vc_123" {
		t.Fatalf("VCID = %q, want vc_123", tc.VCID)
	}
	expected := time.Date(2026, 4, 28, 22, 29, 54, 0, time.UTC)
	if !tc.ReceivedAt.Equal(expected) {
		t.Fatalf("ReceivedAt = %v, want %v", tc.ReceivedAt, expected)
	}
	if input["id"] != "pi_x" {
		t.Fatalf("unwrapped input id = %v, want pi_x", input["id"])
	}
}

func TestUnwrapDirectCallPassesThrough(t *testing.T) {
	body := map[string]any{"id": "pi_x", "amount": float64(4200)}

	input, tc := Unwrap(body)

	if tc != nil {
		t.Fatal("expected nil trigger context for a direct call")
	}
	if input["id"] != "pi_x" {
		t.Fatalf("input id = %v, want pi_x", input["id"])
	}
}

func TestUnwrapMissingReceivedAtFallsBackToNow(t *testing.T) {
	body := map[string]any{
		"event": map[string]any{},
		"_meta": map[string]any{"trigger_id": "tr_1", "source": "cron"},
	}

	before := time.Now().UTC().Add(-time.Second)
	_, tc := Unwrap(body)
	after := time.Now().UTC().Add(time.Second)

	if tc.ReceivedAt.Before(before) || tc.ReceivedAt.After(after) {
		t.Fatalf("ReceivedAt %v not within [%v, %v]", tc.ReceivedAt, before, after)
	}
}

func TestUnwrapUnparseableReceivedAtFallsBackToNow(t *testing.T) {
	body := map[string]any{
		"event": map[string]any{},
		"_meta": map[string]any{
			"trigger_id":  "tr_1",
			"received_at": "not-a-timestamp",
		},
	}

	_, tc := Unwrap(body)
	if tc.ReceivedAt.IsZero() {
		t.Fatal("expected ReceivedAt to fall back to now, got zero time")
	}
}

func TestUnwrapNonObjectEventIsPreserved(t *testing.T) {
	body := map[string]any{
		"event": []any{"a", "b"},
		"_meta": map[string]any{"trigger_id": "tr_1"},
	}

	input, tc := Unwrap(body)
	if tc == nil {
		t.Fatal("expected non-nil trigger context")
	}
	inner, ok := input["event"].([]any)
	if !ok {
		t.Fatalf("expected the list payload to be preserved, got %T", input["event"])
	}
	if len(inner) != 2 {
		t.Fatalf("expected 2 items, got %d", len(inner))
	}
}

func TestContextRoundTripThroughContext(t *testing.T) {
	tc := &Context{TriggerID: "tr_1", Source: "stripe"}

	ctx := NewContext(context.Background(), tc)
	got := FromContext(ctx)

	if got == nil {
		t.Fatal("expected non-nil context from FromContext")
	}
	if got.TriggerID != "tr_1" {
		t.Fatalf("TriggerID = %q, want tr_1", got.TriggerID)
	}
}

func TestFromContextNilCases(t *testing.T) {
	if FromContext(context.Background()) != nil {
		t.Fatal("expected nil for a plain context")
	}
	//nolint:staticcheck // deliberately passing nil to prove it is handled
	if FromContext(nil) != nil {
		t.Fatal("expected nil for a nil context")
	}
	if NewContext(context.Background(), nil) == nil {
		t.Fatal("NewContext with nil tc should return the parent, not nil")
	}
	if FromContext(NewContext(context.Background(), nil)) != nil {
		t.Fatal("expected nil after NewContext with nil tc")
	}
}

func TestApplyTransformRunsMatchingBinding(t *testing.T) {
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
	tc := &Context{Source: "stripe", EventType: "payment_intent.succeeded"}
	input := map[string]any{
		"data": map[string]any{"object": map[string]any{"id": "pi_1"}},
	}

	out := ApplyTransform(tc, bindings, input)
	obj, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %T", out)
	}
	if obj["id"] != "pi_1" {
		t.Fatalf("id = %v, want pi_1", obj["id"])
	}
}

func TestApplyTransformNoMatchReturnsRawInput(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "github",
			Types:     []string{"push"},
			Transform: func(map[string]any) any { return "transformed" },
		}),
	}
	tc := &Context{Source: "stripe", EventType: "payment_intent.succeeded"}
	input := map[string]any{"id": "pi_1"}

	out := ApplyTransform(tc, bindings, input)
	obj, ok := out.(map[string]any)
	if !ok || obj["id"] != "pi_1" {
		t.Fatalf("expected raw input passthrough, got %v", out)
	}
}

func TestApplyTransformPrefersSpecificOverCatchAll(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "stripe",
			Transform: func(map[string]any) any { return "catch-all" },
		}),
		Event(EventOpts{
			Source:    "stripe",
			Types:     []string{"payment_intent.succeeded"},
			Transform: func(map[string]any) any { return "specific" },
		}),
	}
	tc := &Context{Source: "stripe", EventType: "payment_intent.succeeded"}

	if out := ApplyTransform(tc, bindings, map[string]any{}); out != "specific" {
		t.Fatalf("expected specific binding to win, got %v", out)
	}
}

func TestApplyTransformFallsBackToCatchAll(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "stripe",
			Transform: func(map[string]any) any { return "catch-all" },
		}),
		Event(EventOpts{
			Source:    "stripe",
			Types:     []string{"payment_intent.succeeded"},
			Transform: func(map[string]any) any { return "specific" },
		}),
	}
	tc := &Context{Source: "stripe", EventType: "charge.failed"}

	if out := ApplyTransform(tc, bindings, map[string]any{}); out != "catch-all" {
		t.Fatalf("expected catch-all binding, got %v", out)
	}
}

func TestApplyTransformPrefixMatching(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "github",
			Types:     []string{"pull_request"},
			Transform: func(map[string]any) any { return "matched" },
		}),
	}
	tc := &Context{Source: "github", EventType: "pull_request.opened"}

	if out := ApplyTransform(tc, bindings, map[string]any{}); out != "matched" {
		t.Fatalf("expected prefix match to apply, got %v", out)
	}
}

func TestApplyTransformIgnoresScheduleBindings(t *testing.T) {
	bindings := []Binding{
		Schedule(ScheduleOpts{Cron: "* * * * *"}),
		Event(EventOpts{
			Source:    "stripe",
			Transform: func(map[string]any) any { return "event" },
		}),
	}
	tc := &Context{Source: "stripe"}

	if out := ApplyTransform(tc, bindings, map[string]any{}); out != "event" {
		t.Fatalf("expected the event binding to be used, got %v", out)
	}
}

func TestApplyTransformPanicRecoversToRawInput(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{
			Source:    "stripe",
			Transform: func(map[string]any) any { panic("boom") },
		}),
	}
	tc := &Context{Source: "stripe"}
	input := map[string]any{"id": "pi_1"}

	out := ApplyTransform(tc, bindings, input)
	obj, ok := out.(map[string]any)
	if !ok || obj["id"] != "pi_1" {
		t.Fatalf("expected raw input after panic recovery, got %v", out)
	}
}

func TestApplyTransformNilContextOrEmptyBindings(t *testing.T) {
	input := map[string]any{"id": "x"}

	if out := ApplyTransform(nil, []Binding{}, input); out.(map[string]any)["id"] != "x" {
		t.Fatal("expected passthrough for nil context")
	}
	if out := ApplyTransform(&Context{Source: "stripe"}, nil, input); out.(map[string]any)["id"] != "x" {
		t.Fatal("expected passthrough for empty bindings")
	}
}

func TestApplyTransformMatchedBindingWithoutTransform(t *testing.T) {
	bindings := []Binding{
		Event(EventOpts{Source: "stripe", Types: []string{"charge.succeeded"}}),
	}
	tc := &Context{Source: "stripe", EventType: "charge.succeeded"}
	input := map[string]any{"id": "ch_1"}

	out := ApplyTransform(tc, bindings, input)
	if out.(map[string]any)["id"] != "ch_1" {
		t.Fatalf("expected raw input when the binding has no Transform, got %v", out)
	}
}
