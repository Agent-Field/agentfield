package agent

import (
	"context"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/triggers"
)

// TestDispatch_DirectCallLeavesTriggerContextNil verifies the direct-call
// contract: a flat input reaches the handler unchanged and
// triggers.FromContext(ctx) is nil.
func TestDispatch_DirectCallLeavesTriggerContextNil(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-direct")

	var gotInput map[string]any
	var gotTrigger *triggers.Context

	a.OnEvent(triggers.EventOpts{
		Source: "stripe",
		Types:  []string{"payment_intent.succeeded"},
		Transform: func(evt map[string]any) any {
			// Must NOT run on a direct call.
			return map[string]any{"transformed": true}
		},
	}, "handle_payment", func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		gotTrigger = triggers.FromContext(ctx)
		return map[string]any{"ok": true}, nil
	})

	_, err := a.Execute(context.Background(), "handle_payment", map[string]any{
		"id":     "pi_direct",
		"amount": float64(4200),
	})
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	if gotTrigger != nil {
		t.Fatal("expected nil trigger context on a direct call")
	}
	if gotInput["id"] != "pi_direct" {
		t.Fatalf("input id = %v, want pi_direct (transform must not run)", gotInput["id"])
	}
	if _, transformed := gotInput["transformed"]; transformed {
		t.Fatal("transform must be skipped on direct calls")
	}
}

// TestDispatch_EnvelopeUnwrapsAndInjectsContext verifies the trigger contract:
// the envelope is peeled, the Transform runs, and the handler sees a populated
// *triggers.Context.
func TestDispatch_EnvelopeUnwrapsAndInjectsContext(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-envelope")

	var gotInput map[string]any
	var gotTrigger *triggers.Context

	a.OnEvent(triggers.EventOpts{
		Source: "stripe",
		Types:  []string{"payment_intent.succeeded"},
		Transform: func(evt map[string]any) any {
			data, _ := evt["data"].(map[string]any)
			return data["object"]
		},
	}, "handle_payment", func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		gotTrigger = triggers.FromContext(ctx)
		return map[string]any{"saved": true}, nil
	})

	envelope := map[string]any{
		"event": map[string]any{
			"data": map[string]any{
				"object": map[string]any{"id": "pi_1", "amount": float64(5000)},
			},
		},
		"_meta": map[string]any{
			"trigger_id":      "tr_abc",
			"source":          "stripe",
			"event_type":      "payment_intent.succeeded",
			"event_id":        "evt_1",
			"idempotency_key": "idem_1",
			"received_at":     "2026-04-28T22:29:54Z",
		},
	}

	if _, err := a.Execute(context.Background(), "handle_payment", envelope); err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	if gotTrigger == nil {
		t.Fatal("expected non-nil trigger context for an envelope dispatch")
	}
	if gotTrigger.TriggerID != "tr_abc" {
		t.Fatalf("TriggerID = %q, want tr_abc", gotTrigger.TriggerID)
	}
	if gotTrigger.Source != "stripe" {
		t.Fatalf("Source = %q, want stripe", gotTrigger.Source)
	}
	if gotTrigger.EventID != "evt_1" {
		t.Fatalf("EventID = %q, want evt_1", gotTrigger.EventID)
	}
	// Transform must have run: the handler sees data.object, not the envelope.
	if gotInput["id"] != "pi_1" {
		t.Fatalf("input id = %v, want pi_1 (transform should have run)", gotInput["id"])
	}
	if gotInput["amount"] != float64(5000) {
		t.Fatalf("input amount = %v, want 5000", gotInput["amount"])
	}
}

// TestDispatch_EnvelopeWithoutTransformPassesRawEvent covers a binding that
// declares no Transform: the raw event object reaches the handler.
func TestDispatch_EnvelopeWithoutTransformPassesRawEvent(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-no-transform")

	var gotInput map[string]any
	var gotTrigger *triggers.Context

	a.OnEvent(triggers.EventOpts{
		Source: "github",
		Types:  []string{"pull_request"},
	}, "handle_pr", func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		gotTrigger = triggers.FromContext(ctx)
		return nil, nil
	})

	envelope := map[string]any{
		"event": map[string]any{
			"action": "opened",
			"number": float64(42),
		},
		"_meta": map[string]any{
			"trigger_id": "tr_gh",
			"source":     "github",
			"event_type": "pull_request.opened",
			"event_id":   "evt_gh_1",
		},
	}

	if _, err := a.Execute(context.Background(), "handle_pr", envelope); err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	if gotTrigger == nil {
		t.Fatal("expected non-nil trigger context")
	}
	if gotInput["action"] != "opened" {
		t.Fatalf("action = %v, want opened", gotInput["action"])
	}
	if gotInput["number"] != float64(42) {
		t.Fatalf("number = %v, want 42", gotInput["number"])
	}
}

// TestDispatch_CronEnvelope covers the schedule path, where the event body is
// the synthetic cron tick payload.
func TestDispatch_CronEnvelope(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-cron")

	var gotTrigger *triggers.Context

	a.OnSchedule("* * * * *", "handle_tick", func(ctx context.Context, input map[string]any) (any, error) {
		gotTrigger = triggers.FromContext(ctx)
		return nil, nil
	})

	envelope := map[string]any{
		"event": map[string]any{
			"cron":     "* * * * *",
			"fired_at": "2026-04-28T09:00:00Z",
		},
		"_meta": map[string]any{
			"trigger_id": "tr_cron",
			"source":     "cron",
			"event_type": "tick",
			"event_id":   "evt_cron_1",
		},
	}

	if _, err := a.Execute(context.Background(), "handle_tick", envelope); err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	if gotTrigger == nil {
		t.Fatal("expected non-nil trigger context for a cron dispatch")
	}
	if gotTrigger.Source != "cron" {
		t.Fatalf("Source = %q, want cron", gotTrigger.Source)
	}
	if gotTrigger.EventType != "tick" {
		t.Fatalf("EventType = %q, want tick", gotTrigger.EventType)
	}
}

// TestDispatch_TransformReturningNonObjectIsWrapped verifies that a Transform
// returning a scalar or slice is still delivered (wrapped under "input")
// rather than dropped, since HandlerFunc takes map[string]any.
func TestDispatch_TransformReturningNonObjectIsWrapped(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-scalar-transform")

	var gotInput map[string]any

	a.OnEvent(triggers.EventOpts{
		Source: "generic_hmac",
		Transform: func(evt map[string]any) any {
			return []any{"a", "b"}
		},
	}, "handle_list", func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		return nil, nil
	})

	envelope := map[string]any{
		"event": map[string]any{"items": []any{1, 2}},
		"_meta": map[string]any{
			"trigger_id": "tr_h",
			"source":     "generic_hmac",
		},
	}

	if _, err := a.Execute(context.Background(), "handle_list", envelope); err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	list, ok := gotInput["input"].([]any)
	if !ok {
		t.Fatalf("expected the slice to be wrapped under \"input\", got %#v", gotInput)
	}
	if len(list) != 2 {
		t.Fatalf("expected 2 items, got %d", len(list))
	}
}

// TestDispatch_ReasonerWithoutBindingsStillUnwraps proves the unwrap is not
// gated on having declared bindings: a reasoner registered without triggers
// that nonetheless receives an envelope still gets the peeled event and a
// populated context.
func TestDispatch_ReasonerWithoutBindingsStillUnwraps(t *testing.T) {
	a := newTriggerTestAgent(t, "test-dispatch-no-bindings")

	var gotInput map[string]any
	var gotTrigger *triggers.Context

	a.RegisterReasoner("handle_inbound", func(ctx context.Context, input map[string]any) (any, error) {
		gotInput = input
		gotTrigger = triggers.FromContext(ctx)
		return nil, nil
	})

	envelope := map[string]any{
		"event": map[string]any{"payload": "value"},
		"_meta": map[string]any{
			"trigger_id": "tr_ui",
			"source":     "slack",
			"event_type": "app_mention",
		},
	}

	if _, err := a.Execute(context.Background(), "handle_inbound", envelope); err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}

	if gotTrigger == nil {
		t.Fatal("expected non-nil trigger context even without declared bindings")
	}
	if gotInput["payload"] != "value" {
		t.Fatalf("payload = %v, want value", gotInput["payload"])
	}
}
