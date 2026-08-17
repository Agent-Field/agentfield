// Package triggers provides types and factories for declaring inbound webhook
// and cron-schedule bindings on Go SDK reasoners.
//
// A reasoner declares external event sources via WithTriggers on
// RegisterReasoner. The canonical form passes typed Binding values
// created by Event() / Schedule() factories.
//
// The control plane registers a code-managed Trigger row per binding when
// the agent registers, so the agent never has to provision webhooks itself.
//
// Field-for-field equivalent of sdk/python/agentfield/triggers.py and
// sdk/typescript/src/triggers/types.ts.
package triggers

import (
	"encoding/json"
	"time"
)

// Context is the webhook-trigger metadata exposed to reasoners at runtime.
//
// Retrieve it inside a handler with FromContext(ctx). It is nil when the
// reasoner was invoked directly (app.Call, Execute with a flat input) rather
// than dispatched by an inbound event, so a nil check distinguishes the two:
//
//	if tc := triggers.FromContext(ctx); tc != nil {
//	    // dispatched via a trigger
//	}
//
// VCID may be empty until the DID/VC chain wiring lands (tracked separately
// under SDK Feature Parity).
type Context struct {
	// AgentField trigger row ID; stable, equals the public URL slug.
	TriggerID string
	// Provider source ("stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer").
	Source string
	// Provider's event type (or "" for cron tick).
	EventType string
	// AgentField inbound_event ID (replay key).
	EventID string
	// Provider's idempotency key (e.g. evt_xxx).
	IdempotencyKey string
	// When the control plane received the inbound event.
	ReceivedAt time.Time
	// Trigger event VC ID, if DID enabled.
	VCID string
}

// Transform is an optional sync function to convert a raw provider event
// into the reasoner's input. Must be synchronous.
//
// When a binding declaring a Transform matches the dispatched event, the SDK
// runs Transform(rawEvent) and the handler's input is the return value rather
// than the raw event. A Transform that panics degrades to pass-through: the
// handler receives the raw event instead of failing the dispatch.
//
// Transforms are only applied to trigger dispatches, never to direct calls.
type Transform func(rawEvent map[string]any) any

// EventOpts configures an event trigger binding.
type EventOpts struct {
	// Registered Source name (e.g. "stripe", "github", "slack",
	// "generic_hmac", "generic_bearer").
	Source string
	// Event types the reasoner cares about. Empty means "all".
	// Supports prefix-match: "pull_request" matches "pull_request.opened".
	Types []string
	// Name of the env var on the control plane that holds the provider's
	// webhook secret. Required for Sources whose secret_required is true.
	SecretEnv string
	// Source-specific JSON config (timestamp tolerance, custom header names, etc).
	Config json.RawMessage
	// Optional sync transform to convert raw provider event to reasoner input.
	// Runs before the handler on trigger dispatches, skipped on direct calls.
	// See Transform.
	Transform Transform
}

// ScheduleOpts configures a cron schedule trigger binding.
type ScheduleOpts struct {
	// Cron is the 5-field cron expression (minute hour dom month dow).
	Cron string
	// IANA timezone name. Defaults to "UTC".
	Timezone string
	// Optional source-specific config, merged into the binding config. Must
	// marshal to a JSON object; malformed or non-object config is ignored.
	// The "expression" and "timezone" keys are controlled by the Cron and
	// Timezone fields: a custom "expression" is always overridden by Cron,
	// and a custom "timezone" is kept only when Timezone is empty.
	Config json.RawMessage
}

// Binding is a typed trigger binding — either an event trigger or a schedule
// trigger. Created via Event() or Schedule() factory functions. Carries both
// the wire-serialisable payload and the non-serialisable Transform.
type Binding struct {
	// Source is the provider name (for quick filtering without inspecting Wire).
	Source string
	// EventTypes is the set of subscribed event types (empty = all).
	EventTypes []string
	// SecretEnv is the env var name for the provider secret.
	SecretEnv string
	// Config is source-specific JSON config.
	Config json.RawMessage
	// Transform is the optional sync transform (not serialised to wire).
	TransformFn Transform
	// CodeOrigin is the source file:line where the binding was declared.
	CodeOrigin string
	// Kind distinguishes event from schedule bindings.
	Kind BindingKind
}

// BindingKind enumerates the types of trigger bindings.
type BindingKind int

const (
	// EventBinding represents a webhook event trigger.
	EventBinding BindingKind = iota
	// ScheduleBinding represents a cron schedule trigger.
	ScheduleBinding
)

// Event creates an event trigger binding from the given options.
func Event(opts EventOpts) Binding {
	return Binding{
		Source:      opts.Source,
		EventTypes:  opts.Types,
		SecretEnv:   opts.SecretEnv,
		Config:      opts.Config,
		TransformFn: opts.Transform,
		Kind:        EventBinding,
	}
}

// Schedule creates a schedule (cron) trigger binding from the given options.
// The expression and timezone are always merged into Config so the control
// plane sees them regardless of whether custom Config was provided.
func Schedule(opts ScheduleOpts) Binding {
	base := map[string]any{}
	// Custom keys are merged first so Cron and Timezone stay authoritative.
	if len(opts.Config) > 0 {
		var custom map[string]any
		if err := json.Unmarshal(opts.Config, &custom); err == nil {
			for k, v := range custom {
				base[k] = v
			}
		}
	}
	base["expression"] = opts.Cron
	if opts.Timezone != "" {
		base["timezone"] = opts.Timezone
	} else if _, ok := base["timezone"]; !ok {
		base["timezone"] = "UTC"
	}
	cfg, _ := json.Marshal(base)
	return Binding{
		Source: "cron",
		Config: cfg,
		Kind:   ScheduleBinding,
	}
}
