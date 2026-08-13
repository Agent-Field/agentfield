package agent

import (
	"context"

	"github.com/Agent-Field/agentfield/sdk/go/triggers"
	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// OnEvent is sugar for registering an event-triggered reasoner.
//
// Equivalent to:
//
//	app.RegisterReasoner(name, handler, WithTriggers(triggers.Event(opts)))
//
// The reasoner name and handler are required. The trigger binding is
// auto-populated with code_origin from the caller's file:line.
func (a *Agent) OnEvent(opts triggers.EventOpts, name string, handler HandlerFunc) {
	binding := triggers.Event(opts)
	binding.CodeOrigin = captureCodeOrigin(2)
	a.RegisterReasoner(name, handler, withTriggersBinding(binding))
}

// OnSchedule is sugar for registering a cron-triggered reasoner.
//
// Equivalent to:
//
//	app.RegisterReasoner(name, handler, WithTriggers(triggers.Schedule(opts)))
//
// The expression follows the standard 5-field cron format (minute hour dom month dow).
func (a *Agent) OnSchedule(expression string, name string, handler HandlerFunc, opts ...OnScheduleOption) {
	schedOpts := triggers.ScheduleOpts{Cron: expression}
	for _, o := range opts {
		o(&schedOpts)
	}
	binding := triggers.Schedule(schedOpts)
	binding.CodeOrigin = captureCodeOrigin(2)
	a.RegisterReasoner(name, handler, withTriggersBinding(binding))
}

// OnScheduleOption configures optional parameters for OnSchedule.
type OnScheduleOption func(*triggers.ScheduleOpts)

// WithTimezone sets the IANA timezone for a schedule trigger.
func WithTimezone(tz string) OnScheduleOption {
	return func(opts *triggers.ScheduleOpts) {
		opts.Timezone = tz
	}
}

// withTriggersBinding converts a triggers.Binding into a ReasonerOption that
// appends it to the reasoner's trigger list. This bridges the triggers package
// types with the agent registration machinery.
func withTriggersBinding(bindings ...triggers.Binding) ReasonerOption {
	return func(r *Reasoner) {
		for _, b := range bindings {
			tb := bindingToWire(b)
			r.Triggers = append(r.Triggers, tb)
		}
		// Store the bindings with their Transform on the reasoner for
		// dispatch-time use (Transform is not serialised to wire).
		r.triggerBindings = append(r.triggerBindings, bindings...)
	}
}

// bindingToWire converts a triggers.Binding to the wire-level types.TriggerBinding.
func bindingToWire(b triggers.Binding) types.TriggerBinding {
	return types.TriggerBinding{
		Source:       b.Source,
		EventTypes:   b.EventTypes,
		Config:       b.Config,
		SecretEnvVar: b.SecretEnv,
		CodeOrigin:   b.CodeOrigin,
	}
}

// applyTriggerDispatch detects a dispatcher trigger envelope in the raw input,
// unwraps it, applies the matching binding's Transform, and returns the
// resolved input plus a context carrying the *triggers.Context.
//
// For direct calls (no envelope) the input is returned unchanged and the
// context is untouched, so triggers.FromContext(ctx) yields nil.
//
// This is the single place the dispatch-side trigger handling lives; both the
// sync HTTP path and the async goroutine path route through it so their
// behaviour cannot drift.
func applyTriggerDispatch(
	ctx context.Context,
	reasoner *Reasoner,
	input map[string]any,
) (context.Context, map[string]any) {
	unwrapped, tc := triggers.Unwrap(input)
	if tc == nil {
		// Direct call — nothing to do.
		return ctx, input
	}

	// Run the matching binding's Transform (if any) against the event.
	transformed := triggers.ApplyTransform(tc, reasoner.triggerBindings, unwrapped)

	resolved, ok := transformed.(map[string]any)
	if !ok {
		// A Transform may legitimately return a non-object (e.g. a slice or
		// scalar). The HandlerFunc contract is map[string]any, so wrap it
		// under a conventional key rather than dropping the value.
		resolved = map[string]any{"input": transformed}
	}

	return triggers.NewContext(ctx, tc), resolved
}
