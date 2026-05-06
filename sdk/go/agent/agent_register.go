package agent

import "encoding/json"

// RegisterReasoner makes a handler available at /reasoners/{name}.
func (a *Agent) RegisterReasoner(name string, handler HandlerFunc, opts ...ReasonerOption) {
	if handler == nil {
		panic("nil handler supplied")
	}

	meta := &Reasoner{
		Name:         name,
		Handler:      handler,
		InputSchema:  json.RawMessage(`{"type":"object","additionalProperties":true}`),
		OutputSchema: json.RawMessage(`{"type":"object","additionalProperties":true}`),
	}
	for _, opt := range opts {
		opt(meta)
	}

	// Auto-set accepts_webhook=true when the reasoner declared inbound
	// trigger bindings but didn't pass an explicit accepts_webhook option.
	// Mirrors the Python SDK's normalisation in agent.py — without this,
	// the control plane rejects registration with a missing-flag error
	// when the reasoner is wired to a webhook source.
	if meta.AcceptsWebhook == nil && len(meta.Triggers) > 0 {
		flag := "true"
		meta.AcceptsWebhook = &flag
	}

	if meta.DefaultCLI {
		if a.defaultCLIReasoner != "" && a.defaultCLIReasoner != name {
			a.logger.Printf("warn: default CLI reasoner already set to %s, ignoring default flag on %s", a.defaultCLIReasoner, name)
			meta.DefaultCLI = false
		} else {
			a.defaultCLIReasoner = name
		}
	}

	if meta.RequireRealtimeValidation {
		a.realtimeValidationFunctions[name] = struct{}{}
	}

	a.reasoners[name] = meta
}
