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
