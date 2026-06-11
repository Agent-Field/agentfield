package agent

type SessionDefinition struct {
	Name       string         `json:"name"`
	Provider   string         `json:"provider"`
	Transport  string         `json:"transport"`
	Model      string         `json:"model,omitempty"`
	Modalities []string       `json:"modalities"`
	Voice      string         `json:"voice,omitempty"`
	Tools      []string       `json:"tools"`
	Metadata   map[string]any `json:"metadata"`
}

type SessionOption func(*SessionDefinition)

func WithSessionModel(model string) SessionOption {
	return func(s *SessionDefinition) { s.Model = model }
}

func WithSessionModalities(modalities ...string) SessionOption {
	return func(s *SessionDefinition) { s.Modalities = append([]string(nil), modalities...) }
}

func WithSessionVoice(voice string) SessionOption {
	return func(s *SessionDefinition) { s.Voice = voice }
}

func WithSessionTools(tools ...string) SessionOption {
	return func(s *SessionDefinition) { s.Tools = append([]string(nil), tools...) }
}

func WithSessionMetadata(metadata map[string]any) SessionOption {
	return func(s *SessionDefinition) {
		s.Metadata = map[string]any{}
		for key, value := range metadata {
			s.Metadata[key] = value
		}
	}
}

func (a *Agent) RegisterSession(name string, provider string, transport string, opts ...SessionOption) error {
	capability, err := ValidateSessionTransport(provider, transport)
	if err != nil {
		return err
	}

	definition := SessionDefinition{
		Name:       name,
		Provider:   capability.Provider,
		Transport:  capability.Transport,
		Modalities: []string{"audio", "text"},
		Tools:      []string{},
		Metadata:   map[string]any{},
	}
	for _, opt := range opts {
		opt(&definition)
	}
	if len(definition.Modalities) == 0 {
		definition.Modalities = []string{"audio", "text"}
	}
	if definition.Tools == nil {
		definition.Tools = []string{}
	}
	if definition.Metadata == nil {
		definition.Metadata = map[string]any{}
	}

	a.sessions[name] = definition
	return nil
}

func (a *Agent) SessionDefinitions() []SessionDefinition {
	sessions := make([]SessionDefinition, 0, len(a.sessions))
	for _, session := range a.sessions {
		sessions = append(sessions, session)
	}
	return sessions
}
