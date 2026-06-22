package agent

import (
	"fmt"
	"sort"
	"strings"
)

var SupportedSessionTransports = map[string][]string{
	"openai":     {"webrtc", "websocket"},
	"openrouter": {"audio_turns"},
}

type SessionTransportCapability struct {
	Provider  string
	Transport string
}

type SessionTransportError struct {
	Provider  string
	Transport string
	Supported []string
}

func (e *SessionTransportError) Error() string {
	return fmt.Sprintf(
		"unsupported session transport %q for provider %q. Supported transports: %s. AgentField does not infer or switch providers; set provider and transport explicitly.",
		e.Transport,
		e.Provider,
		strings.Join(e.Supported, ", "),
	)
}

func NormalizeSessionTransportValue(value string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(value)), "-", "_")
}

func ValidateSessionTransport(provider string, transport string) (SessionTransportCapability, error) {
	normalizedProvider := NormalizeSessionTransportValue(provider)
	normalizedTransport := NormalizeSessionTransportValue(transport)

	if normalizedProvider == "" {
		return SessionTransportCapability{}, fmt.Errorf("session provider is required; AgentField does not infer providers")
	}
	if normalizedTransport == "" {
		return SessionTransportCapability{}, fmt.Errorf("session transport is required; AgentField does not infer transports")
	}

	supported, ok := SupportedSessionTransports[normalizedProvider]
	if !ok {
		known := make([]string, 0, len(SupportedSessionTransports))
		for knownProvider := range SupportedSessionTransports {
			known = append(known, knownProvider)
		}
		sort.Strings(known)
		return SessionTransportCapability{}, fmt.Errorf(
			"unknown session provider %q. Known providers: %s. Register provider capabilities before using a custom session provider",
			provider,
			strings.Join(known, ", "),
		)
	}

	for _, candidate := range supported {
		if candidate == normalizedTransport {
			return SessionTransportCapability{
				Provider:  normalizedProvider,
				Transport: normalizedTransport,
			}, nil
		}
	}

	return SessionTransportCapability{}, &SessionTransportError{
		Provider:  normalizedProvider,
		Transport: normalizedTransport,
		Supported: append([]string(nil), supported...),
	}
}
