package types

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateSessionTransportAcceptsExplicitSupportedPairs(t *testing.T) {
	capability, err := ValidateSessionTransport("OpenAI", "WebRTC")
	if err != nil {
		t.Fatalf("ValidateSessionTransport returned error: %v", err)
	}

	if capability.Provider != "openai" {
		t.Fatalf("provider = %q, want openai", capability.Provider)
	}
	if capability.Transport != "webrtc" {
		t.Fatalf("transport = %q, want webrtc", capability.Transport)
	}
}

func TestValidateSessionTransportRejectsProviderTransportMismatch(t *testing.T) {
	_, err := ValidateSessionTransport("openrouter", "webrtc")
	if err == nil {
		t.Fatal("expected error")
	}

	var transportErr *SessionTransportError
	if !errors.As(err, &transportErr) {
		t.Fatalf("error type = %T, want *SessionTransportError", err)
	}
	if transportErr.Provider != "openrouter" {
		t.Fatalf("provider = %q, want openrouter", transportErr.Provider)
	}
	if transportErr.Transport != "webrtc" {
		t.Fatalf("transport = %q, want webrtc", transportErr.Transport)
	}
	if got := err.Error(); got == "" || !containsAll(got, []string{"audio_turns", "does not infer or switch providers"}) {
		t.Fatalf("unexpected error message: %q", got)
	}
}

func TestValidateSessionTransportRequiresExplicitTransport(t *testing.T) {
	_, err := ValidateSessionTransport("openai", "")
	if err == nil || !containsAll(err.Error(), []string{"transport is required"}) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateSessionTransportRejectsUnknownProvider(t *testing.T) {
	_, err := ValidateSessionTransport("custom", "webrtc")
	if err == nil || !containsAll(err.Error(), []string{"unknown session provider", "custom"}) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func containsAll(value string, needles []string) bool {
	for _, needle := range needles {
		if !strings.Contains(value, needle) {
			return false
		}
	}
	return true
}
