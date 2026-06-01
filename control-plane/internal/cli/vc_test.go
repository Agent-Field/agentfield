package cli

import (
	"strings"
	"testing"
)

func TestResolveWebDIDAcceptsEncodedPort(t *testing.T) {
	_, err := resolveWebDID("did:web:example.com%3A8443:agents:test-agent")
	if err == nil {
		t.Fatalf("expected online resolution to be disabled")
	}
	if strings.Contains(err.Error(), "invalid did:web domain") || strings.Contains(err.Error(), "invalid URL") {
		t.Fatalf("expected encoded port to be accepted before offline rejection, got %v", err)
	}
}

func TestResolveDIDRejectsOnlineOptions(t *testing.T) {
	_, err := resolveDID("did:key:test", nil, VerifyOptions{ResolveWeb: true})
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("expected resolve-web option to be rejected, got %v", err)
	}

	_, err = resolveDID("did:key:test", nil, VerifyOptions{Resolver: "https://resolver.example"})
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("expected custom resolver option to be rejected, got %v", err)
	}
}
