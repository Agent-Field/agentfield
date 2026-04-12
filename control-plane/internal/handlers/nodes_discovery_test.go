package handlers

import (
	"testing"
)

func TestNormalizeCandidateAddsDefaults(t *testing.T) {
	normalized, err := normalizeCandidate("example.com", "8080")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if normalized != "http://example.com:8080" {
		t.Fatalf("unexpected normalization: %s", normalized)
	}

	normalized, err = normalizeCandidate("https://example.com", "8080")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if normalized != "https://example.com:8080" {
		t.Fatalf("expected default port applied, got %s", normalized)
	}
}

func TestResolveCallbackCandidatesSuccess(t *testing.T) {
	resolved, normalized, results := resolveCallbackCandidates([]string{"http://agent:7000"}, "")

	if resolved == "" {
		t.Fatalf("expected resolved callback URL")
	}
	if len(normalized) != 1 {
		t.Fatalf("expected exactly one normalized candidate, got %d", len(normalized))
	}
	if results != nil {
		t.Fatalf("expected registration-time probing to be disabled, got %+v", results)
	}
}
