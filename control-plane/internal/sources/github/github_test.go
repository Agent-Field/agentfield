package github

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func sign(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func req(body []byte, headers map[string]string) *sources.RawRequest {
	h := http.Header{}
	for k, v := range headers {
		h.Set(k, v)
	}
	return &sources.RawRequest{
		Headers: h,
		Body:    body,
		URL:     &url.URL{Path: "/sources/abc"},
		Method:  "POST",
	}
}

func TestGitHub_VerifiesValidSignature(t *testing.T) {
	secret := "whsec_test"
	body := []byte(`{"action":"opened","pull_request":{"id":1}}`)
	r := req(body, map[string]string{
		"X-Hub-Signature-256": sign(body, secret),
		"X-GitHub-Event":      "pull_request",
		"X-GitHub-Delivery":   "delivery-123",
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "pull_request.opened" {
		t.Errorf("want type pull_request.opened, got %q", events[0].Type)
	}
	if events[0].IdempotencyKey != "delivery-123" {
		t.Errorf("want idempotency=delivery-123, got %q", events[0].IdempotencyKey)
	}
}

func TestGitHub_FallsBackToBareEventTypeWhenNoAction(t *testing.T) {
	secret := "whsec_test"
	body := []byte(`{"zen":"Approachable is better than simple."}`)
	r := req(body, map[string]string{
		"X-Hub-Signature-256": sign(body, secret),
		"X-GitHub-Event":      "ping",
		"X-GitHub-Delivery":   "ping-1",
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].Type != "ping" {
		t.Errorf("want bare event type ping, got %q", events[0].Type)
	}
}

func TestGitHub_RejectsTamperedBody(t *testing.T) {
	secret := "whsec_test"
	body := []byte(`{"action":"opened"}`)
	signed := sign(body, secret)
	tampered := []byte(`{"action":"closed"}`) // different body, original signature
	r := req(tampered, map[string]string{
		"X-Hub-Signature-256": signed,
		"X-GitHub-Event":      "pull_request",
	})

	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected signature mismatch error, got %v", err)
	}
}

func TestGitHub_RejectsMissingHeader(t *testing.T) {
	r := req([]byte(`{}`), map[string]string{"X-GitHub-Event": "ping"})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "secret")
	if err == nil {
		t.Fatal("expected error for missing signature header")
	}
}

func TestGitHub_RejectsWrongPrefix(t *testing.T) {
	secret := "whsec_test"
	body := []byte(`{}`)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	r := req(body, map[string]string{
		"X-Hub-Signature-256": "sha1=" + hex.EncodeToString(mac.Sum(nil)),
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "sha256 prefix") {
		t.Fatalf("expected sha256 prefix error, got %v", err)
	}
}

func TestGitHub_RejectsMissingSecret(t *testing.T) {
	body := []byte(`{}`)
	r := req(body, map[string]string{"X-Hub-Signature-256": sign(body, "x")})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "")
	if err == nil {
		t.Fatal("expected error for missing secret")
	}
}

func TestGitHub_RegistryMetadata(t *testing.T) {
	s := &source{}
	if s.Name() != "github" {
		t.Errorf("name=%q, want github", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Errorf("kind=%v, want HTTP", s.Kind())
	}
	if !s.SecretRequired() {
		t.Error("github should require secret")
	}
	if len(s.ConfigSchema()) == 0 {
		t.Fatal("github should expose a config schema")
	}
	if err := s.Validate([]byte(`{`)); err != nil {
		t.Fatalf("github has no config knobs, validate should ignore payload: %v", err)
	}
}
