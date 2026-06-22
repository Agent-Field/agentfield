package sentry

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func sign(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

func rawReq(body []byte, headers map[string]string) *sources.RawRequest {
	h := http.Header{}
	for k, v := range headers {
		h.Set(k, v)
	}
	return &sources.RawRequest{
		Headers: h,
		Body:    body,
		URL:     &url.URL{Path: "/sources/sentry"},
		Method:  "POST",
	}
}

func TestSentry_VerifiesValidSignature(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123"}},"actor":{"type":"user","name":"Ada"}}`)
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Sentry-Hook-Timestamp": time.Now().UTC().Format(time.RFC3339),
		"Request-ID":            "req-123",
	})

	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "issue.created" {
		t.Fatalf("type=%q, want issue.created", events[0].Type)
	}
	if events[0].IdempotencyKey != "req-123" {
		t.Fatalf("idempotency=%q", events[0].IdempotencyKey)
	}
	if !strings.Contains(string(events[0].Normalized), `"resource":"issue"`) {
		t.Fatalf("normalized missing resource: %s", events[0].Normalized)
	}
}

func TestSentry_DerivesIdempotencyWhenRequestIDMissing(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"resolved"}`)
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Sentry-Hook-Timestamp": time.Now().UTC().Format(time.RFC3339),
	})

	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events[0].IdempotencyKey) != 64 {
		t.Fatalf("expected sha256 idempotency fallback, got %q", events[0].IdempotencyKey)
	}
}

func TestSentry_RejectsTamperedBody(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created"}`)
	tampered := []byte(`{"action":"resolved"}`)
	req := rawReq(tampered, map[string]string{"Sentry-Hook-Signature": sign(body, secret)})

	_, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected signature mismatch, got %v", err)
	}
}

func TestSentry_RejectsMissingSecret(t *testing.T) {
	body := []byte(`{"action":"created"}`)
	req := rawReq(body, map[string]string{"Sentry-Hook-Signature": sign(body, "x")})
	if _, err := (&source{}).HandleRequest(context.Background(), req, nil, ""); err == nil {
		t.Fatal("expected missing secret error")
	}
}

func TestSentry_RegistryMetadata(t *testing.T) {
	s := &source{}
	if s.Name() != "sentry" {
		t.Fatalf("name=%q", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("kind=%v", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("sentry should require a secret")
	}
	if len(s.ConfigSchema()) == 0 {
		t.Fatal("sentry should expose a config schema")
	}
}

func TestSentry_RejectsStaleTimestamp(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123"}},"actor":{"type":"user","name":"Ada"}}`)
	
	// Set timestamp to 10 minutes ago
	staleTime := time.Now().UTC().Add(-10 * time.Minute).Format(time.RFC3339)
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Sentry-Hook-Timestamp": staleTime,
		"Request-ID":            "req-123",
	})
	
	config := []byte(`{"tolerance_seconds":300}`)
	_, err := (&source{}).HandleRequest(context.Background(), req, config, secret)
	if err == nil || !strings.Contains(err.Error(), "outside tolerance") {
		t.Fatalf("expected timestamp outside tolerance error, got %v", err)
	}
}

func TestSentry_AllowsDisabledTolerance(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123"}},"actor":{"type":"user","name":"Ada"}}`)
	
	// Set timestamp to 10 minutes ago
	staleTime := time.Now().UTC().Add(-10 * time.Minute).Format(time.RFC3339)
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Sentry-Hook-Timestamp": staleTime,
		"Request-ID":            "req-123",
	})
	
	// tolerance_seconds = 0 disables the check
	config := []byte(`{"tolerance_seconds":0}`)
	events, err := (&source{}).HandleRequest(context.Background(), req, config, secret)
	if err != nil {
		t.Fatalf("unexpected error with disabled tolerance: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}

func TestSentry_RejectsMissingTimestamp(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123"}},"actor":{"type":"user","name":"Ada"}}`)
	
	// No Sentry-Hook-Timestamp header, default tolerance applies
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Request-ID":            "req-123",
	})
	
	_, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "missing or invalid") {
		t.Fatalf("expected missing timestamp error, got %v", err)
	}
}

func TestSentry_AcceptsUnixSecondsTimestamp(t *testing.T) {
	secret := "sentry_client_secret"
	body := []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123"}},"actor":{"type":"user","name":"Ada"}}`)
	
	// Set timestamp to current time as unix seconds string
	unixSecsStr := fmt.Sprintf("%d", time.Now().Unix())
	req := rawReq(body, map[string]string{
		"Sentry-Hook-Signature": sign(body, secret),
		"Sentry-Hook-Resource":  "issue",
		"Sentry-Hook-Timestamp": unixSecsStr,
		"Request-ID":            "req-123",
	})
	
	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error with unix seconds timestamp: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}
