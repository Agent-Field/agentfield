package genericbearer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

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

func TestBearer_DefaultSchemeAccepted(t *testing.T) {
	secret := "tok_test"
	r := req([]byte(`{}`), map[string]string{"Authorization": "Bearer " + secret})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}

func TestBearer_MetadataValidateAndDefaults(t *testing.T) {
	s := &source{}
	if s.Name() != "generic_bearer" {
		t.Fatalf("Name() = %q, want generic_bearer", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("Kind() = %v, want HTTP", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("generic_bearer should require a secret")
	}
	var schema map[string]any
	if err := json.Unmarshal(s.ConfigSchema(), &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	if err := s.Validate(nil); err != nil {
		t.Fatalf("empty config should validate: %v", err)
	}
	if err := s.Validate([]byte(`{`)); err == nil {
		t.Fatal("expected invalid config error")
	}

	parsed := parseConfig(json.RawMessage(`{"header":"","scheme":""}`))
	if parsed.Header != "Authorization" {
		t.Fatalf("empty header should default to Authorization, got %q", parsed.Header)
	}
}

func TestBearer_CustomHeaderAndScheme(t *testing.T) {
	secret := "tok_test"
	cfg := json.RawMessage(`{"header":"X-API-Key","scheme":""}`)
	r := req([]byte(`{}`), map[string]string{"X-API-Key": secret})
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}

func TestBearer_RejectsWrongToken(t *testing.T) {
	r := req([]byte(`{}`), map[string]string{"Authorization": "Bearer wrong-token"})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "right-token")
	if err == nil || !strings.Contains(err.Error(), "token mismatch") {
		t.Fatalf("expected mismatch error, got %v", err)
	}
}

func TestBearer_RejectsMissingScheme(t *testing.T) {
	secret := "tok_test"
	r := req([]byte(`{}`), map[string]string{"Authorization": secret}) // no Bearer prefix
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "scheme prefix") {
		t.Fatalf("expected scheme prefix error, got %v", err)
	}
}

func TestBearer_RejectsMissingHeader(t *testing.T) {
	r := req([]byte(`{}`), nil)
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "secret")
	if err == nil || !strings.Contains(err.Error(), "missing auth header") {
		t.Fatalf("expected missing header error, got %v", err)
	}
}

func TestBearer_RejectsMissingSecret(t *testing.T) {
	r := req([]byte(`{}`), map[string]string{"Authorization": "Bearer x"})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "")
	if err == nil {
		t.Fatal("expected error for missing secret")
	}
}

func TestBearer_PopulatesEventTypeAndIdempotencyFromHeaders(t *testing.T) {
	secret := "tok_test"
	cfg := json.RawMessage(`{"event_type_header":"X-Event-Type","idempotency_header":"X-Delivery"}`)
	r := req([]byte(`{}`), map[string]string{
		"Authorization": "Bearer " + secret,
		"X-Event-Type":  "thing.happened",
		"X-Delivery":    "del-1",
	})
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].Type != "thing.happened" {
		t.Errorf("event type=%q want thing.happened", events[0].Type)
	}
	if events[0].IdempotencyKey != "del-1" {
		t.Errorf("idempotency=%q want del-1", events[0].IdempotencyKey)
	}
}
