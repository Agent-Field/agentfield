package slack

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func slackSig(body []byte, secret string, ts int64) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(fmt.Sprintf("v0:%d:", ts)))
	mac.Write(body)
	return "v0=" + hex.EncodeToString(mac.Sum(nil))
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

func TestSlack_VerifiesAndUnwrapsEventCallback(t *testing.T) {
	secret := "slack_signing_secret"
	body, _ := json.Marshal(map[string]any{
		"type":     "event_callback",
		"event_id": "Ev123",
		"event":    map[string]any{"type": "app_mention", "text": "hi"},
	})
	ts := time.Now().Unix()

	r := req(body, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         slackSig(body, secret, ts),
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "app_mention" {
		t.Errorf("want type app_mention (unwrapped), got %q", events[0].Type)
	}
	if events[0].IdempotencyKey != "Ev123" {
		t.Errorf("want event_id Ev123, got %q", events[0].IdempotencyKey)
	}
}

func TestSlack_MetadataAndValidateBranches(t *testing.T) {
	s := &source{}
	if s.Name() != "slack" {
		t.Fatalf("Name() = %q, want slack", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("Kind() = %v, want HTTP", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("slack should require a secret")
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
}

func TestSlack_KeepsTopLevelTypeWhenNotEventCallback(t *testing.T) {
	secret := "slack_signing_secret"
	body, _ := json.Marshal(map[string]any{
		"type":      "url_verification",
		"challenge": "abc",
	})
	ts := time.Now().Unix()
	r := req(body, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         slackSig(body, secret, ts),
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].Type != "url_verification" {
		t.Errorf("want url_verification, got %q", events[0].Type)
	}
}

func TestSlack_RejectsExpiredTimestamp(t *testing.T) {
	secret := "slack_signing_secret"
	body := []byte(`{"type":"event_callback","event":{"type":"x"}}`)
	ts := time.Now().Add(-1 * time.Hour).Unix() // outside default 5min tolerance
	r := req(body, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         slackSig(body, secret, ts),
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "tolerance") {
		t.Fatalf("expected tolerance error, got %v", err)
	}
}

func TestSlack_RejectsTamperedBody(t *testing.T) {
	secret := "slack_signing_secret"
	body := []byte(`{"type":"event_callback","event":{"type":"x"}}`)
	ts := time.Now().Unix()
	signed := slackSig(body, secret, ts)
	tampered := []byte(`{"type":"event_callback","event":{"type":"y"}}`)
	r := req(tampered, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         signed,
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected signature mismatch, got %v", err)
	}
}

func TestSlack_RejectsMissingHeaders(t *testing.T) {
	r := req([]byte(`{}`), nil)
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "secret")
	if err == nil || !strings.Contains(err.Error(), "missing signature headers") {
		t.Fatalf("expected missing-headers error, got %v", err)
	}
}

func TestSlack_RejectsMissingV0Prefix(t *testing.T) {
	secret := "slack_signing_secret"
	body := []byte(`{"type":"x"}`)
	ts := time.Now().Unix()
	r := req(body, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         "v1=" + strings.TrimPrefix(slackSig(body, secret, ts), "v0="),
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "v0 prefix") {
		t.Fatalf("expected v0 prefix error, got %v", err)
	}
}

func TestSlack_ConfigToleranceOverride(t *testing.T) {
	cfg := json.RawMessage(`{"tolerance_seconds":-1}`)
	if err := (&source{}).Validate(cfg); err == nil {
		t.Fatal("expected validation failure for negative tolerance")
	}
}

func TestSlack_HandleRequestAdditionalRejects(t *testing.T) {
	secret := "slack_signing_secret"
	body := []byte(`{"type":"event_callback","event":{"type":"x"}}`)

	r := req(body, map[string]string{
		"X-Slack-Request-Timestamp": "not-a-number",
		"X-Slack-Signature":         "v0=deadbeef",
	})
	if _, err := (&source{}).HandleRequest(context.Background(), r, nil, secret); err == nil || !strings.Contains(err.Error(), "invalid timestamp") {
		t.Fatalf("expected invalid timestamp, got %v", err)
	}

	ts := time.Now().Unix()
	r = req([]byte(`{`), map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(ts, 10),
		"X-Slack-Signature":         slackSig([]byte(`{`), secret, ts),
	})
	if _, err := (&source{}).HandleRequest(context.Background(), r, nil, secret); err == nil || !strings.Contains(err.Error(), "invalid event JSON") {
		t.Fatalf("expected invalid JSON, got %v", err)
	}

	_, err := (&source{}).HandleRequest(context.Background(), req(body, nil), nil, "")
	if err == nil || !strings.Contains(err.Error(), "missing signing secret") {
		t.Fatalf("expected missing secret, got %v", err)
	}

	// A zero tolerance disables timestamp freshness checks but still verifies
	// the HMAC and parses the payload.
	oldTS := time.Now().Add(-24 * time.Hour).Unix()
	r = req(body, map[string]string{
		"X-Slack-Request-Timestamp": strconv.FormatInt(oldTS, 10),
		"X-Slack-Signature":         slackSig(body, secret, oldTS),
	})
	events, err := (&source{}).HandleRequest(context.Background(), r, json.RawMessage(`{"tolerance_seconds":0}`), secret)
	if err != nil {
		t.Fatalf("zero tolerance should accept old signed request: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}
