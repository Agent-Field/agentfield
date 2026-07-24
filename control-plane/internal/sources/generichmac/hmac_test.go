package generichmac

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

func sign(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
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

func TestGenericHMAC_DefaultHeader(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"hello":"world"}`)
	r := req(body, map[string]string{
		"X-Signature": sign(body, secret),
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
}

func TestGenericHMAC_MetadataValidateAndDefaults(t *testing.T) {
	s := &source{}
	if s.Name() != "generic_hmac" {
		t.Fatalf("Name() = %q, want generic_hmac", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("Kind() = %v, want HTTP", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("generic_hmac should require a secret")
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

	parsed := parseConfig(json.RawMessage(`{"signature_header":""}`))
	if parsed.SignatureHeader != "X-Signature" {
		t.Fatalf("empty signature header should default, got %q", parsed.SignatureHeader)
	}
}

func TestGenericHMAC_CustomHeaderAndPrefix(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"k":"v"}`)
	cfg := json.RawMessage(`{
        "signature_header":"X-Custom-Sig",
        "signature_prefix":"sha256=",
        "event_type_header":"X-Event-Type",
        "idempotency_header":"X-Delivery-ID"
    }`)
	r := req(body, map[string]string{
		"X-Custom-Sig":  "sha256=" + sign(body, secret),
		"X-Event-Type":  "order.created",
		"X-Delivery-ID": "del-99",
	})

	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].Type != "order.created" {
		t.Errorf("want event type from header, got %q", events[0].Type)
	}
	if events[0].IdempotencyKey != "del-99" {
		t.Errorf("want idempotency from header, got %q", events[0].IdempotencyKey)
	}
}

func TestGenericHMAC_RejectsWrongPrefix(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{}`)
	cfg := json.RawMessage(`{"signature_header":"X-Sig","signature_prefix":"sha256="}`)
	r := req(body, map[string]string{
		"X-Sig": sign(body, secret), // missing sha256= prefix
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "configured prefix") {
		t.Fatalf("expected prefix error, got %v", err)
	}
}

func TestGenericHMAC_RejectsTamperedSignature(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"k":"v"}`)
	r := req(body, map[string]string{
		"X-Signature": "deadbeef",
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected mismatch error, got %v", err)
	}
}

func TestGenericHMAC_RejectsMissingSecret(t *testing.T) {
	r := req([]byte(`{}`), map[string]string{"X-Signature": "x"})
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "")
	if err == nil {
		t.Fatal("expected error for missing secret")
	}
}

func TestGenericHMAC_RejectsMissingHeader(t *testing.T) {
	r := req([]byte(`{}`), nil)
	_, err := (&source{}).HandleRequest(context.Background(), r, nil, "secret")
	if err == nil || !strings.Contains(err.Error(), "missing signature header") {
		t.Fatalf("expected missing header error, got %v", err)
	}
}

// signWithTimestamp computes HMAC-SHA256 over "timestamp.body" (Stripe-style).
func signWithTimestamp(body []byte, secret, timestamp string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

func TestGenericHMAC_TimestampBoundToSignature(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	ts := strconv.FormatInt(time.Now().Unix(), 10)
	sig := signWithTimestamp(body, secret, ts)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
}

func TestGenericHMAC_ReplayWithFreshTimestampRejected(t *testing.T) {
	// Simulates a replay attack: attacker captures body+signature, then
	// replays with a fresh timestamp. Because the timestamp is bound into
	// the HMAC, the signature won't match the new timestamp.
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	originalTS := strconv.FormatInt(time.Now().Unix()-10, 10)
	sig := signWithTimestamp(body, secret, originalTS)

	// Attacker rewrites timestamp to "now" but keeps the original signature
	freshTS := strconv.FormatInt(time.Now().Unix(), 10)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": freshTS, // forged fresh timestamp
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected signature mismatch on replay, got %v", err)
	}
}

func TestGenericHMAC_TimestampRejectsStale(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	// 10 minutes ago — outside default 300s tolerance
	ts := strconv.FormatInt(time.Now().Unix()-600, 10)
	sig := signWithTimestamp(body, secret, ts)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "outside tolerance window") {
		t.Fatalf("expected tolerance error, got %v", err)
	}
}

func TestGenericHMAC_TimestampRejectsFuture(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	// 10 minutes in the future
	ts := strconv.FormatInt(time.Now().Unix()+600, 10)
	sig := signWithTimestamp(body, secret, ts)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "outside tolerance window") {
		t.Fatalf("expected tolerance error, got %v", err)
	}
}

func TestGenericHMAC_TimestampMissingHeaderRejects(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	sig := sign(body, secret) // signed without timestamp

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		// no X-Timestamp
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "missing timestamp header") {
		t.Fatalf("expected missing timestamp error, got %v", err)
	}
}

func TestGenericHMAC_TimestampInvalidFormat(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	ts := "not-a-number"
	sig := signWithTimestamp(body, secret, ts)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "not a valid Unix epoch") {
		t.Fatalf("expected invalid timestamp error, got %v", err)
	}
}

func TestGenericHMAC_CustomToleranceSeconds(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	// 30 seconds ago
	ts := strconv.FormatInt(time.Now().Unix()-30, 10)
	sig := signWithTimestamp(body, secret, ts)

	// Custom tolerance of 10 seconds — should reject 30s old request
	tolerance := 10
	cfg, _ := json.Marshal(config{
		SignatureHeader:  "X-Signature",
		TimestampHeader:  "X-Timestamp",
		ToleranceSeconds: &tolerance,
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	_, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err == nil || !strings.Contains(err.Error(), "outside tolerance window") {
		t.Fatalf("expected tolerance error with custom window, got %v", err)
	}
}

func TestGenericHMAC_ZeroToleranceDisablesCheck(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	// Very old timestamp
	ts := strconv.FormatInt(time.Now().Unix()-9999, 10)
	sig := signWithTimestamp(body, secret, ts)

	tolerance := 0
	cfg, _ := json.Marshal(config{
		SignatureHeader:  "X-Signature",
		TimestampHeader:  "X-Timestamp",
		ToleranceSeconds: &tolerance,
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": ts,
	})
	// tolerance=0 disables the check, so this should succeed
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("expected success with tolerance=0, got %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
}

func TestGenericHMAC_ValidateRejectsNegativeTolerance(t *testing.T) {
	s := &source{}
	if err := s.Validate([]byte(`{"tolerance_seconds": -1}`)); err == nil {
		t.Fatal("expected error for negative tolerance_seconds")
	}
	if err := s.Validate([]byte(`{"tolerance_seconds": 0}`)); err != nil {
		t.Fatalf("expected zero tolerance_seconds to be valid, got %v", err)
	}
}

func TestGenericHMAC_WithoutTimestampHeaderBodyOnlySigning(t *testing.T) {
	// When no timestamp_header is configured, signing is body-only (backward compatible)
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	sig := sign(body, secret) // body-only HMAC

	// No timestamp_header in config
	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
	})

	r := req(body, map[string]string{
		"X-Signature": sig,
	})
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("expected success for body-only signing, got %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
}

func TestGenericHMAC_TimestampWhitespaceHandled(t *testing.T) {
	secret := "supersecret"
	body := []byte(`{"event":"test"}`)
	ts := strconv.FormatInt(time.Now().Unix(), 10)
	sig := signWithTimestamp(body, secret, ts)

	cfg, _ := json.Marshal(config{
		SignatureHeader: "X-Signature",
		TimestampHeader: "X-Timestamp",
	})

	// Timestamp with leading/trailing whitespace
	r := req(body, map[string]string{
		"X-Signature": sig,
		"X-Timestamp": fmt.Sprintf("  %s  ", ts),
	})
	events, err := (&source{}).HandleRequest(context.Background(), r, cfg, secret)
	if err != nil {
		t.Fatalf("expected success with whitespace-padded timestamp, got %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
}
