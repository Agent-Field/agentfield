package linear

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
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

func rawReq(body []byte, headers map[string]string) *sources.RawRequest {
	h := http.Header{}
	for k, v := range headers {
		h.Set(k, v)
	}
	return &sources.RawRequest{
		Headers: h,
		Body:    body,
		URL:     &url.URL{Path: "/sources/linear"},
		Method:  "POST",
	}
}

func TestLinear_VerifiesValidSignature(t *testing.T) {
	secret := "linear_secret"
	body := []byte(`{"action":"create","type":"Issue","createdAt":"2026-06-15T12:00:00Z","webhookTimestamp":` + nowMilli() + `,"webhookId":"hook-1","data":{"id":"issue-1"}}`)
	req := rawReq(body, map[string]string{
		"Linear-Signature": sign(body, secret),
		"Linear-Delivery":  "delivery-123",
		"Linear-Event":     "Issue",
	})

	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "issue.create" {
		t.Fatalf("type=%q, want issue.create", events[0].Type)
	}
	if events[0].IdempotencyKey != "delivery-123" {
		t.Fatalf("idempotency=%q", events[0].IdempotencyKey)
	}
	if !strings.Contains(string(events[0].Normalized), `"delivery":"delivery-123"`) {
		t.Fatalf("normalized missing delivery: %s", events[0].Normalized)
	}
}

func TestLinear_RejectsTamperedBody(t *testing.T) {
	secret := "linear_secret"
	body := []byte(`{"action":"create","type":"Issue","webhookTimestamp":` + nowMilli() + `}`)
	tampered := []byte(`{"action":"update","type":"Issue","webhookTimestamp":` + nowMilli() + `}`)
	req := rawReq(tampered, map[string]string{"Linear-Signature": sign(body, secret)})

	_, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err == nil || !strings.Contains(err.Error(), "signature mismatch") {
		t.Fatalf("expected signature mismatch, got %v", err)
	}
}

func TestLinear_RejectsStaleTimestamp(t *testing.T) {
	secret := "linear_secret"
	body := []byte(`{"action":"update","type":"Issue","webhookTimestamp":` + staleMilli() + `}`)
	req := rawReq(body, map[string]string{"Linear-Signature": sign(body, secret)})

	_, err := (&source{}).HandleRequest(context.Background(), req, []byte(`{"tolerance_seconds":60}`), secret)
	if err == nil || !strings.Contains(err.Error(), "outside tolerance") {
		t.Fatalf("expected tolerance error, got %v", err)
	}
}

func TestLinear_AllowsDisabledTimestampTolerance(t *testing.T) {
	secret := "linear_secret"
	body := []byte(`{"action":"update","type":"Issue","webhookTimestamp":` + staleMilli() + `}`)
	req := rawReq(body, map[string]string{"Linear-Signature": sign(body, secret), "Linear-Delivery": "old"})

	events, err := (&source{}).HandleRequest(context.Background(), req, []byte(`{"tolerance_seconds":0}`), secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].IdempotencyKey != "old" {
		t.Fatalf("idempotency=%q", events[0].IdempotencyKey)
	}
}

func TestLinear_RejectsInvalidConfig(t *testing.T) {
	if err := (&source{}).Validate([]byte(`{"tolerance_seconds":-1}`)); err == nil {
		t.Fatal("expected invalid config error")
	}
}

func TestLinear_RegistryMetadata(t *testing.T) {
	s := &source{}
	if s.Name() != "linear" {
		t.Fatalf("name=%q", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("kind=%v", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("linear should require a secret")
	}
	if len(s.ConfigSchema()) == 0 {
		t.Fatal("linear should expose a config schema")
	}
}

func TestLinear_HashesIdempotencyWhenDeliveryMissing(t *testing.T) {
	secret := "linear_secret"
	currentTs := nowMilli()
	body := []byte(`{"action":"create","type":"Issue","createdAt":"2026-06-15T12:00:00Z","webhookTimestamp":` + currentTs + `,"webhookId":"hook-123","data":{"id":"issue-1"}}`)
	req := rawReq(body, map[string]string{
		"Linear-Signature": sign(body, secret),
		// No Linear-Delivery header
		"Linear-Event": "Issue",
	})

	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}

	// IdempotencyKey should be a 64-char hex string (SHA256 hash)
	key := events[0].IdempotencyKey
	if len(key) != 64 {
		t.Fatalf("idempotency key length=%d, want 64 (hex SHA256)", len(key))
	}
	for _, c := range key {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Fatalf("idempotency key contains non-hex char: %c", c)
		}
	}

	// Same inputs should produce same key
	events2, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if events2[0].IdempotencyKey != events[0].IdempotencyKey {
		t.Fatalf("same inputs produced different keys: %q vs %q", events[0].IdempotencyKey, events2[0].IdempotencyKey)
	}

	// Different action should produce different key
	bodyDiffAction := []byte(`{"action":"update","type":"Issue","createdAt":"2026-06-15T12:00:00Z","webhookTimestamp":` + currentTs + `,"webhookId":"hook-123","data":{"id":"issue-1"}}`)
	reqDiffAction := rawReq(bodyDiffAction, map[string]string{
		"Linear-Signature": sign(bodyDiffAction, secret),
		"Linear-Event":     "Issue",
	})
	eventsDiff, err := (&source{}).HandleRequest(context.Background(), reqDiffAction, nil, secret)
	if err != nil {
		t.Fatalf("diff action call: %v", err)
	}
	if eventsDiff[0].IdempotencyKey == events[0].IdempotencyKey {
		t.Fatalf("different actions produced same key: %q", events[0].IdempotencyKey)
	}
}

func TestLinear_PreferrsLinearDeliveryHeaderOverHash(t *testing.T) {
	secret := "linear_secret"
	currentTs := nowMilli()
	body := []byte(`{"action":"create","type":"Issue","createdAt":"2026-06-15T12:00:00Z","webhookTimestamp":` + currentTs + `,"webhookId":"hook-123","data":{"id":"issue-1"}}`)
	req := rawReq(body, map[string]string{
		"Linear-Signature": sign(body, secret),
		"Linear-Delivery":  "explicit-delivery-uuid",
		"Linear-Event":     "Issue",
	})

	events, err := (&source{}).HandleRequest(context.Background(), req, nil, secret)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events[0].IdempotencyKey != "explicit-delivery-uuid" {
		t.Fatalf("idempotency=%q, want explicit-delivery-uuid", events[0].IdempotencyKey)
	}
}

func nowMilli() string {
	return strconvFormat(time.Now().UnixMilli())
}

func staleMilli() string {
	return strconvFormat(time.Now().Add(-5 * time.Minute).UnixMilli())
}

func strconvFormat(value int64) string {
	return strconv.FormatInt(value, 10)
}
