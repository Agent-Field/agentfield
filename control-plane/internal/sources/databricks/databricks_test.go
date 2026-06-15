package databricks

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func TestMetadataAndSchema(t *testing.T) {
	s := &source{}
	if s.Name() != "databricks" {
		t.Fatalf("Name() = %q, want databricks", s.Name())
	}
	if s.Kind() != sources.KindHTTP {
		t.Fatalf("Kind() = %v, want http", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("databricks should require a secret")
	}
	var schema map[string]any
	if err := json.Unmarshal(s.ConfigSchema(), &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
}

func TestValidateConfig(t *testing.T) {
	valid := []byte(`{"auth_mode":"basic","event_type_path":"run_state.life_cycle_state","event_id_path":"run_id"}`)
	if err := (&source{}).Validate(valid); err != nil {
		t.Fatalf("Validate(valid) = %v", err)
	}
	for _, raw := range []string{
		`{`,
		`{"mode":"poll"}`,
		`{"auth_mode":"none"}`,
		`{"event_type_path":"bad..path"}`,
	} {
		if err := (&source{}).Validate([]byte(raw)); err == nil {
			t.Fatalf("Validate(%s) expected error", raw)
		}
	}
}

func TestHandleRequestBasicAuthNormalizesJobNotification(t *testing.T) {
	body := []byte(`{"run_id":123,"run_state":{"life_cycle_state":"TERMINATED"},"workspace_id":"7474"}`)
	headers := http.Header{}
	reqForAuth := &http.Request{Header: headers}
	reqForAuth.SetBasicAuth("agentfield", "secret")
	req := &sources.RawRequest{
		Headers: headers,
		Body:    body,
		URL:     mustURL(t, "https://agentfield.test/webhooks/databricks"),
		Method:  http.MethodPost,
	}
	events, err := (&source{}).HandleRequest(context.Background(), req, json.RawMessage(`{
		"basic_username":"agentfield",
		"event_type_path":"run_state.life_cycle_state",
		"event_id_path":"run_id",
		"workspace_path":"workspace_id"
	}`), "secret")
	if err != nil {
		t.Fatalf("HandleRequest: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	if events[0].Type != "TERMINATED" || events[0].IdempotencyKey != "123" {
		t.Fatalf("bad event metadata: %+v", events[0])
	}
	var normalized map[string]any
	if err := json.Unmarshal(events[0].Normalized, &normalized); err != nil {
		t.Fatalf("normalized JSON: %v", err)
	}
	db := normalized["databricks"].(map[string]any)
	if db["workspace"] != "7474" || db["auth_mode"] != "basic" {
		t.Fatalf("bad databricks metadata: %#v", db)
	}
}

func TestHandleRequestBearerAuthAndFallbackID(t *testing.T) {
	headers := http.Header{"Authorization": []string{"Bearer token"}}
	events, err := (&source{}).HandleRequest(context.Background(), &sources.RawRequest{
		Headers: headers,
		Body:    []byte(`{"status":"TRIGGERED","message":"alert"}`),
		Method:  http.MethodPost,
	}, json.RawMessage(`{"auth_mode":"bearer"}`), "token")
	if err != nil {
		t.Fatalf("HandleRequest: %v", err)
	}
	if events[0].Type != "TRIGGERED" || !strings.HasPrefix(events[0].IdempotencyKey, "databricks_") {
		t.Fatalf("bad fallback metadata: %+v", events[0])
	}
}

func TestHandleRequestRejectsBadAuthAndBody(t *testing.T) {
	_, err := (&source{}).HandleRequest(context.Background(), &sources.RawRequest{
		Headers: http.Header{"Authorization": []string{"Bearer nope"}},
		Body:    []byte(`{}`),
		Method:  http.MethodPost,
	}, json.RawMessage(`{"auth_mode":"bearer"}`), "token")
	if err == nil || !strings.Contains(err.Error(), "invalid bearer") {
		t.Fatalf("bad bearer error = %v", err)
	}
	_, err = (&source{}).HandleRequest(context.Background(), &sources.RawRequest{
		Headers: http.Header{"Authorization": []string{"Bearer token"}},
		Body:    []byte(`not-json`),
		Method:  http.MethodPost,
	}, json.RawMessage(`{"auth_mode":"bearer"}`), "token")
	if err == nil || !strings.Contains(err.Error(), "must be JSON") {
		t.Fatalf("bad body error = %v", err)
	}
}

func mustURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse URL: %v", err)
	}
	return u
}
