// Package slack implements the Slack Events API Source.
//
// Slack signs requests with HMAC-SHA256 over "v0:<timestamp>:<body>" using the
// signing secret. The signature lives in X-Slack-Signature ("v0=<hex>") and
// the timestamp in X-Slack-Request-Timestamp.
//
// Slack also issues url_verification challenges during endpoint setup; the
// Source short-circuits these by emitting a synthetic event the dispatcher can
// recognize, but production deployments typically respond to the challenge at
// the HTTP layer. Here we surface the challenge as a normal event so it's
// auditable and the agent can choose to ignore it.
package slack

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

const defaultToleranceSeconds = 300

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "slack" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "tolerance_seconds":{"type":"integer","minimum":0,"default":300}
        },
        "additionalProperties": false
    }`)
}

func (s *source) Validate(cfg json.RawMessage) error {
	if len(cfg) == 0 {
		return nil
	}
	var parsed struct {
		ToleranceSeconds *int `json:"tolerance_seconds"`
	}
	if err := json.Unmarshal(cfg, &parsed); err != nil {
		return fmt.Errorf("invalid slack config: %w", err)
	}
	if parsed.ToleranceSeconds != nil && *parsed.ToleranceSeconds < 0 {
		return errors.New("tolerance_seconds must be >= 0")
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("slack: missing signing secret")
	}
	tolerance := defaultToleranceSeconds
	if len(cfg) > 0 {
		var parsed struct {
			ToleranceSeconds *int `json:"tolerance_seconds"`
		}
		if err := json.Unmarshal(cfg, &parsed); err == nil && parsed.ToleranceSeconds != nil {
			tolerance = *parsed.ToleranceSeconds
		}
	}

	tsHeader := req.Headers.Get("X-Slack-Request-Timestamp")
	sig := req.Headers.Get("X-Slack-Signature")
	if tsHeader == "" || sig == "" {
		return nil, errors.New("slack: missing signature headers")
	}
	ts, err := strconv.ParseInt(tsHeader, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("slack: invalid timestamp: %w", err)
	}
	if tolerance > 0 {
		if diff := time.Now().Unix() - ts; diff > int64(tolerance) || diff < -int64(tolerance) {
			return nil, errors.New("slack: timestamp outside tolerance window")
		}
	}
	if !strings.HasPrefix(sig, "v0=") {
		return nil, errors.New("slack: signature missing v0 prefix")
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(fmt.Sprintf("v0:%d:", ts)))
	mac.Write(req.Body)
	expected := "v0=" + hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(sig), []byte(expected)) {
		return nil, errors.New("slack: signature mismatch")
	}

	var envelope struct {
		Type      string          `json:"type"`
		EventID   string          `json:"event_id"`
		Challenge string          `json:"challenge"`
		Event     json.RawMessage `json:"event"`
	}
	if err := json.Unmarshal(req.Body, &envelope); err != nil {
		return nil, fmt.Errorf("slack: invalid event JSON: %w", err)
	}

	eventType := envelope.Type
	if envelope.Type == "event_callback" {
		var inner struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(envelope.Event, &inner)
		if inner.Type != "" {
			eventType = inner.Type
		}
	}

	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: envelope.EventID,
		Raw:            req.Body,
		Normalized:     req.Body,
	}}, nil
}
