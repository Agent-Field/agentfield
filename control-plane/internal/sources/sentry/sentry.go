// Package sentry implements the Sentry integration-platform webhook Source.
//
// Sentry sends JSON webhooks with Sentry-Hook-Resource, Request-ID,
// Sentry-Hook-Timestamp, and Sentry-Hook-Signature headers. The signature is
// an HMAC-SHA256 digest computed with the integration Client Secret.
package sentry

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

func (s *source) Name() string         { return "sentry" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "tolerance_seconds":{"type":"integer","minimum":0,"default":300,"description":"Max age of Sentry-Hook-Timestamp before rejection. Set to 0 to disable."}
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
		return fmt.Errorf("invalid sentry config: %w", err)
	}
	if parsed.ToleranceSeconds != nil && *parsed.ToleranceSeconds < 0 {
		return errors.New("tolerance_seconds must be >= 0")
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("sentry: missing client secret")
	}
	signature := req.Headers.Get("Sentry-Hook-Signature")
	if signature == "" {
		return nil, errors.New("sentry: missing Sentry-Hook-Signature header")
	}
	if err := verifySignature(req.Body, signature, secret); err != nil {
		return nil, err
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
	if tolerance > 0 {
		timestamp := req.Headers.Get("Sentry-Hook-Timestamp")
		if timestamp == "" {
			return nil, errors.New("sentry: missing or invalid Sentry-Hook-Timestamp")
		}
		parsedTime, err := parseTimestamp(timestamp)
		if err != nil {
			return nil, errors.New("sentry: missing or invalid Sentry-Hook-Timestamp")
		}
		if diff := time.Since(parsedTime); diff > time.Duration(tolerance)*time.Second || diff < -time.Duration(tolerance)*time.Second {
			return nil, errors.New("sentry: Sentry-Hook-Timestamp outside tolerance window")
		}
	}

	var payload struct {
		Action       string          `json:"action"`
		Installation json.RawMessage `json:"installation"`
		Data         json.RawMessage `json:"data"`
		Actor        json.RawMessage `json:"actor"`
	}
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("sentry: invalid event JSON: %w", err)
	}
	resource := strings.ToLower(strings.TrimSpace(req.Headers.Get("Sentry-Hook-Resource")))
	eventType := resource
	if action := strings.ToLower(strings.TrimSpace(payload.Action)); action != "" {
		if eventType == "" {
			eventType = action
		} else {
			eventType += "." + action
		}
	}
	requestID := req.Headers.Get("Request-ID")
	if requestID == "" {
		requestID = bodyDigest(resource, req.Body)
	}
	normalized, _ := json.Marshal(map[string]any{
		"action":       payload.Action,
		"resource":     resource,
		"installation": rawOrNil(payload.Installation),
		"data":         rawOrNil(payload.Data),
		"actor":        rawOrNil(payload.Actor),
		"sentry": map[string]string{
			"request_id": requestID,
			"timestamp":  req.Headers.Get("Sentry-Hook-Timestamp"),
			"resource":   req.Headers.Get("Sentry-Hook-Resource"),
		},
	})

	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: requestID,
		Raw:            req.Body,
		Normalized:     normalized,
	}}, nil
}

func verifySignature(body []byte, header, secret string) error {
	got, err := hex.DecodeString(strings.TrimSpace(header))
	if err != nil {
		return fmt.Errorf("sentry: invalid Sentry-Hook-Signature header: %w", err)
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	if !hmac.Equal(got, mac.Sum(nil)) {
		return errors.New("sentry: signature mismatch")
	}
	return nil
}

func parseTimestamp(ts string) (time.Time, error) {
	// Try RFC3339 first
	if t, err := time.Parse(time.RFC3339, ts); err == nil {
		return t, nil
	}

	// Try Unix seconds/milliseconds as string
	if secs, err := strconv.ParseInt(ts, 10, 64); err == nil {
		// Check if it's milliseconds (> 10^12)
		if secs > 1e12 {
			return time.UnixMilli(secs), nil
		}
		return time.Unix(secs, 0), nil
	}

	return time.Time{}, errors.New("invalid timestamp format")
}

func bodyDigest(resource string, body []byte) string {
	mac := sha256.New()
	mac.Write([]byte(resource))
	mac.Write([]byte{0})
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

func rawOrNil(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}
