// Package linear implements the Linear webhook Source.
//
// Linear signs deliveries with HMAC-SHA256 over the raw body using the webhook
// signing secret. The hex digest is sent in Linear-Signature, the entity family
// in Linear-Event, and a unique delivery UUID in Linear-Delivery.
package linear

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

const defaultToleranceSeconds = 60

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "linear" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "tolerance_seconds":{"type":"integer","minimum":0,"default":60,"description":"Max age of Linear webhookTimestamp before rejection"}
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
		return fmt.Errorf("invalid linear config: %w", err)
	}
	if parsed.ToleranceSeconds != nil && *parsed.ToleranceSeconds < 0 {
		return errors.New("tolerance_seconds must be >= 0")
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("linear: missing webhook signing secret")
	}
	signature := req.Headers.Get("Linear-Signature")
	if signature == "" {
		return nil, errors.New("linear: missing Linear-Signature header")
	}
	if err := verifySignature(req.Body, signature, secret); err != nil {
		return nil, err
	}

	var payload struct {
		Action           string          `json:"action"`
		Type             string          `json:"type"`
		Actor            json.RawMessage `json:"actor"`
		CreatedAt        string          `json:"createdAt"`
		Data             json.RawMessage `json:"data"`
		URL              string          `json:"url"`
		UpdatedFrom      json.RawMessage `json:"updatedFrom"`
		WebhookID        string          `json:"webhookId"`
		WebhookTimestamp int64           `json:"webhookTimestamp"`
	}
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("linear: invalid event JSON: %w", err)
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
		if payload.WebhookTimestamp == 0 {
			return nil, errors.New("linear: missing webhookTimestamp")
		}
		sentAt := time.UnixMilli(payload.WebhookTimestamp)
		if diff := time.Since(sentAt); diff > time.Duration(tolerance)*time.Second || diff < -time.Duration(tolerance)*time.Second {
			return nil, errors.New("linear: webhookTimestamp outside tolerance window")
		}
	}

	eventType := normalizedEventType(firstNonBlank(payload.Type, req.Headers.Get("Linear-Event")), payload.Action)
	normalized, _ := json.Marshal(map[string]any{
		"action":            payload.Action,
		"type":              payload.Type,
		"actor":             rawOrNil(payload.Actor),
		"created_at":        payload.CreatedAt,
		"data":              rawOrNil(payload.Data),
		"url":               payload.URL,
		"updated_from":      rawOrNil(payload.UpdatedFrom),
		"webhook_id":        payload.WebhookID,
		"webhook_timestamp": payload.WebhookTimestamp,
		"linear": map[string]string{
			"delivery": req.Headers.Get("Linear-Delivery"),
			"event":    req.Headers.Get("Linear-Event"),
		},
	})

	// Determine idempotency key: use Linear-Delivery if present, otherwise compute hash
	idempotencyKey := req.Headers.Get("Linear-Delivery")
	if idempotencyKey == "" {
		idempotencyKey = deliveryHash(payload.WebhookID, payload.WebhookTimestamp, payload.Type, payload.Action)
	}

	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: idempotencyKey,
		Raw:            req.Body,
		Normalized:     normalized,
	}}, nil
}

func verifySignature(body []byte, header, secret string) error {
	got, err := hex.DecodeString(strings.TrimSpace(header))
	if err != nil {
		return fmt.Errorf("linear: invalid Linear-Signature header: %w", err)
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	if !hmac.Equal(got, mac.Sum(nil)) {
		return errors.New("linear: signature mismatch")
	}
	return nil
}

func deliveryHash(webhookID string, ts int64, entityType, action string) string {
	h := sha256.New()
	h.Write([]byte(webhookID))
	h.Write([]byte{0})
	h.Write([]byte(strconv.FormatInt(ts, 10)))
	h.Write([]byte{0})
	h.Write([]byte(entityType))
	h.Write([]byte{0})
	h.Write([]byte(action))
	return hex.EncodeToString(h.Sum(nil))
}

func normalizedEventType(entityType, action string) string {
	entityType = strings.ToLower(strings.TrimSpace(entityType))
	action = strings.ToLower(strings.TrimSpace(action))
	if entityType == "" {
		return action
	}
	if action == "" {
		return entityType
	}
	return entityType + "." + action
}

func rawOrNil(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
