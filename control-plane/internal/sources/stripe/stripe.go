// Package stripe implements the Stripe webhook Source.
//
// Verification follows Stripe's documented scheme: the Stripe-Signature header
// carries a timestamp and one or more v1 HMAC-SHA256 signatures of
// "<timestamp>.<raw_body>" computed with the endpoint's signing secret.
package stripe

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

const (
	defaultToleranceSeconds = 300
)

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string { return "stripe" }

func (s *source) Kind() sources.Kind { return sources.KindHTTP }

func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "tolerance_seconds":{"type":"integer","minimum":0,"default":300,"description":"Max age of signed timestamp before rejection"}
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
		return fmt.Errorf("invalid stripe config: %w", err)
	}
	if parsed.ToleranceSeconds != nil && *parsed.ToleranceSeconds < 0 {
		return errors.New("tolerance_seconds must be >= 0")
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("stripe: missing webhook secret")
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

	sig := req.Headers.Get("Stripe-Signature")
	if sig == "" {
		return nil, errors.New("stripe: missing Stripe-Signature header")
	}
	if err := verifySignature(req.Body, sig, secret, tolerance, time.Now()); err != nil {
		return nil, err
	}

	var envelope struct {
		ID   string          `json:"id"`
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(req.Body, &envelope); err != nil {
		return nil, fmt.Errorf("stripe: invalid event JSON: %w", err)
	}

	normalized, _ := json.Marshal(map[string]any{
		"id":   envelope.ID,
		"type": envelope.Type,
		"data": envelope.Data,
	})

	return []sources.Event{{
		Type:           envelope.Type,
		IdempotencyKey: envelope.ID,
		Raw:            req.Body,
		Normalized:     normalized,
	}}, nil
}

// verifySignature parses the Stripe-Signature header and confirms at least one
// v1 signature matches HMAC-SHA256(secret, "<t>.<body>"). It also enforces the
// tolerance window so replayed signatures past their freshness are rejected.
func verifySignature(body []byte, header, secret string, toleranceSeconds int, now time.Time) error {
	var (
		ts  int64
		sigs []string
	)
	for _, part := range strings.Split(header, ",") {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) != 2 {
			continue
		}
		switch kv[0] {
		case "t":
			parsed, err := strconv.ParseInt(kv[1], 10, 64)
			if err != nil {
				return fmt.Errorf("stripe: invalid timestamp: %w", err)
			}
			ts = parsed
		case "v1":
			sigs = append(sigs, kv[1])
		}
	}
	if ts == 0 || len(sigs) == 0 {
		return errors.New("stripe: signature header missing required fields")
	}
	if toleranceSeconds > 0 {
		if diff := now.Unix() - ts; diff > int64(toleranceSeconds) || diff < -int64(toleranceSeconds) {
			return errors.New("stripe: signature timestamp outside tolerance window")
		}
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(strconv.FormatInt(ts, 10)))
	mac.Write([]byte("."))
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))
	for _, sig := range sigs {
		if hmac.Equal([]byte(sig), []byte(expected)) {
			return nil
		}
	}
	return errors.New("stripe: signature mismatch")
}
