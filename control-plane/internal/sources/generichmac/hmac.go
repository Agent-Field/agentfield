// Package generichmac implements a configurable HMAC-SHA256 webhook Source.
//
// Use it for providers whose signing scheme is "HMAC of the raw body" with a
// configurable header and optional prefix. The config selects the header name
// and prefix; the secret comes from the trigger's secret_env_var.
package generichmac

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "generic_hmac" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "signature_header":{"type":"string","default":"X-Signature","description":"Header carrying the HMAC-SHA256 hex digest"},
          "signature_prefix":{"type":"string","default":"","description":"Optional prefix on the signature value, e.g. 'sha256='"},
          "event_type_header":{"type":"string","default":"","description":"Optional header to copy into the event type"},
          "idempotency_header":{"type":"string","default":"","description":"Optional header to use as the idempotency key"}
        },
        "additionalProperties": false
    }`)
}

type config struct {
	SignatureHeader   string `json:"signature_header"`
	SignaturePrefix   string `json:"signature_prefix"`
	EventTypeHeader   string `json:"event_type_header"`
	IdempotencyHeader string `json:"idempotency_header"`
}

func parseConfig(cfg json.RawMessage) config {
	c := config{SignatureHeader: "X-Signature"}
	if len(cfg) == 0 {
		return c
	}
	_ = json.Unmarshal(cfg, &c)
	if c.SignatureHeader == "" {
		c.SignatureHeader = "X-Signature"
	}
	return c
}

func (s *source) Validate(cfg json.RawMessage) error {
	if len(cfg) == 0 {
		return nil
	}
	var c config
	if err := json.Unmarshal(cfg, &c); err != nil {
		return fmt.Errorf("invalid generic_hmac config: %w", err)
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("generic_hmac: missing secret")
	}
	c := parseConfig(cfg)

	provided := req.Headers.Get(c.SignatureHeader)
	if provided == "" {
		return nil, fmt.Errorf("generic_hmac: missing signature header %q", c.SignatureHeader)
	}
	if c.SignaturePrefix != "" {
		if !strings.HasPrefix(provided, c.SignaturePrefix) {
			return nil, errors.New("generic_hmac: signature missing configured prefix")
		}
		provided = strings.TrimPrefix(provided, c.SignaturePrefix)
	}

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(req.Body)
	expected := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(provided), []byte(expected)) {
		return nil, errors.New("generic_hmac: signature mismatch")
	}

	eventType := ""
	if c.EventTypeHeader != "" {
		eventType = req.Headers.Get(c.EventTypeHeader)
	}
	idempotency := ""
	if c.IdempotencyHeader != "" {
		idempotency = req.Headers.Get(c.IdempotencyHeader)
	}

	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: idempotency,
		Raw:            req.Body,
		Normalized:     req.Body,
	}}, nil
}
