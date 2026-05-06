// Package genericbearer implements a webhook Source that authenticates inbound
// requests by comparing a bearer token in the Authorization header to a shared
// secret. Use it for providers that don't sign payloads but expect a static
// shared token.
package genericbearer

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "generic_bearer" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "header":{"type":"string","default":"Authorization","description":"Header to read the bearer token from"},
          "scheme":{"type":"string","default":"Bearer","description":"Auth scheme prefix (set to empty string for raw tokens)"},
          "event_type_header":{"type":"string","default":""},
          "idempotency_header":{"type":"string","default":""}
        },
        "additionalProperties": false
    }`)
}

type config struct {
	Header            string `json:"header"`
	Scheme            string `json:"scheme"`
	EventTypeHeader   string `json:"event_type_header"`
	IdempotencyHeader string `json:"idempotency_header"`
}

func parseConfig(cfg json.RawMessage) config {
	c := config{Header: "Authorization", Scheme: "Bearer"}
	if len(cfg) == 0 {
		return c
	}
	_ = json.Unmarshal(cfg, &c)
	if c.Header == "" {
		c.Header = "Authorization"
	}
	return c
}

func (s *source) Validate(cfg json.RawMessage) error {
	if len(cfg) == 0 {
		return nil
	}
	var c config
	if err := json.Unmarshal(cfg, &c); err != nil {
		return errors.New("invalid generic_bearer config")
	}
	return nil
}

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("generic_bearer: missing secret")
	}
	c := parseConfig(cfg)

	provided := req.Headers.Get(c.Header)
	if provided == "" {
		return nil, errors.New("generic_bearer: missing auth header")
	}
	if c.Scheme != "" {
		prefix := c.Scheme + " "
		if !strings.HasPrefix(provided, prefix) {
			return nil, errors.New("generic_bearer: auth header missing scheme prefix")
		}
		provided = strings.TrimPrefix(provided, prefix)
	}
	if subtle.ConstantTimeCompare([]byte(provided), []byte(secret)) != 1 {
		return nil, errors.New("generic_bearer: token mismatch")
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
