// Package github implements the GitHub webhook Source.
//
// GitHub signs deliveries with HMAC-SHA256 over the raw body using the
// webhook secret, sent in X-Hub-Signature-256 as "sha256=<hex>". The event
// type lives in X-GitHub-Event and the unique delivery ID in X-GitHub-Delivery.
package github

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

func (s *source) Name() string         { return "github" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{},
        "additionalProperties": false
    }`)
}

func (s *source) Validate(cfg json.RawMessage) error { return nil }

func (s *source) HandleRequest(ctx context.Context, req *sources.RawRequest, cfg json.RawMessage, secret string) ([]sources.Event, error) {
	if secret == "" {
		return nil, errors.New("github: missing webhook secret")
	}
	sig := req.Headers.Get("X-Hub-Signature-256")
	if sig == "" {
		return nil, errors.New("github: missing X-Hub-Signature-256 header")
	}
	if !strings.HasPrefix(sig, "sha256=") {
		return nil, errors.New("github: signature missing sha256 prefix")
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(req.Body)
	expected := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(strings.TrimPrefix(sig, "sha256=")), []byte(expected)) {
		return nil, errors.New("github: signature mismatch")
	}

	eventType := req.Headers.Get("X-GitHub-Event")
	delivery := req.Headers.Get("X-GitHub-Delivery")

	// GitHub payloads include an "action" field for many event families; combine
	// "<event>.<action>" so reasoners can filter on specific actions like
	// "pull_request.opened" while the bare event still matches the family.
	var probe struct {
		Action string `json:"action"`
	}
	_ = json.Unmarshal(req.Body, &probe)
	if probe.Action != "" {
		eventType = fmt.Sprintf("%s.%s", eventType, probe.Action)
	}

	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: delivery,
		Raw:            req.Body,
		Normalized:     req.Body,
	}}, nil
}
