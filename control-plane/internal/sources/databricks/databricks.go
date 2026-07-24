// Package databricks implements an HTTP Source for Databricks notification
// destination webhooks and normalizes them into AgentField inbound events.
package databricks

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

const (
	modeWebhookNotification = "webhook_notification"
	authModeBasic           = "basic"
	authModeBearer          = "bearer"
)

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "databricks" }
func (s *source) Kind() sources.Kind   { return sources.KindHTTP }
func (s *source) SecretRequired() bool { return true }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "mode":{"type":"string","enum":["webhook_notification"],"default":"webhook_notification"},
          "auth_mode":{"type":"string","enum":["basic","bearer"],"default":"basic"},
          "basic_username":{"type":"string","description":"Optional expected Basic auth username. Password is read from the trigger secret."},
          "event_type_path":{"type":"string","description":"Optional dotted JSON path used as event type."},
          "event_id_path":{"type":"string","description":"Optional dotted JSON path used as idempotency key."},
          "workspace_path":{"type":"string","description":"Optional dotted JSON path for workspace URL or ID."}
        },
        "additionalProperties": false
    }`)
}

type config struct {
	Mode          string `json:"mode"`
	AuthMode      string `json:"auth_mode"`
	BasicUsername string `json:"basic_username"`
	EventTypePath string `json:"event_type_path"`
	EventIDPath   string `json:"event_id_path"`
	WorkspacePath string `json:"workspace_path"`
}

func parseConfig(raw json.RawMessage) (config, error) {
	var c config
	if len(bytes.TrimSpace(raw)) > 0 {
		if err := json.Unmarshal(raw, &c); err != nil {
			return c, fmt.Errorf("databricks: invalid config: %w", err)
		}
	}
	c.Mode = defaultIfBlank(c.Mode, modeWebhookNotification)
	c.AuthMode = defaultIfBlank(c.AuthMode, authModeBasic)
	c.BasicUsername = strings.TrimSpace(c.BasicUsername)
	c.EventTypePath = strings.TrimSpace(c.EventTypePath)
	c.EventIDPath = strings.TrimSpace(c.EventIDPath)
	c.WorkspacePath = strings.TrimSpace(c.WorkspacePath)
	return c, nil
}

func (s *source) Validate(raw json.RawMessage) error {
	c, err := parseConfig(raw)
	if err != nil {
		return err
	}
	if c.Mode != modeWebhookNotification {
		return fmt.Errorf("databricks: unsupported mode %q", c.Mode)
	}
	switch c.AuthMode {
	case authModeBasic, authModeBearer:
	default:
		return fmt.Errorf("databricks: unsupported auth_mode %q", c.AuthMode)
	}
	for name, value := range map[string]string{
		"event_type_path": c.EventTypePath,
		"event_id_path":   c.EventIDPath,
		"workspace_path":  c.WorkspacePath,
	} {
		if value != "" && !validPath(value) {
			return fmt.Errorf("databricks: %s must be a dotted JSON path", name)
		}
	}
	return nil
}

func (s *source) HandleRequest(_ context.Context, req *sources.RawRequest, raw json.RawMessage, secret string) ([]sources.Event, error) {
	if req == nil {
		return nil, errors.New("databricks: request is required")
	}
	if req.Method != "" && req.Method != http.MethodPost {
		return nil, fmt.Errorf("databricks: unsupported method %s", req.Method)
	}
	if strings.TrimSpace(secret) == "" {
		return nil, errors.New("databricks: trigger secret is required")
	}
	c, err := parseConfig(raw)
	if err != nil {
		return nil, err
	}
	if err := s.Validate(raw); err != nil {
		return nil, err
	}
	if err := verifyAuth(req.Headers, c, secret); err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(req.Body, &payload); err != nil {
		return nil, fmt.Errorf("databricks: webhook body must be JSON: %w", err)
	}
	eventType := firstNonBlank(
		pathString(payload, c.EventTypePath),
		pathString(payload, "event_type"),
		pathString(payload, "eventType"),
		pathString(payload, "type"),
		pathString(payload, "state"),
		pathString(payload, "status"),
		pathString(payload, "run_state.life_cycle_state"),
		"databricks.notification",
	)
	eventID := firstNonBlank(
		pathString(payload, c.EventIDPath),
		pathString(payload, "event_id"),
		pathString(payload, "id"),
		pathString(payload, "message_id"),
		pathString(payload, "run_id"),
		pathString(payload, "run.run_id"),
		pathString(payload, "job.run_id"),
		hashBody(req.Body),
	)
	normalized, err := json.Marshal(map[string]any{
		"event_id":   eventID,
		"event_type": eventType,
		"payload":    payload,
		"databricks": map[string]any{
			"source":       "notification_destination",
			"auth_mode":    c.AuthMode,
			"workspace":    pathString(payload, c.WorkspacePath),
			"received_at":  time.Now().UTC().Format(time.RFC3339Nano),
			"request_path": requestPath(req),
		},
	})
	if err != nil {
		return nil, err
	}
	rawBody := append([]byte(nil), req.Body...)
	return []sources.Event{{
		Type:           eventType,
		IdempotencyKey: eventID,
		Raw:            rawBody,
		Normalized:     normalized,
	}}, nil
}

func verifyAuth(headers http.Header, c config, secret string) error {
	switch c.AuthMode {
	case authModeBearer:
		got := strings.TrimSpace(headers.Get("Authorization"))
		prefix := "Bearer "
		if !strings.HasPrefix(got, prefix) {
			return errors.New("databricks: missing bearer authorization")
		}
		if !constantEqual(strings.TrimSpace(strings.TrimPrefix(got, prefix)), secret) {
			return errors.New("databricks: invalid bearer token")
		}
		return nil
	case authModeBasic:
		r := &http.Request{Header: headers}
		username, password, ok := r.BasicAuth()
		if !ok {
			return errors.New("databricks: missing basic authorization")
		}
		if c.BasicUsername != "" && !constantEqual(username, c.BasicUsername) {
			return errors.New("databricks: invalid basic username")
		}
		if !constantEqual(password, secret) {
			return errors.New("databricks: invalid basic password")
		}
		return nil
	default:
		return fmt.Errorf("databricks: unsupported auth_mode %q", c.AuthMode)
	}
}

func constantEqual(a, b string) bool {
	return hmac.Equal([]byte(a), []byte(b))
}

func defaultIfBlank(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func validPath(path string) bool {
	for _, part := range strings.Split(path, ".") {
		if strings.TrimSpace(part) == "" {
			return false
		}
	}
	return true
}

func pathString(payload map[string]any, path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	var current any = payload
	for _, part := range strings.Split(path, ".") {
		m, ok := current.(map[string]any)
		if !ok {
			return ""
		}
		current, ok = m[part]
		if !ok {
			return ""
		}
	}
	switch value := current.(type) {
	case string:
		return strings.TrimSpace(value)
	case json.Number:
		return value.String()
	case float64:
		return strconvFloat(value)
	default:
		data, _ := json.Marshal(value)
		return strings.TrimSpace(string(data))
	}
}

func strconvFloat(value float64) string {
	if value == float64(int64(value)) {
		return fmt.Sprintf("%d", int64(value))
	}
	return fmt.Sprintf("%v", value)
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func hashBody(body []byte) string {
	sum := sha256.Sum256(body)
	return "databricks_" + hex.EncodeToString(sum[:16])
}

func requestPath(req *sources.RawRequest) string {
	if req == nil || req.URL == nil {
		return ""
	}
	return req.URL.Path
}
