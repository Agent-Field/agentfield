package types

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
)

// RunMetadataNamespace is the namespace key inside workflow_runs.metadata.
const RunMetadataNamespace = "run"

const (
	MaxRunDisplayNameRunes   = 200
	MaxRunLabels             = 20
	MaxRunLabelRunes         = 64
	MaxRunLinks              = 10
	MaxRunLinkLabelRunes     = 64
	MaxRunLinkURLBytes       = 2048
	MaxRunMetadataSetByRunes = 200
)

type RunMetadataLink struct {
	Label string `json:"label,omitempty"`
	URL   string `json:"url"`
}

type RunMetadata struct {
	DisplayName string            `json:"display_name,omitempty"`
	Labels      []string          `json:"labels,omitempty"`
	Links       []RunMetadataLink `json:"links,omitempty"`
	SetBy       string            `json:"set_by,omitempty"`
	UpdatedAt   string            `json:"updated_at,omitempty"`
}

// ParseRunMetadata decodes workflow_runs.metadata and returns its run namespace.
func ParseRunMetadata(raw json.RawMessage) *RunMetadata {
	var namespaces map[string]json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &namespaces) != nil {
		return nil
	}
	value, ok := namespaces[RunMetadataNamespace]
	if !ok {
		return nil
	}
	value = bytes.TrimSpace(value)
	if len(value) == 0 || value[0] != '{' {
		return nil
	}
	var metadata RunMetadata
	if json.Unmarshal(value, &metadata) != nil {
		return nil
	}
	return &metadata
}

// ValidateRunLinkURL permits ordinary HTTP(S) links with a host and no credentials.
// services.ValidateWebhookURL is deliberately not used: its private-address blocking
// would reject legitimate internal PR and ticket links.
func ValidateRunLinkURL(raw string) error {
	if len(raw) > MaxRunLinkURLBytes {
		return fmt.Errorf("url exceeds %d bytes", MaxRunLinkURLBytes)
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("url scheme must be http or https")
	}
	if parsed.Host == "" {
		return fmt.Errorf("url must include a host")
	}
	if parsed.User != nil {
		return fmt.Errorf("url must not include credentials")
	}
	return nil
}
