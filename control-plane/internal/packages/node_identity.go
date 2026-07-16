package packages

import (
	"encoding/json"
	"strings"
)

// HealthNodeID extracts the node_id field from an agent health payload.
// Returns "" when the body is not JSON or carries no node_id — custom
// healthcheck endpoints are not required to identify themselves.
func HealthNodeID(body []byte) string {
	var payload struct {
		NodeID string `json:"node_id"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return ""
	}
	return payload.NodeID
}

// NodeIDsEquivalent compares node identifiers with the same tolerance the
// registry uses for name drift: case-insensitive, hyphens and underscores
// interchangeable.
func NodeIDsEquivalent(a, b string) bool {
	fold := func(s string) string {
		return strings.ToLower(strings.ReplaceAll(s, "-", "_"))
	}
	return fold(a) == fold(b)
}
