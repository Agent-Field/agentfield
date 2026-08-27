package packages

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const healthIdentityTimeout = 2 * time.Second

// HealthIdentity is the ownership information returned by a node health
// endpoint. Healthy remains true when the endpoint answers successfully but
// omits node_id, which lets callers distinguish an occupied port from a silent
// one without trusting the listener as the recorded node.
type HealthIdentity struct {
	Healthy bool
	NodeID  string
}

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

// ProbeHealthIdentity asks the manifest health endpoint for its node ID and
// falls back to /health when a custom endpoint is silent or does not identify
// the node. The entire probe is bounded so status reads and maintenance cannot
// hang behind an unresponsive local listener.
func ProbeHealthIdentity(ctx context.Context, port int, healthPath string) HealthIdentity {
	probeCtx, cancel := context.WithTimeout(ctx, healthIdentityTimeout)
	defer cancel()

	path := normalizeHealthPath(healthPath)
	identity := probeHealthPath(probeCtx, port, path)
	if identity.NodeID != "" || path == "/health" {
		return identity
	}
	fallback := probeHealthPath(probeCtx, port, "/health")
	if fallback.Healthy {
		return fallback
	}
	return identity
}

func normalizeHealthPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "/health"
	}
	if !strings.HasPrefix(path, "/") {
		return "/" + path
	}
	return path
}

func probeHealthPath(ctx context.Context, port int, path string) HealthIdentity {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d%s", port, path), nil)
	if err != nil {
		return HealthIdentity{}
	}
	response, err := NewNodeHTTPClient(healthIdentityTimeout).Do(request)
	if err != nil {
		return HealthIdentity{}
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return HealthIdentity{}
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return HealthIdentity{}
	}
	return HealthIdentity{Healthy: true, NodeID: HealthNodeID(body)}
}
