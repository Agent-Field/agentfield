package launchdsvc

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// probeTimeout keeps an install responsive when nothing is listening. The
// endpoints are local and answer in milliseconds; a longer wait would only
// stall `curl … | bash` behind a dead socket.
const probeTimeout = 1500 * time.Millisecond

// HealthURL / ActiveExecutionsURL address the local control plane.
func HealthURL(port int) string { return fmt.Sprintf("http://localhost:%d/health", port) }

// ActiveExecutionsURL matches the route registered in
// internal/server/routes_core.go: agentAPI.GET("/executions/active", …) under
// the /api/v1 group.
func ActiveExecutionsURL(port int) string {
	return fmt.Sprintf("http://localhost:%d/api/v1/executions/active", port)
}

// ServerHealthy reports whether a control plane answers /health on port.
func ServerHealthy(port int) bool {
	client := &http.Client{Timeout: probeTimeout}
	resp, err := client.Get(HealthURL(port))
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// activeExecutionsResponse is the shape of GET /api/v1/executions/active.
type activeExecutionsResponse struct {
	Count int `json:"count"`
	Runs  []struct {
		RunID string `json:"run_id"`
	} `json:"runs"`
}

// ActiveExecutions returns the number of in-flight runs, and whether the answer
// is trustworthy. A server that does not answer, or answers with an auth error,
// reports ok=false — and the caller then treats the server as not-busy rather
// than blocking an install forever on an unreadable endpoint.
func ActiveExecutions(port int, apiKey string) (int, bool) {
	client := &http.Client{Timeout: probeTimeout}
	req, err := http.NewRequest(http.MethodGet, ActiveExecutionsURL(port), nil)
	if err != nil {
		return 0, false
	}
	if apiKey != "" {
		req.Header.Set("X-API-Key", apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, false
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return 0, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, false
	}
	var parsed activeExecutionsResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 0, false
	}
	// count is authoritative; fall back to the run list when it is absent.
	if parsed.Count > 0 {
		return parsed.Count, true
	}
	return len(parsed.Runs), true
}

// FileSHA256 hashes a file, reporting ok=false when it cannot be read. Used to
// tell an upgrade from a re-run of the same version.
func FileSHA256(path string) (string, bool) {
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", false
	}
	return fmt.Sprintf("%x", h.Sum(nil)), true
}

// BytesSHA256 hashes in-memory contents for comparison against FileSHA256.
func BytesSHA256(data []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

// FileHasContents reports whether path already holds exactly data.
func FileHasContents(path string, data []byte) bool {
	sum, ok := FileSHA256(path)
	if !ok {
		return false
	}
	return sum == BytesSHA256(data)
}
