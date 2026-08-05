package launchdsvc

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// probeTimeout keeps an install responsive when nothing is listening. The
// endpoints are local and answer in milliseconds; a longer wait would only
// stall `curl … | bash` behind a dead socket.
const probeTimeout = 1500 * time.Millisecond

// defaultActiveWindow is how recently a run must have done something to count
// as genuinely in flight.
//
// A live workflow touches latest_activity on every reasoner event, so a healthy
// run refreshes it constantly. A run that has not moved in half an hour is
// either wedged or was abandoned by a server that died mid-flight — and with
// execution cleanup disabled it can sit in the active list indefinitely. One
// such zombie used to make the busy probe permanently true, which meant an
// upgrade of the SAME install would defer its restart forever and never pick up
// a new binary.
const defaultActiveWindow = 30 * time.Minute

// ActiveWindowEnv overrides defaultActiveWindow with a Go duration ("2h").
const ActiveWindowEnv = "AGENTFIELD_INSTALL_ACTIVE_WINDOW"

// ActiveWindow resolves the freshness window. An unparseable or non-positive
// value falls back to the default rather than failing an install.
func ActiveWindow() time.Duration {
	raw := strings.TrimSpace(os.Getenv(ActiveWindowEnv))
	if raw == "" {
		return defaultActiveWindow
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return defaultActiveWindow
	}
	return d
}

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
	Count int         `json:"count"`
	Runs  []activeRun `json:"runs"`
}

type activeRun struct {
	RunID          string `json:"run_id"`
	RootStatus     string `json:"root_status"`
	StartedAt      string `json:"started_at"`
	LatestActivity string `json:"latest_activity"`
}

// fresh reports whether this run has done something inside the window, and is
// therefore work an install must not interrupt.
//
// The rule when evidence is missing runs toward STALE: this fleet has a
// history of runs wedged in "running" while doing nothing, and a run that
// cannot demonstrate liveness must not pin upgrades forever — that is the
// exact bug the staleness split exists to fix. A run with no usable
// latest_activity gets one fallback, its start time, so a demonstrably young
// run (an older server that reports no activity stamps) is still protected;
// with no usable evidence at all it is assumed stale.
func (r activeRun) fresh(now time.Time, window time.Duration) bool {
	if t, ok := parseStamp(r.LatestActivity); ok {
		return now.Sub(t) <= window
	}
	if t, ok := parseStamp(r.StartedAt); ok {
		return now.Sub(t) <= window
	}
	return false
}

// parseStamp parses an RFC3339 timestamp, reporting ok=false for the missing
// and malformed shapes fresh() must fall back on.
func parseStamp(stamp string) (time.Time, bool) {
	stamp = strings.TrimSpace(stamp)
	if stamp == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, stamp)
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

// ActiveExecutions reports how many runs are genuinely in flight (fresh), how
// many are listed but have gone quiet (stale, and therefore ignored), and
// whether the answer is trustworthy at all. A server that does not answer, or
// answers with an auth error, reports ok=false — and the caller then treats the
// server as not-busy rather than blocking an install forever on an unreadable
// endpoint.
func ActiveExecutions(port int, apiKey string) (fresh, stale int, ok bool) {
	client := &http.Client{Timeout: probeTimeout}
	req, err := http.NewRequest(http.MethodGet, ActiveExecutionsURL(port), nil)
	if err != nil {
		return 0, 0, false
	}
	if apiKey != "" {
		req.Header.Set("X-API-Key", apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, false
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return 0, 0, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, 0, false
	}
	var parsed activeExecutionsResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 0, 0, false
	}
	return splitActiveRuns(parsed, time.Now(), ActiveWindow())
}

// splitActiveRuns is the pure half of the probe, so the staleness rules are
// testable without a clock or a socket.
func splitActiveRuns(parsed activeExecutionsResponse, now time.Time, window time.Duration) (fresh, stale int, ok bool) {
	for _, run := range parsed.Runs {
		if run.fresh(now, window) {
			fresh++
			continue
		}
		stale++
	}
	// An older server may report a count without the run detail needed to age
	// it. Unconfirmable liveness counts as stale (the owner's rule): a count
	// with no evidence behind it must not pin upgrades, and the stale figure
	// keeps it visible in the install message and `af service status`.
	if len(parsed.Runs) == 0 && parsed.Count > 0 {
		return 0, parsed.Count, true
	}
	return fresh, stale, true
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
