package main

// This file holds the platform-neutral logic behind the tray: health polling,
// path resolution, launchd plist / Info.plist generation, launchctl argument
// construction, and atomic file writes. It has NO GUI (systray/CGO) dependency
// and compiles on every platform, so it can be unit-tested directly in CI
// (which runs on Linux). The OS-specific glue — the systray event loop and the
// exec.Command("launchctl", …) calls — lives in the _darwin files and calls
// into these helpers.

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	trayLabel   = "ai.agentfield.tray"
	serverLabel = "ai.agentfield.server"
)

// ---- Paths -----------------------------------------------------------------

func home() string {
	h, _ := os.UserHomeDir()
	return h
}

func agentfieldDir() string   { return filepath.Join(home(), ".agentfield") }
func binDir() string          { return filepath.Join(agentfieldDir(), "bin") }
func logsDir() string         { return filepath.Join(agentfieldDir(), "logs") }
func launchAgentsDir() string { return filepath.Join(home(), "Library", "LaunchAgents") }
func appBundleDir() string    { return filepath.Join(home(), "Applications", "AgentField.app") }
func serverLogPath() string   { return filepath.Join(logsDir(), "control-plane.log") }
func trayLogPath() string     { return filepath.Join(logsDir(), "tray.log") }
func trayPlistPath() string   { return filepath.Join(launchAgentsDir(), trayLabel+".plist") }
func serverPlistPath() string { return filepath.Join(launchAgentsDir(), serverLabel+".plist") }

// credentialsPath is where the tray persists an API key entered by the user.
// It is written 0600 and is deliberately separate from any server config: the
// server may receive its key via env/config that the tray's launchd context
// cannot see, so the tray keeps its own copy for talking to the local API.
func credentialsPath() string { return filepath.Join(agentfieldDir(), "tray-apikey") }

func trayBundleBinaryPath() string {
	return filepath.Join(appBundleDir(), "Contents", "MacOS", "af-tray")
}

// serverBinaryPath finds the control-plane binary the launchd agent should run.
// It prefers the installed copy, then falls back to whatever is on PATH.
func serverBinaryPath() string {
	cand := filepath.Join(binDir(), "agentfield")
	if isExecutable(cand) {
		return cand
	}
	if p, err := exec.LookPath("af"); err == nil {
		return p
	}
	if p, err := exec.LookPath("agentfield"); err == nil {
		return p
	}
	return cand // best effort; may not exist yet.
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Mode()&0o111 != 0
}

// ---- Health / URLs ---------------------------------------------------------

// serverPort returns the port the control plane is expected to listen on.
func serverPort() int {
	if v := os.Getenv("AGENTFIELD_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			return p
		}
	}
	return 8080
}

func healthURL() string    { return fmt.Sprintf("http://localhost:%d/health", serverPort()) }
func dashboardURL() string { return fmt.Sprintf("http://localhost:%d", serverPort()) }

// uiPageURL deep-links to a page in the embedded web UI (served under /ui/), so
// a metric row can open the dashboard view it summarizes.
func uiPageURL(page string) string {
	return fmt.Sprintf("http://localhost:%d/ui/%s", serverPort(), page)
}

// ---- Desktop app deep links --------------------------------------------------
//
// The AgentField desktop app (desktop/ in this repo) registers the
// agentfield:// URL scheme. When it is installed, the tray opens views there;
// when it is not, `open agentfield://…` fails fast (no handler registered)
// and the tray falls back to the web UI in the browser (see openPage in
// tray_darwin.go).

// deepLinkForPage maps a web-UI page name to the desktop-app view showing the
// same thing. Unknown/empty pages land on the app's dashboard.
func deepLinkForPage(page string) string {
	switch page {
	case "agents":
		return "agentfield://agents"
	case "executions":
		return "agentfield://activity"
	default:
		return "agentfield://dashboard"
	}
}

// browserURLForPage is the web-UI fallback for the same page: the server root
// for the default page, /ui/<page> otherwise.
func browserURLForPage(page string) string {
	if page == "" {
		return dashboardURL()
	}
	return uiPageURL(page)
}

// checkHealth reports whether the given URL answers HTTP 200 within a short
// timeout. The control plane's /health endpoint returns 200 when healthy and
// 503 when not, so only a 200 counts as "running".
func checkHealth(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

func serverHealthy() bool { return checkHealth(healthURL()) }

// ---- Fleet (registered agents) ---------------------------------------------

// nodesURL lists every registered node (show_all=true bypasses the default
// active-only filter so we can report online vs. total).
func nodesURL() string {
	return fmt.Sprintf("http://localhost:%d/api/v1/nodes?show_all=true", serverPort())
}

// fleetStatus is the outcome of trying to read the fleet from the control plane.
type fleetStatus int

const (
	fleetOK           fleetStatus = iota // agents read successfully
	fleetAuthRequired                    // server demands an API key we don't have (or ours was rejected)
	fleetUnavailable                     // server unreachable / unexpected response
)

// agentInfo is the slice of a registered node the tray cares about.
type agentInfo struct {
	ID        string
	Online    bool
	Skills    int
	Reasoners int
	Group     string
	Version   string
}

// fleetSummary is the digest the tray renders: counts plus the agent list.
type fleetSummary struct {
	Status fleetStatus
	Online int
	Total  int
	Skills int // total skills + reasoners across all agents
	Agents []agentInfo
}

// parseNodes extracts the agent list from a GET /api/v1/nodes response body.
func parseNodes(body []byte) ([]agentInfo, error) {
	var payload struct {
		Nodes []struct {
			ID           string            `json:"id"`
			HealthStatus string            `json:"health_status"`
			GroupID      string            `json:"group_id"`
			Version      string            `json:"version"`
			Skills       []json.RawMessage `json:"skills"`
			Reasoners    []json.RawMessage `json:"reasoners"`
		} `json:"nodes"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	agents := make([]agentInfo, 0, len(payload.Nodes))
	for _, n := range payload.Nodes {
		agents = append(agents, agentInfo{
			ID:        n.ID,
			Online:    n.HealthStatus == "active",
			Skills:    len(n.Skills),
			Reasoners: len(n.Reasoners),
			Group:     n.GroupID,
			Version:   n.Version,
		})
	}
	return agents, nil
}

// summarizeFleet rolls a parsed agent list up into counts. Skills are summed
// over online agents only — offline agents' capabilities aren't callable right
// now, so counting them would overstate what's actually available.
func summarizeFleet(agents []agentInfo) fleetSummary {
	s := fleetSummary{Status: fleetOK, Total: len(agents), Agents: agents}
	for _, a := range agents {
		if a.Online {
			s.Online++
			s.Skills += a.Skills + a.Reasoners
		}
	}
	return s
}

// fetchFleet reads the fleet from the local control plane, authenticating with
// apiKey when non-empty. A 401/403 becomes fleetAuthRequired so the tray can
// prompt for (or re-prompt for) a key; anything else unexpected is
// fleetUnavailable and rendered as a transient "unavailable" state.
func fetchFleet(apiKey string) fleetSummary {
	req, err := http.NewRequest(http.MethodGet, nodesURL(), nil)
	if err != nil {
		return fleetSummary{Status: fleetUnavailable}
	}
	if apiKey != "" {
		req.Header.Set("X-API-Key", apiKey)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fleetSummary{Status: fleetUnavailable}
	}
	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusOK:
		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			return fleetSummary{Status: fleetUnavailable}
		}
		agents, err := parseNodes(body)
		if err != nil {
			return fleetSummary{Status: fleetUnavailable}
		}
		return summarizeFleet(agents)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fleetSummary{Status: fleetAuthRequired}
	default:
		return fleetSummary{Status: fleetUnavailable}
	}
}

// ---- Execution stats -------------------------------------------------------

// execStatsURL is the UI stats endpoint. It sits behind the API key (unlike
// /health and /metrics), so it takes the same key as the nodes fetch.
func execStatsURL() string {
	return fmt.Sprintf("http://localhost:%d/api/ui/v1/executions/stats", serverPort())
}

// execStats is the slice of the executions summary the tray renders.
type execStats struct {
	OK         bool // false when the fetch failed / was unauthorized
	Total      int
	Successful int
	Failed     int
	Running    int
	AvgMS      float64
}

func parseExecStats(body []byte) (execStats, error) {
	var payload struct {
		Total      int     `json:"total_executions"`
		Successful int     `json:"successful_count"`
		Failed     int     `json:"failed_count"`
		Running    int     `json:"running_count"`
		AvgMS      float64 `json:"average_duration_ms"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return execStats{}, err
	}
	return execStats{
		OK:         true,
		Total:      payload.Total,
		Successful: payload.Successful,
		Failed:     payload.Failed,
		Running:    payload.Running,
		AvgMS:      payload.AvgMS,
	}, nil
}

// fetchExecStats is best-effort: any failure (including auth) yields OK=false so
// the caller simply omits the metrics rows. Auth is already gated by the nodes
// fetch, which runs first and shows the key prompt.
func fetchExecStats(apiKey string) execStats {
	req, err := http.NewRequest(http.MethodGet, execStatsURL(), nil)
	if err != nil {
		return execStats{}
	}
	if apiKey != "" {
		req.Header.Set("X-API-Key", apiKey)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return execStats{}
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return execStats{}
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return execStats{}
	}
	stats, err := parseExecStats(body)
	if err != nil {
		return execStats{}
	}
	return stats
}

// ---- API key storage -------------------------------------------------------

// effectiveAPIKey is the key the tray should present to the API: an explicit
// env var wins (mirrors the `af` CLI), otherwise the key the user saved via the
// tray. Empty means "no key available".
func effectiveAPIKey() string {
	if k := strings.TrimSpace(os.Getenv("AGENTFIELD_API_KEY")); k != "" {
		return k
	}
	return storedAPIKey()
}

func storedAPIKey() string {
	data, err := os.ReadFile(credentialsPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func saveAPIKey(key string) error {
	return writeFileAtomic(credentialsPath(), []byte(strings.TrimSpace(key)+"\n"), 0o600)
}

// ---- Presentation helpers (pure, so they're unit-tested on any OS) ----------
//
// Design: the menu is a calm dashboard — a two-line header, then one fact per
// row (never crammed), reading "Label — value". Each row is prefixed with a
// real monochrome Lucide icon (set as a macOS template image by the caller),
// not an emoji; status is shown with small colored dot icons. Detail and
// controls live in submenus so the top level stays quiet. These helpers produce
// only the text — the icons are applied in the darwin tray code.

// statusLine is the header's second line: the state word and, when running,
// where to reach the control plane. The colored state dot is an icon.
func statusLine(healthy bool, port int) string {
	if !healthy {
		return "Stopped"
	}
	return fmt.Sprintf("Running · localhost:%d", port)
}

// agentsHeadline titles the Agents submenu, e.g. "Agents — 4 of 7 online".
func agentsHeadline(f fleetSummary) string {
	switch f.Status {
	case fleetUnavailable:
		return "Agents — unavailable"
	default:
		if f.Total == 0 {
			return "Agents — none registered"
		}
		return fmt.Sprintf("Agents — %d of %d online", f.Online, f.Total)
	}
}

// agentLine renders one agent row inside the submenu. The colored online/offline
// dot is an icon; online agents show their live capability count, offline ones
// read "offline" since their skills aren't callable right now.
func agentLine(a agentInfo) string {
	if !a.Online {
		return fmt.Sprintf("%s — offline", a.ID)
	}
	caps := a.Skills + a.Reasoners
	if caps > 0 {
		return fmt.Sprintf("%s — %d skill%s", a.ID, caps, plural(caps))
	}
	return a.ID
}

// metricSuccess is the success-rate row: "Success — 83% (20 of 24)". The
// fraction carries the run volume and, implicitly, the failures, so success and
// activity fit one clean row.
func metricSuccess(s execStats) string {
	if !s.OK || s.Total == 0 {
		return "Success — no runs yet"
	}
	rate := int(math.Round(100 * float64(s.Successful) / float64(s.Total)))
	return fmt.Sprintf("Success — %d%% (%d of %d)", rate, s.Successful, s.Total)
}

// metricResponse is the latency row, or "" when there's nothing to average
// (which tells the caller to hide the row).
func metricResponse(s execStats) string {
	if !s.OK || s.Total == 0 {
		return ""
	}
	return fmt.Sprintf("Response — %d ms avg", int(math.Round(s.AvgMS)))
}

// metricMemory is the server-memory row, or "" when unknown.
func metricMemory(memMB int) string {
	if memMB <= 0 {
		return ""
	}
	return fmt.Sprintf("Memory — %d MB", memMB)
}

// metricLevel is a traffic-light rating for a stat, used to pick its icon color.
type metricLevel int

const (
	levelNeutral metricLevel = iota // no data → monochrome icon
	levelGood                       // green
	levelWarn                       // yellow
	levelBad                        // red
)

// Thresholds. These encode the product's rules of thumb for what's healthy:
//   - success rate ≥ 60% is good, 30–59% is a warning, below that is bad;
//   - average response ≤ 100ms is good, ≤ 500ms is a warning, above is bad;
//   - server memory < 1GB is good, < 2GB is a warning, at/above 2GB is bad.
const (
	successGoodPct = 60
	successWarnPct = 30

	responseGoodMS = 100
	responseWarnMS = 500

	memoryGoodMB = 1024
	memoryWarnMB = 2048
)

// successLevel rates the success rate; no runs yet is neutral.
func successLevel(s execStats) metricLevel {
	if !s.OK || s.Total == 0 {
		return levelNeutral
	}
	rate := int(math.Round(100 * float64(s.Successful) / float64(s.Total)))
	switch {
	case rate >= successGoodPct:
		return levelGood
	case rate >= successWarnPct:
		return levelWarn
	default:
		return levelBad
	}
}

// responseLevel rates average latency; no runs yet is neutral.
func responseLevel(s execStats) metricLevel {
	if !s.OK || s.Total == 0 {
		return levelNeutral
	}
	switch {
	case s.AvgMS <= responseGoodMS:
		return levelGood
	case s.AvgMS <= responseWarnMS:
		return levelWarn
	default:
		return levelBad
	}
}

// memoryLevel rates server memory; unknown is neutral.
func memoryLevel(memMB int) metricLevel {
	if memMB <= 0 {
		return levelNeutral
	}
	switch {
	case memMB < memoryGoodMB:
		return levelGood
	case memMB < memoryWarnMB:
		return levelWarn
	default:
		return levelBad
	}
}

// enterKeyTitle labels the key-prompt item; it doubles as the "auth required"
// message so the menu needs no separate status line for it.
func enterKeyTitle(haveKey bool) string {
	if haveKey {
		return "API key expired — re-enter…"
	}
	return "API key required — enter…"
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// sortAgents orders online agents first, then alphabetically by id, so the most
// relevant rows fill the (bounded) menu slots.
func sortAgents(in []agentInfo) []agentInfo {
	out := make([]agentInfo, len(in))
	copy(out, in)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Online != out[j].Online {
			return out[i].Online // online (true) sorts before offline (false)
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// ---- launchctl argument construction ---------------------------------------

func guiDomain() string         { return fmt.Sprintf("gui/%d", os.Getuid()) }
func svcTarget(l string) string { return guiDomain() + "/" + l }

// kickstartArgs builds the argv for `launchctl kickstart`. The -k flag forces a
// restart of an already-running service (kill then relaunch); without it,
// kickstart only starts a loaded-but-idle service.
func kickstartArgs(label string, kill bool) []string {
	args := []string{"kickstart"}
	if kill {
		args = append(args, "-k")
	}
	return append(args, svcTarget(label))
}

// ---- Files -----------------------------------------------------------------

// writeFileAtomic writes data to a temp file in the destination directory and
// renames it into place, so a reader (or a running binary being replaced) never
// sees a half-written file.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".af-tray-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// ---- plist / Info.plist templates ------------------------------------------

func infoPlist() string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>AgentField</string>
  <key>CFBundleDisplayName</key><string>AgentField</string>
  <key>CFBundleIdentifier</key><string>%s</string>
  <key>CFBundleVersion</key><string>%s</string>
  <key>CFBundleShortVersionString</key><string>%s</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>af-tray</string>
  <key>CFBundleIconFile</key><string>appicon</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>10.15</string>
</dict>
</plist>
`, trayLabel, version, version)
}

// serverPlist is the control-plane launchd agent.
//   - RunAtLoad starts it at login.
//   - KeepAlive={SuccessfulExit: false} restarts it only on a crash, so a
//     graceful "Stop" (SIGTERM → clean exit) actually stays stopped.
//   - --open=false stops it opening a browser every time it starts under launchd.
func serverPlist() string {
	log := serverLogPath()
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>server</string>
    <string>--open=false</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>WorkingDirectory</key><string>%s</string>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
`, serverLabel, serverBinaryPath(), agentfieldDir(), log, log)
}

// trayPlist is the menu-bar tray launchd agent. KeepAlive={Crashed: true} means
// a genuine crash relaunches it, but a clean exit (the "Quit" menu item, or the
// no-GUI-session early exit) does not — so it never crash-loops.
func trayPlist() string {
	log := trayLogPath()
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>run</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>Crashed</key><true/></dict>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
`, trayLabel, trayBundleBinaryPath(), log, log)
}
