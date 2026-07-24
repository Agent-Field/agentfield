package main

// This file holds the platform-neutral logic behind the tray: health polling,
// path resolution, launchd plist / Info.plist generation, launchctl argument
// construction, and atomic file writes. It has NO GUI (systray/CGO) dependency
// and compiles on every platform, so it can be unit-tested directly in CI
// (which runs on Linux). The OS-specific glue — the systray event loop and the
// exec.Command("launchctl", …) calls — lives in the _darwin files and calls
// into these helpers.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
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

// ---- Usage stats (tokens / cost) -------------------------------------------

// usageStatsURL is the UI usage endpoint. Like the executions stats endpoint it
// sits behind the API key, so it takes the same key. The window is one of
// 1h|24h|7d|30d|all; the server defaults it to 24h when omitted/invalid.
func usageStatsURL(window string) string {
	return fmt.Sprintf("http://localhost:%d/api/ui/v1/usage/stats?window=%s", serverPort(), window)
}

// usageStatus mirrors fleetStatus: the outcome of trying to read usage stats.
type usageStatus int

const (
	usageOK           usageStatus = iota // stats read successfully
	usageAuthRequired                    // server demands a key we don't have (or ours was rejected)
	usageUnavailable                     // server unreachable / unexpected response
	usageAbsent                          // endpoint not found (older server) → hide the section entirely
)

// usageGroup is one grouped bucket (by model / provider / harness). Provider is
// only populated for model groups. CostUSD is a pointer because the server emits
// null when no priced usage was recorded.
type usageGroup struct {
	Key          string
	Provider     string
	CostUSD      *float64
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	Entries      int64
}

// seriesPoint is one bucket of the 24h token timeseries the tray charts. T is
// the bucket start; CostUSD is nil when no priced usage landed in the bucket.
type seriesPoint struct {
	T           time.Time
	TotalTokens int64
	CostUSD     *float64
}

// modelSeries is one model's per-bucket token contribution to the stacked chart,
// aligned to the same bucket grid as usageStats.Series. Key is the model id (or
// the literal "other" for the aggregated long tail). Tokens is zero-filled to
// the grid, so it can be stacked directly.
type modelSeries struct {
	Key    string
	Tokens []float64
}

// usageStats is the slice of the usage aggregation the tray renders.
type usageStats struct {
	Status        usageStatus
	Window        string
	CostUSD       *float64
	InputTokens   int64
	OutputTokens  int64
	TotalTokens   int64
	Executions    int64
	ByModel       []usageGroup
	ByProvider    []usageGroup
	ByHarness     []usageGroup
	LastUpdated   *time.Time
	BucketSeconds int           // width of each series bucket; 0 when no series
	Series        []seriesPoint // ascending, zero-filled; empty when the server omits it
	// SeriesByModel is the per-model breakdown of the timeseries (top-3 desc plus
	// an optional final "other"), on the same bucket grid as Series. Empty when
	// the server omits "series_by_model" (older control plane), in which case the
	// tray falls back to the single-series chart.
	SeriesByModel []modelSeries
}

// hasData reports whether there is anything worth rendering: a successful fetch
// that actually recorded some tokens. Empty windows collapse the whole section.
func (u usageStats) hasData() bool {
	return u.Status == usageOK && u.TotalTokens > 0
}

// hasSeries reports whether the server returned a token timeseries (newer
// control planes only). An older server omits the "series" key entirely, in
// which case this is false and the chart row is simply hidden.
func (u usageStats) hasSeries() bool {
	return len(u.Series) > 0
}

// hasSeriesByModel reports whether the server returned a per-model breakdown of
// the timeseries (newest control planes only). When false the tray falls back to
// the single-series area chart.
func (u usageStats) hasSeriesByModel() bool {
	for _, m := range u.SeriesByModel {
		if len(m.Tokens) > 0 {
			return true
		}
	}
	return false
}

// showUsageChart is the (GUI-free, unit-tested) predicate the darwin tray uses to
// decide whether to show a timeseries chart row: there must be usage to chart
// and a series to chart it from. When the series is present but all-zero the
// renderer still draws a flat baseline, so this stays true in that case.
func showUsageChart(u usageStats) bool {
	return u.hasData() && u.hasSeries()
}

// showStackedChart reports whether to draw the richer stacked-by-model chart
// (preferred) rather than the single-series fallback: it needs usage plus a
// per-model breakdown.
func showStackedChart(u usageStats) bool {
	return u.hasData() && u.hasSeriesByModel()
}

// stackedChartData assembles the numeric layers and hues the stacked-area
// renderer consumes, in bottom-to-top draw order (rank 0 at the bottom, "other"
// gray on top). It is GUI-free so it can be unit-tested; the renderer itself
// lives in chart_render.go. The chart is pure graphics — it carries no text — so
// no axis labels or peak value are computed here.
func stackedChartData(u usageStats) (layers [][]float64, colors []color.NRGBA) {
	// Normalize every model series to the common bucket count.
	n := len(u.Series)
	for _, m := range u.SeriesByModel {
		if len(m.Tokens) > n {
			n = len(m.Tokens)
		}
	}
	rank := 0
	for _, m := range u.SeriesByModel {
		vals := make([]float64, n)
		copy(vals, m.Tokens)
		layers = append(layers, vals)
		colors = append(colors, stackedLayerColor(m.Key, rank))
		if m.Key != "other" {
			rank++
		}
	}
	return layers, colors
}

// seriesTokenValues projects the timeseries onto the per-bucket token counts the
// chart renderer consumes.
func seriesTokenValues(pts []seriesPoint) []float64 {
	out := make([]float64, len(pts))
	for i, p := range pts {
		out[i] = float64(p.TotalTokens)
	}
	return out
}

// usageGroupJSON is the wire shape of one grouped bucket.
type usageGroupJSON struct {
	Key          string   `json:"key"`
	Provider     string   `json:"provider"`
	CostUSD      *float64 `json:"cost_usd"`
	InputTokens  int64    `json:"input_tokens"`
	OutputTokens int64    `json:"output_tokens"`
	TotalTokens  int64    `json:"total_tokens"`
	Entries      int64    `json:"entries"`
}

func toUsageGroups(in []usageGroupJSON) []usageGroup {
	out := make([]usageGroup, 0, len(in))
	for _, g := range in {
		out = append(out, usageGroup{
			Key:          g.Key,
			Provider:     g.Provider,
			CostUSD:      g.CostUSD,
			InputTokens:  g.InputTokens,
			OutputTokens: g.OutputTokens,
			TotalTokens:  g.TotalTokens,
			Entries:      g.Entries,
		})
	}
	return out
}

// parseUsageStats extracts the usage aggregation from a
// GET /api/ui/v1/usage/stats response body. It tolerates null/missing fields:
// unknown keys are ignored and absent numbers default to zero.
func parseUsageStats(body []byte) (usageStats, error) {
	var payload struct {
		Window string `json:"window"`
		Totals struct {
			CostUSD             *float64 `json:"cost_usd"`
			InputTokens         int64    `json:"input_tokens"`
			OutputTokens        int64    `json:"output_tokens"`
			TotalTokens         int64    `json:"total_tokens"`
			ExecutionsWithUsage int64    `json:"executions_with_usage"`
		} `json:"totals"`
		ByModel     []usageGroupJSON `json:"by_model"`
		ByProvider  []usageGroupJSON `json:"by_provider"`
		ByHarness   []usageGroupJSON `json:"by_harness"`
		LastUpdated *string          `json:"last_updated"`
		// Series is a pointer so an absent "series" key (older server) is
		// distinguishable from a present-but-empty one, and both degrade to no
		// chart. Newer servers send bucket_seconds + an ascending, zero-filled
		// points array.
		Series *struct {
			BucketSeconds int `json:"bucket_seconds"`
			Points        []struct {
				T           string   `json:"t"`
				TotalTokens int64    `json:"total_tokens"`
				CostUSD     *float64 `json:"cost_usd"`
			} `json:"points"`
		} `json:"series"`
		// SeriesByModel is the per-model breakdown of the timeseries. It is a
		// slice (absent → nil → no stacked chart) of {key, points[]} where each
		// point mirrors a bucket. Parsed tolerantly: a bad entry contributes zero
		// rather than failing the whole parse.
		SeriesByModel []struct {
			Key    string `json:"key"`
			Points []struct {
				TotalTokens int64 `json:"total_tokens"`
			} `json:"points"`
		} `json:"series_by_model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return usageStats{}, err
	}
	u := usageStats{
		Status:       usageOK,
		Window:       payload.Window,
		CostUSD:      payload.Totals.CostUSD,
		InputTokens:  payload.Totals.InputTokens,
		OutputTokens: payload.Totals.OutputTokens,
		TotalTokens:  payload.Totals.TotalTokens,
		Executions:   payload.Totals.ExecutionsWithUsage,
		ByModel:      toUsageGroups(payload.ByModel),
		ByProvider:   toUsageGroups(payload.ByProvider),
		ByHarness:    toUsageGroups(payload.ByHarness),
		LastUpdated:  parseTimePtr(payload.LastUpdated),
	}
	if payload.Series != nil {
		u.BucketSeconds = payload.Series.BucketSeconds
		pts := make([]seriesPoint, 0, len(payload.Series.Points))
		for _, p := range payload.Series.Points {
			sp := seriesPoint{TotalTokens: p.TotalTokens, CostUSD: p.CostUSD}
			// The timestamp is informational (the chart plots by position); a
			// bad/absent one just leaves the zero time rather than failing.
			if t := parseTimePtr(&p.T); t != nil {
				sp.T = *t
			}
			pts = append(pts, sp)
		}
		u.Series = pts
	}
	for _, m := range payload.SeriesByModel {
		if m.Key == "" && len(m.Points) == 0 {
			continue
		}
		toks := make([]float64, 0, len(m.Points))
		for _, p := range m.Points {
			toks = append(toks, float64(p.TotalTokens))
		}
		u.SeriesByModel = append(u.SeriesByModel, modelSeries{Key: m.Key, Tokens: toks})
	}
	return u, nil
}

// fetchUsageStats reads usage for a window, authenticating with apiKey when
// non-empty. It mirrors fetchFleet's status mapping — 401/403 → auth required —
// and additionally maps 404 to usageAbsent so an older control plane without the
// endpoint simply hides the section instead of showing an error.
func fetchUsageStats(window, apiKey string) usageStats {
	// Request the 24h token timeseries (buckets=48 → 30-min buckets over 24h)
	// used by the chart row. usageStatsURL already carries "?window=…", so this
	// appends as an extra query param; older servers ignore it and simply omit
	// the "series" key, which parseUsageStats tolerates.
	req, err := http.NewRequest(http.MethodGet, usageStatsURL(window)+"&buckets="+strconv.Itoa(usageSeriesBuckets)+"&series_by=model", nil)
	if err != nil {
		return usageStats{Status: usageUnavailable}
	}
	if apiKey != "" {
		req.Header.Set("X-API-Key", apiKey)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return usageStats{Status: usageUnavailable}
	}
	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusOK:
		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			return usageStats{Status: usageUnavailable}
		}
		stats, err := parseUsageStats(body)
		if err != nil {
			return usageStats{Status: usageUnavailable}
		}
		return stats
	case http.StatusUnauthorized, http.StatusForbidden:
		return usageStats{Status: usageAuthRequired}
	case http.StatusNotFound:
		return usageStats{Status: usageAbsent}
	default:
		return usageStats{Status: usageUnavailable}
	}
}

// ---- Usage presentation helpers (pure) -------------------------------------

// formatTokens renders a token count compactly: "1.2M", "45.3k", "999".
func formatTokens(n int64) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1_000)
	default:
		return strconv.FormatInt(n, 10)
	}
}

// formatCost renders a dollar cost: "$1.23" normally, more precision for tiny
// amounts ("$0.0042"), and an em dash when the cost is unknown (null).
func formatCost(c *float64) string {
	if c == nil {
		return "—"
	}
	v := *c
	switch {
	case v == 0:
		return "$0.00"
	case v < 0.01:
		return fmt.Sprintf("$%.4f", v)
	default:
		return fmt.Sprintf("$%.2f", v)
	}
}

// shortModelName trims a model id down to something that fits a menu row: it
// drops any provider path prefix ("openrouter/anthropic/claude-…" → "claude-…"),
// strips the common "claude-" vendor prefix, and truncates the rest.
func shortModelName(key string) string {
	s := key
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimPrefix(s, "claude-")
	const max = 22
	if len(s) > max {
		s = strings.TrimSpace(s[:max-1]) + "…"
	}
	if s == "" {
		return key
	}
	return s
}

// usageHeadline titles the Usage submenu parent, e.g.
// "Usage (24h) — $1.23 · 3.0M tokens", dropping the cost when it is unknown.
func usageHeadline(u usageStats) string {
	label := "Usage (" + u.Window + ")"
	if u.CostUSD != nil {
		return fmt.Sprintf("%s — %s · %s tokens", label, formatCost(u.CostUSD), formatTokens(u.TotalTokens))
	}
	return fmt.Sprintf("%s — %s tokens", label, formatTokens(u.TotalTokens))
}

// usageSeriesBuckets is how many buckets the tray asks the usage endpoint to
// split the 24h window into for the chart row: 48 → 30-minute buckets.
const usageSeriesBuckets = 48

// usageModelTitle is the per-model row title used alongside a rendered bar image
// (the image carries the proportion, so the text drops the Unicode bar):
// "opus-4-8 — 1.2M · $0.90".
func usageModelTitle(g usageGroup) string {
	return fmt.Sprintf("%s — %s · %s",
		shortModelName(g.Key), formatTokens(g.TotalTokens), formatCost(g.CostUSD))
}

// modelShare is the model's fraction (0..1) of the window's total tokens, used
// to size its bar image.
func modelShare(g usageGroup, windowTotalTokens int64) float64 {
	if windowTotalTokens <= 0 {
		return 0
	}
	return float64(g.TotalTokens) / float64(windowTotalTokens)
}

// relativeTime renders how long ago t was relative to now, in friendly units:
// "just now", "8 minutes ago", "3 hours ago", "2 days ago".
func relativeTime(t, now time.Time) string {
	d := now.Sub(t)
	if d < 0 {
		d = 0
	}
	switch {
	case d < 45*time.Second:
		return "just now"
	case d < 90*time.Second:
		return "1 minute ago"
	case d < time.Hour:
		return fmt.Sprintf("%d minutes ago", int(math.Round(d.Minutes())))
	case d < 2*time.Hour:
		return "1 hour ago"
	case d < 24*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "1 day ago"
	default:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	}
}

// usageFooter is the "Last updated: …" footer row, or "" (hidden) when the
// server reported no last-updated timestamp.
func usageFooter(t *time.Time, now time.Time) string {
	if t == nil {
		return ""
	}
	return "Last updated: " + relativeTime(*t, now)
}

// parseTimePtr parses an optional RFC3339 timestamp pointer, yielding nil for a
// nil or unparseable value so callers can simply omit the corresponding row.
func parseTimePtr(s *string) *time.Time {
	if s == nil || *s == "" {
		return nil
	}
	parsed, err := time.Parse(time.RFC3339, *s)
	if err != nil {
		return nil
	}
	return &parsed
}

// ---- Claude subscription quota (best-effort, read-only) --------------------
//
// These helpers read the user's existing Claude Code OAuth token from the macOS
// Keychain (in the _darwin file) and query Anthropic's OAuth usage endpoint for
// the 5-hour and 7-day rate-limit windows. Everything here is best-effort and
// degrades silently: a missing token, a failed request, or an unexpected shape
// simply hides the rows. The token is never logged, persisted, or sent anywhere
// except api.anthropic.com.

// claudeUsageURL is Anthropic's OAuth usage endpoint. It is queried with a
// bearer token and the oauth beta header.
const claudeUsageURL = "https://api.anthropic.com/api/oauth/usage"

// claudeOAuthBetaHeader is the beta gate the usage endpoint requires.
const claudeOAuthBetaHeader = "oauth-2025-04-20"

// claudeQuota is the slice of the OAuth usage response the tray renders: the two
// rolling rate-limit windows, each a utilization percentage and a reset time.
type claudeQuota struct {
	OK            bool
	FiveHourPct   float64
	FiveHourReset *time.Time
	SevenDayPct   float64
	SevenDayReset *time.Time
}

// parseClaudeCodeToken extracts the OAuth access token from the JSON blob the
// macOS Keychain stores under the "Claude Code-credentials" item.
func parseClaudeCodeToken(keychainJSON []byte) (string, error) {
	var payload struct {
		ClaudeAiOauth struct {
			AccessToken string `json:"accessToken"`
		} `json:"claudeAiOauth"`
	}
	if err := json.Unmarshal(keychainJSON, &payload); err != nil {
		return "", err
	}
	if payload.ClaudeAiOauth.AccessToken == "" {
		return "", fmt.Errorf("no access token in keychain payload")
	}
	return payload.ClaudeAiOauth.AccessToken, nil
}

// parseClaudeQuota extracts the 5h/7d utilization + reset times from the OAuth
// usage response. It parses defensively: the endpoint returns many optional
// fields, and any that are missing/null leave the corresponding value zeroed or
// nil so the row can degrade gracefully.
func parseClaudeQuota(body []byte) (claudeQuota, error) {
	var payload struct {
		FiveHour struct {
			Utilization *float64 `json:"utilization"`
			ResetsAt    *string  `json:"resets_at"`
		} `json:"five_hour"`
		SevenDay struct {
			Utilization *float64 `json:"utilization"`
			ResetsAt    *string  `json:"resets_at"`
		} `json:"seven_day"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return claudeQuota{}, err
	}
	q := claudeQuota{OK: true}
	if payload.FiveHour.Utilization != nil {
		q.FiveHourPct = *payload.FiveHour.Utilization
	}
	q.FiveHourReset = parseTimePtr(payload.FiveHour.ResetsAt)
	if payload.SevenDay.Utilization != nil {
		q.SevenDayPct = *payload.SevenDay.Utilization
	}
	q.SevenDayReset = parseTimePtr(payload.SevenDay.ResetsAt)
	return q, nil
}

// fetchClaudeQuota queries the OAuth usage endpoint at baseURL with the given
// token. Any failure (no token, transport error, non-200, bad body) yields
// OK=false so the caller hides the rows. baseURL is a parameter purely so the
// function is testable against an httptest server; production always passes
// claudeUsageURL.
func fetchClaudeQuota(baseURL, token string) claudeQuota {
	if token == "" {
		return claudeQuota{}
	}
	req, err := http.NewRequest(http.MethodGet, baseURL, nil)
	if err != nil {
		return claudeQuota{}
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("anthropic-beta", claudeOAuthBetaHeader)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return claudeQuota{}
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return claudeQuota{}
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return claudeQuota{}
	}
	q, err := parseClaudeQuota(body)
	if err != nil {
		return claudeQuota{}
	}
	return q
}

// claudeQuotaTitle is the rate-limit row title used alongside a rendered bar
// image (the image carries the proportion, so the text drops the Unicode bar):
// "Claude 5h — 24% (resets 3:00 PM)". The reset clock is omitted when unknown.
func claudeQuotaTitle(label string, pct float64, reset *time.Time, now time.Time) string {
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	line := fmt.Sprintf("%s — %d%%", label, int(math.Round(pct)))
	if reset != nil {
		line += fmt.Sprintf(" (resets %s)", reset.Local().Format("3:04 PM"))
	}
	return line
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

// ---- Tray settings (persisted preferences) ---------------------------------

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
	return fmt.Sprintf("Response — %s avg", formatDurationMS(s.AvgMS))
}

// formatDurationMS renders an average duration at a human scale: raw
// milliseconds under a second, seconds under a minute, then minutes+seconds
// ("742 ms", "8.3 s", "2m 43s"). Long-running agent workflows make multi-minute
// averages common, where "162867 ms" is unreadable.
func formatDurationMS(ms float64) string {
	if ms < 0 {
		ms = 0
	}
	switch {
	case ms < 1000:
		return fmt.Sprintf("%d ms", int(math.Round(ms)))
	case ms < 60_000:
		return fmt.Sprintf("%.1f s", ms/1000)
	default:
		total := int(math.Round(ms / 1000))
		return fmt.Sprintf("%dm %ds", total/60, total%60)
	}
}

// sparkIconSize is the pixel size of menu-item icons (16pt @2x), matching the
// embedded PNGs in assets/icons.
const sparkIconSize = 32

// sparkMaxSamples caps the history ring feeding the sparkline: one column per
// icon pixel, so older samples would not be visible anyway.
const sparkMaxSamples = sparkIconSize

// pushSparkSample appends v to a bounded sample ring, dropping the oldest
// sample once the ring is full.
func pushSparkSample(ring []float64, v float64) []float64 {
	ring = append(ring, v)
	if len(ring) > sparkMaxSamples {
		ring = ring[len(ring)-sparkMaxSamples:]
	}
	return ring
}

// levelSparkColor maps a traffic-light rating to the sparkline tint, matching
// the palette of the colored dot/glyph icons.
func levelSparkColor(lvl metricLevel) color.NRGBA {
	switch lvl {
	case levelGood:
		return color.NRGBA{48, 209, 88, 255}
	case levelWarn:
		return color.NRGBA{255, 204, 0, 255}
	case levelBad:
		return color.NRGBA{255, 69, 58, 255}
	default:
		return color.NRGBA{152, 152, 157, 255}
	}
}

// sparklineIconPNG renders a 32×32 menu-item icon: a 2px line over a faint
// area fill, on a transparent background, in the given color. With fewer than
// two samples (or an all-equal series) it draws a flat mid-height line, so the
// row shows a calm baseline until history accumulates rather than nothing.
func sparklineIconPNG(samples []float64, c color.NRGBA) []byte {
	const (
		size = sparkIconSize
		top  = 7  // keep vertical margins so the line doesn't touch the box
		bot  = 25 // baseline of the area fill
	)
	img := image.NewNRGBA(image.Rect(0, 0, size, size))

	lo, hi := math.Inf(1), math.Inf(-1)
	for _, v := range samples {
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	flat := len(samples) < 2 || hi-lo < 1e-9

	// yAt maps the sample interpolated at column x to a pixel row.
	yAt := func(x int) int {
		if flat {
			return (top + bot) / 2
		}
		pos := float64(x) / float64(size-1) * float64(len(samples)-1)
		i := int(pos)
		frac := pos - float64(i)
		v := samples[i]
		if i+1 < len(samples) {
			v += frac * (samples[i+1] - samples[i])
		}
		return bot - int(math.Round((v-lo)/(hi-lo)*float64(bot-top)))
	}

	fill := color.NRGBA{c.R, c.G, c.B, 56}
	prevY := yAt(0)
	for x := 0; x < size; x++ {
		y := yAt(x)
		// Faint area under the line down to the baseline.
		for fy := y + 1; fy <= bot; fy++ {
			img.SetNRGBA(x, fy, fill)
		}
		// 2px line, connecting vertically to the previous column so steep
		// segments stay continuous.
		yLo, yHi := y, prevY
		if yLo > yHi {
			yLo, yHi = yHi, yLo
		}
		for ly := yLo; ly <= yHi+1; ly++ {
			img.SetNRGBA(x, ly, c)
		}
		prevY = y
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil
	}
	return buf.Bytes()
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
