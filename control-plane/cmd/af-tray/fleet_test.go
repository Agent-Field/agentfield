package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

// These tests pin the behaviours the enhanced status menu relies on. Like the
// rest of the af-tray tests they avoid any GUI/CGO dependency, so they run on
// the Linux CI even though the menu itself is macOS-only.

// Contract: the nodes URL targets the resolved port and asks for every node
// (show_all=true), not just the active ones.
func TestNodesURL(t *testing.T) {
	t.Setenv("AGENTFIELD_PORT", "9091")
	if got, want := nodesURL(), "http://localhost:9091/api/v1/nodes?show_all=true"; got != want {
		t.Errorf("nodesURL() = %q, want %q", got, want)
	}
}

// Contract: metric rows deep-link into the embedded UI under /ui/<page>.
func TestUIPageURL(t *testing.T) {
	t.Setenv("AGENTFIELD_PORT", "8080")
	if got, want := uiPageURL("executions"), "http://localhost:8080/ui/executions"; got != want {
		t.Errorf("uiPageURL() = %q, want %q", got, want)
	}
}

// Contract: a node counts as online only when health_status == "active", and a
// node's capability count is skills + reasoners.
func TestParseAndSummarizeNodes(t *testing.T) {
	body := []byte(`{"nodes":[
		{"id":"weather","health_status":"active","skills":[{},{}],"reasoners":[{}]},
		{"id":"research","health_status":"active","skills":[],"reasoners":[{},{},{}]},
		{"id":"stale","health_status":"inactive","skills":[{}],"reasoners":[]}
	],"count":3}`)

	agents, err := parseNodes(body)
	if err != nil {
		t.Fatalf("parseNodes: %v", err)
	}
	if len(agents) != 3 {
		t.Fatalf("parsed %d agents, want 3", len(agents))
	}

	s := summarizeFleet(agents)
	if s.Total != 3 {
		t.Errorf("Total = %d, want 3", s.Total)
	}
	if s.Online != 2 {
		t.Errorf("Online = %d, want 2 (only active nodes)", s.Online)
	}
	// online only: weather 2+1, research 0+3 = 6 (stale/inactive excluded)
	if s.Skills != 6 {
		t.Errorf("Skills = %d, want 6 (skills+reasoners across ONLINE nodes)", s.Skills)
	}
	if s.Status != fleetOK {
		t.Errorf("Status = %v, want fleetOK", s.Status)
	}
}

func TestParseNodesBadJSON(t *testing.T) {
	if _, err := parseNodes([]byte("not json")); err == nil {
		t.Error("parseNodes on garbage = nil error, want error")
	}
}

// Contract: fetchFleet reads agents on 200, maps 401/403 to fleetAuthRequired
// so the tray can prompt for a key, and maps everything else to
// fleetUnavailable. It must send X-API-Key when (and only when) a key is given.
func TestFetchFleet(t *testing.T) {
	var gotKey string
	var mode string // controlled per-subtest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-API-Key")
		switch mode {
		case "ok":
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"nodes":[{"id":"a","health_status":"active","skills":[{}],"reasoners":[]}]}`)
		case "401":
			w.WriteHeader(http.StatusUnauthorized)
		case "403":
			w.WriteHeader(http.StatusForbidden)
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGENTFIELD_PORT", u.Port())

	t.Run("ok sends key and parses", func(t *testing.T) {
		mode = "ok"
		s := fetchFleet("secret-key")
		if s.Status != fleetOK {
			t.Fatalf("Status = %v, want fleetOK", s.Status)
		}
		if gotKey != "secret-key" {
			t.Errorf("X-API-Key = %q, want %q", gotKey, "secret-key")
		}
		if s.Online != 1 || s.Total != 1 {
			t.Errorf("Online/Total = %d/%d, want 1/1", s.Online, s.Total)
		}
	})

	t.Run("no key sends no header", func(t *testing.T) {
		mode = "ok"
		gotKey = "sentinel"
		_ = fetchFleet("")
		if gotKey != "" {
			t.Errorf("X-API-Key = %q, want empty when no key given", gotKey)
		}
	})

	t.Run("401 -> auth required", func(t *testing.T) {
		mode = "401"
		if s := fetchFleet("bad"); s.Status != fleetAuthRequired {
			t.Errorf("Status = %v, want fleetAuthRequired", s.Status)
		}
	})

	t.Run("403 -> auth required", func(t *testing.T) {
		mode = "403"
		if s := fetchFleet("bad"); s.Status != fleetAuthRequired {
			t.Errorf("Status = %v, want fleetAuthRequired", s.Status)
		}
	})

	t.Run("500 -> unavailable", func(t *testing.T) {
		mode = "500"
		if s := fetchFleet(""); s.Status != fleetUnavailable {
			t.Errorf("Status = %v, want fleetUnavailable", s.Status)
		}
	})
}

// Contract: a server that isn't listening reads as unavailable, not a crash.
func TestFetchFleetUnreachable(t *testing.T) {
	down := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	u, _ := url.Parse(down.URL)
	down.Close()
	t.Setenv("AGENTFIELD_PORT", u.Port())
	if s := fetchFleet(""); s.Status != fleetUnavailable {
		t.Errorf("Status = %v, want fleetUnavailable", s.Status)
	}
}

// Contract: env var wins over stored file; with neither, the key is empty. This
// mirrors the `af` CLI, where AGENTFIELD_API_KEY overrides everything.
func TestEffectiveAPIKey(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)

	_ = os.Unsetenv("AGENTFIELD_API_KEY")
	if got := effectiveAPIKey(); got != "" {
		t.Errorf("effectiveAPIKey() with nothing set = %q, want empty", got)
	}

	if err := saveAPIKey("  from-file  "); err != nil {
		t.Fatalf("saveAPIKey: %v", err)
	}
	if got := effectiveAPIKey(); got != "from-file" {
		t.Errorf("effectiveAPIKey() = %q, want trimmed stored key", got)
	}

	// The file is written owner-only.
	info, err := os.Stat(credentialsPath())
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("credentials perm = %v, want 0600", info.Mode().Perm())
	}

	t.Setenv("AGENTFIELD_API_KEY", "from-env")
	if got := effectiveAPIKey(); got != "from-env" {
		t.Errorf("effectiveAPIKey() = %q, want env to win", got)
	}
}

func TestCredentialsPathUnderHome(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	if got, want := credentialsPath(), filepath.Join(root, ".agentfield", "tray-apikey"); got != want {
		t.Errorf("credentialsPath() = %q, want %q", got, want)
	}
}

// Contract: the header status line shows a colored dot + state, and the address
// only when running.
func TestStatusLine(t *testing.T) {
	if got, want := statusLine(true, 8080), "Running · localhost:8080"; got != want {
		t.Errorf("statusLine(healthy) = %q, want %q", got, want)
	}
	if got, want := statusLine(false, 8080), "Stopped"; got != want {
		t.Errorf("statusLine(stopped) = %q, want %q", got, want)
	}
}

// Contract: the Agents submenu title reflects each fleet state and reports
// online/total when data is available.
func TestAgentsHeadline(t *testing.T) {
	cases := []struct {
		name string
		in   fleetSummary
		want string
	}{
		{"unavailable", fleetSummary{Status: fleetUnavailable}, "Agents — unavailable"},
		{"empty", fleetSummary{Status: fleetOK, Total: 0}, "Agents — none registered"},
		{"counts", fleetSummary{Status: fleetOK, Online: 2, Total: 3}, "Agents — 2 of 3 online"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := agentsHeadline(tc.in); got != tc.want {
				t.Errorf("agentsHeadline() = %q, want %q", got, tc.want)
			}
		})
	}
}

// Contract: online agents show a green dot + capability count; offline agents
// show a hollow dot and read "offline" (their skills aren't callable).
func TestAgentLine(t *testing.T) {
	cases := []struct {
		name string
		in   agentInfo
		want string
	}{
		{"online plural", agentInfo{ID: "weather", Online: true, Skills: 2, Reasoners: 1}, "weather — 3 skills"},
		{"online singular", agentInfo{ID: "solo", Online: true, Skills: 1}, "solo — 1 skill"},
		{"offline", agentInfo{ID: "stale", Online: false, Skills: 4}, "stale — offline"},
		{"online no caps", agentInfo{ID: "bare", Online: true}, "bare"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := agentLine(tc.in); got != tc.want {
				t.Errorf("agentLine() = %q, want %q", got, tc.want)
			}
		})
	}
}

// Contract: the success row shows rate + fraction (which carries volume and
// failures) on one clean line, and reads gracefully with no executions.
func TestMetricSuccess(t *testing.T) {
	cases := []struct {
		name string
		in   execStats
		want string
	}{
		{"ok-false", execStats{OK: false}, "Success — no runs yet"},
		{"zero", execStats{OK: true, Total: 0}, "Success — no runs yet"},
		{"clean", execStats{OK: true, Total: 10, Successful: 10}, "Success — 100% (10 of 10)"},
		{"with-failures", execStats{OK: true, Total: 24, Successful: 20, Failed: 4}, "Success — 83% (20 of 24)"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := metricSuccess(tc.in); got != tc.want {
				t.Errorf("metricSuccess() = %q, want %q", got, tc.want)
			}
		})
	}
}

// Contract: the response row shows rounded average latency, or "" (hidden) when
// there's nothing to average.
func TestMetricResponse(t *testing.T) {
	if got, want := metricResponse(execStats{OK: true, Total: 3, AvgMS: 42.4}), "Response — 42 ms avg"; got != want {
		t.Errorf("metricResponse() = %q, want %q", got, want)
	}
	if got := metricResponse(execStats{OK: true, Total: 0}); got != "" {
		t.Errorf("metricResponse(no runs) = %q, want empty", got)
	}
	if got := metricResponse(execStats{OK: false}); got != "" {
		t.Errorf("metricResponse(!ok) = %q, want empty", got)
	}
}

// Contract: the memory row shows MB, or "" (hidden) when unknown.
func TestMetricMemory(t *testing.T) {
	if got, want := metricMemory(37), "Memory — 37 MB"; got != want {
		t.Errorf("metricMemory() = %q, want %q", got, want)
	}
	if got := metricMemory(0); got != "" {
		t.Errorf("metricMemory(0) = %q, want empty", got)
	}
}

// Contract: success rate maps to green ≥60%, yellow 30–59%, red <30%, neutral
// with no runs — including the boundary values.
func TestSuccessLevel(t *testing.T) {
	cases := []struct {
		name string
		in   execStats
		want metricLevel
	}{
		{"no-runs", execStats{OK: true, Total: 0}, levelNeutral},
		{"not-ok", execStats{OK: false}, levelNeutral},
		{"good-100", execStats{OK: true, Total: 10, Successful: 10}, levelGood},
		{"good-boundary-60", execStats{OK: true, Total: 100, Successful: 60}, levelGood},
		{"warn-59", execStats{OK: true, Total: 100, Successful: 59}, levelWarn},
		{"warn-boundary-30", execStats{OK: true, Total: 100, Successful: 30}, levelWarn},
		{"bad-29", execStats{OK: true, Total: 100, Successful: 29}, levelBad},
		{"bad-0", execStats{OK: true, Total: 10, Successful: 0}, levelBad},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := successLevel(tc.in); got != tc.want {
				t.Errorf("successLevel() = %v, want %v", got, tc.want)
			}
		})
	}
}

// Contract: response maps to green ≤100ms, yellow ≤500ms, red >500ms, neutral
// with no runs — including the boundaries.
func TestResponseLevel(t *testing.T) {
	cases := []struct {
		name string
		in   execStats
		want metricLevel
	}{
		{"no-runs", execStats{OK: true, Total: 0}, levelNeutral},
		{"good-42", execStats{OK: true, Total: 5, AvgMS: 42}, levelGood},
		{"good-boundary-100", execStats{OK: true, Total: 5, AvgMS: 100}, levelGood},
		{"warn-101", execStats{OK: true, Total: 5, AvgMS: 101}, levelWarn},
		{"warn-boundary-500", execStats{OK: true, Total: 5, AvgMS: 500}, levelWarn},
		{"bad-501", execStats{OK: true, Total: 5, AvgMS: 501}, levelBad},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := responseLevel(tc.in); got != tc.want {
				t.Errorf("responseLevel() = %v, want %v", got, tc.want)
			}
		})
	}
}

// Contract: memory maps to green <1GB, yellow <2GB, red ≥2GB, neutral unknown.
func TestMemoryLevel(t *testing.T) {
	cases := []struct {
		name  string
		memMB int
		want  metricLevel
	}{
		{"unknown", 0, levelNeutral},
		{"good-34", 34, levelGood},
		{"good-boundary-1023", 1023, levelGood},
		{"warn-1024", 1024, levelWarn},
		{"warn-2047", 2047, levelWarn},
		{"bad-2048", 2048, levelBad},
		{"bad-4096", 4096, levelBad},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := memoryLevel(tc.memMB); got != tc.want {
				t.Errorf("memoryLevel(%d) = %v, want %v", tc.memMB, got, tc.want)
			}
		})
	}
}

// Contract: the key prompt title distinguishes "need a key" from "your key
// stopped working".
func TestEnterKeyTitle(t *testing.T) {
	if got, want := enterKeyTitle(false), "API key required — enter…"; got != want {
		t.Errorf("enterKeyTitle(false) = %q, want %q", got, want)
	}
	if got, want := enterKeyTitle(true), "API key expired — re-enter…"; got != want {
		t.Errorf("enterKeyTitle(true) = %q, want %q", got, want)
	}
}

// Contract: execution stats parse from the UI endpoint's JSON shape.
func TestParseExecStats(t *testing.T) {
	body := []byte(`{"total_executions":24,"successful_count":20,"failed_count":4,"running_count":1,"average_duration_ms":42.5}`)
	s, err := parseExecStats(body)
	if err != nil {
		t.Fatalf("parseExecStats: %v", err)
	}
	if !s.OK || s.Total != 24 || s.Successful != 20 || s.Failed != 4 || s.Running != 1 || s.AvgMS != 42.5 {
		t.Errorf("parseExecStats = %+v, want 24/20/4/1/42.5", s)
	}
	if _, err := parseExecStats([]byte("nope")); err == nil {
		t.Error("parseExecStats on garbage = nil error, want error")
	}
}

// Contract: fetchExecStats sends the key and returns OK only on 200; any error
// or non-200 (including auth) yields OK=false so the caller omits the rows.
func TestFetchExecStats(t *testing.T) {
	var mode, gotKey string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-API-Key")
		switch mode {
		case "ok":
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"total_executions":2,"successful_count":2,"failed_count":0,"average_duration_ms":10}`)
		case "401":
			w.WriteHeader(http.StatusUnauthorized)
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer ts.Close()
	u, _ := url.Parse(ts.URL)
	t.Setenv("AGENTFIELD_PORT", u.Port())

	mode = "ok"
	if s := fetchExecStats("k"); !s.OK || s.Total != 2 {
		t.Errorf("fetchExecStats ok = %+v, want OK/Total=2", s)
	}
	if gotKey != "k" {
		t.Errorf("X-API-Key = %q, want %q", gotKey, "k")
	}
	mode = "401"
	if s := fetchExecStats("k"); s.OK {
		t.Error("fetchExecStats on 401 = OK, want OK=false")
	}
	mode = "500"
	if s := fetchExecStats("k"); s.OK {
		t.Error("fetchExecStats on 500 = OK, want OK=false")
	}
}

// Contract: online agents sort before offline, then ties break alphabetically,
// and the input slice is not mutated.
func TestSortAgents(t *testing.T) {
	in := []agentInfo{
		{ID: "zeta", Online: true},
		{ID: "alpha", Online: false},
		{ID: "beta", Online: true},
		{ID: "gamma", Online: false},
	}
	got := sortAgents(in)
	wantOrder := []string{"beta", "zeta", "alpha", "gamma"}
	for i, w := range wantOrder {
		if got[i].ID != w {
			t.Errorf("sortAgents()[%d] = %q, want %q (full: %v)", i, got[i].ID, w, ids(got))
		}
	}
	if in[0].ID != "zeta" {
		t.Error("sortAgents mutated its input slice")
	}
}

func ids(in []agentInfo) []string {
	out := make([]string, len(in))
	for i, a := range in {
		out[i] = a.ID
	}
	return out
}
