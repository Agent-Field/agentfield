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

// Contract: the headline reflects each fleet state, and the OK line reports
// online/total/skills.
func TestFleetHeadline(t *testing.T) {
	cases := []struct {
		name string
		in   fleetSummary
		want string
	}{
		{"auth", fleetSummary{Status: fleetAuthRequired}, "🔒 API key required"},
		{"unavailable", fleetSummary{Status: fleetUnavailable}, "Agents unavailable"},
		{"empty", fleetSummary{Status: fleetOK, Total: 0}, "No agents registered yet"},
		{"counts", fleetSummary{Status: fleetOK, Online: 2, Total: 3, Skills: 7}, "2 of 3 agents online · 7 skills"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := fleetHeadline(tc.in); got != tc.want {
				t.Errorf("fleetHeadline() = %q, want %q", got, tc.want)
			}
		})
	}
}

// Contract: an online agent shows a filled dot, an offline one a hollow dot, and
// the capability count is pluralized correctly.
func TestAgentLine(t *testing.T) {
	cases := []struct {
		name string
		in   agentInfo
		want string
	}{
		{"online plural", agentInfo{ID: "weather", Online: true, Skills: 2, Reasoners: 1}, "●  weather — 3 skills"},
		{"online singular", agentInfo{ID: "solo", Online: true, Skills: 1}, "●  solo — 1 skill"},
		{"offline", agentInfo{ID: "stale", Online: false, Skills: 4}, "○  stale — 4 skills"},
		{"no caps", agentInfo{ID: "bare", Online: true}, "●  bare"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := agentLine(tc.in); got != tc.want {
				t.Errorf("agentLine() = %q, want %q", got, tc.want)
			}
		})
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
