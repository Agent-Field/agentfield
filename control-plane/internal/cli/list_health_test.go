package cli

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// Contract: reconcileHealth renders agreement plainly, flags every disagreement,
// and never treats an unreachable control plane as an error.
func TestReconcileHealth(t *testing.T) {
	cases := []struct {
		name           string
		registryStatus string
		cpHealth       string
		found          bool
		reachable      bool
		wantDisplay    string
		wantDiscrep    bool
	}{
		{"unreachable", "running", "", false, false, "unknown (control plane unreachable)", false},
		{"agree-active", "running", "active", true, true, "active", false},
		{"agree-degraded", "running", "degraded", true, true, "degraded", false},
		{"running-but-inactive", "running", "inactive", true, true, "inactive (mismatch)", true},
		{"running-but-absent", "running", "", false, true, "not on control plane (mismatch)", true},
		{"stopped-but-active", "stopped", "active", true, true, "active (mismatch)", true},
		{"agree-stopped-absent", "stopped", "", false, true, "—", false},
		{"stopped-and-inactive", "stopped", "inactive", true, true, "inactive", false},
		{"found-empty-health-running", "running", "", true, true, "unknown (mismatch)", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			display, discrep := reconcileHealth(tc.registryStatus, tc.cpHealth, tc.found, tc.reachable)
			if display != tc.wantDisplay {
				t.Errorf("display = %q, want %q", display, tc.wantDisplay)
			}
			if discrep != tc.wantDiscrep {
				t.Errorf("discrepancy = %v, want %v", discrep, tc.wantDiscrep)
			}
		})
	}
}

func nodesServer(t *testing.T, nodes []controlPlaneNode) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/nodes" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.URL.Query().Get("show_all") != "true" {
			t.Errorf("expected show_all=true so inactive nodes are visible, got %q", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"nodes": nodes, "count": len(nodes)})
	}))
}

func fixtureRegistry() *packages.InstallationRegistry {
	// Paths are intentionally nonexistent so metadata resolution falls back to
	// the registry name (which equals the control-plane id in these fixtures).
	mk := func(status string) packages.InstalledPackage {
		return packages.InstalledPackage{Status: status, Path: "/nonexistent/path"}
	}
	return &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"agree-running":  mk("running"),
			"dead-node":      mk("running"), // CP says inactive
			"orphan-running": mk("running"), // absent from CP
			"revived":        mk("stopped"), // CP says active
			"agree-stopped":  mk("stopped"), // absent from CP
		},
	}
}

// Contract: af list merges the local registry with /api/v1/nodes — agreeing
// statuses render plainly, disagreements are flagged.
func TestResolveNodeHealth_Merge(t *testing.T) {
	srv := nodesServer(t, []controlPlaneNode{
		{ID: "agree-running", HealthStatus: "active"},
		{ID: "dead-node", HealthStatus: "inactive"},
		{ID: "revived", HealthStatus: "active"},
	})
	defer srv.Close()

	got := resolveNodeHealth(srv.URL, fixtureRegistry())

	checks := map[string]struct {
		display string
		discrep bool
	}{
		"agree-running":  {"active", false},
		"dead-node":      {"inactive (mismatch)", true},
		"orphan-running": {"not on control plane (mismatch)", true},
		"revived":        {"active (mismatch)", true},
		"agree-stopped":  {"—", false},
	}
	for name, want := range checks {
		g := got[name]
		if g.Display != want.display || g.Discrepancy != want.discrep {
			t.Errorf("%s: got (%q, %v), want (%q, %v)", name, g.Display, g.Discrepancy, want.display, want.discrep)
		}
	}
}

// Contract: an unreachable control plane yields "unknown" health for every node
// without erroring.
func TestResolveNodeHealth_Unreachable(t *testing.T) {
	got := resolveNodeHealth("http://127.0.0.1:1", fixtureRegistry())
	if len(got) != 5 {
		t.Fatalf("expected 5 nodes, got %d", len(got))
	}
	for name, h := range got {
		if !strings.Contains(h.Display, "unknown") {
			t.Errorf("%s: expected unknown health, got %q", name, h.Display)
		}
		if h.Discrepancy {
			t.Errorf("%s: unreachable control plane must not flag a discrepancy", name)
		}
	}
}

func TestFetchControlPlaneNodes_Reachable(t *testing.T) {
	srv := nodesServer(t, []controlPlaneNode{{ID: "n1", HealthStatus: "active"}})
	defer srv.Close()

	nodes, reachable := fetchControlPlaneNodes(srv.URL)
	if !reachable {
		t.Fatalf("expected reachable=true")
	}
	if n, ok := nodes["n1"]; !ok || n.HealthStatus != "active" {
		t.Errorf("expected n1 active, got %+v (ok=%v)", n, ok)
	}
}

func TestFetchControlPlaneNodes_Unreachable(t *testing.T) {
	if _, reachable := fetchControlPlaneNodes("http://127.0.0.1:1"); reachable {
		t.Errorf("expected reachable=false for a dead control plane")
	}
	if _, reachable := fetchControlPlaneNodes(""); reachable {
		t.Errorf("expected reachable=false for an empty URL")
	}
}
