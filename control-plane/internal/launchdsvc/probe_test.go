package launchdsvc

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// probeServer starts an httptest server and returns its port. The probes build
// their own URLs as http://localhost:<port>/…, which resolves to the same
// loopback address httptest listens on — so no URL seam is needed to point them
// at a stub.
func probeServer(t *testing.T, h http.Handler) int {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	addr, ok := srv.Listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("unexpected listener address %T", srv.Listener.Addr())
	}
	return addr.Port
}

// freePort returns a port with nothing listening on it, so a probe against it
// fails to connect the way it would against a stopped control plane.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func TestURLBuilders(t *testing.T) {
	if got := HealthURL(9111); got != "http://localhost:9111/health" {
		t.Errorf("HealthURL = %q", got)
	}
	if got := ActiveExecutionsURL(9111); got != "http://localhost:9111/api/v1/executions/active" {
		t.Errorf("ActiveExecutionsURL = %q", got)
	}
}

func TestServerHealthy(t *testing.T) {
	t.Run("responding", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/health" {
				t.Errorf("unexpected path %q", r.URL.Path)
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		}))
		if !ServerHealthy(port) {
			t.Error("a 200 on /health must report healthy")
		}
	})

	t.Run("error status is not healthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		if ServerHealthy(port) {
			t.Error("503 must not report healthy")
		}
	})

	t.Run("nothing listening", func(t *testing.T) {
		if ServerHealthy(freePort(t)) {
			t.Error("a refused connection must not report healthy")
		}
	})
}

func TestActiveExecutions(t *testing.T) {
	t.Run("counts in-flight runs", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/executions/active" {
				t.Errorf("unexpected path %q", r.URL.Path)
			}
			_, _ = w.Write([]byte(freshRuns(3)))
		}))
		n, stale, ok := ActiveExecutions(port, "")
		if !ok || n != 3 || stale != 0 {
			t.Fatalf("ActiveExecutions = (%d, %d, %v), want (3, 0, true)", n, stale, ok)
		}
	})

	t.Run("idle server reports zero", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"count":0,"runs":[]}`))
		}))
		n, stale, ok := ActiveExecutions(port, "")
		if !ok || n != 0 || stale != 0 {
			t.Fatalf("ActiveExecutions = (%d, %d, %v), want (0, 0, true)", n, stale, ok)
		}
	})

	t.Run("counts runs when count is absent", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(freshRuns(2)))
		}))
		n, stale, ok := ActiveExecutions(port, "")
		if !ok || n != 2 || stale != 0 {
			t.Fatalf("ActiveExecutions = (%d, %d, %v), want (2, 0, true)", n, stale, ok)
		}
	})

	t.Run("forwards the api key", func(t *testing.T) {
		var seen string
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			seen = r.Header.Get("X-API-Key")
			_, _ = w.Write([]byte(`{"count":1}`))
		}))
		if _, _, ok := ActiveExecutions(port, "sekret"); !ok {
			t.Fatal("expected a usable answer")
		}
		if seen != "sekret" {
			t.Errorf("X-API-Key = %q, want it forwarded", seen)
		}
	})

	// Every un-interpretable answer must report ok=false, because the install
	// path treats "unknown" as not-busy rather than blocking forever.
	t.Run("unauthorized is not trustworthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"nope"}`))
		}))
		if n, _, ok := ActiveExecutions(port, ""); ok {
			t.Fatalf("401 reported as trustworthy (n=%d)", n)
		}
	})

	t.Run("malformed json is not trustworthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"count": `))
		}))
		if _, _, ok := ActiveExecutions(port, ""); ok {
			t.Fatal("truncated JSON reported as trustworthy")
		}
	})

	t.Run("nothing listening is not trustworthy", func(t *testing.T) {
		if _, _, ok := ActiveExecutions(freePort(t), ""); ok {
			t.Fatal("a refused connection reported as trustworthy")
		}
	})
}

func TestFileSHA256AndBytesSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f")
	body := []byte("agentfield")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}

	sum, ok := FileSHA256(path)
	if !ok {
		t.Fatal("FileSHA256 failed on a readable file")
	}
	if sum != BytesSHA256(body) {
		t.Errorf("FileSHA256 = %q, BytesSHA256 = %q — must agree", sum, BytesSHA256(body))
	}
	if len(sum) != 64 || strings.ContainsAny(sum, "ghijklmnopqrstuvwxyz") {
		t.Errorf("not a hex sha256: %q", sum)
	}

	if _, ok := FileSHA256(filepath.Join(dir, "absent")); ok {
		t.Error("a missing file must report ok=false")
	}
	// A directory is openable but not readable as a stream.
	if _, ok := FileSHA256(dir); ok {
		t.Error("a directory must report ok=false")
	}
}

func TestFileHasContentsMatrix(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "plist")
	body := []byte(fmt.Sprintf("<plist>%s</plist>", ServerLabel))
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if !FileHasContents(path, body) {
		t.Error("identical contents must match")
	}
	if FileHasContents(path, append(body, '\n')) {
		t.Error("a trailing byte must not match")
	}
	if FileHasContents(filepath.Join(dir, "absent"), body) {
		t.Error("a missing file must not match")
	}
}

// freshRuns builds an executions/active payload whose runs all reported
// activity just now.
func freshRuns(n int) string {
	now := time.Now().UTC().Format(time.RFC3339)
	runs := make([]string, 0, n)
	for i := 0; i < n; i++ {
		runs = append(runs, fmt.Sprintf(
			`{"run_id":"r%d","root_status":"running","started_at":%q,"latest_activity":%q}`,
			i, now, now))
	}
	return fmt.Sprintf(`{"count":%d,"runs":[%s]}`, n, strings.Join(runs, ","))
}

// run builds one entry of the executions/active payload. latest_activity is
// passed verbatim so tests can supply a missing or malformed value.
func run(id, latestActivity string) string {
	return fmt.Sprintf(
		`{"run_id":%q,"root_status":"running","started_at":"2026-08-05T02:29:00Z","latest_activity":%s}`,
		id, latestActivity)
}

func quoted(t time.Time) string { return fmt.Sprintf("%q", t.UTC().Format(time.RFC3339)) }

func payload(runs ...string) string {
	return fmt.Sprintf(`{"count":%d,"runs":[%s]}`, len(runs), strings.Join(runs, ","))
}

// TestActiveExecutionsStaleness is the zombie-run regression. Execution cleanup
// is disabled server-side, so a wedged run sits in the active list forever; if
// it counted as busy, an upgrade of the same install would defer its restart
// permanently and never land a new binary.
func TestActiveExecutionsStaleness(t *testing.T) {
	now := time.Now()
	fresh := quoted(now.Add(-2 * time.Minute))
	// The live example: started 02:29, last touched 03:05, observed >10h later.
	zombie := quoted(now.Add(-10 * time.Hour))

	cases := []struct {
		name      string
		body      string
		wantFresh int
		wantStale int
	}{
		{
			name:      "all fresh runs block a restart",
			body:      payload(run("a", fresh), run("b", fresh)),
			wantFresh: 2, wantStale: 0,
		},
		{
			name:      "all stale runs are ignored, so the server is not busy",
			body:      payload(run("zombie", zombie)),
			wantFresh: 0, wantStale: 1,
		},
		{
			name:      "mixed: only the fresh one blocks",
			body:      payload(run("live", fresh), run("zombie", zombie), run("zombie2", zombie)),
			wantFresh: 1, wantStale: 2,
		},
		{
			// Unconfirmable liveness is stale (the owner's rule): the fixture's
			// started_at is hours old, so with no activity stamp the run cannot
			// demonstrate it is alive and must not pin an upgrade.
			name:      "missing latest_activity with an old start is stale",
			body:      payload(`{"run_id":"x","root_status":"running","started_at":"2026-08-05T02:29:00Z"}`),
			wantFresh: 0, wantStale: 1,
		},
		{
			name:      "empty latest_activity with an old start is stale",
			body:      payload(run("x", `""`)),
			wantFresh: 0, wantStale: 1,
		},
		{
			name:      "malformed latest_activity with an old start is stale",
			body:      payload(run("x", `"not-a-timestamp"`)),
			wantFresh: 0, wantStale: 1,
		},
		{
			// The one fallback before assuming stale: a demonstrably young run
			// (older server reporting no activity stamps) is still protected.
			name: "missing latest_activity with a young start is fresh",
			body: payload(fmt.Sprintf(
				`{"run_id":"x","root_status":"running","started_at":%s}`,
				quoted(now.Add(-2*time.Minute)))),
			wantFresh: 1, wantStale: 0,
		},
		{
			name:      "no usable timestamps at all is stale",
			body:      payload(`{"run_id":"x","root_status":"running","started_at":"garbage"}`),
			wantFresh: 0, wantStale: 1,
		},
		{
			// An older server may report a count with no run detail to age it.
			// No evidence of liveness → stale, kept visible via the stale count.
			name:      "count without runs is unconfirmable, so stale",
			body:      `{"count":2,"runs":[]}`,
			wantFresh: 0, wantStale: 2,
		},
		{
			name:      "idle server",
			body:      `{"count":0,"runs":[]}`,
			wantFresh: 0, wantStale: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := tc.body
			port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(body))
			}))
			f, s, ok := ActiveExecutions(port, "")
			if !ok {
				t.Fatal("expected a trustworthy answer")
			}
			if f != tc.wantFresh || s != tc.wantStale {
				t.Fatalf("ActiveExecutions = (fresh %d, stale %d), want (%d, %d)",
					f, s, tc.wantFresh, tc.wantStale)
			}
		})
	}
}

// TestActiveWindow covers the env override and its failure modes: a bad value
// must fall back rather than fail an install.
func TestActiveWindow(t *testing.T) {
	t.Setenv(ActiveWindowEnv, "")
	if got := ActiveWindow(); got != defaultActiveWindow {
		t.Errorf("default = %v, want %v", got, defaultActiveWindow)
	}
	t.Setenv(ActiveWindowEnv, "2h")
	if got := ActiveWindow(); got != 2*time.Hour {
		t.Errorf("override = %v, want 2h", got)
	}
	t.Setenv(ActiveWindowEnv, "  90s  ")
	if got := ActiveWindow(); got != 90*time.Second {
		t.Errorf("trimmed override = %v, want 90s", got)
	}
	for _, bad := range []string{"soon", "-5m", "0", "12"} {
		t.Setenv(ActiveWindowEnv, bad)
		if got := ActiveWindow(); got != defaultActiveWindow {
			t.Errorf("%q should fall back to %v, got %v", bad, defaultActiveWindow, got)
		}
	}
}

// TestActiveWindowHonouredByProbe: the window is what decides, end to end.
func TestActiveWindowHonouredByProbe(t *testing.T) {
	body := payload(run("a", quoted(time.Now().Add(-45*time.Minute))))
	port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))

	t.Setenv(ActiveWindowEnv, "") // default 30m → 45m old is stale
	if f, s, _ := ActiveExecutions(port, ""); f != 0 || s != 1 {
		t.Fatalf("default window: fresh=%d stale=%d, want 0/1", f, s)
	}
	t.Setenv(ActiveWindowEnv, "2h") // widened → the same run is fresh again
	if f, s, _ := ActiveExecutions(port, ""); f != 1 || s != 0 {
		t.Fatalf("2h window: fresh=%d stale=%d, want 1/0", f, s)
	}
}

// TestSplitActiveRunsIsPure pins the boundary condition without a socket: a run
// exactly at the window edge still counts as fresh.
func TestSplitActiveRunsIsPure(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	parsed := activeExecutionsResponse{Runs: []activeRun{
		{RunID: "edge", LatestActivity: now.Add(-30 * time.Minute).Format(time.RFC3339)},
		{RunID: "just-over", LatestActivity: now.Add(-31 * time.Minute).Format(time.RFC3339)},
	}}
	f, s, ok := splitActiveRuns(parsed, now, 30*time.Minute)
	if !ok || f != 1 || s != 1 {
		t.Fatalf("splitActiveRuns = (%d, %d, %v), want (1, 1, true)", f, s, ok)
	}
}
