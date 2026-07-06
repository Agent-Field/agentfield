package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These tests validate the behaviours the tray depends on, expressed as
// contracts rather than by mirroring the implementation. They live in a
// build-tag-free file so they compile and run on the Linux CI even though the
// tray UI itself is macOS-only.

// Contract: the port comes from AGENTFIELD_PORT when set to a valid integer,
// and defaults to 8080 otherwise (including when the value is garbage).
func TestServerPort(t *testing.T) {
	cases := []struct {
		name string
		env  string
		set  bool
		want int
	}{
		{"default when unset", "", false, 8080},
		{"honors valid env", "9090", true, 9090},
		{"falls back on garbage", "not-a-number", true, 8080},
		{"falls back on empty", "", true, 8080},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv("AGENTFIELD_PORT", tc.env)
			} else {
				_ = os.Unsetenv("AGENTFIELD_PORT")
			}
			if got := serverPort(); got != tc.want {
				t.Fatalf("serverPort() = %d, want %d", got, tc.want)
			}
		})
	}
}

// Contract: the health/dashboard URLs target the resolved port on localhost.
func TestURLsUsePort(t *testing.T) {
	t.Setenv("AGENTFIELD_PORT", "1234")
	if got, want := healthURL(), "http://localhost:1234/health"; got != want {
		t.Errorf("healthURL() = %q, want %q", got, want)
	}
	if got, want := dashboardURL(), "http://localhost:1234"; got != want {
		t.Errorf("dashboardURL() = %q, want %q", got, want)
	}
}

// Contract: a server is "running" only when /health answers HTTP 200; a 503
// (unhealthy) or an unreachable server both read as not running.
func TestCheckHealth(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ok.Close()
	if !checkHealth(ok.URL) {
		t.Error("checkHealth on a 200 server = false, want true")
	}

	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer unhealthy.Close()
	if checkHealth(unhealthy.URL) {
		t.Error("checkHealth on a 503 server = true, want false")
	}

	// Unreachable: start a server, capture its URL, then close it.
	down := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	downURL := down.URL
	down.Close()
	if checkHealth(downURL) {
		t.Error("checkHealth on a closed server = true, want false")
	}
}

// Contract: serverHealthy() wires serverPort()→healthURL()→checkHealth end to
// end against a live server.
func TestServerHealthyEndToEnd(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGENTFIELD_PORT", u.Port()) // localhost:<port> resolves to the test server.
	if !serverHealthy() {
		t.Error("serverHealthy() = false against a live /health, want true")
	}
}

// Contract: kickstart uses -k (force restart) only when a kill is requested,
// and always targets gui/<uid>/<label>.
func TestKickstartArgs(t *testing.T) {
	got := kickstartArgs(serverLabel, false)
	if len(got) < 2 || got[0] != "kickstart" {
		t.Fatalf("kickstartArgs(kill=false) = %v, want it to start with 'kickstart'", got)
	}
	for _, a := range got {
		if a == "-k" {
			t.Errorf("kickstartArgs(kill=false) unexpectedly contains -k: %v", got)
		}
	}
	if last := got[len(got)-1]; last != svcTarget(serverLabel) {
		t.Errorf("target = %q, want %q", last, svcTarget(serverLabel))
	}

	killed := kickstartArgs(serverLabel, true)
	if !contains(killed, "-k") {
		t.Errorf("kickstartArgs(kill=true) = %v, want it to contain -k", killed)
	}
	if last := killed[len(killed)-1]; last != svcTarget(serverLabel) {
		t.Errorf("target = %q, want %q", last, svcTarget(serverLabel))
	}
}

func TestSvcTargetFormat(t *testing.T) {
	target := svcTarget("ai.example.thing")
	if !strings.HasPrefix(target, "gui/") || !strings.HasSuffix(target, "/ai.example.thing") {
		t.Errorf("svcTarget = %q, want gui/<uid>/ai.example.thing", target)
	}
}

// Contract: writeFileAtomic creates the file with the requested permissions and
// fully replaces prior content on a second write.
func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "file.txt")

	if err := writeFileAtomic(path, []byte("first"), 0o600); err != nil {
		t.Fatalf("first write: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("perm = %v, want 0600", info.Mode().Perm())
	}

	if err := writeFileAtomic(path, []byte("second"), 0o644); err != nil {
		t.Fatalf("second write: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "second" {
		t.Errorf("content = %q, want %q (content must be fully replaced)", got, "second")
	}

	// No stray temp files left behind in the directory.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".af-tray-") {
			t.Errorf("leftover temp file: %s", e.Name())
		}
	}
}

// Contract: paths are anchored under $HOME with the expected layout.
func TestPaths(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)

	checks := map[string]string{
		"agentfieldDir":    filepath.Join(root, ".agentfield"),
		"binDir":           filepath.Join(root, ".agentfield", "bin"),
		"logsDir":          filepath.Join(root, ".agentfield", "logs"),
		"serverLogPath":    filepath.Join(root, ".agentfield", "logs", "control-plane.log"),
		"trayLogPath":      filepath.Join(root, ".agentfield", "logs", "tray.log"),
		"launchAgentsDir":  filepath.Join(root, "Library", "LaunchAgents"),
		"appBundleDir":     filepath.Join(root, "Applications", "AgentField.app"),
		"trayBundleBinary": filepath.Join(root, "Applications", "AgentField.app", "Contents", "MacOS", "af-tray"),
		"serverPlistPath":  filepath.Join(root, "Library", "LaunchAgents", serverLabel+".plist"),
		"trayPlistPath":    filepath.Join(root, "Library", "LaunchAgents", trayLabel+".plist"),
	}
	got := map[string]string{
		"agentfieldDir":    agentfieldDir(),
		"binDir":           binDir(),
		"logsDir":          logsDir(),
		"serverLogPath":    serverLogPath(),
		"trayLogPath":      trayLogPath(),
		"launchAgentsDir":  launchAgentsDir(),
		"appBundleDir":     appBundleDir(),
		"trayBundleBinary": trayBundleBinaryPath(),
		"serverPlistPath":  serverPlistPath(),
		"trayPlistPath":    trayPlistPath(),
	}
	for k, want := range checks {
		if got[k] != want {
			t.Errorf("%s = %q, want %q", k, got[k], want)
		}
	}
	if home() != root {
		t.Errorf("home() = %q, want %q", home(), root)
	}
}

// Contract: serverBinaryPath prefers an installed, executable copy under
// ~/.agentfield/bin/agentfield.
func TestServerBinaryPathPrefersInstalled(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	installed := filepath.Join(root, ".agentfield", "bin", "agentfield")
	if err := os.MkdirAll(filepath.Dir(installed), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(installed, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := serverBinaryPath(); got != installed {
		t.Errorf("serverBinaryPath() = %q, want the installed copy %q", got, installed)
	}
	if !isExecutable(installed) {
		t.Error("isExecutable on a 0755 file = false, want true")
	}
	if isExecutable(filepath.Dir(installed)) {
		t.Error("isExecutable on a directory = true, want false")
	}
}

// Contract: with no installed copy, serverBinaryPath falls back to `af` on
// PATH, then to `agentfield`, then to the (nonexistent) installed candidate.
func TestServerBinaryPathFallbacks(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root) // no ~/.agentfield/bin/agentfield present

	// A fake `af` on PATH is preferred over the missing installed copy.
	pathDir := t.TempDir()
	afPath := filepath.Join(pathDir, "af")
	if err := os.WriteFile(afPath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathDir)
	if got := serverBinaryPath(); got != afPath {
		t.Errorf("serverBinaryPath() = %q, want PATH copy %q", got, afPath)
	}

	// Only `agentfield` on PATH (no `af`) → returns the agentfield copy.
	pathDir2 := t.TempDir()
	agPath := filepath.Join(pathDir2, "agentfield")
	if err := os.WriteFile(agPath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathDir2)
	if got := serverBinaryPath(); got != agPath {
		t.Errorf("serverBinaryPath() = %q, want agentfield copy %q", got, agPath)
	}

	// Nothing installed and nothing on PATH → the installed candidate path.
	t.Setenv("PATH", t.TempDir())
	want := filepath.Join(root, ".agentfield", "bin", "agentfield")
	if got := serverBinaryPath(); got != want {
		t.Errorf("serverBinaryPath() fallback = %q, want %q", got, want)
	}
}

// Contract: writeFileAtomic surfaces an error when the destination directory
// can't be created (here: a parent path component is a regular file).
func TestWriteFileAtomicMkdirError(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(filepath.Join(blocker, "sub", "f"), []byte("y"), 0o644); err == nil {
		t.Error("writeFileAtomic under a file-as-parent = nil error, want error")
	}

	// Renaming the temp file over an existing directory fails.
	target := filepath.Join(dir, "adir")
	if err := os.Mkdir(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(target, []byte("z"), 0o644); err == nil {
		t.Error("writeFileAtomic onto an existing directory = nil error, want error")
	}
}

// Contract: the control-plane launchd agent must (a) run the `server`
// subcommand of the resolved binary, (b) NOT open a browser, (c) restart only
// on crash (KeepAlive SuccessfulExit=false) so a graceful Stop sticks, and
// (d) run at load.
func TestServerPlistContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	p := serverPlist()

	mustContain(t, p, "<string>"+serverLabel+"</string>")
	mustContain(t, p, "<string>server</string>")
	mustContain(t, p, "<string>--open=false</string>")
	mustContain(t, p, "<key>RunAtLoad</key><true/>")
	// KeepAlive only on failure → graceful stop is not relaunched.
	mustContain(t, p, "<key>SuccessfulExit</key><false/>")
	mustContain(t, p, "<string>"+serverBinaryPath()+"</string>")
}

// Contract: the tray launchd agent restarts only on crash (Crashed=true) so a
// clean Quit / no-GUI exit is never relaunched, runs at load, and launches the
// bundled binary with `run`.
func TestTrayPlistContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	p := trayPlist()

	mustContain(t, p, "<string>"+trayLabel+"</string>")
	mustContain(t, p, "<string>run</string>")
	mustContain(t, p, "<key>RunAtLoad</key><true/>")
	mustContain(t, p, "<key>Crashed</key><true/>")
	mustContain(t, p, "<string>"+trayBundleBinaryPath()+"</string>")
}

// Contract: the .app Info.plist marks the tray as a menu-bar-only agent
// (LSUIElement) with the expected identity.
func TestInfoPlistContract(t *testing.T) {
	p := infoPlist()
	mustContain(t, p, "<key>LSUIElement</key><true/>")
	mustContain(t, p, "<key>CFBundleExecutable</key><string>af-tray</string>")
	mustContain(t, p, "<string>"+trayLabel+"</string>")
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func mustContain(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("expected output to contain %q\n---\n%s", needle, haystack)
	}
}
