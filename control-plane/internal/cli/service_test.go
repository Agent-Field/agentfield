package cli

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
)

// serviceTestServer starts a stub control plane and points servicePort() at it
// via AGENTFIELD_PORT, which is how the real command finds the server.
func serviceTestServer(t *testing.T, h http.Handler) int {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	port := srv.Listener.Addr().(*net.TCPAddr).Port
	return port
}

func TestServiceCommandWiring(t *testing.T) {
	cmd := NewServiceCommand()
	if cmd.Use != "service" {
		t.Fatalf("Use = %q", cmd.Use)
	}
	want := map[string]bool{"status": false, "stop": false, "restart": false, "uninstall": false}
	for _, sub := range cmd.Commands() {
		name := strings.Fields(sub.Use)[0]
		if _, ok := want[name]; ok {
			want[name] = true
		}
	}
	for name, found := range want {
		if !found {
			t.Errorf("subcommand %q not registered", name)
		}
	}
	// The group's help must name the trap it exists to explain.
	if !strings.Contains(cmd.Long, "killing the process does") {
		t.Error("service help should explain that a plain kill is not enough")
	}
}

// TestServiceCommandRegisteredOnRoot pins that `af service` is reachable, on
// every platform, so the command self-documents even where launchd is absent.
func TestServiceCommandRegisteredOnRoot(t *testing.T) {
	root := NewRootCommand(func(*cobra.Command, []string) {}, VersionInfo{})
	for _, c := range root.Commands() {
		if strings.Fields(c.Use)[0] == "service" {
			return
		}
	}
	t.Fatal("`service` is not registered on the root command")
}

func TestServiceHelpRuns(t *testing.T) {
	cmd := NewServiceCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--help"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("service --help: %v", err)
	}
	for _, want := range []string{"status", "stop", "restart", "uninstall"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}

func TestServicePort(t *testing.T) {
	t.Setenv("AGENTFIELD_PORT", "")
	if got := servicePort(); got != 8080 {
		t.Errorf("default port = %d, want 8080", got)
	}
	t.Setenv("AGENTFIELD_PORT", "9433")
	if got := servicePort(); got != 9433 {
		t.Errorf("port from env = %d, want 9433", got)
	}
	t.Setenv("AGENTFIELD_PORT", "not-a-number")
	if got := servicePort(); got != 8080 {
		t.Errorf("garbage env should fall back to 8080, got %d", got)
	}
}

func TestServiceHome(t *testing.T) {
	if serviceHome() == "" {
		t.Error("serviceHome should resolve a home directory")
	}
}

func TestServiceServerVersion(t *testing.T) {
	t.Run("reads the version field", func(t *testing.T) {
		port := serviceTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"status":"healthy","version":"1.2.3"}`))
		}))
		if got := serviceServerVersion(port); got != "1.2.3" {
			t.Errorf("version = %q, want 1.2.3", got)
		}
	})

	t.Run("absent field is empty, not an error", func(t *testing.T) {
		port := serviceTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		}))
		if got := serviceServerVersion(port); got != "" {
			t.Errorf("version = %q, want empty", got)
		}
	})

	t.Run("malformed body is empty", func(t *testing.T) {
		port := serviceTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`not json`))
		}))
		if got := serviceServerVersion(port); got != "" {
			t.Errorf("version = %q, want empty", got)
		}
	})

	t.Run("no server is empty", func(t *testing.T) {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := l.Addr().(*net.TCPAddr).Port
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
		if got := serviceServerVersion(port); got != "" {
			t.Errorf("version = %q, want empty", got)
		}
	})
}

// TestPrintServiceStatus walks every rendering branch. printServiceStatus is
// what a user actually reads, so each state has to produce the right sentence.
func TestPrintServiceStatus(t *testing.T) {
	cases := []struct {
		name string
		st   serviceStatus
		want []string
		deny []string
	}{
		{
			name: "loaded healthy busy with version",
			st: serviceStatus{
				Supported: true, Loaded: true, Healthy: true, Port: 8080,
				Program: "/u/.agentfield/bin/agentfield", Version: "1.0.0",
				ActiveExecutions: 2, ActiveKnown: true,
			},
			want: []string{"loaded (starts at login)", "/u/.agentfield/bin/agentfield",
				"responding on :8080 (version 1.0.0)", "2 workflow(s)"},
		},
		{
			name: "loaded healthy idle without version",
			st: serviceStatus{
				Supported: true, Loaded: true, Healthy: true, Port: 8080, ActiveKnown: true,
			},
			want: []string{"responding on :8080", "0 workflow(s)"},
			deny: []string{"version"},
		},
		{
			name: "stale runs are named alongside the live count",
			st: serviceStatus{
				Supported: true, Loaded: true, Healthy: true, Port: 8080,
				ActiveExecutions: 2, StaleExecutions: 1, ActiveKnown: true,
			},
			want: []string{"2 workflow(s) (plus 1 stale, idle >"},
		},
		{
			name: "healthy but in-flight count unreadable",
			st:   serviceStatus{Supported: true, Loaded: true, Healthy: true, Port: 8080},
			want: []string{"unknown (endpoint unreadable"},
		},
		{
			name: "registered but not answering",
			st:   serviceStatus{Supported: true, Loaded: true, Port: 8080},
			want: []string{"not responding on :8080"},
			deny: []string{"In flight"},
		},
		{
			name: "not loaded",
			st:   serviceStatus{Supported: true, Port: 8080},
			want: []string{"not loaded"},
		},
		{
			name: "unsupported platform",
			st:   serviceStatus{Port: 8080},
			want: []string{"not applicable"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := captureStdoutCLI(t, func() { printServiceStatus(tc.st) })
			for _, w := range tc.want {
				if !strings.Contains(out, w) {
					t.Errorf("output missing %q:\n%s", w, out)
				}
			}
			for _, d := range tc.deny {
				if strings.Contains(out, d) {
					t.Errorf("output should not mention %q:\n%s", d, out)
				}
			}
		})
	}
}

func serviceItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestServiceStatusRunE drives the command end to end against a stub server,
// in both output modes. It never touches launchd: on macOS the registration
// probe is read-only, and the assertions below only concern health fields.
func TestServiceStatusRunE(t *testing.T) {
	// Stub the registration probe: status assembly is exercised on every
	// platform, and no test ever shells out to launchctl.
	orig := agentLoadedFn
	agentLoadedFn = func(string) bool { return true }
	t.Cleanup(func() { agentLoadedFn = orig })

	port := serviceTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_, _ = w.Write([]byte(`{"status":"healthy","version":"9.9.9"}`))
		case "/api/v1/executions/active":
			now := time.Now().UTC().Format(time.RFC3339)
			old := time.Now().Add(-10 * time.Hour).UTC().Format(time.RFC3339)
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"count":5,"runs":[{"run_id":"a","latest_activity":%q},`+
					`{"run_id":"b","latest_activity":%q},`+
					`{"run_id":"c","latest_activity":%q},`+
					`{"run_id":"d","latest_activity":%q},`+
					`{"run_id":"zombie","latest_activity":%q}]}`,
				now, now, now, now, old)))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	home := t.TempDir()
	t.Setenv("AGENTFIELD_PORT", serviceItoa(port))
	t.Setenv("HOME", home)

	// A plist on disk exercises the branch that reports which binary launchd
	// would run — the field that made the takeover bug visible.
	agents := filepath.Join(home, "Library", "LaunchAgents")
	if err := os.MkdirAll(agents, 0o755); err != nil {
		t.Fatal(err)
	}
	plist := `<plist><dict><key>ProgramArguments</key><array>` +
		`<string>/opt/demo/bin/agentfield</string><string>server</string></array>` +
		`<key>WorkingDirectory</key><string>/opt/demo</string></dict></plist>`
	if err := os.WriteFile(filepath.Join(agents, "ai.agentfield.server.plist"),
		[]byte(plist), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("text", func(t *testing.T) {
		out := captureStdoutCLI(t, func() {
			cmd := newServiceStatusCmd()
			cmd.SetArgs(nil)
			if err := cmd.Execute(); err != nil {
				t.Fatalf("status: %v", err)
			}
		})
		for _, want := range []string{"version 9.9.9", "4 workflow(s) (plus 1 stale, idle >30m0s)", "/opt/demo/bin/agentfield"} {
			if !strings.Contains(out, want) {
				t.Errorf("status output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("json", func(t *testing.T) {
		out := captureStdoutCLI(t, func() {
			cmd := newServiceStatusCmd()
			cmd.SetArgs([]string{"--json"})
			if err := cmd.Execute(); err != nil {
				t.Fatalf("status --json: %v", err)
			}
		})
		for _, want := range []string{`"healthy": true`, `"version": "9.9.9"`,
			`"active_executions": 4`, `"stale_executions": 1`, `"active_known": true`} {
			if !strings.Contains(out, want) {
				t.Errorf("json output missing %q:\n%s", want, out)
			}
		}
	})
}
