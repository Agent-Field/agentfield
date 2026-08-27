package cli

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"reflect"
	"testing"
	"time"
)

// Contract: classifyProbe maps (exit code, stdout, timed-out) to the four probe
// statuses, with timeout taking precedence over a non-zero exit and an empty
// completion on a clean exit reported distinctly from an error.
func TestClassifyProbe(t *testing.T) {
	cases := []struct {
		name     string
		exitCode int
		stdout   string
		timedOut bool
		want     string
	}{
		{"ok", 0, "OK\n", false, "ok"},
		{"empty exit zero", 0, "", false, "empty"},
		{"empty whitespace only", 0, "   \n\t", false, "empty"},
		{"error nonzero", 1, "partial", false, "error"},
		{"error nonzero no output", 127, "", false, "error"},
		{"timeout wins over exit code", -1, "", true, "timeout"},
		{"timeout wins even with output", 1, "some", true, "timeout"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyProbe(tc.exitCode, tc.stdout, tc.timedOut); got != tc.want {
				t.Errorf("classifyProbe(%d, %q, %v) = %q, want %q", tc.exitCode, tc.stdout, tc.timedOut, got, tc.want)
			}
		})
	}
}

func TestHarnessProviders_ProbeInputContract(t *testing.T) {
	wantStdinArgs := []string{"--print", "--mode", "json"}
	for _, provider := range harnessProviders {
		switch provider.Name {
		case "pi", "omp":
			if !reflect.DeepEqual(provider.ProbeArgs, wantStdinArgs) {
				t.Errorf("%s ProbeArgs = %v, want %v", provider.Name, provider.ProbeArgs, wantStdinArgs)
			}
			if provider.ProbeStdin == "" {
				t.Errorf("%s ProbeStdin must be non-empty", provider.Name)
			}
			for _, arg := range provider.ProbeArgs {
				if arg == provider.ProbeStdin {
					t.Errorf("%s prompt %q must not appear in ProbeArgs", provider.Name, provider.ProbeStdin)
				}
			}
		default:
			if provider.ProbeStdin != "" {
				t.Errorf("positional-prompt provider %s ProbeStdin = %q, want empty", provider.Name, provider.ProbeStdin)
			}
		}
	}
}

// End-to-end wiring of runProbeCommand -> classifyProbe over real processes, so
// each classification path is exercised through the actual command runner.
func TestProbeHarnessProvider_RealProcesses(t *testing.T) {
	cases := []struct {
		name    string
		bin     string
		args    []string
		stdin   string
		timeout time.Duration
		want    string
	}{
		{"ok", "echo", []string{"OK"}, "", 5 * time.Second, "ok"},
		{"empty", "true", nil, "", 5 * time.Second, "empty"},
		{"error", "false", nil, "", 5 * time.Second, "error"},
		{"timeout", "sleep", []string{"5"}, "", 200 * time.Millisecond, "timeout"},
		{"stdin", "cat", nil, "Say OK", 5 * time.Second, "ok"},
		{"empty stdin", "cat", nil, "", 5 * time.Second, "empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := exec.LookPath(tc.bin); err != nil {
				t.Skipf("%s not available: %v", tc.bin, err)
			}
			res := probeHarnessProvider("prov-"+tc.name, tc.bin, tc.args, tc.stdin, tc.timeout)
			if res.Status != tc.want {
				t.Errorf("status = %q, want %q (detail=%q)", res.Status, tc.want, res.Detail)
			}
			if res.Provider != "prov-"+tc.name {
				t.Errorf("provider label lost: %q", res.Provider)
			}
		})
	}
}

// Contract: probes run ONLY for providers doctor already detected — unavailable
// providers are never invoked.
func TestRunHarnessProbes_SkipsUndetected(t *testing.T) {
	report := DoctorReport{
		HarnessProviders: map[string]ToolStatus{
			"claude-code": {Available: false},
			"codex":       {Available: false},
			"gemini":      {Available: false},
			"opencode":    {Available: false},
			"pi":          {Available: false},
			"omp":         {Available: false},
		},
	}
	got := runHarnessProbes(report)
	if len(got) != 0 {
		t.Errorf("no detected providers should mean no probes, got %v", got)
	}
}

// Contract: `af doctor` without --probe performs no probe (no harness_probes in
// the report).
func TestDoctorCommand_NoProbeByDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	out := captureStdoutCLI(t, func() {
		cmd := NewDoctorCommand()
		cmd.SetArgs([]string{"--json", "--server", srv.URL})
		if err := cmd.Execute(); err != nil {
			t.Fatalf("doctor failed: %v", err)
		}
	})

	var report DoctorReport
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatalf("parse report: %v\noutput:\n%s", err, out)
	}
	if len(report.HarnessProbes) != 0 {
		t.Errorf("without --probe there must be no probes, got %v", report.HarnessProbes)
	}
}

// captureStdoutCLI captures os.Stdout while fn runs.
func captureStdoutCLI(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var b bytes.Buffer
		_, _ = io.Copy(&b, r)
		done <- b.String()
	}()
	fn()
	_ = w.Close()
	os.Stdout = orig
	return <-done
}
