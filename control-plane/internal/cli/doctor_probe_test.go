package cli

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
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
			if got := classifyProbe(tc.exitCode, tc.stdout, tc.timedOut, false); got != tc.want {
				t.Errorf("classifyProbe(%d, %q, %v) = %q, want %q", tc.exitCode, tc.stdout, tc.timedOut, got, tc.want)
			}
		})
	}
}

func TestClassifyProbe_JSONLStream(t *testing.T) {
	cases := []struct {
		name   string
		stdout string
		want   string
	}{
		{"session only", `{"type":"session","id":"s1"}`, "empty"},
		{"assistant text", "{\"type\":\"session\",\"id\":\"s1\"}\n{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":\"OK\",\"stopReason\":\"stop\"}}", "ok"},
		{"provider error", `{"type":"message_end","message":{"role":"assistant","content":"partial","stopReason":"error","errorMessage":"provider failed"}}`, "error"},
		{"recovered", "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":\"partial\",\"stopReason\":\"error\"}}\n{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"OK\"}],\"stopReason\":\"stop\"}}", "ok"},
		{"empty text part", `{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":""}],"stopReason":"stop"}}`, "empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyProbe(0, tc.stdout, false, true); got != tc.want {
				t.Errorf("classifyProbe JSONL = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestHarnessProviders_JSONLStream(t *testing.T) {
	var got []string
	for _, provider := range harnessProviders {
		if provider.JSONLStream {
			got = append(got, provider.Name)
		}
	}
	if want := []string{"pi", "omp"}; !reflect.DeepEqual(got, want) {
		t.Errorf("JSONLStream providers = %v, want %v", got, want)
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
			if provider.Name == "aforge" && (len(provider.ProbeArgs) != 0 || provider.ProbeStdin != "") {
				t.Errorf("aforge must not declare probe input")
			}
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
			res := probeHarnessProvider("prov-"+tc.name, tc.bin, tc.args, tc.stdin, tc.timeout, false)
			if res.Status != tc.want {
				t.Errorf("status = %q, want %q (detail=%q)", res.Status, tc.want, res.Detail)
			}
			if res.Provider != "prov-"+tc.name {
				t.Errorf("provider label lost: %q", res.Provider)
			}
		})
	}
}

// Contract: aforge is surveyed by doctor but never smoke-tested — its only
// one-shot is a full coding-agent run, so it deliberately declares no ProbeArgs.
func TestHarnessProviders_AforgeIsSurveyedButNotProbed(t *testing.T) {
	if harnessProviders[0].Name != "aforge" {
		t.Fatalf("aforge must lead the provider list, got %q", harnessProviders[0].Name)
	}
	if len(harnessProviders[0].ProbeArgs) != 0 || harnessProviders[0].ProbeStdin != "" {
		t.Errorf("aforge must declare no probe input, got %+v", harnessProviders[0])
	}
}

// Contract: buildDoctorReport detects aforge through its harness specification,
// including its managed install location and aforge-specific version argument.
func TestBuildDoctorReport_AforgeDetectionUsesHarnessSpec(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	t.Run("found in AGENTFIELD_HOME/bin when not on PATH", func(t *testing.T) {
		t.Setenv("PATH", t.TempDir())
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)
		managedBin := filepath.Join(home, "bin")
		if err := os.MkdirAll(managedBin, 0o755); err != nil {
			t.Fatalf("create managed bin: %v", err)
		}
		writeHarnessTestScript(t, managedBin, "aforge", "printf 'aforge 1.2.3\\n'")

		report := buildDoctorReport(srv.URL)
		status := report.HarnessProviders["aforge"]
		if !status.Available {
			t.Errorf("aforge should be available, got %+v", status)
		}
		if status.Version != "aforge 1.2.3" {
			t.Errorf("aforge version = %q, want %q", status.Version, "aforge 1.2.3")
		}
		if want := filepath.Join(home, "bin", "aforge"); status.Path != want {
			t.Errorf("aforge path = %q, want %q", status.Path, want)
		}
	})

	t.Run("version comes from the aforge-specific version argument", func(t *testing.T) {
		binDir := t.TempDir()
		t.Setenv("PATH", binDir)
		t.Setenv("AGENTFIELD_HOME", t.TempDir())
		writeHarnessTestScript(t, binDir, "aforge", "[ \"$1\" = version ] && { printf 'aforge 9.9.9\\n'; exit 0; }; exit 1")

		report := buildDoctorReport(srv.URL)
		status := report.HarnessProviders["aforge"]
		if !status.Available {
			t.Errorf("aforge should be available, got %+v", status)
		}
		if status.Version != "aforge 9.9.9" {
			t.Errorf("aforge version = %q, want %q", status.Version, "aforge 9.9.9")
		}
	})
}

// Contract: a provider with no ProbeArgs is skipped by --probe even when
// detected, while a provider that declares them is still probed. The registry
// is swapped for synthetic entries so no real coding-agent CLI is invoked.
func TestRunHarnessProbes_SkipsProvidersWithoutProbeArgs(t *testing.T) {
	original := harnessProviders
	t.Cleanup(func() { harnessProviders = original })
	harnessProviders = []doctorHarnessProvider{
		{Name: "no-probe", Binary: "agentfield-absent-no-probe"},
		{Name: "with-probe", Binary: "agentfield-absent-with-probe", ProbeArgs: []string{"--version"}},
	}

	report := DoctorReport{HarnessProviders: map[string]ToolStatus{
		"no-probe":   {Available: true},
		"with-probe": {Available: true},
	}}
	got := runHarnessProbes(report)
	if _, ok := got["no-probe"]; ok {
		t.Error("a provider without ProbeArgs must be skipped even when available")
	}
	if _, ok := got["with-probe"]; !ok {
		t.Error("an available provider with ProbeArgs must produce a result")
	}
}

// Contract: probes run ONLY for providers doctor already detected — unavailable
// providers are never invoked.
func TestRunHarnessProbes_SkipsUndetected(t *testing.T) {
	report := DoctorReport{
		HarnessProviders: map[string]ToolStatus{
			"aforge":      {Available: false},
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
