package packages

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func startStopTestProcess(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestProcessStartIdentityChild$")
	cmd.Env = append(os.Environ(), "AGENTFIELD_PROCESS_IDENTITY_CHILD=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	return cmd
}

func writeRunningLegacyRegistry(t *testing.T, home string, port, pid int) {
	t.Helper()
	data, err := yaml.Marshal(InstallationRegistry{Installed: map[string]InstalledPackage{
		"demo": {
			Name: "demo", Status: "running", Path: filepath.Join(home, "packages", "demo"),
			Runtime: RuntimeInfo{Port: &port, PID: &pid},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o640); err != nil {
		t.Fatal(err)
	}
}

func TestStopPackageForReinstallOwnsLegacyProcessThroughHealthAndShutsDownFirst(t *testing.T) {
	home := t.TempDir()
	cmd := startStopTestProcess(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	var shutdowns atomic.Int32
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/health" {
			_, _ = w.Write([]byte(`{"node_id":"demo"}`))
			return
		}
		if request.URL.Path != "/shutdown" || request.Method != http.MethodPost {
			t.Errorf("unexpected shutdown request %s %s", request.Method, request.URL.Path)
			http.NotFound(w, request)
			return
		}
		shutdowns.Add(1)
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		w.WriteHeader(http.StatusOK)
	})}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	writeRunningLegacyRegistry(t, home, port, cmd.Process.Pid)

	state, err := StopPackageForReinstall(context.Background(), home, "demo")
	if err != nil {
		t.Fatal(err)
	}
	if !state.WasRunning || state.PreferredPort != port || shutdowns.Load() != 1 {
		t.Fatalf("state=%+v shutdowns=%d", state, shutdowns.Load())
	}
	entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["demo"]
	if entry.Status != "stopped" || entry.DesiredState != DesiredStateRunning || entry.Runtime.PID != nil || entry.Runtime.Port == nil || *entry.Runtime.Port != port {
		t.Fatalf("stopped entry=%+v", entry)
	}
}

func TestE7StopPackageForReinstallRefusesSilentProcessWithoutIdentity(t *testing.T) {
	home := t.TempDir()
	cmd := startStopTestProcess(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	writeRunningLegacyRegistry(t, home, port, cmd.Process.Pid)

	state, err := StopPackageForReinstall(context.Background(), home, "demo")
	if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("process %d", cmd.Process.Pid)) || !strings.Contains(err.Error(), "stop it manually") {
		t.Fatalf("state=%+v err=%v", state, err)
	}
	if !processExists(cmd.Process.Pid) {
		t.Fatal("silent legacy port caused the unidentified PID to be signalled")
	}
	entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["demo"]
	if entry.Status != "running" || entry.Runtime.PID == nil || *entry.Runtime.PID != cmd.Process.Pid {
		t.Fatalf("record changed despite unverifiable identity: %+v", entry)
	}
}

func TestStopPackageForReinstallAnonymousTypeScriptHealthFallsBackAfterMissingShutdown(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: demo\nversion: 1.0.0\nlanguage: typescript\nagent_node:\n  node_id: demo\n"
	if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { _, _ = cmd.Process.Wait(); close(done) }()
	t.Cleanup(func() {
		select {
		case <-done:
		default:
			_ = cmd.Process.Kill()
			<-done
		}
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	var shutdowns atomic.Int32
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/health" {
			_, _ = w.Write([]byte(`{"status":"ok"}`))
			return
		}
		shutdowns.Add(1)
		http.NotFound(w, request)
	})}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	writeRunningLegacyRegistry(t, home, port, cmd.Process.Pid)
	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	entry := registry.Installed["demo"]
	entry.Path = pkgDir
	registry.Installed["demo"] = entry
	data, err := yaml.Marshal(registry)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o640); err != nil {
		t.Fatal(err)
	}

	state, err := StopPackageForReinstall(context.Background(), home, "demo")
	if err != nil || !state.WasRunning || shutdowns.Load() != 1 {
		t.Fatalf("state=%+v shutdowns=%d err=%v", state, shutdowns.Load(), err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("reinstall stop left owned TypeScript process alive after 404 fallback")
	}
}

func TestStopPackageForReinstallOwnershipAndCompatibilityFallbacks(t *testing.T) {
	t.Run("foreign node leaves recorded PID untouched and reconciles for reinstall", func(t *testing.T) {
		home := t.TempDir()
		cmd := startStopTestProcess(t)
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := listener.Addr().(*net.TCPAddr).Port
		var shutdowns atomic.Int32
		server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
			if request.URL.Path == "/health" {
				_, _ = w.Write([]byte(`{"node_id":"somebody-else"}`))
				return
			}
			shutdowns.Add(1)
		})}
		go func() { _ = server.Serve(listener) }()
		t.Cleanup(func() { _ = server.Close() })
		writeRunningLegacyRegistry(t, home, port, cmd.Process.Pid)

		state, err := StopPackageForReinstall(context.Background(), home, "demo")
		if err != nil || !state.WasRunning || shutdowns.Load() != 0 {
			t.Fatalf("state=%+v shutdowns=%d err=%v", state, shutdowns.Load(), err)
		}
		if !processExists(cmd.Process.Pid) {
			t.Fatal("foreign listener caused the recorded PID to be signalled")
		}
		entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["demo"]
		if entry.Status != "stopped" || entry.Runtime.PID != nil || entry.DesiredState != DesiredStateRunning {
			t.Fatalf("reinstall entry=%+v", entry)
		}
	})

	t.Run("identified node with missing shutdown falls back to interrupt", func(t *testing.T) {
		home := t.TempDir()
		pkgDir := filepath.Join(home, "packages", "demo")
		if err := os.MkdirAll(pkgDir, 0o755); err != nil {
			t.Fatal(err)
		}
		manifest := "name: demo\nversion: 1.0.0\nlanguage: python\nagent_node:\n  node_id: demo\n"
		if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(manifest), 0o600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		done := make(chan struct{})
		go func() { _, _ = cmd.Process.Wait(); close(done) }()
		t.Cleanup(func() {
			select {
			case <-done:
			default:
				_ = cmd.Process.Kill()
				<-done
			}
		})

		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := listener.Addr().(*net.TCPAddr).Port
		var shutdowns atomic.Int32
		server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
			if request.URL.Path == "/health" {
				_, _ = w.Write([]byte(`{"node_id":"demo"}`))
				return
			}
			shutdowns.Add(1)
			http.NotFound(w, request)
		})}
		go func() { _ = server.Serve(listener) }()
		t.Cleanup(func() { _ = server.Close() })

		writeRunningLegacyRegistry(t, home, port, cmd.Process.Pid)
		registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
		entry := registry.Installed["demo"]
		entry.Path = pkgDir
		registry.Installed["demo"] = entry
		data, err := yaml.Marshal(registry)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o640); err != nil {
			t.Fatal(err)
		}

		if _, err := StopPackageForReinstall(context.Background(), home, "demo"); err != nil {
			t.Fatal(err)
		}
		if shutdowns.Load() != 1 {
			t.Fatalf("shutdown requests=%d, want one 404 compatibility probe", shutdowns.Load())
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("404 shutdown did not fall back to interrupt")
		}
	})
}

func TestStopRecordedProcessRejectedShutdownFallsBackToSignals(t *testing.T) {
	cmd := startStopTestProcess(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/health" {
			_, _ = w.Write([]byte(`{"node_id":"demo"}`))
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	})}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })

	entry := InstalledPackage{Name: "demo", Runtime: RuntimeInfo{Port: &port, PID: &cmd.Process.Pid}}
	result, err := StopRecordedProcess(context.Background(), "demo", entry)
	if err != nil || !result.Owned || !result.HTTPAttempted || result.HTTPAccepted || !result.InterruptSent {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestAssessRecordedProcessUnknownIdentityMatrix(t *testing.T) {
	pid, port := 42, 8123
	entry := InstalledPackage{Name: "demo-node", Runtime: RuntimeInfo{PID: &pid, Port: &port}}
	for _, test := range []struct {
		name       string
		state      RuntimeProcessState
		health     HealthIdentity
		want       RecordedProcessOwnership
		wantSignal bool
	}{
		{name: "Go-shaped anonymous health is ours", state: RuntimeProcessUnknown, health: HealthIdentity{Healthy: true}, want: RecordedProcessOursHealthy, wantSignal: true},
		{name: "equivalent node id is ours", state: RuntimeProcessUnknown, health: HealthIdentity{Healthy: true, NodeID: "DEMO_NODE"}, want: RecordedProcessOursHealthy, wantSignal: true},
		{name: "different node id is foreign", state: RuntimeProcessUnknown, health: HealthIdentity{Healthy: true, NodeID: "somebody-else"}, want: RecordedProcessForeign},
		{name: "silent port keeps an existing unknown process", state: RuntimeProcessUnknown, health: HealthIdentity{}, want: RecordedProcessUnknown},
		{name: "known dead PID with anonymous healthy replacement is ours but cannot be signalled", state: RuntimeProcessDead, health: HealthIdentity{Healthy: true}, want: RecordedProcessOursHealthy},
	} {
		t.Run(test.name, func(t *testing.T) {
			assessment := AssessRecordedProcessWith(
				context.Background(), "demo-node", entry,
				func(RuntimeInfo) RuntimeProcessState { return test.state },
				func(context.Context, int, string) HealthIdentity { return test.health },
				ProcessConfirmationPolicy{Attempts: 1, ProcessExists: func(int) bool { return test.state != RuntimeProcessDead }},
			)
			if assessment.Ownership != test.want || assessment.SignalAllowed != test.wantSignal {
				t.Fatalf("assessment=%+v, want ownership=%v signal=%v", assessment, test.want, test.wantSignal)
			}
		})
	}
}

func TestWindowsStopLadderWithInjectedProcessPrimitives(t *testing.T) {
	t.Run("graceful taskkill succeeds without force", func(t *testing.T) {
		forced := false
		interruptSent, forceNeeded, err := stopProcessWith(
			42,
			func(int) error { return nil },
			func(int) bool { return false },
			func(int) error { forced = true; return nil },
		)
		if err != nil || !interruptSent || forceNeeded || forced {
			t.Fatalf("interrupt=%v forceNeeded=%v forced=%v err=%v", interruptSent, forceNeeded, forced, err)
		}
	})

	t.Run("failed graceful taskkill escalates to forced taskkill", func(t *testing.T) {
		forced := false
		interruptSent, forceNeeded, err := stopProcessWith(
			42,
			func(int) error { return errors.New("taskkill failed") },
			func(int) bool { return true },
			func(int) error { forced = true; return nil },
		)
		if err != nil || interruptSent || !forceNeeded || !forced {
			t.Fatalf("interrupt=%v forceNeeded=%v forced=%v err=%v", interruptSent, forceNeeded, forced, err)
		}
	})

	t.Run("already finished is success and reinstall registry is stopped", func(t *testing.T) {
		home := t.TempDir()
		writeRunningLegacyRegistry(t, home, 8123, 42)
		state, err := stopPackageForReinstallWith(
			context.Background(), home, "demo",
			func(context.Context, string, InstalledPackage) (StopProcessResult, error) {
				interruptSent, forceNeeded, err := stopProcessWith(
					42,
					func(int) error { return os.ErrProcessDone },
					func(int) bool { t.Fatal("already-finished error must not probe or force"); return true },
					func(int) error { t.Fatal("already-finished error must not force"); return nil },
				)
				return StopProcessResult{InterruptSent: interruptSent, ForceKillNeeded: forceNeeded}, err
			},
		)
		if err != nil || !state.WasRunning {
			t.Fatalf("state=%+v err=%v", state, err)
		}
		entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["demo"]
		if entry.Status != "stopped" || entry.Runtime.PID != nil {
			t.Fatalf("already-finished registry entry=%+v", entry)
		}
	})
}

func TestStopRecordedProcessAnonymousAndNoShutdownRoutesUseSignalFallback(t *testing.T) {
	for _, test := range []struct {
		name          string
		manifest      string
		health        string
		installName   string
		wantHTTPCalls int32
	}{
		{
			name:        "Go SDK anonymous health skips unsupported shutdown",
			manifest:    "name: demo\nversion: 1.0.0\nlanguage: go\nagent_node:\n  node_id: demo\n",
			health:      `{"status":"ok"}`,
			installName: "demo",
		},
		{
			name:          "anonymous health with missing shutdown falls back",
			manifest:      "name: demo\nversion: 1.0.0\nlanguage: python\nagent_node:\n  node_id: demo\n",
			health:        `{"status":"ok"}`,
			installName:   "demo",
			wantHTTPCalls: 1,
		},
		{
			name:          "TypeScript-shaped equivalent identity with missing shutdown falls back",
			manifest:      "name: demo-node\nversion: 1.0.0\nlanguage: typescript\nagent_node:\n  node_id: demo-node\n",
			health:        `{"status":"ok","node_id":"DEMO_NODE"}`,
			installName:   "demo-node",
			wantHTTPCalls: 1,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			go func() { _, _ = cmd.Process.Wait(); close(done) }()
			t.Cleanup(func() {
				select {
				case <-done:
				default:
					_ = cmd.Process.Kill()
					<-done
				}
			})

			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(test.manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			port := listener.Addr().(*net.TCPAddr).Port
			var shutdownCalls atomic.Int32
			server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				if request.URL.Path == "/health" {
					_, _ = w.Write([]byte(test.health))
					return
				}
				shutdownCalls.Add(1)
				http.NotFound(w, request)
			})}
			go func() { _ = server.Serve(listener) }()
			t.Cleanup(func() { _ = server.Close() })

			entry := InstalledPackage{Name: test.installName, Path: dir, Runtime: RuntimeInfo{Port: &port, PID: &cmd.Process.Pid}}
			result, err := StopRecordedProcess(context.Background(), test.installName, entry)
			if err != nil || !result.Owned || !result.InterruptSent || shutdownCalls.Load() != test.wantHTTPCalls {
				t.Fatalf("result=%+v shutdown calls=%d err=%v", result, shutdownCalls.Load(), err)
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("owned node remained alive after signal fallback")
			}
		})
	}
}

func TestStopRecordedProcessLegacyAcceptedShutdownFallsBackToInterrupt(t *testing.T) {
	cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { _, _ = cmd.Process.Wait(); close(done) }()
	t.Cleanup(func() {
		select {
		case <-done:
		default:
			_ = cmd.Process.Kill()
			<-done
		}
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/health":
			_, _ = w.Write([]byte(`{"node_id":"demo"}`))
		case "/shutdown":
			w.WriteHeader(http.StatusAccepted)
		default:
			http.NotFound(w, request)
		}
	})}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })

	entry := InstalledPackage{Name: "demo", Runtime: RuntimeInfo{Port: &port, PID: &cmd.Process.Pid}}
	result, err := StopRecordedProcess(context.Background(), "demo", entry)
	if err != nil || !result.Owned || !result.HTTPAccepted || !result.InterruptSent {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("legacy process remained alive after the accepted shutdown fallback")
	}
}

func TestRestartPackageAfterReinstallTreatsPreviousPortAsPreference(t *testing.T) {
	for _, test := range []struct {
		name      string
		available bool
		wantPort  int
	}{
		{name: "previous port free", available: true, wantPort: 8123},
		{name: "previous port occupied", available: false, wantPort: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			gotPort := -1
			err := restartPackageAfterReinstall(
				ReinstallState{WasRunning: true, PreferredPort: 8123},
				func(int) bool { return test.available },
				func(port int) error { gotPort = port; return nil },
			)
			if err != nil || gotPort != test.wantPort {
				t.Fatalf("port=%d err=%v, want %d", gotPort, err, test.wantPort)
			}
		})
	}
}

func TestStopRecordedProcessKnownIdentityBranches(t *testing.T) {
	t.Run("foreign listener disowns known PID", func(t *testing.T) {
		cmd := startStopTestProcess(t)
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := listener.Addr().(*net.TCPAddr).Port
		server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"node_id":"foreign"}`))
		})}
		go func() { _ = server.Serve(listener) }()
		t.Cleanup(func() { _ = server.Close() })
		startTime := CurrentProcessStartTime(cmd.Process.Pid)
		result, err := StopRecordedProcess(context.Background(), "demo", InstalledPackage{
			Runtime: RuntimeInfo{Port: &port, PID: &cmd.Process.Pid, StartTime: startTime},
		})
		if err != nil || result.Owned || !processExists(cmd.Process.Pid) {
			t.Fatalf("result=%+v err=%v", result, err)
		}
	})

	t.Run("dead known PID is already stopped", func(t *testing.T) {
		pid := 99_999_999
		result, err := StopRecordedProcess(context.Background(), "demo", InstalledPackage{
			Runtime: RuntimeInfo{PID: &pid, StartTime: "gone"},
		})
		if err != nil || result.Owned {
			t.Fatalf("result=%+v err=%v", result, err)
		}
	})

	t.Run("known PID receives interrupt without an HTTP port", func(t *testing.T) {
		cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		done := make(chan struct{})
		go func() { _, _ = cmd.Process.Wait(); close(done) }()
		t.Cleanup(func() {
			select {
			case <-done:
			default:
				_ = cmd.Process.Kill()
				<-done
			}
		})
		startTime := CurrentProcessStartTime(cmd.Process.Pid)
		result, err := StopRecordedProcess(context.Background(), "demo", InstalledPackage{
			Runtime: RuntimeInfo{PID: &cmd.Process.Pid, StartTime: startTime},
		})
		if err != nil || !result.Owned || !result.InterruptSent || result.HTTPAttempted {
			t.Fatalf("result=%+v err=%v", result, err)
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("interrupted process did not exit")
		}
	})
}

func TestProcessControlErrorAndNoopContracts(t *testing.T) {
	home := t.TempDir()
	if _, err := PackageReinstallState(home, "demo"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing state error=%v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := PackageReinstallState(home, "demo"); err == nil {
		t.Fatal("invalid registry was accepted")
	}
	writeRunningLegacyRegistry(t, home, 8123, os.Getpid())
	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	entry := registry.Installed["demo"]
	entry.Status = "stopped"
	registry.Installed["demo"] = entry
	data, err := yaml.Marshal(registry)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	state, err := PackageReinstallState(home, "demo")
	if err != nil || state.WasRunning {
		t.Fatalf("stopped state=%+v err=%v", state, err)
	}
	if _, err := StopPackageForReinstall(context.Background(), t.TempDir(), "demo"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing stop registry error=%v", err)
	}
	invalidHome := t.TempDir()
	if err := os.WriteFile(filepath.Join(invalidHome, "installed.yaml"), []byte("installed: ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := StopPackageForReinstall(context.Background(), invalidHome, "demo"); err == nil {
		t.Fatal("invalid stop registry was accepted")
	}
	pid := os.Getpid()
	if result, err := StopRecordedProcess(context.Background(), "demo", InstalledPackage{Runtime: RuntimeInfo{PID: &pid}}); err != nil || result.Owned {
		t.Fatalf("legacy PID without port result=%+v err=%v", result, err)
	}
	if err := restartPackageAfterReinstall(ReinstallState{}, func(int) bool { return true }, func(int) error {
		return errors.New("must not run")
	}); err != nil {
		t.Fatalf("stopped restart error=%v", err)
	}
	if err := RestartPackageAfterReinstall(t.TempDir(), "missing", ReinstallState{WasRunning: true}); err == nil {
		t.Fatal("missing package restart unexpectedly succeeded")
	}
	if !isFinishedProcessError(os.ErrProcessDone) || !isFinishedProcessError(errors.New("no such process")) || isFinishedProcessError(errors.New("other")) {
		t.Fatal("finished-process error classification mismatch")
	}
	if !waitForProcessExit(99_999_999, time.Millisecond) {
		t.Fatal("missing PID did not count as exited")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if PortAvailable(port) {
		t.Fatal("occupied port was available")
	}
	_ = listener.Close()
	if !PortAvailable(port) {
		t.Fatal("released port was unavailable")
	}
}

func TestPackageHealthIdentityUsesManifestMetadata(t *testing.T) {
	dir := t.TempDir()
	manifest := "name: demo\nversion: 1.0.0\nentrypoint:\n  healthcheck: readyz\nagent_node:\n  node_id: runtime_demo\n"
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	entry := InstalledPackage{Path: dir}
	if path := PackageHealthcheckPath(entry); path != "readyz" {
		t.Fatalf("health path=%q", path)
	}
	if nodeID := PackageNodeID(entry, "fallback"); nodeID != "runtime_demo" {
		t.Fatalf("node id=%q", nodeID)
	}
	if path := PackageHealthcheckPath(InstalledPackage{Path: filepath.Join(dir, "missing")}); path != "/health" {
		t.Fatalf("fallback health path=%q", path)
	}
	if nodeID := PackageNodeID(InstalledPackage{Path: filepath.Join(dir, "missing")}, "fallback"); nodeID != "fallback" {
		t.Fatalf("fallback node id=%q", nodeID)
	}
}

func TestRequestHTTPShutdownHonorsExpiredContext(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	accepted, timedOut := requestHTTPShutdown(ctx, 1)
	if accepted || !timedOut {
		t.Fatalf("accepted=%v timedOut=%v", accepted, timedOut)
	}
}
