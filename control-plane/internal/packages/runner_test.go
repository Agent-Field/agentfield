package packages

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func waitForHTTPServer(t *testing.T, addr string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 50*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("server %s did not start", addr)
}

func TestAgentNodeRunnerPortEnvAndRegistry(t *testing.T) {
	t.Parallel()

	home := t.TempDir()
	runner := &AgentNodeRunner{AgentFieldHome: home}

	port, err := runner.getFreePort()
	if err != nil {
		t.Fatalf("getFreePort: %v", err)
	}
	if !runner.isPortAvailable(port) {
		t.Fatalf("expected port %d to be available", port)
	}

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	if runner.isPortAvailable(port) {
		_ = listener.Close()
		t.Fatalf("expected port %d to be unavailable while listening", port)
	}
	_ = listener.Close()

	pkgPath := filepath.Join(t.TempDir(), "pkg")
	if err := os.MkdirAll(pkgPath, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(pkgPath, ".env"), []byte(strings.Join([]string{
		"PLAIN=value",
		`QUOTED="double"`,
		"SINGLE='single'",
		"# comment",
		"",
	}, "\n")), 0644); err != nil {
		t.Fatalf("write env: %v", err)
	}

	envVars, err := runner.loadPackageEnvFile(pkgPath)
	if err != nil {
		t.Fatalf("loadPackageEnvFile: %v", err)
	}
	if envVars["PLAIN"] != "value" || envVars["QUOTED"] != "double" || envVars["SINGLE"] != "single" {
		t.Fatalf("unexpected env vars: %#v", envVars)
	}

	if _, err := runner.loadPackageEnvFile(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatalf("expected missing env error")
	}

	registry := &InstallationRegistry{
		Installed: map[string]InstalledPackage{
			"demo": {Name: "demo"},
		},
	}
	data, err := yaml.Marshal(registry)
	if err != nil {
		t.Fatalf("marshal registry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0644); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	if err := runner.updateRuntimeInfo("demo", 8123, 4321); err != nil {
		t.Fatalf("updateRuntimeInfo: %v", err)
	}

	loaded, err := runner.loadRegistry()
	if err != nil {
		t.Fatalf("loadRegistry: %v", err)
	}
	if loaded.Installed["demo"].Status != "running" {
		t.Fatalf("expected running status")
	}
	if loaded.Installed["demo"].Runtime.Port == nil || *loaded.Installed["demo"].Runtime.Port != 8123 {
		t.Fatalf("unexpected port: %#v", loaded.Installed["demo"].Runtime.Port)
	}
}

func TestAgentNodeRunnerWaitDisplayAndStartProcess(t *testing.T) {
	// Cannot use t.Parallel() — this test modifies PATH via t.Setenv,
	// which is process-global and unsafe to share with parallel tests.

	runner := &AgentNodeRunner{}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/reasoners", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"reasoners":[{"id":"reason-a"},{"id":"reason-b"}]}`))
	})
	mux.HandleFunc("/skills", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"skills":[{"id":"skill-a"}]}`))
	})
	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(func() {
		_ = server.Close()
	})
	waitForHTTPServer(t, fmt.Sprintf("127.0.0.1:%d", port))

	if err := runner.waitForAgentNode(port, 2*time.Second); err != nil {
		t.Fatalf("waitForAgentNode: %v", err)
	}
	if err := runner.displayCapabilities(InstalledPackage{Name: "demo"}, port); err != nil {
		t.Fatalf("displayCapabilities: %v", err)
	}

	unusedPortListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen unused port: %v", err)
	}
	unusedPort := unusedPortListener.Addr().(*net.TCPAddr).Port
	_ = unusedPortListener.Close()
	if err := runner.waitForAgentNode(unusedPort, 600*time.Millisecond); err == nil || !strings.Contains(err.Error(), "did not become ready") {
		t.Fatalf("expected timeout error, got %v", err)
	}

	tempDir := t.TempDir()
	fakeBin := filepath.Join(tempDir, "bin")
	if err := os.MkdirAll(fakeBin, 0755); err != nil {
		t.Fatalf("mkdir fake bin: %v", err)
	}
	fakePython := filepath.Join(fakeBin, "python")
	if err := os.WriteFile(fakePython, []byte("#!/bin/sh\nsleep 5\n"), 0755); err != nil {
		t.Fatalf("write fake python: %v", err)
	}

	origPath := os.Getenv("PATH")
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+origPath)
	t.Setenv("AGENTFIELD_SERVER", "http://control-plane.test")

	pkgPath := filepath.Join(tempDir, "package")
	if err := os.MkdirAll(pkgPath, 0755); err != nil {
		t.Fatalf("mkdir package: %v", err)
	}
	if err := os.WriteFile(filepath.Join(pkgPath, "main.py"), []byte("print('ok')\n"), 0644); err != nil {
		t.Fatalf("write main: %v", err)
	}
	if err := os.WriteFile(filepath.Join(pkgPath, ".env"), []byte("EXTRA=value\n"), 0644); err != nil {
		t.Fatalf("write env: %v", err)
	}
	logPath := filepath.Join(tempDir, "runner.log")
	cmd, err := runner.startAgentNodeProcess(InstalledPackage{
		Path: pkgPath,
		Runtime: RuntimeInfo{
			LogFile: logPath,
		},
	}, 9001)
	if err != nil {
		t.Fatalf("startAgentNodeProcess: %v", err)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	})

	badLogCmd, err := runner.startAgentNodeProcess(InstalledPackage{
		Path: pkgPath,
		Runtime: RuntimeInfo{
			LogFile: filepath.Join(tempDir, "missing", "runner.log"),
		},
	}, 9002)
	if badLogCmd != nil {
		t.Fatalf("expected nil cmd on log open failure")
	}
	if err == nil || !strings.Contains(err.Error(), "failed to open log file") {
		t.Fatalf("expected log file error, got %v", err)
	}
}

func TestAgentNodeRunnerRunAgentNode(t *testing.T) {
	home := t.TempDir()
	logDir := filepath.Join(home, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		t.Fatalf("mkdir logs: %v", err)
	}
	python3Path, err := exec.LookPath("python3")
	if err != nil {
		t.Fatalf("python3 not found: %v", err)
	}
	fakeBin := filepath.Join(home, "bin")
	if err := os.MkdirAll(fakeBin, 0755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}
	pythonWrapper := filepath.Join(fakeBin, "python")
	if err := os.WriteFile(pythonWrapper, []byte("#!/bin/sh\nexec \""+python3Path+"\" \"$@\"\n"), 0755); err != nil {
		t.Fatalf("write python wrapper: %v", err)
	}
	origPath := os.Getenv("PATH")
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+origPath)

	pkgPath := filepath.Join(home, "package")
	if err := os.MkdirAll(pkgPath, 0755); err != nil {
		t.Fatalf("mkdir package: %v", err)
	}

	pythonApp := `import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ["PORT"])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/reasoners":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reasoners":[{"id":"reasoner.one"}]}).encode())
        elif self.path == "/skills":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"skills":[{"id":"skill.one"}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
`
	if err := os.WriteFile(filepath.Join(pkgPath, "main.py"), []byte(pythonApp), 0644); err != nil {
		t.Fatalf("write main.py: %v", err)
	}

	registry := &InstallationRegistry{
		Installed: map[string]InstalledPackage{
			"demo-node": {
				Name:   "demo-node",
				Path:   pkgPath,
				Status: "stopped",
				Runtime: RuntimeInfo{
					LogFile: filepath.Join(logDir, "demo-node.log"),
				},
			},
		},
	}
	data, err := yaml.Marshal(registry)
	if err != nil {
		t.Fatalf("marshal registry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0644); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	runner := &AgentNodeRunner{AgentFieldHome: home}
	if err := runner.RunAgentNode("missing"); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("expected missing package error, got %v", err)
	}

	runningPort := 8124
	registry.Installed["already-running"] = InstalledPackage{
		Name:   "already-running",
		Path:   pkgPath,
		Status: "running",
		Runtime: RuntimeInfo{
			Port:    &runningPort,
			LogFile: filepath.Join(logDir, "already-running.log"),
		},
	}
	data, err = yaml.Marshal(registry)
	if err != nil {
		t.Fatalf("marshal running registry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0644); err != nil {
		t.Fatalf("write running registry: %v", err)
	}
	if err := runner.RunAgentNode("already-running"); err == nil || !strings.Contains(err.Error(), "already running") {
		t.Fatalf("expected already running error, got %v", err)
	}

	if err := runner.RunAgentNode("demo-node"); err != nil {
		t.Fatalf("RunAgentNode success path: %v", err)
	}

	loaded, err := runner.loadRegistry()
	if err != nil {
		t.Fatalf("loadRegistry: %v", err)
	}
	pkg := loaded.Installed["demo-node"]
	if pkg.Runtime.PID == nil || pkg.Runtime.Port == nil {
		t.Fatalf("expected runtime info after run: %+v", pkg.Runtime)
	}

	process, err := os.FindProcess(*pkg.Runtime.PID)
	if err != nil {
		t.Fatalf("find process: %v", err)
	}
	if err := process.Kill(); err != nil {
		t.Fatalf("kill process: %v", err)
	}
	_, _ = process.Wait()
}

func TestSpinners(t *testing.T) {
	installer := &PackageInstaller{}
	spinner := installer.newSpinner("working")
	spinner.Start()
	time.Sleep(20 * time.Millisecond)
	spinner.Stop()

	gitSpinner := (&GitInstaller{}).newSpinner("git")
	gitSpinner.Start()
	time.Sleep(20 * time.Millisecond)
	gitSpinner.Success("ok")

	githubSpinner := (&GitHubInstaller{}).newSpinner("github")
	githubSpinner.Start()
	time.Sleep(20 * time.Millisecond)
	githubSpinner.Error("failed")

	if _, err := exec.LookPath("sh"); err != nil {
		t.Fatalf("expected shell in PATH: %v", err)
	}
}
