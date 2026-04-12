//go:build !windows

package services

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestRunDevHelperProcess(t *testing.T) {
	if os.Getenv("AF_TEST_HELPER_MODE") == "" {
		return
	}

	port := os.Getenv("PORT")
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/reasoners", func(w http.ResponseWriter, r *http.Request) {
		if os.Getenv("AF_TEST_HELPER_MODE") == "serve-capabilities-error" {
			_, _ = fmt.Fprint(w, `{"reasoners":[`)
			return
		}
		_, _ = fmt.Fprint(w, `{"reasoners":[{"id":"helper-reasoner"}]}`)
	})
	mux.HandleFunc("/skills", func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"skills":[{"id":"helper-skill"}]}`)
	})

	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		os.Exit(2)
	}

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(ln)
	}()

	// Sleep is inherent to the test: this helper process must stay alive long enough
	// for the parent test to exercise health checks and shutdown against it.
	time.Sleep(1200 * time.Millisecond)
	_ = server.Shutdown(context.Background())

	if os.Getenv("AF_TEST_HELPER_MODE") == "serve-exit-error" {
		os.Exit(3)
	}
}

func TestDevServiceRunDev(t *testing.T) {
	// These subtests spawn a real python3 subprocess that serves HTTP on a
	// port in the 8001-8999 range. discoverAgentPort scans that range with a
	// 120s internal timeout, so if the subprocess fails to start (e.g., python3
	// heredoc issues, port already taken), the test hangs for minutes.
	// Guard with a per-test deadline to prevent that from killing the suite.
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available, skipping dev service subprocess tests")
	}
	if testing.Short() {
		t.Skip("skipping slow dev service subprocess tests in short mode")
	}

	t.Run("successful run", func(t *testing.T) {
		dir := t.TempDir()
		venvBin := filepath.Join(dir, "venv", "bin")
		require.NoError(t, os.MkdirAll(venvBin, 0o755))

		port := findPortInAgentRange(t)
		t.Setenv("AF_TEST_HELPER_MODE", "serve-success")
		script := "#!/bin/sh\npython3 - <<'PY'\nimport os, sys, threading, time\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\nport = int(os.environ['PORT'])\nmode = os.environ.get('AF_TEST_HELPER_MODE', 'serve-success')\nclass Handler(BaseHTTPRequestHandler):\n    def log_message(self, fmt, *args):\n        pass\n    def do_GET(self):\n        if self.path == '/health':\n            self.send_response(200)\n            self.end_headers()\n        elif self.path == '/reasoners':\n            self.send_response(200)\n            self.end_headers()\n            if mode == 'serve-capabilities-error':\n                self.wfile.write(b'{\"reasoners\":[')\n            else:\n                self.wfile.write(b'{\"reasoners\":[{\"id\":\"helper-reasoner\"}]}')\n        elif self.path == '/skills':\n            self.send_response(200)\n            self.end_headers()\n            self.wfile.write(b'{\"skills\":[{\"id\":\"helper-skill\"}]}')\n        else:\n            self.send_response(404)\n            self.end_headers()\nserver = HTTPServer(('127.0.0.1', port), Handler)\nthread = threading.Thread(target=server.serve_forever)\nthread.daemon = True\nthread.start()\ntime.sleep(1.2)\nserver.shutdown()\nserver.server_close()\nsys.exit(3 if mode == 'serve-exit-error' else 0)\nPY\n"
		require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte(script), 0o755))

		service := &DefaultDevService{}
		err := service.runDev(dir, domain.DevOptions{Port: port})
		require.NoError(t, err)
	})

	t.Run("process exits with error after startup", func(t *testing.T) {
		dir := t.TempDir()
		venvBin := filepath.Join(dir, "venv", "bin")
		require.NoError(t, os.MkdirAll(venvBin, 0o755))

		port := findPortInAgentRange(t)
		t.Setenv("AF_TEST_HELPER_MODE", "serve-exit-error")
		script := "#!/bin/sh\npython3 - <<'PY'\nimport os, sys, threading, time\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\nport = int(os.environ['PORT'])\nmode = os.environ.get('AF_TEST_HELPER_MODE', 'serve-success')\nclass Handler(BaseHTTPRequestHandler):\n    def log_message(self, fmt, *args):\n        pass\n    def do_GET(self):\n        if self.path == '/health':\n            self.send_response(200)\n            self.end_headers()\n        elif self.path == '/reasoners':\n            self.send_response(200)\n            self.end_headers()\n            if mode == 'serve-capabilities-error':\n                self.wfile.write(b'{\"reasoners\":[')\n            else:\n                self.wfile.write(b'{\"reasoners\":[{\"id\":\"helper-reasoner\"}]}')\n        elif self.path == '/skills':\n            self.send_response(200)\n            self.end_headers()\n            self.wfile.write(b'{\"skills\":[{\"id\":\"helper-skill\"}]}')\n        else:\n            self.send_response(404)\n            self.end_headers()\nserver = HTTPServer(('127.0.0.1', port), Handler)\nthread = threading.Thread(target=server.serve_forever)\nthread.daemon = True\nthread.start()\ntime.sleep(1.2)\nserver.shutdown()\nserver.server_close()\nsys.exit(3 if mode == 'serve-exit-error' else 0)\nPY\n"
		require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte(script), 0o755))

		service := &DefaultDevService{}
		err := service.runDev(dir, domain.DevOptions{Port: port})
		require.NoError(t, err)
	})

	t.Run("start error", func(t *testing.T) {
		dir := t.TempDir()
		venvBin := filepath.Join(dir, "venv", "bin")
		require.NoError(t, os.MkdirAll(venvBin, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte("#!/bin/sh\nexit 0\n"), 0o644))

		service := &DefaultDevService{}
		err := service.runDev(dir, domain.DevOptions{Port: findPortInAgentRange(t)})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to start agent")
	})

	t.Run("capabilities error does not fail run", func(t *testing.T) {
		dir := t.TempDir()
		venvBin := filepath.Join(dir, "venv", "bin")
		require.NoError(t, os.MkdirAll(venvBin, 0o755))

		port := findPortInAgentRange(t)
		t.Setenv("AF_TEST_HELPER_MODE", "serve-capabilities-error")
		script := "#!/bin/sh\npython3 - <<'PY'\nimport os, sys, threading, time\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\nport = int(os.environ['PORT'])\nmode = os.environ.get('AF_TEST_HELPER_MODE', 'serve-success')\nclass Handler(BaseHTTPRequestHandler):\n    def log_message(self, fmt, *args):\n        pass\n    def do_GET(self):\n        if self.path == '/health':\n            self.send_response(200)\n            self.end_headers()\n        elif self.path == '/reasoners':\n            self.send_response(200)\n            self.end_headers()\n            if mode == 'serve-capabilities-error':\n                self.wfile.write(b'{\"reasoners\":[')\n            else:\n                self.wfile.write(b'{\"reasoners\":[{\"id\":\"helper-reasoner\"}]}')\n        elif self.path == '/skills':\n            self.send_response(200)\n            self.end_headers()\n            self.wfile.write(b'{\"skills\":[{\"id\":\"helper-skill\"}]}')\n        else:\n            self.send_response(404)\n            self.end_headers()\nserver = HTTPServer(('127.0.0.1', port), Handler)\nthread = threading.Thread(target=server.serve_forever)\nthread.daemon = True\nthread.start()\ntime.sleep(1.2)\nserver.shutdown()\nserver.server_close()\nsys.exit(3 if mode == 'serve-exit-error' else 0)\nPY\n"
		require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte(script), 0o755))

		service := &DefaultDevService{}
		err := service.runDev(dir, domain.DevOptions{Port: port})
		require.NoError(t, err)
	})
}

func TestAgentServiceStopAgentAdditionalCoverage(t *testing.T) {
	t.Run("http shutdown success updates registry", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sleep", "60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		t.Cleanup(func() {
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
		})

		port := 8131
		startedAt := time.Now().Add(-time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"http-agent": {
					Name:   "http-agent",
					Status: "running",
					Runtime: packages.RuntimeInfo{
						Port:      &port,
						PID:       &pid,
						StartedAt: &startedAt,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		agentClient := newMockAgentClient()
		agentClient.shutdownFunc = func(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
			return &interfaces.AgentShutdownResponse{Status: "shutting_down"}, nil
		}

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), agentClient, home).(*DefaultAgentService)
		require.NoError(t, service.StopAgent("http-agent"))

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		assert.Equal(t, "stopped", updated.Installed["http-agent"].Status)
		assert.Nil(t, updated.Installed["http-agent"].Runtime.PID)
		assert.Nil(t, updated.Installed["http-agent"].Runtime.Port)
		assert.Nil(t, updated.Installed["http-agent"].Runtime.StartedAt)
	})

	t.Run("running agent without pid returns error", func(t *testing.T) {
		home := t.TempDir()
		port := 8132
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"missing-pid-agent": {
					Name:   "missing-pid-agent",
					Status: "running",
					Runtime: packages.RuntimeInfo{
						Port: &port,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
		err = service.StopAgent("missing-pid-agent")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not running")
	})

	t.Run("running agent without port returns error", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sleep", "60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		t.Cleanup(func() {
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
		})

		startedAt := time.Now().Add(-time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"missing-port-agent": {
					Name:   "missing-port-agent",
					Status: "running",
					Runtime: packages.RuntimeInfo{
						PID:       &pid,
						StartedAt: &startedAt,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
		err = service.StopAgent("missing-port-agent")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no port found")
	})

	t.Run("stopped agent returns not running", func(t *testing.T) {
		home := t.TempDir()
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"already-stopped": {
					Name:   "already-stopped",
					Status: "stopped",
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
		err = service.StopAgent("already-stopped")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not running")
	})

	t.Run("process already finished during fallback is handled", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sh", "-c", "sleep 0.1")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		port := 8133
		startedAt := time.Now().Add(-time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"finished-agent": {
					Name:   "finished-agent",
					Status: "running",
					Runtime: packages.RuntimeInfo{
						Port:      &port,
						PID:       &pid,
						StartedAt: &startedAt,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		agentClient := newMockAgentClient()
		agentClient.shutdownFunc = func(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
			// Sleep is inherent to the test: simulates a slow shutdown response.
			time.Sleep(300 * time.Millisecond)
			return nil, fmt.Errorf("shutdown unavailable")
		}

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), agentClient, home).(*DefaultAgentService)
		require.NoError(t, service.StopAgent("finished-agent"))
		_, _ = cmd.Process.Wait()
	})
}

func TestPackageServiceAdditionalCoverage(t *testing.T) {
	t.Run("install local package force reinstall success", func(t *testing.T) {
		home := t.TempDir()
		sourcePath := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(sourcePath, "agentfield-package.yaml"), []byte("name: force-package\nversion: 1.0.0\nmain: main.py\n"), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(sourcePath, "main.py"), []byte("print('ok')\n"), 0o644))

		existing := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"force-package": {
					Name:   "force-package",
					Path:   filepath.Join(home, "packages", "force-package"),
					Status: "stopped",
				},
			},
		}
		data, err := yaml.Marshal(existing)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		require.NoError(t, service.installLocalPackage(sourcePath, true, false))

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		assert.Equal(t, filepath.Join(home, "packages", "force-package"), updated.Installed["force-package"].Path)
	})

	t.Run("install local package copy error", func(t *testing.T) {
		home := t.TempDir()
		sourcePath := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(sourcePath, "agentfield-package.yaml"), []byte("name: broken-copy\nversion: 1.0.0\nmain: main.py\n"), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(sourcePath, "main.py"), []byte("print('ok')\n"), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(home, "packages"), []byte("blocker"), 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		err := service.installLocalPackage(sourcePath, false, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to copy package")
	})

	t.Run("force uninstall continues when stop fails", func(t *testing.T) {
		home := t.TempDir()
		packagePath := filepath.Join(home, "packages", "force-remove")
		logPath := filepath.Join(home, "logs", "force-remove.log")
		require.NoError(t, os.MkdirAll(packagePath, 0o755))
		require.NoError(t, os.MkdirAll(filepath.Dir(logPath), 0o755))
		require.NoError(t, os.WriteFile(logPath, []byte("log"), 0o644))

		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"force-remove": {
					Name:   "force-remove",
					Path:   packagePath,
					Status: "running",
					Runtime: packages.RuntimeInfo{
						LogFile: logPath,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		require.NoError(t, service.uninstallPackage("force-remove", true))
		_, err = os.Stat(packagePath)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("update registry with invalid existing yaml", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		err := service.updateRegistry(&packages.PackageMetadata{Name: "pkg", Version: "1.0.0"}, t.TempDir(), filepath.Join(home, "packages", "pkg"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse registry")
	})

	t.Run("update registry preserves existing packages", func(t *testing.T) {
		home := t.TempDir()
		existing := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"existing": {
					Name:   "existing",
					Path:   filepath.Join(home, "packages", "existing"),
					Status: "stopped",
				},
			},
		}
		data, err := yaml.Marshal(existing)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		err = service.updateRegistry(&packages.PackageMetadata{Name: "new-package", Version: "2.0.0"}, t.TempDir(), filepath.Join(home, "packages", "new-package"))
		require.NoError(t, err)

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		_, existingOK := updated.Installed["existing"]
		_, newOK := updated.Installed["new-package"]
		assert.True(t, existingOK)
		assert.True(t, newOK)
	})

	t.Run("save registry write error", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("blocking file"), 0o644))

		service := &DefaultPackageService{agentfieldHome: filepath.Join(home, "installed.yaml")}
		err := service.saveRegistry(&packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to write registry")
	})

	t.Run("copy file missing source", func(t *testing.T) {
		service := &DefaultPackageService{}
		err := service.copyFile(filepath.Join(t.TempDir(), "missing.txt"), filepath.Join(t.TempDir(), "dst.txt"))
		require.Error(t, err)
	})

	t.Run("copy package missing source", func(t *testing.T) {
		service := &DefaultPackageService{}
		err := service.copyPackage(filepath.Join(t.TempDir(), "missing"), filepath.Join(t.TempDir(), "dst"))
		require.Error(t, err)
	})

	t.Run("stop agent node kills live process", func(t *testing.T) {
		cmd := exec.Command("sleep", "60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		agentNode := &packages.InstalledPackage{
			Name: "live-agent",
			Runtime: packages.RuntimeInfo{
				PID: &pid,
			},
		}

		service := &DefaultPackageService{}
		require.NoError(t, service.stopAgentNode(agentNode))
		waitErr := cmd.Wait()
		assert.Error(t, waitErr)
	})

	t.Run("force uninstall running package with live pid", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sleep", "60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid

		packagePath := filepath.Join(home, "packages", "live-remove")
		logPath := filepath.Join(home, "logs", "live-remove.log")
		require.NoError(t, os.MkdirAll(packagePath, 0o755))
		require.NoError(t, os.MkdirAll(filepath.Dir(logPath), 0o755))
		require.NoError(t, os.WriteFile(logPath, []byte("log"), 0o644))

		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"live-remove": {
					Name:   "live-remove",
					Path:   packagePath,
					Status: "running",
					Runtime: packages.RuntimeInfo{
						PID:     &pid,
						LogFile: logPath,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		require.NoError(t, service.uninstallPackage("live-remove", true))
		waitErr := cmd.Wait()
		assert.Error(t, waitErr)
	})
}

func TestAgentServiceBuildProcessConfigAdditionalCoverage(t *testing.T) {
	t.Run("windows style virtualenv", func(t *testing.T) {
		dir := t.TempDir()
		scriptsDir := filepath.Join(dir, "venv", "Scripts")
		require.NoError(t, os.MkdirAll(scriptsDir, 0o755))
		pythonPath := filepath.Join(scriptsDir, "python.exe")
		require.NoError(t, os.WriteFile(pythonPath, []byte("stub"), 0o755))

		service := &DefaultAgentService{}
		cfg := service.buildProcessConfig(packages.InstalledPackage{
			Name: "windows-agent",
			Path: dir,
			Runtime: packages.RuntimeInfo{
				LogFile: filepath.Join(dir, "agent.log"),
			},
		}, 8141)

		assert.Equal(t, pythonPath, cfg.Command)
		assert.Contains(t, cfg.Env, "VIRTUAL_ENV="+filepath.Join(dir, "venv"))
		assert.Contains(t, cfg.Env, "PYTHONHOME=")
		assert.Contains(t, cfg.Env, "PYTHONPATH="+filepath.Join(dir, "venv", "Lib", "site-packages"))
		assertEnvWithPrefix(t, cfg.Env, "PATH=", scriptsDir)
	})

	t.Run("system python fallback and lookpath", func(t *testing.T) {
		dir := t.TempDir()
		fakeBin := filepath.Join(dir, "bin")
		require.NoError(t, os.MkdirAll(fakeBin, 0o755))
		fakePython := filepath.Join(fakeBin, "python3")
		require.NoError(t, os.WriteFile(fakePython, []byte("#!/bin/sh\nexit 0\n"), 0o755))
		t.Setenv("PATH", fakeBin)

		service := &DefaultAgentService{}
		assert.Equal(t, fakePython, service.findPythonExecutable())

		cfg := service.buildProcessConfig(packages.InstalledPackage{
			Name: "fallback-agent",
			Path: dir,
		}, 8142)
		assert.Equal(t, fakePython, cfg.Command)
	})

	t.Run("no python executable found", func(t *testing.T) {
		dir := t.TempDir()
		t.Setenv("PATH", dir)
		service := &DefaultAgentService{}
		assert.Empty(t, service.findPythonExecutable())
	})
}

func TestAgentServiceRunAgentAdditionalCoverage(t *testing.T) {
	t.Run("port allocation error", func(t *testing.T) {
		home := t.TempDir()
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"alloc-agent": {
					Name:   "alloc-agent",
					Path:   t.TempDir(),
					Status: "stopped",
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		portManager := newMockPortManager()
		portManager.findFreePortFunc = func(startPort int) (int, error) {
			return 0, fmt.Errorf("no ports")
		}

		service := NewAgentService(newMockProcessManager(), portManager, newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		_, err = service.RunAgent("alloc-agent", domain.RunOptions{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to allocate port")
	})

	t.Run("process start error", func(t *testing.T) {
		home := t.TempDir()
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"start-agent": {
					Name:   "start-agent",
					Path:   t.TempDir(),
					Status: "stopped",
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		processManager := newMockProcessManager()
		processManager.startFunc = func(config interfaces.ProcessConfig) (int, error) {
			return 0, fmt.Errorf("boom")
		}

		service := NewAgentService(processManager, newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		_, err = service.RunAgent("start-agent", domain.RunOptions{Port: 8143})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to start agent node")
	})

	t.Run("already running agent", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sleep", "60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		t.Cleanup(func() {
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
		})

		port := 8144
		startedAt := time.Now().Add(-time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"running-agent": {
					Name:   "running-agent",
					Path:   t.TempDir(),
					Status: "running",
					Runtime: packages.RuntimeInfo{
						Port:      &port,
						PID:       &pid,
						StartedAt: &startedAt,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		_, err = service.RunAgent("running-agent", domain.RunOptions{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "already running")
	})

	t.Run("runtime update failure", func(t *testing.T) {
		home := t.TempDir()
		agentPath := t.TempDir()
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"update-agent": {
					Name:   "update-agent",
					Path:   agentPath,
					Status: "stopped",
					Runtime: packages.RuntimeInfo{
						LogFile: filepath.Join(home, "update-agent.log"),
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		registryPath := filepath.Join(home, "installed.yaml")
		require.NoError(t, os.WriteFile(registryPath, data, 0o444))

		server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/health":
				w.WriteHeader(http.StatusOK)
			case "/reasoners":
				_, _ = fmt.Fprint(w, `{"reasoners":[{"id":"reasoner-a"}]}`)
			case "/skills":
				_, _ = fmt.Fprint(w, `{"skills":[{"id":"skill-a"}]}`)
			default:
				http.NotFound(w, r)
			}
		}))
		defer server.Close()

		processManager := newMockProcessManager()
		processManager.startFunc = func(config interfaces.ProcessConfig) (int, error) {
			return 5151, nil
		}

		service := NewAgentService(processManager, newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		_, err = service.RunAgent("update-agent", domain.RunOptions{Port: port})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to update runtime info")
	})
}

func TestAgentServiceRegistryHelpersAdditionalCoverage(t *testing.T) {
	t.Run("load registry invalid yaml", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))
		service := &DefaultAgentService{agentfieldHome: home}
		_, err := service.loadRegistryDirect()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse registry")
	})

	t.Run("save registry write error", func(t *testing.T) {
		home := t.TempDir()
		blocker := filepath.Join(home, "blocker")
		require.NoError(t, os.WriteFile(blocker, []byte("x"), 0o644))
		service := &DefaultAgentService{agentfieldHome: blocker}
		err := service.saveRegistryDirect(&packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{}})
		require.Error(t, err)
	})

	t.Run("update runtime info invalid yaml", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))
		service := &DefaultAgentService{agentfieldHome: home}
		err := service.updateRuntimeInfo("agent", 8145, 5152)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse registry")
	})
}
