package services

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

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestResolveServerURL(t *testing.T) {
	t.Setenv("AGENTFIELD_SERVER", "")
	t.Setenv("AGENTFIELD_SERVER_URL", "")
	assert.Equal(t, "http://localhost:8080", resolveServerURL())

	t.Setenv("AGENTFIELD_SERVER_URL", "http://from-url")
	assert.Equal(t, "http://from-url", resolveServerURL())

	t.Setenv("AGENTFIELD_SERVER", "http://from-server")
	assert.Equal(t, "http://from-server", resolveServerURL())
}

func TestAgentServiceFindAgentInRegistry(t *testing.T) {
	service := &DefaultAgentService{}
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"deep-research-agent": {Name: "deep-research-agent"},
			"exact-name":          {Name: "exact-name"},
		},
	}

	t.Run("exact match", func(t *testing.T) {
		pkg, actualName, ok := service.findAgentInRegistry(registry, "exact-name")
		require.True(t, ok)
		assert.Equal(t, "exact-name", actualName)
		assert.Equal(t, "exact-name", pkg.Name)
	})

	t.Run("normalized match", func(t *testing.T) {
		pkg, actualName, ok := service.findAgentInRegistry(registry, "deepresearchagent")
		require.True(t, ok)
		assert.Equal(t, "deep-research-agent", actualName)
		assert.Equal(t, "deep-research-agent", pkg.Name)
	})

	t.Run("missing agent", func(t *testing.T) {
		_, actualName, ok := service.findAgentInRegistry(registry, "missing")
		assert.False(t, ok)
		assert.Empty(t, actualName)
	})
}

func TestAgentServiceLoadPackageEnvFile(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, ".env")
	require.NoError(t, os.WriteFile(envPath, []byte(strings.Join([]string{
		"# comment",
		"FOO=bar",
		`QUOTED="value with spaces"`,
		"SINGLE='quoted'",
		"INVALID_LINE",
		"",
	}, "\n")), 0o644))

	service := &DefaultAgentService{}
	envVars, err := service.loadPackageEnvFile(dir)
	require.NoError(t, err)
	assert.Equal(t, map[string]string{
		"FOO":    "bar",
		"QUOTED": "value with spaces",
		"SINGLE": "quoted",
	}, envVars)
}

func TestAgentServiceBuildProcessConfig(t *testing.T) {
	dir := t.TempDir()
	venvBin := filepath.Join(dir, "venv", "bin")
	require.NoError(t, os.MkdirAll(venvBin, 0o755))
	pythonPath := filepath.Join(venvBin, "python")
	require.NoError(t, os.WriteFile(pythonPath, []byte("#!/bin/sh\nexit 0\n"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN=secret\n"), 0o644))

	logFile := filepath.Join(dir, "agent.log")
	service := &DefaultAgentService{}
	cfg := service.buildProcessConfig(packages.InstalledPackage{
		Name: "agent",
		Path: dir,
		Runtime: packages.RuntimeInfo{
			LogFile: logFile,
		},
	}, 8123)

	assert.Equal(t, pythonPath, cfg.Command)
	assert.Equal(t, []string{"main.py"}, cfg.Args)
	assert.Equal(t, dir, cfg.WorkDir)
	assert.Equal(t, logFile, cfg.LogFile)
	assert.Contains(t, cfg.Env, "PORT=8123")
	assert.Contains(t, cfg.Env, "AGENTFIELD_SERVER_URL=http://localhost:8080")
	assert.Contains(t, cfg.Env, "TOKEN=secret")
	assert.Contains(t, cfg.Env, "VIRTUAL_ENV="+filepath.Join(dir, "venv"))
	assertEnvWithPrefix(t, cfg.Env, "PATH=", venvBin)
	assert.Contains(t, cfg.Env, "PYTHONHOME=")
	assert.Contains(t, cfg.Env, "PYTHONPATH="+filepath.Join(dir, "venv", "lib"))
}

func TestAgentServiceWaitForAgentNode(t *testing.T) {
	t.Run("ready", func(t *testing.T) {
		server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/health" {
				http.NotFound(w, r)
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		service := &DefaultAgentService{}
		require.NoError(t, service.waitForAgentNode(port, 2*time.Second))
	})

	t.Run("timeout", func(t *testing.T) {
		port := findFreePortInRange(t)
		service := &DefaultAgentService{}
		err := service.waitForAgentNode(port, 750*time.Millisecond)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "did not become ready")
	})
}

func TestAgentServiceUpdateRuntimeInfo(t *testing.T) {
	home := t.TempDir()
	registryPath := filepath.Join(home, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"agent": {Name: "agent", Status: "stopped"},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(registryPath, data, 0o644))

	service := &DefaultAgentService{agentfieldHome: home}
	require.NoError(t, service.updateRuntimeInfo("agent", 8123, 4567))

	updatedData, err := os.ReadFile(registryPath)
	require.NoError(t, err)

	var updated packages.InstallationRegistry
	require.NoError(t, yaml.Unmarshal(updatedData, &updated))
	assert.Equal(t, "running", updated.Installed["agent"].Status)
	require.NotNil(t, updated.Installed["agent"].Runtime.Port)
	require.NotNil(t, updated.Installed["agent"].Runtime.PID)
	require.NotNil(t, updated.Installed["agent"].Runtime.StartedAt)
	assert.Equal(t, 8123, *updated.Installed["agent"].Runtime.Port)
	assert.Equal(t, 4567, *updated.Installed["agent"].Runtime.PID)
}

func TestPackageServiceHelpers(t *testing.T) {
	service := &DefaultPackageService{agentfieldHome: t.TempDir()}

	t.Run("parse metadata default main", func(t *testing.T) {
		dir := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte("name: sample\nversion: 1.2.3\n"), 0o644))

		metadata, err := service.parsePackageMetadata(dir)
		require.NoError(t, err)
		assert.Equal(t, "sample", metadata.Name)
		assert.Equal(t, "1.2.3", metadata.Version)
		assert.Equal(t, "main.py", metadata.Main)
	})

	t.Run("parse metadata validation errors", func(t *testing.T) {
		tests := []struct {
			name        string
			content     string
			errContains string
		}{
			{name: "missing name", content: "version: 1.0.0\nmain: main.py\n", errContains: "package name is required"},
			{name: "missing version", content: "name: sample\nmain: main.py\n", errContains: "package version is required"},
			{name: "invalid yaml", content: "name: [", errContains: "failed to parse"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				dir := t.TempDir()
				require.NoError(t, os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(tc.content), 0o644))
				_, err := service.parsePackageMetadata(dir)
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errContains)
			})
		}
	})

	t.Run("validate package", func(t *testing.T) {
		dir := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte("name: sample\nversion: 1.0.0\nmain: main.py\n"), 0o644))
		err := service.validatePackage(dir)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "main.py not found")

		require.NoError(t, os.WriteFile(filepath.Join(dir, "main.py"), []byte("print('ok')\n"), 0o644))
		require.NoError(t, service.validatePackage(dir))
	})

	t.Run("requirements and installed registry helpers", func(t *testing.T) {
		dir := t.TempDir()
		assert.False(t, service.hasRequirementsFile(dir))
		require.NoError(t, os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte("pytest\n"), 0o644))
		assert.True(t, service.hasRequirementsFile(dir))

		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"present": {Name: "present"},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(service.agentfieldHome, "installed.yaml"), data, 0o644))
		assert.True(t, service.isPackageInstalled("present"))

		require.NoError(t, os.WriteFile(filepath.Join(service.agentfieldHome, "installed.yaml"), []byte("invalid: ["), 0o644))
		assert.False(t, service.isPackageInstalled("present"))
	})

	t.Run("copy package", func(t *testing.T) {
		source := t.TempDir()
		destParent := t.TempDir()
		dest := filepath.Join(destParent, "copied")
		require.NoError(t, os.MkdirAll(filepath.Join(source, "nested"), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(source, "agentfield-package.yaml"), []byte("name: copy\nversion: 1.0.0\nmain: main.py\n"), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(source, "nested", "main.py"), []byte("print('nested')\n"), 0o644))

		require.NoError(t, service.copyPackage(source, dest))

		data, err := os.ReadFile(filepath.Join(dest, "nested", "main.py"))
		require.NoError(t, err)
		assert.Equal(t, "print('nested')\n", string(data))
	})

	t.Run("install dependencies with fake python", func(t *testing.T) {
		packagePath := t.TempDir()
		logPath := filepath.Join(packagePath, "pip.log")
		fakeBin := filepath.Join(packagePath, "fake-bin")
		require.NoError(t, os.MkdirAll(fakeBin, 0o755))
		python3Path := filepath.Join(fakeBin, "python3")
		python3Script := fmt.Sprintf(`#!/bin/sh
venv_path="$3"
mkdir -p "$venv_path/bin"
cat > "$venv_path/bin/pip" <<'EOF'
#!/bin/sh
echo "$@" >> %s
exit 0
EOF
chmod +x "$venv_path/bin/pip"
exit 0
`, logPath)
		require.NoError(t, os.WriteFile(python3Path, []byte(python3Script), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(packagePath, "requirements.txt"), []byte("pytest\n"), 0o644))
		t.Setenv("PATH", fakeBin+":"+os.Getenv("PATH"))

		metadata := &packages.PackageMetadata{
			Dependencies: packages.DependencyConfig{
				Python: []string{"dep-one"},
				System: []string{"sys-lib"},
			},
		}

		require.NoError(t, service.installDependencies(packagePath, metadata))

		logData, err := os.ReadFile(logPath)
		require.NoError(t, err)
		logText := string(logData)
		assert.Contains(t, logText, "install --upgrade pip")
		assert.Contains(t, logText, "install -r")
		assert.Contains(t, logText, "install dep-one")
	})

	t.Run("update registry writes package metadata", func(t *testing.T) {
		home := t.TempDir()
		localService := &DefaultPackageService{agentfieldHome: home}
		metadata := &packages.PackageMetadata{
			Name:        "registry-agent",
			Version:     "2.0.0",
			Description: "test package",
		}
		sourcePath := t.TempDir()
		destPath := filepath.Join(home, "packages", metadata.Name)

		require.NoError(t, localService.updateRegistry(metadata, sourcePath, destPath))

		data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var registry packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(data, &registry))
		assert.Equal(t, "registry-agent", registry.Installed["registry-agent"].Name)
		assert.Equal(t, destPath, registry.Installed["registry-agent"].Path)
		assert.Equal(t, filepath.Join(home, "logs", "registry-agent.log"), registry.Installed["registry-agent"].Runtime.LogFile)
	})

	t.Run("uninstall package success", func(t *testing.T) {
		home := t.TempDir()
		localService := &DefaultPackageService{agentfieldHome: home}
		packagePath := filepath.Join(home, "packages", "remove-me")
		logPath := filepath.Join(home, "logs", "remove-me.log")
		require.NoError(t, os.MkdirAll(filepath.Dir(logPath), 0o755))
		require.NoError(t, os.MkdirAll(packagePath, 0o755))
		require.NoError(t, os.WriteFile(logPath, []byte("log"), 0o644))

		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"remove-me": {
					Name:   "remove-me",
					Path:   packagePath,
					Status: "stopped",
					Runtime: packages.RuntimeInfo{
						LogFile: logPath,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		require.NoError(t, localService.uninstallPackage("remove-me", false))
		_, err = os.Stat(packagePath)
		assert.True(t, os.IsNotExist(err))
		_, err = os.Stat(logPath)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("environment variables and status helpers", func(t *testing.T) {
		assert.NotEmpty(t, service.red("err"))
		assert.NotEmpty(t, service.yellow("warn"))
		assert.NotEmpty(t, service.cyan("info"))
		assert.Equal(t, statusError, service.statusError())

		t.Setenv("OPTIONAL_KEY", "configured")
		metadata := &packages.PackageMetadata{
			Name: "env-agent",
			UserEnvironment: packages.UserEnvironmentConfig{
				Required: []packages.UserEnvironmentVar{{Name: "REQUIRED_KEY"}},
				Optional: []packages.UserEnvironmentVar{{Name: "OPTIONAL_KEY", Description: "optional", Default: "fallback"}},
			},
		}
		service.checkEnvironmentVariables(metadata)
	})

	t.Run("spinner error", func(t *testing.T) {
		spinner := service.newSpinner("working")
		spinner.Start()
		// Sleep is inherent to the test: let the spinner goroutine animate briefly before stopping.
		time.Sleep(50 * time.Millisecond)
		spinner.Error("failed")
	})
}

func TestDevServiceLoadDevEnvFile(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".env"), []byte(strings.Join([]string{
		"# comment",
		"FOO=bar",
		`QUOTED="hello world"`,
		"SINGLE='quoted'",
		"INVALID",
	}, "\n")), 0o644))

	service := &DefaultDevService{}
	envVars, err := service.loadDevEnvFile(dir)
	require.NoError(t, err)
	assert.Equal(t, map[string]string{
		"FOO":    "bar",
		"QUOTED": "hello world",
		"SINGLE": "quoted",
	}, envVars)
}

func TestDevServiceStartDevProcess(t *testing.T) {
	dir := t.TempDir()
	venvBin := filepath.Join(dir, "venv", "bin")
	require.NoError(t, os.MkdirAll(venvBin, 0o755))
	outputPath := filepath.Join(dir, "env-output.txt")
	script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' \"$PORT\" > %s\nprintf '%%s\\n' \"$AGENTFIELD_SERVER_URL\" >> %s\nprintf '%%s\\n' \"$AGENTFIELD_DEV_MODE\" >> %s\nprintf '%%s\\n' \"$TOKEN\" >> %s\n", outputPath, outputPath, outputPath, outputPath)
	require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte(script), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN=dev-secret\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "main.py"), []byte("print('ignored')\n"), 0o644))

	service := &DefaultDevService{}
	cmd, err := service.startDevProcess(dir, 8124, domain.DevOptions{Verbose: true})
	require.NoError(t, err)
	require.NoError(t, cmd.Wait())

	data, err := os.ReadFile(outputPath)
	require.NoError(t, err)
	assert.Equal(t, "8124\nhttp://localhost:8080\ntrue\ndev-secret\n", string(data))
}

func TestDevServicePortHelpersWithoutManager(t *testing.T) {
	service := &DefaultDevService{}

	port, err := service.getFreePort()
	require.NoError(t, err)
	assert.True(t, port >= 8001 && port <= 8999)

	busyListener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer busyListener.Close()
	busyPort := busyListener.Addr().(*net.TCPAddr).Port
	assert.False(t, service.isPortAvailable(busyPort))

	freeListener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	freePort := freeListener.Addr().(*net.TCPAddr).Port
	require.NoError(t, freeListener.Close())
	assert.True(t, service.isPortAvailable(freePort))
}

func TestAgentServiceRunStopAndStatusWithLiveProcess(t *testing.T) {
	t.Run("run agent success", func(t *testing.T) {
		home := t.TempDir()
		agentPath := t.TempDir()
		venvBin := filepath.Join(agentPath, "venv", "bin")
		require.NoError(t, os.MkdirAll(venvBin, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(venvBin, "python"), []byte("#!/bin/sh\nexit 0\n"), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(agentPath, ".env"), []byte("TOKEN=run-secret\n"), 0o644))

		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"run-agent": {
					Name:    "run-agent",
					Version: "1.0.0",
					Path:    agentPath,
					Status:  "stopped",
					Runtime: packages.RuntimeInfo{
						LogFile: filepath.Join(home, "logs", "run-agent.log"),
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

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
			assert.Equal(t, filepath.Join(venvBin, "python"), config.Command)
			assert.Contains(t, config.Env, "TOKEN=run-secret")
			return 4242, nil
		}

		portManager := newMockPortManager()
		portManager.findFreePortFunc = func(startPort int) (int, error) {
			return port, nil
		}

		service := NewAgentService(processManager, portManager, newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		runningAgent, err := service.RunAgent("run-agent", domain.RunOptions{})
		require.NoError(t, err)
		assert.Equal(t, "run-agent", runningAgent.Name)
		assert.Equal(t, 4242, runningAgent.PID)
		assert.Equal(t, port, runningAgent.Port)

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		assert.Equal(t, "running", updated.Installed["run-agent"].Status)
	})

	t.Run("stop agent fallback success", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sh", "-c", "trap '' INT; sleep 60")
		require.NoError(t, cmd.Start())
		pid := cmd.Process.Pid
		t.Cleanup(func() {
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
		})

		port := 8126
		startedAt := time.Now().Add(-2 * time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"stop-agent": {
					Name:    "stop-agent",
					Version: "1.0.0",
					Path:    home,
					Status:  "running",
					Runtime: packages.RuntimeInfo{
						Port:      &port,
						PID:       &pid,
						StartedAt: &startedAt,
						LogFile:   filepath.Join(home, "agent.log"),
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
		require.NoError(t, service.StopAgent("stop-agent"))

		waitErr := cmd.Wait()
		assert.Error(t, waitErr)

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		assert.Equal(t, "stopped", updated.Installed["stop-agent"].Status)
		assert.Nil(t, updated.Installed["stop-agent"].Runtime.PID)
		assert.Nil(t, updated.Installed["stop-agent"].Runtime.Port)
	})

	t.Run("get agent status running", func(t *testing.T) {
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

		port := 8127
		startedAt := time.Now().Add(-90 * time.Second).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"status-agent": {
					Name:    "status-agent",
					Version: "1.0.0",
					Path:    home,
					Status:  "running",
					Runtime: packages.RuntimeInfo{
						Port:      &port,
						PID:       &pid,
						StartedAt: &startedAt,
						LogFile:   filepath.Join(home, "agent.log"),
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
		status, err := service.GetAgentStatus("status-agent")
		require.NoError(t, err)
		assert.True(t, status.IsRunning)
		assert.Equal(t, port, status.Port)
		assert.Equal(t, pid, status.PID)
		assert.NotEmpty(t, status.Uptime)
	})
}

func assertEnvWithPrefix(t *testing.T, env []string, prefix, contains string) {
	t.Helper()
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) && strings.Contains(entry, contains) {
			return
		}
	}
	t.Fatalf("expected environment entry with prefix %q containing %q", prefix, contains)
}

// findFreePortInRange returns an ephemeral port that is free at call time.
// Uses :0 to let the OS assign a port, avoiding hardcoded range conflicts.
// Note: there is an inherent TOCTOU race between closing the listener and
// the caller re-binding. For tests that start an HTTP server, prefer
// startLocalServerOnFreePort which keeps the listener open.
func findFreePortInRange(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	return port
}

// startLocalServerOnFreePort starts an HTTP server on an OS-assigned port,
// avoiding the TOCTOU race of find-port-then-bind. Returns the server and
// the port it is listening on.
func startLocalServerOnFreePort(t *testing.T, handler http.Handler) (*http.Server, int) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := ln.Addr().(*net.TCPAddr).Port

	server := &http.Server{Handler: handler}
	go func() {
		_ = server.Serve(ln)
	}()

	t.Cleanup(func() {
		_ = server.Close()
	})

	return server, port
}

func startLocalServer(t *testing.T, port int, handler http.Handler) *http.Server {
	t.Helper()
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	require.NoError(t, err)

	server := &http.Server{Handler: handler}
	go func() {
		_ = server.Serve(ln)
	}()

	t.Cleanup(func() {
		_ = server.Close()
	})

	return server
}

// findPortInAgentRange finds a free port in the 8001-8999 range that
// discoverAgentPort and the production runner scan. Has an inherent TOCTOU
// race but is required for tests that exercise the hardcoded port range.
func findPortInAgentRange(t *testing.T) int {
	t.Helper()
	for port := 8001; port <= 8999; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err == nil {
			_ = ln.Close()
			return port
		}
	}
	t.Fatal("no free port found in range 8001-8999")
	return 0
}
