//go:build !windows

package services

import (
	"context"
	"errors"
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

func TestCoverageGapInstallDependenciesErrors(t *testing.T) {
	t.Run("virtualenv creation fails for both python commands", func(t *testing.T) {
		packagePath := t.TempDir()
		fakeBin := filepath.Join(packagePath, "fake-bin")
		require.NoError(t, os.MkdirAll(fakeBin, 0o755))

		failScript := []byte("#!/bin/sh\necho venv failed >&2\nexit 1\n")
		require.NoError(t, os.WriteFile(filepath.Join(fakeBin, "python3"), failScript, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(fakeBin, "python"), failScript, 0o755))

		t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))

		service := &DefaultPackageService{}
		err := service.installDependencies(packagePath, &packages.PackageMetadata{
			Dependencies: packages.DependencyConfig{Python: []string{"requests"}},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create virtual environment")
	})

	t.Run("requirements installation fails", func(t *testing.T) {
		packagePath := t.TempDir()
		fakeBin := filepath.Join(packagePath, "fake-bin")
		require.NoError(t, os.MkdirAll(fakeBin, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(packagePath, "requirements.txt"), []byte("broken\n"), 0o644))

		python3Script := []byte(`#!/bin/sh
mkdir -p "$3/bin"
cat > "$3/bin/pip" <<'SH'
#!/bin/sh
if [ "$2" = "--upgrade" ]; then
  exit 0
fi
if [ "$2" = "-r" ]; then
  echo requirements failed >&2
  exit 2
fi
exit 0
SH
chmod +x "$3/bin/pip"
`)
		require.NoError(t, os.WriteFile(filepath.Join(fakeBin, "python3"), python3Script, 0o755))
		t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))

		service := &DefaultPackageService{}
		err := service.installDependencies(packagePath, &packages.PackageMetadata{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to install requirements.txt dependencies")
	})

	t.Run("dependency installation fails", func(t *testing.T) {
		packagePath := t.TempDir()
		fakeBin := filepath.Join(packagePath, "fake-bin")
		require.NoError(t, os.MkdirAll(fakeBin, 0o755))

		python3Script := []byte(`#!/bin/sh
mkdir -p "$3/bin"
cat > "$3/bin/pip" <<'SH'
#!/bin/sh
if [ "$2" = "--upgrade" ]; then
  exit 0
fi
echo dependency failed >&2
exit 3
SH
chmod +x "$3/bin/pip"
`)
		require.NoError(t, os.WriteFile(filepath.Join(fakeBin, "python3"), python3Script, 0o755))
		t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))

		service := &DefaultPackageService{}
		err := service.installDependencies(packagePath, &packages.PackageMetadata{
			Dependencies: packages.DependencyConfig{Python: []string{"broken-dep"}},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to install dependency broken-dep")
	})
}

func TestCoverageGapPackageServiceRegistryAndUninstall(t *testing.T) {
	t.Run("update registry invalid yaml", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		err := service.updateRegistry(&packages.PackageMetadata{Name: "pkg", Version: "1.0.0"}, "/src", "/dest")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse registry")
	})

	t.Run("update registry directory creation fails", func(t *testing.T) {
		blocker := filepath.Join(t.TempDir(), "blocker")
		require.NoError(t, os.WriteFile(blocker, []byte("x"), 0o644))

		service := &DefaultPackageService{agentfieldHome: blocker}
		err := service.updateRegistry(&packages.PackageMetadata{Name: "pkg", Version: "1.0.0"}, "/src", "/dest")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create registry directory")
	})

	t.Run("load registry invalid yaml", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		_, err := service.loadRegistryDirect()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse registry")
	})

	t.Run("uninstall running package without force", func(t *testing.T) {
		home := t.TempDir()
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"busy": {Name: "busy", Path: filepath.Join(home, "packages", "busy"), Status: "running"},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))

		service := &DefaultPackageService{agentfieldHome: home}
		err = service.uninstallPackage("busy", false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "currently running")
	})

	t.Run("uninstall missing package", func(t *testing.T) {
		service := &DefaultPackageService{agentfieldHome: t.TempDir()}
		err := service.uninstallPackage("missing", false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "is not installed")
	})
}

func TestCoverageGapAgentServiceBranches(t *testing.T) {
	t.Run("update runtime info read error", func(t *testing.T) {
		blocker := filepath.Join(t.TempDir(), "blocker")
		require.NoError(t, os.WriteFile(blocker, []byte("x"), 0o644))

		service := &DefaultAgentService{agentfieldHome: blocker}
		err := service.updateRuntimeInfo("agent", 8123, 44)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read registry")
	})

	t.Run("reconcile running state without pid", func(t *testing.T) {
		service := &DefaultAgentService{}
		pkg := &packages.InstalledPackage{Name: "agent", Status: "running"}

		running, reconciled := service.reconcileProcessState(pkg, "agent")
		assert.False(t, running)
		assert.True(t, reconciled)
		assert.Equal(t, "stopped", pkg.Status)
		assert.Nil(t, pkg.Runtime.Port)
		assert.Nil(t, pkg.Runtime.StartedAt)
	})

	t.Run("stop agent falls back after http shutdown error", func(t *testing.T) {
		home := t.TempDir()
		cmd := exec.Command("sh", "-c", "trap '' INT; sleep 60")
		require.NoError(t, cmd.Start())
		t.Cleanup(func() {
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
		})

		pid := cmd.Process.Pid
		port := 8139
		startedAt := time.Now().Add(-time.Minute).Format(time.RFC3339)
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"fallback-agent": {
					Name:   "fallback-agent",
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

		agentClient := &mockAgentClient{
			shutdownFunc: func(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
				return nil, errors.New("http unavailable")
			},
		}

		service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), agentClient, home).(*DefaultAgentService)
		require.NoError(t, service.StopAgent("fallback-agent"))

		updatedData, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
		require.NoError(t, err)
		var updated packages.InstallationRegistry
		require.NoError(t, yaml.Unmarshal(updatedData, &updated))
		assert.Equal(t, "stopped", updated.Installed["fallback-agent"].Status)
		assert.Nil(t, updated.Installed["fallback-agent"].Runtime.PID)
		assert.Nil(t, updated.Installed["fallback-agent"].Runtime.Port)
	})
}

func TestCoverageGapRunInDevModeAbsError(t *testing.T) {
	oldWd, err := os.Getwd()
	require.NoError(t, err)

	tmp := t.TempDir()
	require.NoError(t, os.Chdir(tmp))
	t.Cleanup(func() {
		_ = os.Chdir(oldWd)
	})
	require.NoError(t, os.RemoveAll(tmp))

	service := &DefaultDevService{fileSystem: newMockFileSystemAdapter()}
	err = service.RunInDevMode("pkg", domain.DevOptions{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to resolve path")
}
