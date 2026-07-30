package cli

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"gopkg.in/yaml.v3"
)

func seedUninstallPackage(t *testing.T, home, name, status string, pid *int) string {
	t.Helper()

	packageDir := filepath.Join(home, "packages", name)
	if err := os.MkdirAll(packageDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", packageDir, err)
	}
	manifest := []byte("name: " + name + "\nversion: 1.0.0\nentrypoint:\n  start: echo ready\n")
	if err := os.WriteFile(filepath.Join(packageDir, "agentfield-package.yaml"), manifest, 0o644); err != nil {
		t.Fatalf("WriteFile(manifest): %v", err)
	}

	registry := packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			name: {
				Name:   name,
				Path:   packageDir,
				Status: status,
				Runtime: packages.RuntimeInfo{
					PID: pid,
				},
			},
		},
	}
	data, err := yaml.Marshal(registry)
	if err != nil {
		t.Fatalf("yaml.Marshal(registry): %v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644); err != nil {
		t.Fatalf("WriteFile(installed.yaml): %v", err)
	}
	return packageDir
}

func readUninstallRegistry(t *testing.T, home string) packages.InstallationRegistry {
	t.Helper()

	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		t.Fatalf("ReadFile(installed.yaml): %v", err)
	}
	var registry packages.InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		t.Fatalf("yaml.Unmarshal(installed.yaml): %v", err)
	}
	return registry
}

func executeUninstall(t *testing.T, args ...string) error {
	t.Helper()

	cmd := NewUninstallCommand()
	cmd.SilenceErrors = true
	cmd.SilenceUsage = true
	cmd.SetArgs(args)
	return cmd.Execute()
}

func startUninstallChild(t *testing.T) (*exec.Cmd, <-chan struct{}) {
	t.Helper()

	child := exec.Command("sleep", "300")
	if err := child.Start(); err != nil {
		t.Fatalf("start sleep child: %v", err)
	}
	done := make(chan struct{})
	go func() {
		_ = child.Wait()
		close(done)
	}()
	t.Cleanup(func() {
		_ = child.Process.Kill()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Errorf("timed out reaping sleep child PID %d", child.Process.Pid)
		}
	})
	return child, done
}

func assertProcessAlive(t *testing.T, pid int) {
	t.Helper()
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("process %d is not alive: %v", pid, err)
	}
}

func assertPackagePresent(t *testing.T, home, name, packageDir string) {
	t.Helper()
	if _, ok := readUninstallRegistry(t, home).Installed[name]; !ok {
		t.Fatalf("registry entry %q was removed", name)
	}
	if _, err := os.Stat(packageDir); err != nil {
		t.Fatalf("package directory %q is not present: %v", packageDir, err)
	}
}

func TestUninstallHappyPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	packageDir := seedUninstallPackage(t, home, "demo", "stopped", nil)

	if err := executeUninstall(t, "demo"); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	if _, ok := readUninstallRegistry(t, home).Installed["demo"]; ok {
		t.Fatal("registry entry demo still exists")
	}
	if _, err := os.Stat(packageDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("package directory still exists or stat failed unexpectedly: %v", err)
	}
}

func TestUninstallRunningPackageRefusesWithoutForce(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	child, _ := startUninstallChild(t)
	pid := child.Process.Pid
	packageDir := seedUninstallPackage(t, home, "running-demo", "running", &pid)

	err := executeUninstall(t, "running-demo")
	if err == nil {
		t.Fatal("uninstall succeeded without --force")
	}
	message := strings.ToLower(err.Error())
	if !strings.Contains(message, "running") || !strings.Contains(message, "force") {
		t.Fatalf("uninstall error = %q, want running and force", err)
	}
	assertProcessAlive(t, pid)
	assertPackagePresent(t, home, "running-demo", packageDir)
}

func TestUninstallForceKillsRunningPackage(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	child, done := startUninstallChild(t)
	pid := child.Process.Pid
	packageDir := seedUninstallPackage(t, home, "running-demo", "running", &pid)

	if err := executeUninstall(t, "running-demo", "--force"); err != nil {
		t.Fatalf("forced uninstall: %v", err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("process %d was not killed", pid)
	}
	if _, ok := readUninstallRegistry(t, home).Installed["running-demo"]; ok {
		t.Fatal("registry entry running-demo still exists")
	}
	if _, err := os.Stat(packageDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("package directory still exists or stat failed unexpectedly: %v", err)
	}
}

func TestUninstallUnknownPackage(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	err := executeUninstall(t, "missing")
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "not installed") {
		t.Fatalf("uninstall error = %v, want not installed error", err)
	}
}

func TestUninstallForceFlagDoesNotLeakBetweenCommands(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	seedUninstallPackage(t, home, "stopped-demo", "stopped", nil)
	if err := executeUninstall(t, "stopped-demo", "--force"); err != nil {
		t.Fatalf("first forced uninstall: %v", err)
	}

	child, _ := startUninstallChild(t)
	pid := child.Process.Pid
	packageDir := seedUninstallPackage(t, home, "running-demo", "running", &pid)
	err := executeUninstall(t, "running-demo")
	if err == nil {
		t.Fatal("fresh command inherited --force")
	}
	message := strings.ToLower(err.Error())
	if !strings.Contains(message, "running") || !strings.Contains(message, "force") {
		t.Fatalf("uninstall error = %q, want running and force", err)
	}
	assertProcessAlive(t, pid)
	assertPackagePresent(t, home, "running-demo", packageDir)
}
