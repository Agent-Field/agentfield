package services

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func instantLifecycleConfirmation() packages.ProcessConfirmationPolicy {
	policy := packages.LifecycleConfirmationPolicy()
	policy.Sleep = func(context.Context, time.Duration) error { return nil }
	return policy
}

func TestE1LegacyRecycledPIDIsStoppedAndRunAgentRestoresIt(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the E1 fixture uses Linux /proc wall-clock identity")
	}
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgDir, "name: demo\nversion: 1.0.0\nagent_node:\n  node_id: demo\n")
	recycledPID := os.Getpid()
	processStarted, ok := packages.ProcessStartWallClock(recycledPID)
	require.True(t, ok)
	recorded := processStarted.Add(-time.Minute).Format(time.RFC3339)
	silentPort := findFreePortInRange(t)
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: &recycledPID, Port: &silentPort, StartedAt: &recorded},
		},
	}})

	readyServer, readyPort := startLocalServerOnFreePort(t, contractNodeHandler("demo"))
	defer readyServer.Close()
	manager := newMockProcessManager()
	var child *exec.Cmd
	manager.startFunc = func(interfaces.ProcessConfig) (int, error) {
		child = exec.Command("sleep", "60")
		if err := child.Start(); err != nil {
			return 0, err
		}
		return child.Process.Pid, nil
	}
	t.Cleanup(func() {
		if child != nil && child.Process != nil {
			_ = child.Process.Kill()
			_, _ = child.Process.Wait()
		}
	})
	ports := newMockPortManager()
	ports.findFreePortFunc = func(int) (int, error) { return readyPort, nil }
	service := NewAgentService(manager, ports, newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
	service.confirmation = instantLifecycleConfirmation()

	status, err := service.GetAgentStatus("demo")
	require.NoError(t, err)
	require.False(t, status.IsRunning)
	running, err := service.RunAgent("demo", domain.RunOptions{})
	require.NoError(t, err)
	require.Equal(t, readyPort, running.Port)
}

func TestE2LegacyOwnedSilentProcessIsKeptAndStoppedBySignal(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the signal fixture is Unix-specific")
	}
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgDir, "name: demo\nversion: 1.0.0\nlanguage: go\nagent_node:\n  node_id: demo\n")
	cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
	require.NoError(t, cmd.Start())
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
	started, ok := packages.ProcessStartWallClock(cmd.Process.Pid)
	require.True(t, ok)
	startedAt := started.Format(time.RFC3339)
	port := findFreePortInRange(t)
	pid := cmd.Process.Pid
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartedAt: &startedAt},
		},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	service.confirmation = instantLifecycleConfirmation()

	status, err := service.GetAgentStatus("demo")
	require.NoError(t, err)
	require.True(t, status.IsRunning)
	_, err = service.RunAgent("demo", domain.RunOptions{})
	require.ErrorContains(t, err, "already running")
	require.NoError(t, service.StopAgent("demo"))
	select {
	case <-done:
	case <-time.After(4 * time.Second):
		t.Fatal("owned legacy process was not interrupted")
	}
	registry, err := packages.LoadInstallationRegistry(filepath.Join(home, "installed.yaml"))
	require.NoError(t, err)
	entry := registry.Installed["demo"]
	require.Equal(t, "stopped", entry.Status)
	require.Nil(t, entry.Runtime.PID)
}

func TestE4SlowHealthReadKeepsExistingPIDOnLinuxAndWindows(t *testing.T) {
	service := &DefaultAgentService{}
	for _, goos := range []string{"linux", "windows"} {
		t.Run(goos, func(t *testing.T) {
			pid, port := os.Getpid(), 8123
			entry := packages.InstalledPackage{
				Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning,
				Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartTime: packages.CurrentProcessStartTime(pid)},
			}
			running, reconciled := service.reconcileProcessStateWithProbes(
				&entry, "demo", goos, false, make(map[int]packages.HealthIdentity),
				func(packages.RuntimeInfo) packages.RuntimeProcessState { return packages.RuntimeProcessAliveState },
				func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
			)
			require.True(t, running)
			require.False(t, reconciled)
			require.NotNil(t, entry.Runtime.PID)
			require.Equal(t, pid, *entry.Runtime.PID)
		})
	}
}

// E7 (service level): an explicit stop of a record whose PID is alive but
// cannot be identified — no start_time, no started_at, silent port — must
// say so and leave both the process and the record alone.
func TestE7StopAgentRefusesToGuessAboutAnUnidentifiedLivePID(t *testing.T) {
	home := t.TempDir()
	child := exec.Command("sleep", "60")
	require.NoError(t, child.Start())
	t.Cleanup(func() { _ = child.Process.Kill(); _, _ = child.Process.Wait() })
	pid := child.Process.Pid
	port := closedPort(t)
	registry := &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Version: "1.0.0", Path: filepath.Join(home, "packages", "demo"), Status: "running",
			DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{PID: &pid, Port: &port}},
	}}
	createTestRegistry(t, home, registry)
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)

	err := service.StopAgent("demo")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "could not verify")
	assert.NoError(t, child.Process.Signal(syscall.Signal(0)), "the unidentified process must not be signalled")

	reloaded, loadErr := service.loadRegistryDirect()
	require.NoError(t, loadErr)
	entry := reloaded.Installed["demo"]
	require.NotNil(t, entry.Runtime.PID)
	assert.Equal(t, pid, *entry.Runtime.PID, "the only PID that can recover the node is kept")
	assert.Equal(t, "running", entry.Status)
	assert.Equal(t, packages.DesiredStateRunning, entry.DesiredState, "an honest refusal records no intent change")
}
