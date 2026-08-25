package cli

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
)

func TestStopperAndUtilityHelpers(t *testing.T) {
	t.Run("agent field home dir honors env and creates subdirs", func(t *testing.T) {
		home := filepath.Join(t.TempDir(), "custom-home")
		t.Setenv("AGENTFIELD_HOME", home)

		got := getAgentFieldHomeDir()
		require.Equal(t, home, got)
		for _, subdir := range []string{"packages", "logs", "config"} {
			info, err := os.Stat(filepath.Join(home, subdir))
			require.NoError(t, err)
			require.True(t, info.IsDir())
		}
	})

	t.Run("stopper load save and stop branches", func(t *testing.T) {
		home := t.TempDir()
		stopper := &AgentNodeStopper{AgentFieldHome: home}

		registry, err := stopper.loadRegistry()
		require.NoError(t, err)
		require.Empty(t, registry.Installed)

		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))
		_, err = stopper.loadRegistry()
		require.ErrorContains(t, err, "failed to parse registry")

		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "stopped", nil, nil)))
		saved, err := stopper.loadRegistry()
		require.NoError(t, err)
		require.Contains(t, saved.Installed, "demo")

		err = stopper.StopAgentNode("missing")
		require.ErrorContains(t, err, "not installed")

		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "stopped", nil, nil)))
		output := captureOutput(t, func() {
			require.NoError(t, stopper.StopAgentNode("demo"))
		})
		require.Contains(t, output, "is not running")

		// Stale records — a dead PID or a running-with-no-PID entry — must
		// reconcile to "stopped" rather than error, or stop-then-start flows
		// (desktop restart, login autostart) wedge permanently after reboot.
		pid := 999999
		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", nil, &pid)))
		require.NoError(t, stopper.StopAgentNode("demo"))
		saved, err = stopper.loadRegistry()
		require.NoError(t, err)
		require.Equal(t, "stopped", saved.Installed["demo"].Status)

		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", nil, nil)))
		require.NoError(t, stopper.StopAgentNode("demo"))
		saved, err = stopper.loadRegistry()
		require.NoError(t, err)
		require.Equal(t, "stopped", saved.Installed["demo"].Status)
	})
}

type recordingRestoreAgent struct{ runs int }

func (a *recordingRestoreAgent) RunAgent(string, domain.RunOptions) (*domain.RunningAgent, error) {
	a.runs++
	return &domain.RunningAgent{}, nil
}

func TestExplicitCLIStopDisarmsNextMaintenanceRestore(t *testing.T) {
	for _, observedStatus := range []string{"running", "stopped"} {
		t.Run(observedStatus, func(t *testing.T) {
			home := t.TempDir()
			port := 8123
			stopper := &AgentNodeStopper{AgentFieldHome: home, Quiet: true}
			registry := makeRegistry("demo", observedStatus, &port, nil)
			entry := registry.Installed["demo"]
			entry.DesiredState = packages.DesiredStateRunning
			registry.Installed["demo"] = entry
			require.NoError(t, stopper.saveRegistry(registry))

			require.NoError(t, stopper.StopAgentNode("demo"))
			stopped, err := stopper.loadRegistry()
			require.NoError(t, err)
			require.Equal(t, packages.DesiredStateStopped, stopped.Installed["demo"].DesiredState)

			agent := &recordingRestoreAgent{}
			maintenance := packagemaint.New(packagemaint.Config{
				AgentFieldHome: home,
				Agent:          agent,
				Enabled:        func() (bool, string) { return false, "disabled" },
				ProcessAlive:   func(packages.RuntimeInfo) bool { return false },
				HealthProbe:    func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
			})
			summary := maintenance.RunPass(context.Background())
			require.Zero(t, agent.runs, "a package explicitly stopped by af stop was restored")
			require.Empty(t, summary.Restored)
		})
	}
}

func makeRegistry(name, status string, port, pid *int) *packages.InstallationRegistry {
	startTime := ""
	bootID := ""
	if pid != nil {
		startTime = packages.CurrentProcessStartTime(*pid)
		bootID = packages.CurrentBootID()
	}
	return &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			name: {
				Name:   name,
				Path:   filepath.Join("/tmp", name),
				Status: status,
				Runtime: packages.RuntimeInfo{
					Port:      port,
					PID:       pid,
					LogFile:   filepath.Join("/tmp", name+".log"),
					BootID:    bootID,
					StartTime: startTime,
				},
			},
		},
	}
}
