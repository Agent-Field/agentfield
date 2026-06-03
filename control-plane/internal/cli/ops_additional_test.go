package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
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

		pid := 999999
		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", nil, &pid)))
		err = stopper.StopAgentNode("demo")
		require.Error(t, err)
		require.True(t,
			strings.Contains(err.Error(), "failed to find process") ||
				strings.Contains(err.Error(), "failed to kill process") ||
				strings.Contains(err.Error(), "failed to force kill process"),
			"unexpected error: %v", err)

		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", nil, nil)))
		err = stopper.StopAgentNode("demo")
		require.ErrorContains(t, err, "no PID found")
	})
}

func makeRegistry(name, status string, port, pid *int) *packages.InstallationRegistry {
	return &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			name: {
				Name:   name,
				Path:   filepath.Join("/tmp", name),
				Status: status,
				Runtime: packages.RuntimeInfo{
					Port:    port,
					PID:     pid,
					LogFile: filepath.Join("/tmp", name+".log"),
				},
			},
		},
	}
}
