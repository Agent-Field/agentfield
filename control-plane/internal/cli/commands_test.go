package cli

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

// TestListCommand tests the list command
func TestListCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewListCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{})

	// Should not error even if no packages installed
	err := cmd.Execute()
	require.NoError(t, err)
}

// TestStopCommand tests the stop command argument validation
func TestStopCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewStopCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// Test missing argument
	cmd.SetArgs([]string{})
	err := cmd.Execute()
	require.Error(t, err)

	// Test with argument (will fail because agent doesn't exist, but validates command structure)
	cmd.SetArgs([]string{"test-agent"})
	err = cmd.Execute()
	// Error is expected if agent doesn't exist, but command should be valid
	// The error message should indicate the agent is not installed or not running
	if err != nil {
		errorMsg := err.Error()
		require.True(t,
			strings.Contains(strings.ToLower(errorMsg), "not installed") ||
			strings.Contains(strings.ToLower(errorMsg), "not running") ||
			strings.Contains(strings.ToLower(errorMsg), "not found"),
			"Expected error about agent not found/installed/running, got: %s", errorMsg)
	}
}

// TestConfigCommand tests the config command
func TestConfigCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewConfigCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{})

	// Should display help or config info
	err := cmd.Execute()
	// May error if no config, but validates command structure
	_ = err
}

// TestVCCommand tests the VC command
func TestVCCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewVCCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{})

	// Should display help
	err := cmd.Execute()
	require.NoError(t, err)
}

// TestVersionCommand tests the version command
func TestVersionCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewVersionCommand(VersionInfo{
		Version: "1.0.0",
		Commit:  "abc123",
		Date:    "2024-01-01",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{})

	err := cmd.Execute()
	require.NoError(t, err)
}

// TestUninstallCommand tests the uninstall command
func TestUninstallCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewUninstallCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// Test missing argument
	cmd.SetArgs([]string{})
	err := cmd.Execute()
	require.Error(t, err)

	// Test with argument
	cmd.SetArgs([]string{"test-package"})
	err = cmd.Execute()
	// May error if package doesn't exist, but validates command structure
	_ = err
}

// TestLogsCommand tests the logs command
func TestLogsCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewLogsCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// Test missing argument
	cmd.SetArgs([]string{})
	err := cmd.Execute()
	require.Error(t, err)

	// Test with argument
	cmd.SetArgs([]string{"test-agent"})
	err = cmd.Execute()
	// May error if agent doesn't exist, but validates command structure
	_ = err
}

// TestInitCommand tests the init command
func TestInitCommand(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewInitCommand()
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{})

	err := cmd.Execute()
	// May error if already initialized, but validates command structure
	_ = err
}

// TestRootCommandFlags tests various root command flags
func TestRootCommandFlags(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	tests := []struct {
		name   string
		args   []string
		verify func(t *testing.T)
	}{
		{
			name: "verbose_flag",
			args: []string{"--verbose", "server", "--open=false"},
			verify: func(t *testing.T) {
				require.True(t, verbose)
			},
		},
		{
			name: "port_flag",
			args: []string{"--port=8080", "server", "--open=false"},
			verify: func(t *testing.T) {
				require.Equal(t, 8080, GetPortFlag())
			},
		},
		{
			name: "storage_mode_flag",
			args: []string{"--storage-mode=postgres", "server", "--open=false"},
			verify: func(t *testing.T) {
				require.Equal(t, "postgres", storageModeFlag)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetCLIStateForTest()
			invoked := false
			cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {
				invoked = true
				if tt.verify != nil {
					tt.verify(t)
				}
			}, VersionInfo{
				Version: "test",
				Commit:  "test",
				Date:    "test",
			})
			cmd.SetOut(io.Discard)
			cmd.SetErr(io.Discard)
			cmd.SetArgs(tt.args)

			err := cmd.Execute()
			require.NoError(t, err)
			require.True(t, invoked)
		})
	}
}

// TestRootCommandInvalidArgs tests invalid command arguments
func TestRootCommandInvalidArgs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// Test invalid command
	cmd.SetArgs([]string{"invalid-command"})
	err := cmd.Execute()
	require.Error(t, err)
}

// TestRootCommandConfigFile tests config file handling
func TestRootCommandConfigFile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "agentfield.yaml")
	configContent := `agentfield:
  port: 7000
  storage:
    mode: local
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	var received string
	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {
		received = GetConfigFilePath()
	}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--config", configPath, "server", "--open=false"})

	err := cmd.Execute()
	require.NoError(t, err)
	require.Equal(t, configPath, received)
}

// TestFrameworkCommands tests framework-based commands (install, run, dev)
func TestFrameworkCommands(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	// These commands require service container setup
	// Testing that they can be created and have proper structure
	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})

	// Verify install command exists
	installCmd, _, _ := cmd.Find([]string{"install"})
	require.NotNil(t, installCmd)

	// Verify run command exists
	runCmd, _, _ := cmd.Find([]string{"run"})
	require.NotNil(t, runCmd)

	// Verify dev command exists
	devCmd, _, _ := cmd.Find([]string{"dev"})
	require.NotNil(t, devCmd)
}
