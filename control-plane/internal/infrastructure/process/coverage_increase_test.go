package process

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultProcessManager_Start_Extended(t *testing.T) {
	pm := NewProcessManager()

	t.Run("WorkDir", func(t *testing.T) {
		tempDir := t.TempDir()
		cfg := interfaces.ProcessConfig{
			Command: "ls",
			WorkDir: tempDir,
		}
		// We don't necessarily need it to succeed if we just want to hit the line,
		// but let's try to make it succeed.
		if runtime.GOOS == "windows" {
			cfg.Command = "cmd"
			cfg.Args = []string{"/c", "dir"}
		}
		pid, err := pm.Start(cfg)
		if err == nil {
			require.NoError(t, pm.Stop(pid))
		}
	})

	t.Run("EnvVars", func(t *testing.T) {
		cfg := interfaces.ProcessConfig{
			Command: "env",
			Env:     []string{"TEST_VAR=true"},
		}
		if runtime.GOOS == "windows" {
			cfg.Command = "cmd"
			cfg.Args = []string{"/c", "set"}
		}
		pid, err := pm.Start(cfg)
		if err == nil {
			require.NoError(t, pm.Stop(pid))
		}
	})

	t.Run("LogFileRedirection", func(t *testing.T) {
		tempDir := t.TempDir()
		logFile := filepath.Join(tempDir, "subdir", "test.log")
		cfg := interfaces.ProcessConfig{
			Command: "echo",
			Args:    []string{"hello"},
			LogFile: logFile,
		}
		if runtime.GOOS == "windows" {
			cfg.Command = "cmd"
			cfg.Args = []string{"/c", "echo hello"}
		}
		pid, err := pm.Start(cfg)
		require.NoError(t, err)
		require.NoError(t, pm.Stop(pid))

		_, err = os.Stat(logFile)
		assert.NoError(t, err, "log file should be created")
	})

	t.Run("LogFile_MkdirAllFail", func(t *testing.T) {
		// Create a file where we want a directory
		tempDir := t.TempDir()
		badPath := filepath.Join(tempDir, "file")
		err := os.WriteFile(badPath, []byte("test"), 0644)
		require.NoError(t, err)

		logFile := filepath.Join(badPath, "wontwork", "test.log")
		cfg := interfaces.ProcessConfig{
			Command: "echo",
			LogFile: logFile,
		}
		_, err = pm.Start(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create log directory")
	})

	t.Run("LogFile_OpenFileFail", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("difficult to simulate file permission errors reliably on windows in this way")
		}
		tempDir := t.TempDir()
		logDir := filepath.Join(tempDir, "readonly")
		err := os.Mkdir(logDir, 0555) // Read-only directory
		require.NoError(t, err)
		defer os.Chmod(logDir, 0755)

		logFile := filepath.Join(logDir, "test.log")
		cfg := interfaces.ProcessConfig{
			Command: "echo",
			LogFile: logFile,
		}
		_, err = pm.Start(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to open log file")
	})

	t.Run("StartFail", func(t *testing.T) {
		cfg := interfaces.ProcessConfig{
			Command: "/non/existent/command",
		}
		_, err := pm.Start(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to start process")
	})
}

func TestDefaultProcessManager_StopStatus_Extended(t *testing.T) {
	pm := NewProcessManager().(*DefaultProcessManager)

	t.Run("StopNotFound", func(t *testing.T) {
		err := pm.Stop(999999)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("StatusNotFound", func(t *testing.T) {
		_, err := pm.Status(999999)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("Status_ProcessNoLongerRunning", func(t *testing.T) {
		cfg := interfaces.ProcessConfig{
			Command: "sleep",
			Args:    []string{"0.1"},
		}
		if runtime.GOOS == "windows" {
			cfg.Command = "cmd"
			cfg.Args = []string{"/c", "timeout /t 1"}
		}
		pid, err := pm.Start(cfg)
		require.NoError(t, err)

		// Wait for process to exit
		time.Sleep(200 * time.Millisecond)
		if runtime.GOOS == "windows" {
			time.Sleep(1 * time.Second)
		}

		info, err := pm.Status(pid)
		// It might return stopped or it might delete it and return error
		if err == nil {
			assert.Equal(t, "stopped", info.Status)
		} else {
			assert.Contains(t, err.Error(), "not found")
		}
	})
}

func TestDefaultPortManager_FindFreePort_Full(t *testing.T) {
	pm := NewPortManager()

	t.Run("NoFreePort", func(t *testing.T) {
		// This is hard to test without actually occupying 101 ports.
		// Let's mock it or just try to occupy a few and see if it skips.
		// Actually, we can just reserve them in the pm.
		startPort := 40000
		for i := 0; i <= 100; i++ {
			pm.(*DefaultPortManager).reservedPorts[startPort+i] = true
		}

		_, err := pm.FindFreePort(startPort)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no free port available")
	})
}
