package process

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		_ = flag.Set("test.run", "^(TestDefaultProcessManager_StartStatusStop|TestDefaultProcessManager_StopHandlesExitedProcess|TestDefaultPortManager_FindFreePortSkipsBusyAndReserved|TestDefaultPortManager_ReserveAndReleaseLifecycle|TestDefaultPortManager_ReserveFailsWhenSystemPortBusy|TestInfraProcessCoverage_StartBranches|TestInfraProcessCoverage_StopStatusBranches|TestInfraProcessCoverage_FindFreePortExhausted|TestProcessHelper)$")
	}

	os.Exit(m.Run())
}

func TestInfraProcessCoverage_StartBranches(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX shell assumptions")
	}

	t.Run("success_with_workdir_env_and_logfile", func(t *testing.T) {
		pm := NewProcessManager()
		tempDir := t.TempDir()
		logFile := filepath.Join(tempDir, "logs", "process.log")
		cfg := interfaces.ProcessConfig{
			Command: "sh",
			Args:    []string{"-c", "printf '%s\\n' \"$CUSTOM_VALUE\"; pwd"},
			Env:     []string{"CUSTOM_VALUE=from-test"},
			WorkDir: tempDir,
			LogFile: logFile,
		}

		pid, err := pm.Start(cfg)
		require.NoError(t, err)

		time.Sleep(100 * time.Millisecond)
		require.NoError(t, pm.Stop(pid))

		data, err := os.ReadFile(logFile)
		require.NoError(t, err)
		content := string(data)
		assert.Contains(t, content, "from-test")
		assert.Contains(t, content, tempDir)
	})

	testCases := []struct {
		name       string
		buildCfg   func(t *testing.T) interfaces.ProcessConfig
		wantSubstr string
	}{
		{
			name: "mkdirall_failure",
			buildCfg: func(t *testing.T) interfaces.ProcessConfig {
				tempDir := t.TempDir()
				blockingFile := filepath.Join(tempDir, "occupied")
				require.NoError(t, os.WriteFile(blockingFile, []byte("x"), 0o644))
				return interfaces.ProcessConfig{
					Command: "sh",
					Args:    []string{"-c", "echo unreachable"},
					LogFile: filepath.Join(blockingFile, "nested", "process.log"),
				}
			},
			wantSubstr: "failed to create log directory",
		},
		{
			name: "openfile_failure",
			buildCfg: func(t *testing.T) interfaces.ProcessConfig {
				tempDir := t.TempDir()
				logPath := filepath.Join(tempDir, "existing-dir")
				require.NoError(t, os.Mkdir(logPath, 0o755))
				return interfaces.ProcessConfig{
					Command: "sh",
					Args:    []string{"-c", "echo unreachable"},
					LogFile: logPath,
				}
			},
			wantSubstr: "failed to open log file",
		},
		{
			name: "start_failure",
			buildCfg: func(t *testing.T) interfaces.ProcessConfig {
				return interfaces.ProcessConfig{Command: filepath.Join(t.TempDir(), "missing-command")}
			},
			wantSubstr: "failed to start process",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			pm := NewProcessManager()
			_, err := pm.Start(tc.buildCfg(t))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantSubstr)
		})
	}
}

func TestInfraProcessCoverage_StopStatusBranches(t *testing.T) {
	pm := NewProcessManager().(*DefaultProcessManager)

	t.Run("stop_removes_nil_process", func(t *testing.T) {
		pid := 4242
		pm.runningProcesses[pid] = &exec.Cmd{}

		require.NoError(t, pm.Stop(pid))
		_, exists := pm.runningProcesses[pid]
		assert.False(t, exists)
	})

	t.Run("status_reports_stopped_and_cleans_nil_process", func(t *testing.T) {
		pid := 4343
		pm.runningProcesses[pid] = &exec.Cmd{
			Path: "/bin/example",
			Args: []string{"/bin/example", "--flag", "value"},
		}

		info, err := pm.Status(pid)
		require.NoError(t, err)
		assert.Equal(t, "stopped", info.Status)
		assert.Equal(t, pid, info.PID)
		assert.True(t, strings.Contains(info.Command, "--flag"), "command string should include args")

		_, exists := pm.runningProcesses[pid]
		assert.False(t, exists)
	})
}

func TestInfraProcessCoverage_FindFreePortExhausted(t *testing.T) {
	pm := NewPortManager().(*DefaultPortManager)
	startPort := 40000

	for port := startPort; port <= startPort+100; port++ {
		pm.reservedPorts[port] = true
	}

	port, err := pm.FindFreePort(startPort)
	require.Error(t, err)
	assert.Zero(t, port)
	assert.EqualError(t, err, fmt.Sprintf("no free port available in range %d-%d", startPort, startPort+100))
}
