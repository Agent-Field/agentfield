package process

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func helperProcessConfig(mode string) interfaces.ProcessConfig {
	return interfaces.ProcessConfig{
		Command: os.Args[0],
		Args:    []string{"-test.run=TestProcessHelper"},
		Env: []string{
			"GO_WANT_HELPER_PROCESS=1",
			fmt.Sprintf("HELPER_MODE=%s", mode),
		},
	}
}

func TestDefaultProcessManager_StartStatusStop(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process helper uses POSIX signals")
	}

	pm := NewProcessManager()

	cfg := helperProcessConfig("block")

	pid, err := pm.Start(cfg)
	require.NoError(t, err)
	require.True(t, pid > 0)

	info, err := pm.Status(pid)
	require.NoError(t, err)
	assert.Equal(t, "running", info.Status)

	require.NoError(t, pm.Stop(pid))

	_, err = pm.Status(pid)
	require.Error(t, err, "process should be removed after stop")
}

func TestDefaultProcessManager_StopHandlesExitedProcess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process helper uses POSIX signals")
	}

	pm := NewProcessManager()
	cfg := helperProcessConfig("exit")

	pid, err := pm.Start(cfg)
	require.NoError(t, err)
	require.True(t, pid > 0)

	time.Sleep(50 * time.Millisecond)

	err = pm.Stop(pid)
	require.NoError(t, err, "stopping an already exited process should not fail")

	_, err = pm.Status(pid)
	require.Error(t, err, "process should be removed after stop")
}

func TestC10StopKillsChildThatIgnoresSIGTERMWithinSixSeconds(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process helper uses POSIX signals")
	}
	pm := NewProcessManager().(*DefaultProcessManager)
	pid, err := pm.Start(helperProcessConfig("ignore-term"))
	require.NoError(t, err)
	time.Sleep(100 * time.Millisecond)
	process := pm.runningProcesses[pid].Process
	started := time.Now()
	require.NoError(t, pm.Stop(pid))
	elapsed := time.Since(started)
	assert.Less(t, elapsed, 6*time.Second)
	assert.GreaterOrEqual(t, elapsed, 4*time.Second)
	assert.Error(t, process.Signal(syscall.Signal(0)), "force-killed child still exists")
}

func TestE21FinalReapTimeoutStillRemovesProcessFromManager(t *testing.T) {
	pm := &DefaultProcessManager{
		runningProcesses: map[int]*exec.Cmd{42: {Process: &os.Process{Pid: 42}}},
		stopProcess: func(_ *exec.Cmd, pid int) error {
			return fmt.Errorf("timed out reaping process %d after force kill", pid)
		},
	}
	require.ErrorContains(t, pm.Stop(42), "timed out reaping process 42")
	_, exists := pm.runningProcesses[42]
	require.False(t, exists, "bounded stop returns must release process bookkeeping")
}

func TestProcessHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	mode := os.Getenv("HELPER_MODE")
	if mode == "" {
		mode = "block"
	}

	switch mode {
	case "block":
		select {}
	case "ignore-term":
		signal.Ignore(syscall.SIGTERM)
		select {}
	case "exit":
		// Exit immediately
	default:
	}

	os.Exit(0)
}
