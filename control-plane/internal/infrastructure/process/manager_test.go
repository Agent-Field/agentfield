package process

import (
	"fmt"
	"os"
	"runtime"
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
	case "exit":
		// Exit immediately
	default:
	}

	os.Exit(0)
}
