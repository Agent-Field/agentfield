package process

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
)

// DefaultProcessManager provides a default implementation for managing system processes.
// It keeps track of running processes and provides methods to start, stop, and monitor them.
type DefaultProcessManager struct {
	// mu guards runningProcesses: boot restores and update jobs start and stop
	// nodes from different goroutines.
	mu               sync.Mutex
	runningProcesses map[int]*exec.Cmd
	// stopProcess is a narrow test seam for the platform-specific bounded stop
	// path. Production always uses stopManagedProcess below.
	stopProcess func(*exec.Cmd, int) error
}

const gracefulStopTimeout = 5 * time.Second

// NewProcessManager creates a new instance of DefaultProcessManager.
// It initializes the map for tracking running processes.
func NewProcessManager() interfaces.ProcessManager {
	return &DefaultProcessManager{
		runningProcesses: make(map[int]*exec.Cmd),
	}
}

// Start initiates a new process based on the provided configuration.
// It returns the PID of the started process or an error if the process could not be started.
func (pm *DefaultProcessManager) Start(config interfaces.ProcessConfig) (pid int, err error) {
	// Create the command with arguments
	cmd := exec.Command(config.Command, config.Args...)

	// Set working directory if specified
	if config.WorkDir != "" {
		cmd.Dir = config.WorkDir
	}

	// Set environment variables
	if len(config.Env) > 0 {
		// Start with current environment and add/override with provided variables
		cmd.Env = append(os.Environ(), config.Env...)
	}

	// Handle log file redirection if specified
	if config.LogFile != "" {
		// Ensure log directory exists
		logDir := filepath.Dir(config.LogFile)
		if err := os.MkdirAll(logDir, 0755); err != nil {
			return 0, fmt.Errorf("failed to create log directory %s: %w", logDir, err)
		}

		// Create or open log file
		logFile, err := os.OpenFile(config.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return 0, fmt.Errorf("failed to open log file %s: %w", config.LogFile, err)
		}

		// Redirect stdout and stderr to log file
		cmd.Stdout = logFile
		cmd.Stderr = logFile

		// Note: We don't close the file here as the process needs it
		// The file will be closed when the process exits
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to start process '%s': %w", config.Command, err)
	}

	// Get the PID
	pid = cmd.Process.Pid

	// Track the running process
	pm.mu.Lock()
	pm.runningProcesses[pid] = cmd
	pm.mu.Unlock()

	return pid, nil
}

// Stop terminates a process identified by its PID.
// It attempts graceful termination first, then forceful termination if necessary.
func (pm *DefaultProcessManager) Stop(pid int) error {
	pm.mu.Lock()
	cmd, exists := pm.runningProcesses[pid]
	pm.mu.Unlock()
	if !exists {
		return fmt.Errorf("process with PID %d not found in managed processes", pid)
	}
	// Tracking is bookkeeping, not proof that an unresponsive process was
	// reaped. Every bounded return path must release the slot.
	defer func() {
		pm.mu.Lock()
		delete(pm.runningProcesses, pid)
		pm.mu.Unlock()
	}()

	// Check if process is still running
	if cmd.Process == nil {
		// Process already terminated, clean up
		return nil
	}
	if pm.stopProcess != nil {
		return pm.stopProcess(cmd, pid)
	}
	return stopManagedProcess(cmd, pid)
}

func stopManagedProcess(cmd *exec.Cmd, pid int) error {
	// Start reaping before signalling so every exit path is observed exactly
	// once and a child that ignores SIGTERM cannot wedge the caller forever.
	waited := make(chan struct{}, 1)
	go func() {
		_, _ = cmd.Process.Wait()
		waited <- struct{}{}
	}()

	// Try graceful termination first (SIGTERM).
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		if !errors.Is(err, os.ErrProcessDone) {
			// If SIGTERM fails, try forceful termination (SIGKILL)
			if killErr := cmd.Process.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				return fmt.Errorf("failed to terminate process %d: SIGTERM failed (%v), SIGKILL failed (%v)", pid, err, killErr)
			}
		}
	}

	timer := time.NewTimer(gracefulStopTimeout)
	select {
	case <-waited:
		timer.Stop()
	case <-timer.C:
		if killErr := cmd.Process.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
			return fmt.Errorf("failed to force kill process %d after %s: %w", pid, gracefulStopTimeout, killErr)
		}
		// Kill makes Wait return and reaps the child. Keep this final wait bounded
		// as defense against an unusual platform/process implementation.
		select {
		case <-waited:
		case <-time.After(time.Second):
			return fmt.Errorf("timed out reaping process %d after force kill", pid)
		}
	}

	return nil
}

// Status retrieves the current status and information of a process identified by its PID.
func (pm *DefaultProcessManager) Status(pid int) (interfaces.ProcessInfo, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	cmd, exists := pm.runningProcesses[pid]
	if !exists {
		return interfaces.ProcessInfo{}, fmt.Errorf("process with PID %d not found in managed processes", pid)
	}

	// Build the command string for display
	commandStr := cmd.Path
	if len(cmd.Args) > 1 {
		commandStr = fmt.Sprintf("%s %v", cmd.Path, cmd.Args[1:])
	}

	// Determine status
	status := "stopped"
	if cmd.Process != nil {
		// Check if process is still running by sending signal 0
		if err := cmd.Process.Signal(syscall.Signal(0)); err == nil {
			status = "running"
		} else {
			// Clean up if process is no longer running
			delete(pm.runningProcesses, pid)
		}
	} else {
		delete(pm.runningProcesses, pid)
	}

	return interfaces.ProcessInfo{
		PID:     pid,
		Status:  status,
		Command: commandStr,
	}, nil
}
