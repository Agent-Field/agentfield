package process

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"

	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
)

// DefaultProcessManager provides a default implementation for managing system processes.
// It keeps track of running processes and provides methods to start, stop, and monitor them.
type DefaultProcessManager struct {
	runningProcesses map[int]*exec.Cmd
}

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
	pm.runningProcesses[pid] = cmd

	return pid, nil
}

// Stop terminates a process identified by its PID.
// It attempts graceful termination first, then forceful termination if necessary.
func (pm *DefaultProcessManager) Stop(pid int) error {
	cmd, exists := pm.runningProcesses[pid]
	if !exists {
		return fmt.Errorf("process with PID %d not found in managed processes", pid)
	}

	// Check if process is still running
	if cmd.Process == nil {
		// Process already terminated, clean up
		delete(pm.runningProcesses, pid)
		return nil
	}

	// Try graceful termination first (SIGTERM)
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		// If SIGTERM fails, try forceful termination (SIGKILL)
		if killErr := cmd.Process.Kill(); killErr != nil {
			return fmt.Errorf("failed to terminate process %d: SIGTERM failed (%v), SIGKILL failed (%v)", pid, err, killErr)
		}
	}

	// Wait for the process to actually terminate
	_, waitErr := cmd.Process.Wait()
	if waitErr != nil {
		// Process might have already exited, which is fine
		// We'll still clean up our tracking
	}

	// Clean up tracking
	delete(pm.runningProcesses, pid)

	return nil
}

// Status retrieves the current status and information of a process identified by its PID.
func (pm *DefaultProcessManager) Status(pid int) (interfaces.ProcessInfo, error) {
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
	status := "unknown"
	if cmd.Process != nil {
		// Check if process is still running by sending signal 0
		if err := cmd.Process.Signal(syscall.Signal(0)); err == nil {
			status = "running"
		} else {
			status = "stopped"
			// Clean up if process is no longer running
			delete(pm.runningProcesses, pid)
		}
	} else {
		status = "stopped"
		delete(pm.runningProcesses, pid)
	}

	return interfaces.ProcessInfo{
		PID:     pid,
		Status:  status,
		Command: commandStr,
	}, nil
}
