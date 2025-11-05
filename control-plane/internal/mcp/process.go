package mcp

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// ProcessManager handles MCP server process lifecycle management
type ProcessManager struct {
	projectDir string
	template   *TemplateProcessor
	verbose    bool
}

// NewProcessManager creates a new process manager instance
func NewProcessManager(projectDir string, template *TemplateProcessor, verbose bool) *ProcessManager {
	return &ProcessManager{
		projectDir: projectDir,
		template:   template,
		verbose:    verbose,
	}
}

// StartLocalMCP starts a local MCP server using the run command
func (pm *ProcessManager) StartLocalMCP(config MCPServerConfig) (*MCPProcess, error) {
	// Find available port
	port, err := pm.FindAvailablePort(3001)
	if err != nil {
		return nil, fmt.Errorf("failed to find available port: %w", err)
	}

	// Create template variables
	vars := pm.template.CreateTemplateVars(config, port)

	// Process run command
	processedCmd, err := pm.template.ProcessCommand(config.RunCmd, vars)
	if err != nil {
		return nil, fmt.Errorf("failed to process run command: %w", err)
	}

	// Set working directory
	workingDir := vars.ServerDir
	if config.WorkingDir != "" {
		processedWorkingDir, err := pm.template.ProcessCommand(config.WorkingDir, vars)
		if err != nil {
			return nil, fmt.Errorf("failed to process working directory: %w", err)
		}
		workingDir = processedWorkingDir
	}

	if pm.verbose {
		fmt.Printf("Executing run command: %s\n", processedCmd)
		fmt.Printf("Working directory: %s\n", workingDir)
		fmt.Printf("Port: %d\n", port)
	}

	// Create context for process management
	ctx, cancel := context.WithCancel(context.Background())

	// Execute run command
	cmd := exec.CommandContext(ctx, "sh", "-c", processedCmd)
	cmd.Dir = workingDir

	// Set environment variables
	if len(config.Env) > 0 {
		processedEnv, err := pm.template.ProcessEnvironment(config.Env, vars)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to process environment variables: %w", err)
		}

		for key, value := range processedEnv {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}

	// Set up pipes
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to start process: %w", err)
	}

	// Create process structure
	process := &MCPProcess{
		Config:  config,
		Cmd:     cmd,
		Stdin:   stdin,
		Stdout:  stdout,
		Stderr:  stderr,
		Context: ctx,
		Cancel:  cancel,
		LogFile: vars.LogFile,
	}

	// Update config with process info
	process.Config.Port = port
	process.Config.PID = cmd.Process.Pid
	process.Config.Status = string(StatusRunning)

	return process, nil
}

// ConnectRemoteMCP connects to a remote MCP server
func (pm *ProcessManager) ConnectRemoteMCP(config MCPServerConfig) (*MCPProcess, error) {
	if pm.verbose {
		fmt.Printf("Connecting to remote MCP: %s\n", config.URL)
	}

	// Validate URL connectivity
	if err := pm.validateRemoteURL(config.URL); err != nil {
		return nil, fmt.Errorf("failed to connect to remote MCP: %w", err)
	}

	// Create a minimal process structure for remote MCPs
	process := &MCPProcess{
		Config:  config,
		LogFile: filepath.Join(pm.projectDir, "packages", "mcp", config.Alias, fmt.Sprintf("%s.log", config.Alias)),
	}

	// Update config
	process.Config.Status = string(StatusRunning)

	return process, nil
}

// MonitorProcess monitors a running process
func (pm *ProcessManager) MonitorProcess(process *MCPProcess, onExit func(alias string, err error)) {
	if process.Cmd == nil {
		// Remote MCP - no process to monitor
		return
	}

	// Wait for process to finish
	err := process.Cmd.Wait()

	// Update status
	if err != nil {
		process.Config.Status = string(StatusError)
		if pm.verbose {
			fmt.Printf("Process %s exited with error: %v\n", process.Config.Alias, err)
		}
	} else {
		process.Config.Status = string(StatusStopped)
		if pm.verbose {
			fmt.Printf("Process %s exited normally\n", process.Config.Alias)
		}
	}

	// Call exit callback
	if onExit != nil {
		onExit(process.Config.Alias, err)
	}
}

// ExecuteSetupCommands executes setup commands for an MCP server
func (pm *ProcessManager) ExecuteSetupCommands(config MCPServerConfig) error {
	if len(config.SetupCmds) == 0 {
		return nil
	}

	serverDir := filepath.Join(pm.projectDir, "packages", "mcp", config.Alias)

	// Create template variables (port 0 for setup)
	vars := pm.template.CreateTemplateVars(config, 0)

	// Process setup commands
	processedCmds, err := pm.template.ProcessCommands(config.SetupCmds, vars)
	if err != nil {
		return fmt.Errorf("failed to process setup commands: %w", err)
	}

	workingDir := serverDir
	if config.WorkingDir != "" {
		processedWorkingDir, err := pm.template.ProcessCommand(config.WorkingDir, vars)
		if err != nil {
			return fmt.Errorf("failed to process working directory: %w", err)
		}
		workingDir = processedWorkingDir
	}

	if pm.verbose {
		fmt.Printf("Executing setup commands in %s:\n", workingDir)
	}

	for i, cmd := range processedCmds {
		if pm.verbose {
			fmt.Printf("  [%d/%d] %s\n", i+1, len(processedCmds), cmd)
		}

		execCmd := exec.Command("sh", "-c", cmd)
		execCmd.Dir = workingDir

		// Set environment variables
		if len(config.Env) > 0 {
			processedEnv, err := pm.template.ProcessEnvironment(config.Env, vars)
			if err != nil {
				return fmt.Errorf("failed to process environment variables: %w", err)
			}

			for key, value := range processedEnv {
				execCmd.Env = append(execCmd.Env, fmt.Sprintf("%s=%s", key, value))
			}
		}

		output, err := execCmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("setup command failed: %s\nOutput: %s", err, string(output))
		}

		if pm.verbose && len(output) > 0 {
			fmt.Printf("    Output: %s\n", strings.TrimSpace(string(output)))
		}
	}

	return nil
}

// ExecuteRunCommand executes the run command for an MCP server
func (pm *ProcessManager) ExecuteRunCommand(config MCPServerConfig, port int) (*exec.Cmd, error) {
	// Create template variables
	vars := pm.template.CreateTemplateVars(config, port)

	// Process run command
	processedCmd, err := pm.template.ProcessCommand(config.RunCmd, vars)
	if err != nil {
		return nil, fmt.Errorf("failed to process run command: %w", err)
	}

	// Set working directory
	workingDir := vars.ServerDir
	if config.WorkingDir != "" {
		processedWorkingDir, err := pm.template.ProcessCommand(config.WorkingDir, vars)
		if err != nil {
			return nil, fmt.Errorf("failed to process working directory: %w", err)
		}
		workingDir = processedWorkingDir
	}

	// Create command
	cmd := exec.Command("sh", "-c", processedCmd)
	cmd.Dir = workingDir

	// Set environment variables
	if len(config.Env) > 0 {
		processedEnv, err := pm.template.ProcessEnvironment(config.Env, vars)
		if err != nil {
			return nil, fmt.Errorf("failed to process environment variables: %w", err)
		}

		for key, value := range processedEnv {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}

	return cmd, nil
}

// FindAvailablePort finds an available port starting from the given port
func (pm *ProcessManager) FindAvailablePort(startPort int) (int, error) {
	for port := startPort; port < startPort+100; port++ {
		if pm.IsPortAvailable(port) {
			return port, nil
		}
	}
	return 0, fmt.Errorf("no available ports found in range %d-%d", startPort, startPort+99)
}

// IsPortAvailable checks if a port is available
func (pm *ProcessManager) IsPortAvailable(port int) bool {
	address := fmt.Sprintf(":%d", port)
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return false
	}
	defer listener.Close()
	return true
}

// HealthCheck performs a health check on an MCP server
func (pm *ProcessManager) HealthCheck(config MCPServerConfig, port int) error {
	if config.HealthCheck == "" {
		// No health check configured - assume healthy
		return nil
	}

	// Create template variables
	vars := pm.template.CreateTemplateVars(config, port)

	// Process health check command
	processedCmd, err := pm.template.ProcessCommand(config.HealthCheck, vars)
	if err != nil {
		return fmt.Errorf("failed to process health check command: %w", err)
	}

	if pm.verbose {
		fmt.Printf("Executing health check: %s\n", processedCmd)
	}

	// Set timeout for health check
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Execute health check with timeout
	cmd := exec.CommandContext(ctx, "sh", "-c", processedCmd)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("health check failed: %s\nOutput: %s", err, string(output))
	}

	if pm.verbose {
		fmt.Printf("Health check passed\n")
	}

	return nil
}

// StopProcess stops a running process
func (pm *ProcessManager) StopProcess(process *MCPProcess) error {
	if process.Cancel != nil {
		process.Cancel()
	}

	if process.Cmd != nil && process.Cmd.Process != nil {
		// Try graceful shutdown first
		if err := process.Cmd.Process.Signal(os.Interrupt); err != nil {
			// Force kill if graceful shutdown fails
			return process.Cmd.Process.Kill()
		}

		// Wait a bit for graceful shutdown
		time.Sleep(2 * time.Second)

		// Check if still running
		if pm.IsProcessRunning(process) {
			return process.Cmd.Process.Kill()
		}
	}

	return nil
}

// RestartProcess restarts a running process
func (pm *ProcessManager) RestartProcess(process *MCPProcess) (*MCPProcess, error) {
	// Stop the current process
	if err := pm.StopProcess(process); err != nil {
		return nil, fmt.Errorf("failed to stop process: %w", err)
	}

	// Wait a moment for cleanup
	time.Sleep(1 * time.Second)

	// Start a new process
	if process.Config.URL != "" {
		return pm.ConnectRemoteMCP(process.Config)
	} else {
		return pm.StartLocalMCP(process.Config)
	}
}

// IsProcessRunning checks if a process is still running
func (pm *ProcessManager) IsProcessRunning(process *MCPProcess) bool {
	if process.Cmd == nil || process.Cmd.Process == nil {
		return false
	}

	// Try to signal the process with signal 0 (test if process exists)
	err := process.Cmd.Process.Signal(syscall.Signal(0))
	return err == nil
}

// GetProcessLogs returns a reader for the process logs
func (pm *ProcessManager) GetProcessLogs(alias string, follow bool, lines int) (io.ReadCloser, error) {
	logFile := filepath.Join(pm.projectDir, "packages", "mcp", alias, fmt.Sprintf("%s.log", alias))

	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("log file not found for server %s", alias)
	}

	// For now, just return the file reader
	// TODO: Implement follow and lines functionality
	file, err := os.Open(logFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	return file, nil
}

// validateRemoteURL validates that a remote URL is accessible
func (pm *ProcessManager) validateRemoteURL(url string) error {
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("failed to connect to URL: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("URL returned error status: %d", resp.StatusCode)
	}

	return nil
}
