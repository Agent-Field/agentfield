//go:build !windows

package services

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
)

type DefaultDevService struct {
	processManager interfaces.ProcessManager
	portManager    interfaces.PortManager
	fileSystem     interfaces.FileSystemAdapter
}

func NewDevService(
	processManager interfaces.ProcessManager,
	portManager interfaces.PortManager,
	fileSystem interfaces.FileSystemAdapter,
) interfaces.DevService {
	return &DefaultDevService{
		processManager: processManager,
		portManager:    portManager,
		fileSystem:     fileSystem,
	}
}

func (ds *DefaultDevService) RunInDevMode(path string, options domain.DevOptions) error {
	// Convert to absolute path
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}

	// Check if agentfield.yaml exists
	agentfieldYamlPath := filepath.Join(absPath, "agentfield.yaml")
	if !ds.fileSystem.Exists(agentfieldYamlPath) {
		return fmt.Errorf("no agentfield.yaml found in %s", absPath)
	}

	return ds.runDev(absPath, options)
}

func (ds *DefaultDevService) StopDevMode(path string) error {
	// TODO: Implement dev mode stopping logic
	// This would involve tracking running dev processes and stopping them
	return fmt.Errorf("stop dev mode not yet implemented")
}

func (ds *DefaultDevService) GetDevStatus(path string) (*domain.DevStatus, error) {
	// TODO: Implement dev status retrieval
	// This would involve checking if a dev server is running for the given path
	return nil, fmt.Errorf("get dev status not yet implemented")
}

// runDev starts the agent package in development mode
func (ds *DefaultDevService) runDev(packagePath string, options domain.DevOptions) error {
	fmt.Printf("🔧 Development Mode: %s\n", packagePath)

	// Temporarily disable MCP server startup - let Python SDK manage them
	// Config loading removed since MCP manager is disabled
	var agentCmd *exec.Cmd // Declare agentCmd here to be accessible in defer and signal handler

	// Defer StopAll to ensure it's called on any exit path from runDev
	defer func() {
		// MCP server management disabled - Python SDK handles MCP servers
		fmt.Println("\n✅ MCP server management delegated to Python SDK.")
	}()

	// Setup signal handling to gracefully shut down
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigs
		fmt.Printf("\nReceived signal: %s. Initiating shutdown...\n", sig)
		// Attempt to gracefully terminate the agent process group
		if agentCmd != nil && agentCmd.Process != nil {
			// Send SIGINT to the process group. Negative PID sends to the group.
			if err := syscall.Kill(-agentCmd.Process.Pid, syscall.SIGINT); err != nil {
				// If group signal fails, try to kill the main process directly
				if errKill := agentCmd.Process.Kill(); errKill != nil {
					fmt.Printf("⚠️ Failed to kill agent process: %v\n", errKill)
				}
			}
		}
	}()

	// MCP server startup disabled - Python SDK manages MCP servers
	if options.Verbose {
		fmt.Println("ℹ️ MCP server management delegated to Python SDK.")
	}

	// 1. Start agent process (let Python SDK choose its own port)
	fmt.Printf("📡 Starting agent process...\n")
	var agentStartErr error
	agentCmd, agentStartErr = ds.startDevProcess(packagePath, options.Port, options)
	if agentStartErr != nil {
		return fmt.Errorf("failed to start agent: %w", agentStartErr)
	}

	// 2. Discover the port the agent actually chose
	port, discoverErr := ds.discoverAgentPort(120 * time.Second)
	if discoverErr != nil {
		if agentCmd.Process != nil {
			_ = agentCmd.Process.Kill()
		}
		return fmt.Errorf("failed to discover agent port: %w", discoverErr)
	}

	fmt.Printf("✅ Agent ready on port %d\n", port)

	// 3. Display capabilities
	if err := ds.displayDevCapabilities(port); err != nil {
		fmt.Printf("⚠️  Could not fetch capabilities: %v\n", err)
	}

	// Always run in foreground for dev mode (no detach option)
	fmt.Printf("\n💡 Agent running in foreground\n")
	fmt.Printf("💡 Access at: http://localhost:%d\n", port)
	fmt.Printf("💡 Press Ctrl+C to stop\n\n")

	// Wait for process to complete
	if agentErr := agentCmd.Wait(); agentErr != nil {
		if exitErr, ok := agentErr.(*exec.ExitError); ok {
			if ws, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				if ws.Signaled() && (ws.Signal() == syscall.SIGINT || ws.Signal() == syscall.SIGTERM) {
					fmt.Printf("Agent process terminated by signal: %s\n", ws.Signal())
				} else {
					fmt.Printf("Agent process exited with error: %v\n", agentErr)
				}
			} else {
				fmt.Printf("Agent process exited with error: %v\n", agentErr)
			}
		} else {
			fmt.Printf("Error waiting for agent process: %v\n", agentErr)
		}
	}

	return nil
}

// getFreePort finds an available port in the range 8001-8999.
//
//nolint:unused // retained for future dev-service enhancements
func (ds *DefaultDevService) getFreePort() (int, error) {
	for port := 8001; port <= 8999; port++ {
		if ds.isPortAvailable(port) {
			return port, nil
		}
	}
	return 0, fmt.Errorf("no free port available in range 8001-8999")
}

// isPortAvailable checks if a port is available.
//
//nolint:unused // retained for future dev-service enhancements
func (ds *DefaultDevService) isPortAvailable(port int) bool {
	conn, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// startDevProcess starts the agent process in development mode
func (ds *DefaultDevService) startDevProcess(packagePath string, port int, options domain.DevOptions) (*exec.Cmd, error) {
	// Prepare environment variables
	env := os.Environ()
	// Only set PORT if it's a valid port (> 0), otherwise let Python agent choose its own port
	if port > 0 {
		env = append(env, fmt.Sprintf("PORT=%d", port))
	}
	env = append(env, "AGENTFIELD_SERVER_URL=http://localhost:8080")
	env = append(env, "AGENTFIELD_DEV_MODE=true")

	// Load environment variables from package .env file
	if envVars, err := ds.loadDevEnvFile(packagePath); err == nil {
		for key, value := range envVars {
			env = append(env, fmt.Sprintf("%s=%s", key, value))
		}
		if options.Verbose {
			fmt.Printf("🔧 Loaded %d environment variables from .env file\n", len(envVars))
		}
	}

	// Prepare command - use virtual environment if available
	var pythonPath string
	venvPath := filepath.Join(packagePath, "venv")

	// Check if virtual environment exists
	if _, err := os.Stat(filepath.Join(venvPath, "bin", "python")); err == nil {
		pythonPath = filepath.Join(venvPath, "bin", "python")
		if options.Verbose {
			fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)
		}
	} else if _, err := os.Stat(filepath.Join(venvPath, "Scripts", "python.exe")); err == nil {
		pythonPath = filepath.Join(venvPath, "Scripts", "python.exe") // Windows
		if options.Verbose {
			fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)
		}
	} else {
		// Fallback to system python
		pythonPath = "python"
		if options.Verbose {
			fmt.Printf("⚠️  Virtual environment not found, using system Python\n")
		}
	}

	cmd := exec.Command(pythonPath, "main.py")
	cmd.Dir = packagePath
	cmd.Env = env

	// Show output in terminal for interactive mode
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start process
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start process: %w", err)
	}

	return cmd, nil
}

// discoverAgentPort discovers the port the agent actually chose by scanning common ports
func (ds *DefaultDevService) discoverAgentPort(timeout time.Duration) (int, error) {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)

	fmt.Printf("🔍 Discovering agent port...\n")

	checkCount := 0

	for time.Now().Before(deadline) {
		checkCount++

		// Try ports in range 8001-8999
		for port := 8001; port <= 8999; port++ {
			resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", port))

			if err == nil && resp.StatusCode == 200 {
				resp.Body.Close()
				fmt.Printf("✅ Discovered agent on port %d after %d checks\n", port, checkCount)
				return port, nil
			}

			if resp != nil {
				resp.Body.Close()
			}
		}

		// Log progress every 20 checks to avoid spam
		if checkCount%20 == 0 {
			fmt.Printf("🔄 Port discovery attempt %d...\n", checkCount)
		}

		time.Sleep(500 * time.Millisecond)
	}

	return 0, fmt.Errorf("could not discover agent port within %v after %d attempts", timeout, checkCount)
}

// waitForAgent waits for the agent to become ready in dev mode.
//
//nolint:unused // retained for future dev-service enhancements
func (ds *DefaultDevService) waitForAgent(port int, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)

	fmt.Printf("🔍 Waiting for agent to become ready on port %d...\n", port)

	lastError := ""
	checkCount := 0

	for time.Now().Before(deadline) {
		checkCount++
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", port))

		if err == nil {
			if resp.StatusCode == 200 {
				resp.Body.Close()
				fmt.Printf("✅ Agent is ready after %d health checks\n", checkCount)
				return nil
			} else {
				// Agent is responding but not ready yet - this is progress
				if checkCount%10 == 0 { // Log every 10th check to avoid spam
					fmt.Printf("🔄 Agent responding with status %d (check %d)...\n", resp.StatusCode, checkCount)
				}
				lastError = fmt.Sprintf("HTTP %d", resp.StatusCode)
			}
			resp.Body.Close()
		} else {
			// Connection error - agent may still be starting
			if checkCount%20 == 0 { // Log every 20th check for connection errors
				fmt.Printf("🔄 Waiting for agent to start (check %d): %v\n", checkCount, err)
			}
			lastError = err.Error()
		}

		time.Sleep(500 * time.Millisecond)
	}

	// Provide more detailed error information
	return fmt.Errorf("agent did not become ready within %v (last error: %s, checks: %d)", timeout, lastError, checkCount)
}

// displayDevCapabilities fetches and displays agent capabilities in dev mode
func (ds *DefaultDevService) displayDevCapabilities(port int) error {
	client := &http.Client{Timeout: 5 * time.Second}

	// Get reasoners
	reasonersResp, err := client.Get(fmt.Sprintf("http://localhost:%d/reasoners", port))
	if err != nil {
		return err
	}
	defer reasonersResp.Body.Close()

	var reasonersData map[string]interface{}
	if err := json.NewDecoder(reasonersResp.Body).Decode(&reasonersData); err != nil {
		return err
	}

	// Get skills
	skillsResp, err := client.Get(fmt.Sprintf("http://localhost:%d/skills", port))
	if err != nil {
		return err
	}
	defer skillsResp.Body.Close()

	var skillsData map[string]interface{}
	if err := json.NewDecoder(skillsResp.Body).Decode(&skillsData); err != nil {
		return err
	}

	fmt.Printf("\n🌐 Development server: http://localhost:%d\n", port)
	fmt.Printf("� Available functions:\n")

	// Display reasoners
	if reasoners, ok := reasonersData["reasoners"].([]interface{}); ok && len(reasoners) > 0 {
		fmt.Printf("  🧠 Reasoners: ")
		var reasonerNames []string
		for _, reasoner := range reasoners {
			if r, ok := reasoner.(map[string]interface{}); ok {
				if id, ok := r["id"].(string); ok {
					reasonerNames = append(reasonerNames, id)
				}
			}
		}
		fmt.Printf("%s\n", strings.Join(reasonerNames, ", "))
	}

	// Display skills
	if skills, ok := skillsData["skills"].([]interface{}); ok && len(skills) > 0 {
		fmt.Printf("  🛠️  Skills:    ")
		var skillNames []string
		for _, skill := range skills {
			if s, ok := skill.(map[string]interface{}); ok {
				if id, ok := s["id"].(string); ok {
					skillNames = append(skillNames, id)
				}
			}
		}
		fmt.Printf("%s\n", strings.Join(skillNames, ", "))
	}

	return nil
}

// loadDevEnvFile loads environment variables from package .env file for dev mode
func (ds *DefaultDevService) loadDevEnvFile(packagePath string) (map[string]string, error) {
	envPath := filepath.Join(packagePath, ".env")

	data, err := os.ReadFile(envPath)
	if err != nil {
		return nil, err
	}

	envVars := make(map[string]string)
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			// Remove quotes if present
			if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
				(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
				value = value[1 : len(value)-1]
			}

			envVars[key] = value
		}
	}

	return envVars, nil
}
