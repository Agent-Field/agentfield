package packages

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// AgentNodeRunner handles running agent nodes
type AgentNodeRunner struct {
	HaxenHome string
	Port      int
	Detach    bool
}


// RunAgentNode starts an installed agent node
func (ar *AgentNodeRunner) RunAgentNode(agentNodeName string) error {
	fmt.Printf("🚀 Launching agent node: %s\n", agentNodeName)

	// 1. Check if agent node is installed
	registry, err := ar.loadRegistry()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	agentNode, exists := registry.Installed[agentNodeName]
	if !exists {
		return fmt.Errorf("agent node %s not installed", agentNodeName)
	}

	// 2. Check if already running
	if agentNode.Status == "running" {
		return fmt.Errorf("agent node %s is already running on port %d", agentNodeName, *agentNode.Runtime.Port)
	}

	// 3. Allocate port
	fmt.Printf("🔍 Searching for available port...\n")
	port := ar.Port
	if port == 0 {
		port, err = ar.getFreePort()
		if err != nil {
			return fmt.Errorf("failed to allocate port: %w", err)
		}
	}

	fmt.Printf("✅ Assigned port: %d\n", port)

	// 4. Start agent node process
	fmt.Printf("📡 Starting agent node process...\n")
	cmd, err := ar.startAgentNodeProcess(agentNode, port)
	if err != nil {
		return fmt.Errorf("failed to start agent node: %w", err)
	}

	// 5. Wait for agent node to be ready
	if err := ar.waitForAgentNode(port, 10*time.Second); err != nil {
		cmd.Process.Kill()
		return fmt.Errorf("agent node failed to start: %w", err)
	}

	fmt.Printf("🧠 Agent node registered with Haxen Server\n")

	// 6. Update registry with runtime info
	if err := ar.updateRuntimeInfo(agentNodeName, port, cmd.Process.Pid); err != nil {
		return fmt.Errorf("failed to update runtime info: %w", err)
	}

	// 7. Display agent node capabilities
	if err := ar.displayCapabilities(agentNode, port); err != nil {
		fmt.Printf("⚠️  Could not fetch capabilities: %v\n", err)
	}

	fmt.Printf("\n💡 Agent node running in background (PID: %d)\n", cmd.Process.Pid)
	fmt.Printf("💡 View logs: haxen logs %s\n", agentNodeName)
	fmt.Printf("💡 Stop agent node: haxen stop %s\n", agentNodeName)

	return nil
}

// getFreePort finds an available port in the range 8001-8999
func (ar *AgentNodeRunner) getFreePort() (int, error) {
	for port := 8001; port <= 8999; port++ {
		if ar.isPortAvailable(port) {
			return port, nil
		}
	}
	return 0, fmt.Errorf("no free port available in range 8001-8999")
}

// isPortAvailable checks if a port is available
func (ar *AgentNodeRunner) isPortAvailable(port int) bool {
	conn, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// startAgentNodeProcess starts the agent node process
func (ar *AgentNodeRunner) startAgentNodeProcess(agentNode InstalledPackage, port int) (*exec.Cmd, error) {
	// Prepare environment variables
	env := os.Environ()
	env = append(env, fmt.Sprintf("PORT=%d", port))
	env = append(env, "HAXEN_SERVER_URL=http://localhost:8080")

	// Load environment variables from package .env file
	if envVars, err := ar.loadPackageEnvFile(agentNode.Path); err == nil {
		for key, value := range envVars {
			env = append(env, fmt.Sprintf("%s=%s", key, value))
		}
		fmt.Printf("🔧 Loaded %d environment variables from .env file\n", len(envVars))
	}

	// Prepare command - use virtual environment if available
	var pythonPath string
	venvPath := filepath.Join(agentNode.Path, "venv")
	
	// Check if virtual environment exists
	if _, err := os.Stat(filepath.Join(venvPath, "bin", "python")); err == nil {
		pythonPath = filepath.Join(venvPath, "bin", "python")
		fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)
	} else if _, err := os.Stat(filepath.Join(venvPath, "Scripts", "python.exe")); err == nil {
		pythonPath = filepath.Join(venvPath, "Scripts", "python.exe") // Windows
		fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)
	} else {
		// Fallback to system python
		pythonPath = "python"
		fmt.Printf("⚠️  Virtual environment not found, using system Python\n")
	}
	
	cmd := exec.Command(pythonPath, "main.py")
	cmd.Dir = agentNode.Path
	cmd.Env = env

	// Setup logging
	logFile, err := os.OpenFile(agentNode.Runtime.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	// Start process
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start process: %w", err)
	}

	return cmd, nil
}

// waitForAgentNode waits for the agent node to become ready
func (ar *AgentNodeRunner) waitForAgentNode(port int, timeout time.Duration) error {
	client := &http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", port))
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("agent node did not become ready within %v", timeout)
}

// displayCapabilities fetches and displays agent node capabilities
func (ar *AgentNodeRunner) displayCapabilities(agentNode InstalledPackage, port int) error {
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

	fmt.Printf("\n🌐 Access locally at: http://localhost:%d\n", port)
	fmt.Printf("📖 Available functions:\n")

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

// updateRuntimeInfo updates the registry with runtime information
func (ar *AgentNodeRunner) updateRuntimeInfo(agentNodeName string, port, pid int) error {
	registryPath := filepath.Join(ar.HaxenHome, "installed.yaml")

	// Load registry
	registry := &InstallationRegistry{}
	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	// Update runtime info
	if agentNode, exists := registry.Installed[agentNodeName]; exists {
		startedAt := time.Now().Format(time.RFC3339)
		agentNode.Status = "running"
		agentNode.Runtime.Port = &port
		agentNode.Runtime.PID = &pid
		agentNode.Runtime.StartedAt = &startedAt
		registry.Installed[agentNodeName] = agentNode
	}

	// Save registry
	data, err := yaml.Marshal(registry)
	if err != nil {
		return err
	}

	return os.WriteFile(registryPath, data, 0644)
}

// loadRegistry loads the installation registry
func (ar *AgentNodeRunner) loadRegistry() (*InstallationRegistry, error) {
	registryPath := filepath.Join(ar.HaxenHome, "installed.yaml")

	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nil, fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	return registry, nil
}

// loadPackageEnvFile loads environment variables from package .env file
func (ar *AgentNodeRunner) loadPackageEnvFile(packagePath string) (map[string]string, error) {
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
