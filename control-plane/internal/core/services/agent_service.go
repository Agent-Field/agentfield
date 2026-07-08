// agentfield/internal/core/services/agent_service.go
package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"gopkg.in/yaml.v3"
)

// DefaultAgentService implements the AgentService interface
type DefaultAgentService struct {
	processManager  interfaces.ProcessManager
	portManager     interfaces.PortManager
	registryStorage interfaces.RegistryStorage
	agentClient     interfaces.AgentClient
	agentfieldHome  string
}

// NewAgentService creates a new agent service instance
func NewAgentService(
	processManager interfaces.ProcessManager,
	portManager interfaces.PortManager,
	registryStorage interfaces.RegistryStorage,
	agentClient interfaces.AgentClient,
	agentfieldHome string,
) interfaces.AgentService {
	return &DefaultAgentService{
		processManager:  processManager,
		portManager:     portManager,
		registryStorage: registryStorage,
		agentClient:     agentClient,
		agentfieldHome:  agentfieldHome,
	}
}

// RunAgent starts an installed agent
func (as *DefaultAgentService) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	return as.runAgentGuarded(name, options, map[string]bool{})
}

// runAgentGuarded starts a node; inProgress tracks nodes already being started
// in this dependency chain to break cycles.
func (as *DefaultAgentService) runAgentGuarded(name string, options domain.RunOptions, inProgress map[string]bool) (*domain.RunningAgent, error) {
	fmt.Printf("🚀 Launching agent node: %s\n", name)
	inProgress[name] = true

	// 1. Check if agent node is installed
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	agentNode, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return nil, fmt.Errorf("agent node %s not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName

	// 2. Check current state and reconcile if needed
	actuallyRunning, wasReconciled := as.reconcileProcessState(&agentNode, name)
	if wasReconciled {
		// Save reconciled state
		registry.Installed[name] = agentNode
		if err := as.saveRegistryDirect(registry); err != nil {
			fmt.Printf("Warning: failed to save reconciled registry state: %v\n", err)
		}
	}

	// If actually running after reconciliation, return appropriate message
	if actuallyRunning {
		return nil, fmt.Errorf("agent node %s is already running on port %d", name, *agentNode.Runtime.Port)
	}

	// 2b. Start declared node dependencies first, before allocating this node's
	// port — each dependency fully binds its own port, avoiding collisions.
	as.startNodeDependencies(agentNode, inProgress, options)

	// 3. Allocate port
	fmt.Printf("🔍 Searching for available port...\n")
	port := options.Port
	if port == 0 {
		port, err = as.portManager.FindFreePort(8001)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate port: %w", err)
		}
	}

	fmt.Printf("✅ Assigned port: %d\n", port)

	// 4. Start agent node process
	fmt.Printf("📡 Starting agent node process...\n")
	processConfig, err := as.buildProcessConfig(agentNode, port)
	if err != nil {
		return nil, err
	}
	pid, err := as.processManager.Start(processConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to start agent node: %w", err)
	}

	// 5. Wait for agent node to be ready
	healthPath := "/health"
	if metadata, err := packages.ParsePackageMetadata(agentNode.Path); err == nil {
		healthPath = metadata.HealthcheckPath()
	}
	if err := as.waitForAgentNode(port, healthPath, nodeReadyTimeout()); err != nil {
		// Kill the process if it failed to start properly
		if stopErr := as.processManager.Stop(pid); stopErr != nil {
			return nil, fmt.Errorf("agent node failed to start: %w (additionally failed to stop process: %v)", err, stopErr)
		}
		return nil, fmt.Errorf("agent node failed to start: %w", err)
	}

	fmt.Printf("🧠 Agent node registered with AgentField Server\n")

	// 6. Update registry with runtime info
	if err := as.updateRuntimeInfo(name, port, pid); err != nil {
		return nil, fmt.Errorf("failed to update runtime info: %w", err)
	}

	// 7. Display agent node capabilities
	if err := as.displayCapabilities(agentNode, port); err != nil {
		fmt.Printf("⚠️  Could not fetch capabilities: %v\n", err)
	}

	fmt.Printf("\n💡 Agent node running in background (PID: %d)\n", pid)
	fmt.Printf("💡 View logs: af logs %s\n", name)
	fmt.Printf("💡 Stop agent node: af stop %s\n", name)

	// Convert to domain model and return
	runningAgent := as.convertToRunningAgent(agentNode)
	runningAgent.PID = pid
	runningAgent.Port = port
	runningAgent.StartedAt = time.Now()

	return &runningAgent, nil
}

// StopAgent stops a running agent with robust error handling
func (as *DefaultAgentService) StopAgent(name string) error {
	// Load registry to get agent info
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	pkg, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return fmt.Errorf("agent %s is not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName

	// Check current state and reconcile if needed
	actuallyRunning, wasReconciled := as.reconcileProcessState(&pkg, name)
	if wasReconciled {
		// Save reconciled state
		registry.Installed[name] = pkg
		if err := as.saveRegistryDirect(registry); err != nil {
			fmt.Printf("Warning: failed to save reconciled registry state: %v\n", err)
		}
	}

	// If not actually running after reconciliation, return appropriate message
	if !actuallyRunning {
		if pkg.Status == "stopped" {
			return fmt.Errorf("agent %s is not running", name)
		} else {
			// Was marked as running but process was dead - now reconciled
			fmt.Printf("Agent %s was marked as running but process was not found - state has been corrected\n", name)
			return nil
		}
	}

	// Agent is actually running - proceed with HTTP shutdown
	if pkg.Runtime.Port == nil {
		return fmt.Errorf("no port found for agent %s", name)
	}

	// Try HTTP shutdown first
	httpShutdownSuccess := false
	if as.agentClient != nil {
		fmt.Printf("🛑 Attempting graceful HTTP shutdown for agent %s\n", name)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		// Construct node ID from agent name (assuming they match)
		nodeID := name

		// Try graceful shutdown with 30-second timeout
		shutdownResp, err := as.agentClient.ShutdownAgent(ctx, nodeID, true, 30)
		if err == nil && shutdownResp != nil && shutdownResp.Status == "shutting_down" {
			fmt.Printf("✅ HTTP shutdown request accepted for agent %s\n", name)
			httpShutdownSuccess = true

			// Wait a moment for the agent to shut down gracefully
			time.Sleep(2 * time.Second)
		} else {
			fmt.Printf("⚠️ HTTP shutdown failed for agent %s: %v\n", name, err)
		}
	}

	// If HTTP shutdown failed or not available, fall back to process signals
	if !httpShutdownSuccess {
		fmt.Printf("🔄 Falling back to process signal shutdown for agent %s\n", name)

		if pkg.Runtime.PID == nil {
			return fmt.Errorf("no PID found for agent %s", name)
		}

		// Stop the process
		process, err := os.FindProcess(*pkg.Runtime.PID)
		if err != nil {
			// Process not found - update registry and return success
			fmt.Printf("Process %d not found for agent %s - updating registry\n", *pkg.Runtime.PID, name)
			pkg.Status = "stopped"
			pkg.Runtime.PID = nil
			pkg.Runtime.Port = nil
			pkg.Runtime.StartedAt = nil
			registry.Installed[name] = pkg
			if err := as.saveRegistryDirect(registry); err != nil {
				return fmt.Errorf("failed to update registry: %w", err)
			}
			return nil
		}

		// Send SIGTERM first for graceful shutdown
		if err := process.Signal(os.Interrupt); err != nil {
			// If graceful shutdown fails, force kill
			if err := process.Kill(); err != nil {
				// Handle "process already finished" gracefully
				if strings.Contains(err.Error(), "process already finished") ||
					strings.Contains(err.Error(), "no such process") {
					fmt.Printf("Process %d for agent %s already finished - updating registry\n", *pkg.Runtime.PID, name)
				} else {
					return fmt.Errorf("failed to kill process: %w", err)
				}
			}
		} else {
			// Wait a moment for graceful shutdown, then force kill if needed
			time.Sleep(3 * time.Second)

			// Check if process is still running
			if err := process.Signal(syscall.Signal(0)); err == nil {
				// Process still running, force kill
				fmt.Printf("⚠️ Process %d still running, force killing agent %s\n", *pkg.Runtime.PID, name)
				if err := process.Kill(); err != nil && !strings.Contains(err.Error(), "process already finished") {
					return fmt.Errorf("failed to force kill process: %w", err)
				}
			}
		}
	}

	// Update registry to mark as stopped
	pkg.Status = "stopped"
	pkg.Runtime.PID = nil
	pkg.Runtime.Port = nil
	pkg.Runtime.StartedAt = nil
	registry.Installed[name] = pkg

	// Save registry
	if err := as.saveRegistryDirect(registry); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	return nil
}

// GetAgentStatus returns the status of a specific agent with process reconciliation
func (as *DefaultAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	pkg, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return nil, fmt.Errorf("agent %s is not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName

	// Reconcile registry state with actual process state
	actuallyRunning, reconciled := as.reconcileProcessState(&pkg, name)
	if reconciled {
		// Save updated registry if reconciliation occurred
		registry.Installed[name] = pkg
		if err := as.saveRegistryDirect(registry); err != nil {
			fmt.Printf("Warning: failed to save reconciled registry state: %v\n", err)
		}
	}

	status := &domain.AgentStatus{
		Name:      pkg.Name,
		IsRunning: actuallyRunning,
	}

	if pkg.Runtime.Port != nil {
		status.Port = *pkg.Runtime.Port
	}

	if pkg.Runtime.PID != nil {
		status.PID = *pkg.Runtime.PID
	}

	if pkg.Runtime.StartedAt != nil {
		if startedAt, err := time.Parse(time.RFC3339, *pkg.Runtime.StartedAt); err == nil {
			status.LastSeen = startedAt
			// Calculate uptime if running
			if actuallyRunning {
				uptime := time.Since(startedAt)
				status.Uptime = uptime.String()
			}
		}
	}

	return status, nil
}

// reconcileProcessState checks if the registry state matches actual process state
// Returns (actuallyRunning, wasReconciled)
func (as *DefaultAgentService) reconcileProcessState(pkg *packages.InstalledPackage, name string) (bool, bool) {
	registryRunning := pkg.Status == "running"

	// If registry says not running, trust it (no process to check)
	if !registryRunning {
		return false, false
	}

	// Registry says running - verify the process actually exists
	if pkg.Runtime.PID == nil {
		// Registry says running but no PID - inconsistent state
		fmt.Printf("Warning: Agent %s marked as running but no PID found, marking as stopped\n", name)
		pkg.Status = "stopped"
		pkg.Runtime.Port = nil
		pkg.Runtime.StartedAt = nil
		return false, true
	}

	// Check if process actually exists
	process, err := os.FindProcess(*pkg.Runtime.PID)
	if err != nil {
		// Process not found - mark as stopped
		fmt.Printf("Warning: Agent %s process (PID %d) not found, marking as stopped\n", name, *pkg.Runtime.PID)
		pkg.Status = "stopped"
		pkg.Runtime.PID = nil
		pkg.Runtime.Port = nil
		pkg.Runtime.StartedAt = nil
		return false, true
	}

	// On Unix systems, check if process is actually alive
	if runtime.GOOS != "windows" {
		if err := process.Signal(syscall.Signal(0)); err != nil {
			// Process exists but is not alive (zombie or permission issue)
			fmt.Printf("Warning: Agent %s process (PID %d) not responding, marking as stopped\n", name, *pkg.Runtime.PID)
			pkg.Status = "stopped"
			pkg.Runtime.PID = nil
			pkg.Runtime.Port = nil
			pkg.Runtime.StartedAt = nil
			return false, true
		}
	}

	// Process exists and appears to be running
	return true, false
}

// ListRunningAgents returns a list of all running agents
func (as *DefaultAgentService) ListRunningAgents() ([]domain.RunningAgent, error) {
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	var runningAgents []domain.RunningAgent
	for _, pkg := range registry.Installed {
		if pkg.Status == "running" {
			runningAgents = append(runningAgents, as.convertToRunningAgent(pkg))
		}
	}

	return runningAgents, nil
}

// loadRegistryDirect loads the registry using direct file system access
// TODO: Eventually replace with registryStorage interface usage
func (as *DefaultAgentService) loadRegistryDirect() (*packages.InstallationRegistry, error) {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")

	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nil, fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	return registry, nil
}

// saveRegistryDirect saves the registry using direct file system access
// TODO: Eventually replace with registryStorage interface usage
func (as *DefaultAgentService) saveRegistryDirect(registry *packages.InstallationRegistry) error {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")

	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	return os.WriteFile(registryPath, data, 0644)
}

// convertToRunningAgent converts packages.InstalledPackage to domain.RunningAgent
func (as *DefaultAgentService) convertToRunningAgent(pkg packages.InstalledPackage) domain.RunningAgent {
	agent := domain.RunningAgent{
		Name:   pkg.Name,
		Status: pkg.Status,
	}

	if pkg.Runtime.Port != nil {
		agent.Port = *pkg.Runtime.Port
	}

	if pkg.Runtime.PID != nil {
		agent.PID = *pkg.Runtime.PID
	}

	if pkg.Runtime.StartedAt != nil {
		if startedAt, err := time.Parse(time.RFC3339, *pkg.Runtime.StartedAt); err == nil {
			agent.StartedAt = startedAt
		}
	}

	agent.LogFile = pkg.Runtime.LogFile

	return agent
}

// startNodeDependencies starts a node's installed, not-yet-running node
// dependencies before the node itself. inProgress guards against cycles.
func (as *DefaultAgentService) startNodeDependencies(node packages.InstalledPackage, inProgress map[string]bool, options domain.RunOptions) {
	metadata, err := packages.ParsePackageMetadata(node.Path)
	if err != nil {
		return
	}
	for _, ref := range metadata.Dependencies.Nodes {
		depName := packages.NodeDepName(ref)
		if depName == "" || inProgress[depName] {
			continue
		}
		registry, err := as.loadRegistryDirect()
		if err != nil {
			return
		}
		dep, _, exists := as.findAgentInRegistry(registry, depName)
		if !exists {
			fmt.Printf("⚠️  Node dependency %s is declared but not installed (run: af install %s)\n", depName, ref)
			continue
		}
		if running, _ := as.reconcileProcessState(&dep, depName); running {
			continue
		}
		fmt.Printf("🔗 Starting node dependency: %s\n", depName)
		// Dependencies get an auto-assigned port, not the parent's --port.
		depOptions := options
		depOptions.Port = 0
		if _, err := as.runAgentGuarded(depName, depOptions, inProgress); err != nil {
			fmt.Printf("⚠️  Failed to start node dependency %s: %v\n", depName, err)
		}
	}
}

// buildProcessConfig creates a process configuration for starting an agent.
// It reads the manifest entrypoint and resolves declared environment variables
// from the encrypted secret store (prompting for missing required ones).
func (as *DefaultAgentService) buildProcessConfig(agentNode packages.InstalledPackage, port int) (interfaces.ProcessConfig, error) {
	// Read the manifest for the entrypoint, healthcheck and declared env. Fall
	// back to defaults (python main.py) if no manifest is present.
	metadata, err := packages.ParsePackageMetadata(agentNode.Path)
	if err != nil {
		fmt.Printf("⚠️  No usable manifest (%v); falling back to python main.py\n", err)
		metadata = &packages.PackageMetadata{}
	}

	// Prepare environment variables. Export both AGENTFIELD_SERVER (the var the
	// SDK reads) and the legacy AGENTFIELD_SERVER_URL.
	serverURL := resolveServerURL()
	env := os.Environ()
	env = append(env, fmt.Sprintf("PORT=%d", port))
	// Tell the SDK to bind exactly this port and fail fast if it is unavailable,
	// rather than silently moving to another port that the runner is not polling
	// (the readiness check below targets this exact port). Gated by this signal so
	// standalone `python -m <node>.app` keeps its lenient auto-port fallback.
	env = append(env, "AGENTFIELD_STRICT_PORT=1")
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER=%s", serverURL))
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER_URL=%s", serverURL))

	// Resolve declared variables from the encrypted secret store. Secrets are
	// injected only into this child process — never written to disk in plaintext.
	resolvedEnv, err := as.resolveNodeEnvironment(agentNode.Name, metadata)
	if err != nil {
		return interfaces.ProcessConfig{}, err
	}
	for key, value := range resolvedEnv {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	// Determine Python path - use virtual environment if available
	var pythonPath string
	venvPath := filepath.Join(agentNode.Path, "venv")

	// Check if virtual environment exists (Unix/Linux/macOS)
	if _, err := os.Stat(filepath.Join(venvPath, "bin", "python")); err == nil {
		pythonPath = filepath.Join(venvPath, "bin", "python")
		fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)

		// Complete virtual environment activation for Unix/Linux/macOS
		venvBinPath := filepath.Join(venvPath, "bin")

		// Set VIRTUAL_ENV first (required for proper activation)
		env = append(env, fmt.Sprintf("VIRTUAL_ENV=%s", venvPath))

		// Prepend virtual environment bin to PATH (critical for package resolution)
		currentPath := os.Getenv("PATH")
		env = append(env, fmt.Sprintf("PATH=%s:%s", venvBinPath, currentPath))

		// Unset PYTHONHOME to avoid conflicts with virtual environment
		env = append(env, "PYTHONHOME=")

		// Set PYTHONPATH to ensure proper module resolution
		env = append(env, fmt.Sprintf("PYTHONPATH=%s", filepath.Join(venvPath, "lib")))

		fmt.Printf("✅ Virtual environment fully activated with PATH=%s\n", venvBinPath)

	} else if _, err := os.Stat(filepath.Join(venvPath, "Scripts", "python.exe")); err == nil {
		pythonPath = filepath.Join(venvPath, "Scripts", "python.exe") // Windows
		fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)

		// Complete virtual environment activation for Windows
		venvScriptsPath := filepath.Join(venvPath, "Scripts")

		// Set VIRTUAL_ENV first (required for proper activation)
		env = append(env, fmt.Sprintf("VIRTUAL_ENV=%s", venvPath))

		// Prepend virtual environment Scripts to PATH (critical for package resolution)
		currentPath := os.Getenv("PATH")
		env = append(env, fmt.Sprintf("PATH=%s;%s", venvScriptsPath, currentPath))

		// Unset PYTHONHOME to avoid conflicts with virtual environment
		env = append(env, "PYTHONHOME=")

		// Set PYTHONPATH to ensure proper module resolution
		env = append(env, fmt.Sprintf("PYTHONPATH=%s", filepath.Join(venvPath, "Lib", "site-packages")))

		fmt.Printf("✅ Virtual environment fully activated with PATH=%s\n", venvScriptsPath)

	} else {
		// Try to find python3 or python
		if pythonPath = as.findPythonExecutable(); pythonPath == "" {
			pythonPath = "python" // Final fallback
		}
		fmt.Printf("⚠️  Virtual environment not found at %s, using system Python: %s\n", venvPath, pythonPath)
	}

	// Launch via the manifest entrypoint (e.g. "python -m pr_af.app"). When the
	// program token is python/python3, substitute the resolved interpreter.
	startArgs := metadata.StartCommand()
	command := startArgs[0]
	args := startArgs[1:]
	if command == "python" || command == "python3" {
		command = pythonPath
	}

	return interfaces.ProcessConfig{
		Command: command,
		Args:    args,
		Env:     env,
		WorkDir: agentNode.Path,
		LogFile: agentNode.Runtime.LogFile,
	}, nil
}

// resolveNodeEnvironment resolves a node's declared variables via the encrypted
// secret store, prompting for missing required ones.
func (as *DefaultAgentService) resolveNodeEnvironment(nodeName string, metadata *packages.PackageMetadata) (map[string]string, error) {
	env := metadata.UserEnvironment
	if len(env.Required) == 0 && len(env.Optional) == 0 {
		return map[string]string{}, nil
	}
	store, err := packages.NewSecretStore(as.agentfieldHome)
	if err != nil {
		return nil, fmt.Errorf("failed to open secret store: %w", err)
	}
	resolver := &packages.EnvResolver{Store: store, NodeName: nodeName, Prompter: packages.TTYPrompter{}}
	return resolver.Resolve(env)
}

// nodeReadyTimeout is how long to wait for a freshly started node to answer its
// health check. Import-heavy nodes (large dependency graphs) routinely take more
// than the old hardcoded 10s to boot, which produced spurious "did not become
// ready" failures on nodes that were actually starting fine. Default 30s,
// overridable via AGENTFIELD_NODE_READY_TIMEOUT (whole seconds).
func nodeReadyTimeout() time.Duration {
	if v := os.Getenv("AGENTFIELD_NODE_READY_TIMEOUT"); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return 30 * time.Second
}

// waitForAgentNode waits for the agent node to become ready
func (as *DefaultAgentService) waitForAgentNode(port int, healthPath string, timeout time.Duration) error {
	if healthPath == "" {
		healthPath = "/health"
	}
	client := &http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d%s", port, healthPath))
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

// updateRuntimeInfo updates the registry with runtime information
func (as *DefaultAgentService) updateRuntimeInfo(agentNodeName string, port, pid int) error {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")

	// Load registry
	registry := &packages.InstallationRegistry{}
	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return fmt.Errorf("failed to parse registry: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to read registry: %w", err)
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

// displayCapabilities fetches and displays agent node capabilities
func (as *DefaultAgentService) displayCapabilities(agentNode packages.InstalledPackage, port int) error {
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

// findAgentInRegistry finds an agent in the registry by name, handling name normalization
// Returns the agent package, actual name, and whether it was found
func (as *DefaultAgentService) findAgentInRegistry(registry *packages.InstallationRegistry, name string) (packages.InstalledPackage, string, bool) {
	// Try exact match first
	if agentNode, exists := registry.Installed[name]; exists {
		return agentNode, name, true
	}

	// Try with hyphens converted to no hyphens (deepresearchagent -> deep-research-agent)
	for registryName, agentNode := range registry.Installed {
		normalizedRegistryName := strings.ReplaceAll(registryName, "-", "")
		normalizedInputName := strings.ReplaceAll(name, "-", "")

		if normalizedRegistryName == normalizedInputName {
			return agentNode, registryName, true
		}
	}

	// Not found
	return packages.InstalledPackage{}, "", false
}

// findPythonExecutable tries to find a suitable Python executable
func (as *DefaultAgentService) findPythonExecutable() string {
	// Try common Python executable names in order of preference
	candidates := []string{"python3", "python", "python3.11", "python3.10", "python3.9", "python3.8"}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}

		// Also try to find in PATH
		if path, err := exec.LookPath(candidate); err == nil {
			return path
		}
	}

	return "" // Not found
}
