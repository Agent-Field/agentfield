package packages

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	AgentFieldHome string
	Port           int
	Detach         bool
}

// RunAgentNode starts an installed agent node, bringing up its declared node
// dependencies first.
func (ar *AgentNodeRunner) RunAgentNode(agentNodeName string) error {
	return ar.runAgentNode(agentNodeName, map[string]bool{})
}

// runAgentNode starts a node; inProgress tracks nodes already being started in
// this dependency chain to break cycles.
func (ar *AgentNodeRunner) runAgentNode(agentNodeName string, inProgress map[string]bool) error {
	fmt.Printf("🚀 Launching agent node: %s\n", agentNodeName)
	inProgress[agentNodeName] = true

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

	// 2b. Start declared node dependencies first (best-effort, in dep order).
	ar.startNodeDependencies(agentNode, inProgress)

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
	healthPath := "/health"
	expectedNodeID := agentNodeName
	if metadata, err := ParsePackageMetadata(agentNode.Path); err == nil {
		healthPath = metadata.HealthcheckPath()
		if metadata.AgentNode.NodeID != "" {
			expectedNodeID = metadata.AgentNode.NodeID
		}
	}
	if err := ar.waitForAgentNode(port, healthPath, expectedNodeID, 10*time.Second); err != nil {
		if killErr := cmd.Process.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
			fmt.Printf("⚠️  Failed to kill agent node process: %v\n", killErr)
		}
		return fmt.Errorf("agent node failed to start: %w", err)
	}

	fmt.Printf("🧠 Agent node registered with AgentField Server\n")

	// 6. Update registry with runtime info
	if err := ar.updateRuntimeInfo(agentNodeName, port, cmd.Process.Pid); err != nil {
		return fmt.Errorf("failed to update runtime info: %w", err)
	}

	// 7. Display agent node capabilities
	if err := ar.displayCapabilities(agentNode, port); err != nil {
		fmt.Printf("⚠️  Could not fetch capabilities: %v\n", err)
	}

	fmt.Printf("\n💡 Agent node running in background (PID: %d)\n", cmd.Process.Pid)
	fmt.Printf("💡 View logs: af logs %s\n", agentNodeName)
	fmt.Printf("💡 Stop agent node: af stop %s\n", agentNodeName)

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
	// A successful bind is not proof on Windows: without SO_EXCLUSIVEADDRUSE
	// a probe bind can succeed while another process is actively listening on
	// the same port (observed with uvicorn agent nodes). If something accepts
	// a connection, the port is taken.
	if dial, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 250*time.Millisecond); err == nil {
		dial.Close()
		return false
	}
	return true
}

// startAgentNodeProcess starts the agent node process
func (ar *AgentNodeRunner) startAgentNodeProcess(agentNode InstalledPackage, port int) (*exec.Cmd, error) {
	// Read the package manifest for the entrypoint and declared environment.
	// Fall back to defaults (python main.py, /health, no declared env) if a
	// manifest is missing so legacy installs still start.
	metadata, err := ParsePackageMetadata(agentNode.Path)
	if err != nil {
		fmt.Printf("⚠️  No usable manifest (%v); falling back to python main.py\n", err)
		metadata = &PackageMetadata{}
	}

	// Prepare environment variables. Export both AGENTFIELD_SERVER (what the
	// SDK reads) and the legacy AGENTFIELD_SERVER_URL for back-compat.
	serverURL := resolveServerURL()
	env := os.Environ()
	env = append(env, fmt.Sprintf("PORT=%d", port))
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER=%s", serverURL))
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER_URL=%s", serverURL))
	env = PythonUTF8Env(env)

	// Resolve declared variables from the encrypted secret store, prompting for
	// missing required ones and persisting them. Secrets are only ever injected
	// into this child process — never written to disk in plaintext.
	resolvedEnv, err := ar.resolveEnvironment(agentNode.Name, metadata)
	if err != nil {
		return nil, err
	}
	for key, value := range resolvedEnv {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	// Prepare command - resolve the launcher for the node's language.
	startArgs := metadata.StartCommand()
	program := startArgs[0]
	args := startArgs[1:]

	if metadata.IsGo() {
		// Go nodes launch a compiled binary (built at install time) or `go run`.
		// A package-relative binary path is resolved against the install dir so
		// exec finds it regardless of the parent's working directory.
		if resolved := GoBinaryProgram(agentNode.Path, program); resolved != program {
			program = resolved
			fmt.Printf("🐹 Launching Go binary: %s\n", program)
		}
	} else {
		venvPath := filepath.Join(agentNode.Path, "venv")
		if program == "python" || program == "python3" {
			if p := venvPython(venvPath); p != "" {
				program = p
				fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)
			} else {
				program = "python"
				fmt.Printf("⚠️  Virtual environment not found, using system Python\n")
			}
		}
	}

	cmd := exec.Command(program, args...)
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

// startNodeDependencies starts any installed, not-yet-running node dependencies
// of the given node before the node itself. `inProgress` guards against cycles.
func (ar *AgentNodeRunner) startNodeDependencies(node InstalledPackage, inProgress map[string]bool) {
	metadata, err := ParsePackageMetadata(node.Path)
	if err != nil {
		return
	}
	for _, ref := range metadata.Dependencies.Nodes {
		depName := NodeDepName(ref)
		if depName == "" || inProgress[depName] {
			continue
		}
		registry, err := ar.loadRegistry()
		if err != nil {
			return
		}
		dep, exists := registry.Installed[depName]
		if !exists {
			fmt.Printf("⚠️  Node dependency %s is declared but not installed (run: af install %s)\n", depName, ref)
			continue
		}
		if dep.Status == "running" {
			continue
		}
		fmt.Printf("🔗 Starting node dependency: %s\n", depName)
		depRunner := &AgentNodeRunner{AgentFieldHome: ar.AgentFieldHome}
		if err := depRunner.runAgentNode(depName, inProgress); err != nil {
			fmt.Printf("⚠️  Failed to start node dependency %s: %v\n", depName, err)
		}
	}
}

// PythonUTF8Env appends PYTHONUTF8=1 unless the environment already pins a
// value. Spawned agents log through a redirected stdout/stderr; on Windows
// Python then encodes with the legacy ANSI code page (e.g. cp1252), which
// cannot represent the SDK's emoji log prefixes and floods the log file with
// UnicodeEncodeError tracebacks. UTF-8 mode is a no-op where UTF-8 is already
// the default, so this is safe to set on every platform.
func PythonUTF8Env(env []string) []string {
	for _, kv := range env {
		if strings.HasPrefix(kv, "PYTHONUTF8=") {
			return env
		}
	}
	return append(env, "PYTHONUTF8=1")
}

// venvPython returns the venv python interpreter path, or "" if no venv exists.
func venvPython(venvPath string) string {
	if p := filepath.Join(venvPath, "bin", "python"); fileExists(p) {
		return p
	}
	if p := filepath.Join(venvPath, "Scripts", "python.exe"); fileExists(p) { // Windows
		return p
	}
	return ""
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// resolveEnvironment resolves the node's declared variables via the encrypted
// secret store, prompting for missing required ones.
func (ar *AgentNodeRunner) resolveEnvironment(nodeName string, metadata *PackageMetadata) (map[string]string, error) {
	env := metadata.UserEnvironment
	if len(env.Required) == 0 && len(env.Optional) == 0 && len(env.RequireOneOf) == 0 {
		return map[string]string{}, nil
	}
	store, err := NewSecretStore(ar.AgentFieldHome)
	if err != nil {
		return nil, fmt.Errorf("failed to open secret store: %w", err)
	}
	resolver := &EnvResolver{Store: store, NodeName: nodeName, Prompter: TTYPrompter{}}
	return resolver.Resolve(env)
}

// waitForAgentNode waits for the agent node to become ready. A 200 on the
// health endpoint is only trusted when the payload's node_id (if it carries
// one) matches the node just started — on Windows the port probe can miss an
// existing listener (no SO_EXCLUSIVEADDRUSE), and without this check a
// squatter's health response makes a dead agent look started. An empty
// expectedNodeID or a payload without node_id skips the identity check.
func (ar *AgentNodeRunner) waitForAgentNode(port int, healthPath, expectedNodeID string, timeout time.Duration) error {
	if healthPath == "" {
		healthPath = "/health"
	}
	client := &http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(timeout)

	impostor := ""
	for time.Now().Before(deadline) {
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d%s", port, healthPath))
		if err == nil && resp.StatusCode == 200 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
			resp.Body.Close()
			got := HealthNodeID(body)
			if got == "" || expectedNodeID == "" || NodeIDsEquivalent(got, expectedNodeID) {
				return nil
			}
			impostor = got
		} else if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}

	if impostor != "" {
		return fmt.Errorf("port %d is answering health checks as %q, not %q — another process is using the port", port, impostor, expectedNodeID)
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
	registryPath := filepath.Join(ar.AgentFieldHome, "installed.yaml")

	// Load registry
	registry := &InstallationRegistry{}
	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return fmt.Errorf("failed to parse registry: %w", err)
		}
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
	registryPath := filepath.Join(ar.AgentFieldHome, "installed.yaml")

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
