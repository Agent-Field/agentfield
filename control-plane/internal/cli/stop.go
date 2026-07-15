package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// NewStopCommand creates the stop command
func NewStopCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stop <agent-node-name>",
		Short: "Stop a running AgentField agent node",
		Long: `Stop a running AgentField agent node package.

The agent node process will be terminated gracefully and its status
will be updated in the registry.

Examples:
  af stop email-helper
  af stop data-analyzer`,
		Args: cobra.ExactArgs(1),
		RunE: runStopCommand,
	}

	return cmd
}

func runStopCommand(cmd *cobra.Command, args []string) error {
	agentNodeName := args[0]

	stopper := &AgentNodeStopper{
		AgentFieldHome: getAgentFieldHomeDir(),
	}

	if err := stopper.StopAgentNode(agentNodeName); err != nil {
		return fmt.Errorf("failed to stop agent node: %w", err)
	}

	return nil
}

// AgentNodeStopper handles stopping agent nodes
type AgentNodeStopper struct {
	AgentFieldHome string
}

// StopAgentNode stops a running agent node
func (as *AgentNodeStopper) StopAgentNode(agentNodeName string) error {
	// Load registry
	registry, err := as.loadRegistry()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	agentNode, exists := registry.Installed[agentNodeName]
	if !exists {
		return fmt.Errorf("agent node %s not installed", agentNodeName)
	}

	if agentNode.Status != "running" {
		fmt.Printf("⚠️  Agent node %s is not running\n", agentNodeName)
		return nil
	}

	if agentNode.Runtime.PID == nil {
		fmt.Printf("⚠️  Agent node %s has no recorded PID — clearing stale registry entry\n", agentNodeName)
		return as.markStopped(registry, agentNodeName, agentNode)
	}

	fmt.Printf("🛑 Stopping agent node: %s (PID: %d)\n", agentNodeName, *agentNode.Runtime.PID)

	// A "running" registry entry is a claim, not a fact: after a reboot or a
	// crash the PID is gone — or reassigned to an unrelated process — and the
	// port may be someone else's. Verify before signalling anything. A dead
	// process reconciles to "stopped" instead of erroring, so stop-then-start
	// flows (desktop restart, login autostart) recover on their own.
	process, err := os.FindProcess(*agentNode.Runtime.PID)
	if err != nil || !isProcessAlive(process) {
		fmt.Printf("⚠️  Process %d is not running anymore — clearing stale registry entry\n", *agentNode.Runtime.PID)
		return as.markStopped(registry, agentNodeName, agentNode)
	}

	// If the recorded port answers /health as a DIFFERENT node, the record is
	// stale and the live PID is almost certainly a reused one belonging to
	// someone else — never signal a process we cannot identify as ours.
	if agentNode.Runtime.Port != nil {
		if id := probeHealthNodeID(*agentNode.Runtime.Port); id != "" && !packages.NodeIDsEquivalent(id, agentNodeName) {
			fmt.Printf("⚠️  Port %d belongs to node %q, not %s — clearing stale registry entry without signalling PID %d\n",
				*agentNode.Runtime.Port, id, agentNodeName, *agentNode.Runtime.PID)
			return as.markStopped(registry, agentNodeName, agentNode)
		}
	}

	// Try HTTP shutdown first if port is available
	httpShutdownSuccess := false
	if agentNode.Runtime.Port != nil {
		fmt.Printf("🛑 Attempting graceful HTTP shutdown for agent %s on port %d\n", agentNodeName, *agentNode.Runtime.Port)

		// Construct agent base URL
		baseURL := fmt.Sprintf("http://localhost:%d", *agentNode.Runtime.Port)
		shutdownURL := fmt.Sprintf("%s/shutdown", baseURL)

		// Create shutdown request
		requestBody := map[string]interface{}{
			"graceful":        true,
			"timeout_seconds": 30,
		}

		bodyBytes, err := json.Marshal(requestBody)
		if err == nil {
			req, err := http.NewRequest("POST", shutdownURL, bytes.NewReader(bodyBytes))
			if err == nil {
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("User-Agent", "AgentField-CLI/1.0")

				client := &http.Client{Timeout: 10 * time.Second}
				resp, err := client.Do(req)
				if err == nil {
					defer resp.Body.Close()
					if resp.StatusCode == 200 {
						fmt.Printf("✅ HTTP shutdown request accepted for agent %s\n", agentNodeName)
						httpShutdownSuccess = true

						// Wait a moment for graceful shutdown
						time.Sleep(3 * time.Second)
					} else {
						fmt.Printf("⚠️ HTTP shutdown returned status %d for agent %s\n", resp.StatusCode, agentNodeName)
					}
				} else {
					fmt.Printf("⚠️ HTTP shutdown request failed for agent %s: %v\n", agentNodeName, err)
				}
			}
		}
	}

	// If HTTP shutdown failed or not available, fall back to process signals
	if !httpShutdownSuccess {
		fmt.Printf("🔄 Falling back to process signal shutdown for agent %s\n", agentNodeName)

		// Ask for graceful shutdown (SIGINT on Unix, taskkill on Windows).
		// A process that exits between the aliveness check above and the
		// signal is a success, not a failure.
		if err := signalGracefulStop(process); err != nil {
			// If graceful shutdown fails, force kill
			if err := process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				return fmt.Errorf("failed to kill process: %w", err)
			}
		} else {
			// Wait for graceful shutdown, then check if still running
			time.Sleep(3 * time.Second)

			// Check if process is still running
			if isProcessAlive(process) {
				// Process still running, force kill
				fmt.Printf("⚠️ Process still running, force killing agent %s\n", agentNodeName)
				if err := process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
					return fmt.Errorf("failed to force kill process: %w", err)
				}
			}
		}
	}

	return as.markStopped(registry, agentNodeName, agentNode)
}

// markStopped clears the runtime fields and persists the registry — shared by
// the healthy stop path and every stale-record reconciliation, so `af stop`
// always leaves the registry in a state `af run` can start from.
func (as *AgentNodeStopper) markStopped(registry *packages.InstallationRegistry, name string, node packages.InstalledPackage) error {
	node.Status = "stopped"
	node.Runtime.Port = nil
	node.Runtime.PID = nil
	node.Runtime.StartedAt = nil
	registry.Installed[name] = node

	if err := as.saveRegistry(registry); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("✅ Agent node %s stopped successfully\n", name)
	return nil
}

// probeHealthNodeID asks /health on a local port and returns the node_id the
// responder claims — "" when nothing answers or the payload carries none
// (custom health endpoints are not required to identify themselves).
func probeHealthNodeID(port int) string {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", port))
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return ""
	}
	return packages.HealthNodeID(body)
}

// loadRegistry loads the installation registry
func (as *AgentNodeStopper) loadRegistry() (*packages.InstallationRegistry, error) {
	registryPath := filepath.Join(as.AgentFieldHome, "installed.yaml")

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

// saveRegistry saves the installation registry
func (as *AgentNodeStopper) saveRegistry(registry *packages.InstallationRegistry) error {
	registryPath := filepath.Join(as.AgentFieldHome, "installed.yaml")

	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	return os.WriteFile(registryPath, data, 0644)
}
