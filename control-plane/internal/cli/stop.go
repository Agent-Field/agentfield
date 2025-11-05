package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"syscall"
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
		Run:  runStopCommand,
	}

	return cmd
}

func runStopCommand(cmd *cobra.Command, args []string) {
	agentNodeName := args[0]

	stopper := &AgentNodeStopper{
		AgentFieldHome: getAgentFieldHomeDir(),
	}

	if err := stopper.StopAgentNode(agentNodeName); err != nil {
		fmt.Printf("❌ Failed to stop agent node: %v\n", err)
		os.Exit(1)
	}
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
		return fmt.Errorf("no PID found for agent node %s", agentNodeName)
	}

	fmt.Printf("🛑 Stopping agent node: %s (PID: %d)\n", agentNodeName, *agentNode.Runtime.PID)

	// Find and kill the process
	process, err := os.FindProcess(*agentNode.Runtime.PID)
	if err != nil {
		return fmt.Errorf("failed to find process: %w", err)
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

		// Send SIGTERM for graceful shutdown
		if err := process.Signal(os.Interrupt); err != nil {
			// If graceful shutdown fails, force kill
			if err := process.Kill(); err != nil {
				return fmt.Errorf("failed to kill process: %w", err)
			}
		} else {
			// Wait for graceful shutdown, then check if still running
			time.Sleep(3 * time.Second)

			// Check if process is still running
			if err := process.Signal(syscall.Signal(0)); err == nil {
				// Process still running, force kill
				fmt.Printf("⚠️ Process still running, force killing agent %s\n", agentNodeName)
				if err := process.Kill(); err != nil {
					return fmt.Errorf("failed to force kill process: %w", err)
				}
			}
		}
	}

	// Update registry
	agentNode.Status = "stopped"
	agentNode.Runtime.Port = nil
	agentNode.Runtime.PID = nil
	agentNode.Runtime.StartedAt = nil
	registry.Installed[agentNodeName] = agentNode

	if err := as.saveRegistry(registry); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("✅ Agent node %s stopped successfully\n", agentNodeName)

	return nil
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
