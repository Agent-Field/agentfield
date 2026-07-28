package cli

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var (
	stopJSON  bool
	stopForce bool
)

// NewStopCommand creates the stop command
func NewStopCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stop <agent-node-name>",
		Short: "Stop a running AgentField agent node",
		Long: `Stop a running AgentField agent node package.

The agent node process will be terminated gracefully and its status
will be updated in the registry.

Before stopping, the control plane is queried for executions still running on
the node. On a terminal you are asked to confirm; when not attached to a
terminal a warning is printed and the stop proceeds. Pass --force to skip the
check entirely.

Examples:
  af stop email-helper
  af stop data-analyzer
  af stop email-helper --json
  af stop email-helper --force`,
		Args: cobra.ExactArgs(1),
		RunE: runStopCommand,
	}

	cmd.Flags().BoolVar(&stopJSON, "json", false, "Emit a machine-readable JSON envelope instead of progress output")
	cmd.Flags().BoolVar(&stopForce, "force", false, "Stop immediately without warning about or confirming in-flight executions")
	return cmd
}

func runStopCommand(cmd *cobra.Command, args []string) error {
	agentNodeName := args[0]

	stopper := &AgentNodeStopper{
		AgentFieldHome: getAgentFieldHomeDir(),
		Quiet:          stopJSON,
		Force:          stopForce,
		ServerURL:      GetServerURL(),
		// A JSON invocation is a machine caller: never prompt. Otherwise honor
		// whether stdin is a real terminal.
		Interactive: !stopJSON && isInputTerminal(),
	}

	if err := stopper.StopAgentNode(agentNodeName); err != nil {
		if stopJSON {
			return nodeJSONError("stop_failed", err.Error(), "Run 'af list --json' to see installed nodes and their status.")
		}
		return fmt.Errorf("failed to stop agent node: %w", err)
	}

	if stopJSON {
		return nodeJSONSuccess(map[string]interface{}{
			"node":   agentNodeName,
			"status": stopper.Outcome,
		})
	}
	return nil
}

// AgentNodeStopper handles stopping agent nodes
type AgentNodeStopper struct {
	AgentFieldHome string
	// Quiet suppresses human progress output (used by --json mode).
	Quiet bool
	// Force skips the running-execution check entirely (no query, no warning,
	// no prompt).
	Force bool
	// ServerURL is the control plane queried for in-flight executions. Empty
	// disables the check.
	ServerURL string
	// Interactive is true when the caller may prompt on a real terminal. When
	// false, a running-execution warning is printed and the stop proceeds.
	Interactive bool
	// In is where an interactive confirmation is read from (defaults to stdin).
	In io.Reader
	// ErrOut receives execution warnings/prompts (defaults to stderr) so they
	// never pollute a --json stdout envelope.
	ErrOut io.Writer
	// Outcome records the result of the last StopAgentNode call:
	// "stopped", "not_running", or "aborted".
	Outcome string
}

// printf writes human progress output unless the stopper is in quiet mode.
func (as *AgentNodeStopper) printf(format string, args ...interface{}) {
	if as.Quiet {
		return
	}
	fmt.Printf(format, args...)
}

// warnf writes an execution warning/prompt. Unlike printf it is never gated by
// Quiet — a potential-data-loss warning must not be silently swallowed — and it
// goes to ErrOut (stderr) so --json stdout stays clean.
func (as *AgentNodeStopper) warnf(format string, args ...interface{}) {
	w := as.ErrOut
	if w == nil {
		w = os.Stderr
	}
	fmt.Fprintf(w, format, args...)
}

func (as *AgentNodeStopper) input() io.Reader {
	if as.In != nil {
		return as.In
	}
	return os.Stdin
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
		as.Outcome = "not_running"
		as.printf("⚠️  Agent node %s is not running\n", agentNodeName)
		return nil
	}

	if agentNode.Runtime.PID == nil {
		as.printf("⚠️  Agent node %s has no recorded PID — clearing stale registry entry\n", agentNodeName)
		return as.markStopped(registry, agentNodeName, agentNode)
	}

	as.printf("🛑 Stopping agent node: %s (PID: %d)\n", agentNodeName, *agentNode.Runtime.PID)

	// A "running" registry entry is a claim, not a fact: after a reboot or a
	// crash the PID is gone — or reassigned to an unrelated process — and the
	// port may be someone else's. Verify before signalling anything. A dead
	// process reconciles to "stopped" instead of erroring, so stop-then-start
	// flows (desktop restart, login autostart) recover on their own.
	process, err := os.FindProcess(*agentNode.Runtime.PID)
	if err != nil || !isProcessAlive(process) {
		as.printf("⚠️  Process %d is not running anymore — clearing stale registry entry\n", *agentNode.Runtime.PID)
		return as.markStopped(registry, agentNodeName, agentNode)
	}

	// If the recorded port answers /health as a DIFFERENT node, the record is
	// stale and the live PID is almost certainly a reused one belonging to
	// someone else — never signal a process we cannot identify as ours.
	if agentNode.Runtime.Port != nil {
		if id := probeHealthNodeID(*agentNode.Runtime.Port); id != "" && !packages.NodeIDsEquivalent(id, agentNodeName) {
			as.printf("⚠️  Port %d belongs to node %q, not %s — clearing stale registry entry without signalling PID %d\n",
				*agentNode.Runtime.Port, id, agentNodeName, *agentNode.Runtime.PID)
			return as.markStopped(registry, agentNodeName, agentNode)
		}
	}

	// The process is genuinely ours and running — a graceful stop from here on
	// will interrupt whatever it is doing. Warn about (or, on a terminal,
	// confirm) any in-flight executions before we signal it, so a long build or
	// analysis is not silently killed. Resolve the manifest node_id for the
	// query since the control plane keys executions by node_id, which may differ
	// from the registry install name.
	queryID := agentNodeName
	if md, err := packages.ParsePackageMetadata(agentNode.Path); err == nil && md.AgentNode.NodeID != "" {
		queryID = md.AgentNode.NodeID
	}
	if !as.confirmStopWithRunningExecutions(queryID) {
		as.Outcome = "aborted"
		as.printf("🚫 Stop aborted — %s left running to protect its in-flight executions\n", agentNodeName)
		return nil
	}

	// Try HTTP shutdown first if port is available
	httpShutdownSuccess := false
	if agentNode.Runtime.Port != nil {
		as.printf("🛑 Attempting graceful HTTP shutdown for agent %s on port %d\n", agentNodeName, *agentNode.Runtime.Port)

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
						as.printf("✅ HTTP shutdown request accepted for agent %s\n", agentNodeName)
						httpShutdownSuccess = true

						// Wait a moment for graceful shutdown
						time.Sleep(3 * time.Second)
					} else {
						as.printf("⚠️ HTTP shutdown returned status %d for agent %s\n", resp.StatusCode, agentNodeName)
					}
				} else {
					as.printf("⚠️ HTTP shutdown request failed for agent %s: %v\n", agentNodeName, err)
				}
			}
		}
	}

	// If HTTP shutdown failed or not available, fall back to process signals
	if !httpShutdownSuccess {
		as.printf("🔄 Falling back to process signal shutdown for agent %s\n", agentNodeName)

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
				as.printf("⚠️ Process still running, force killing agent %s\n", agentNodeName)
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

	as.Outcome = "stopped"
	as.printf("✅ Agent node %s stopped successfully\n", name)
	return nil
}

// runningExecution is the subset of an execution record the stop guard needs.
type runningExecution struct {
	ExecutionID string    `json:"execution_id"`
	AgentNodeID string    `json:"agent_node_id"`
	Status      string    `json:"status"`
	StartedAt   time.Time `json:"started_at"`
}

// executionsQueryResponse mirrors the /api/v1/agentic/query envelope for the
// executions resource.
type executionsQueryResponse struct {
	OK   bool `json:"ok"`
	Data struct {
		Results []runningExecution `json:"results"`
		Total   int                `json:"total"`
	} `json:"data"`
}

// queryRunningExecutions asks the control plane for executions still running on
// nodeID. An unreachable control plane or an unusable response returns an error;
// callers treat that as "proceed, best-effort".
func queryRunningExecutions(serverURL, nodeID string) ([]runningExecution, error) {
	serverURL = strings.TrimRight(strings.TrimSpace(serverURL), "/")
	if serverURL == "" {
		return nil, fmt.Errorf("no control plane URL configured")
	}

	payload := map[string]interface{}{
		"resource": "executions",
		"filters":  map[string]string{"status": "running", "agent_id": nodeID},
		"limit":    100,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	client := &http.Client{Timeout: 3 * time.Second}
	req, err := http.NewRequest(http.MethodPost, serverURL+"/api/v1/agentic/query", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := GetAPIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("control plane returned status %d", resp.StatusCode)
	}

	var parsed executionsQueryResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&parsed); err != nil {
		return nil, err
	}
	return parsed.Data.Results, nil
}

// confirmStopWithRunningExecutions decides whether to proceed with stopping a
// node given its in-flight executions. It returns true to proceed, false to
// abort (only reachable via an interactive "no"):
//
//   - --force                     → proceed silently (no query, no warning)
//   - control plane unreachable   → note it, proceed (best-effort, as before)
//   - no running executions       → proceed silently
//   - running executions, TTY     → list them and prompt; abort on anything but yes
//   - running executions, non-TTY → warn with count + ids + age, then proceed
func (as *AgentNodeStopper) confirmStopWithRunningExecutions(nodeID string) bool {
	if as.Force {
		return true
	}

	execs, err := queryRunningExecutions(as.ServerURL, nodeID)
	if err != nil {
		as.warnf("⚠️  Could not check the control plane for running executions on %s (%v) — proceeding with stop\n", nodeID, err)
		return true
	}
	if len(execs) == 0 {
		return true
	}

	as.warnf("⚠️  %d running execution(s) on %s will be interrupted by stopping it:\n", len(execs), nodeID)
	now := time.Now()
	for _, e := range execs {
		age := "unknown age"
		if !e.StartedAt.IsZero() {
			age = "age " + formatExecutionAge(now.Sub(e.StartedAt))
		}
		as.warnf("     • %s (%s)\n", e.ExecutionID, age)
	}

	if as.Interactive {
		as.warnf("Proceed and interrupt %d running execution(s)? [y/N]: ", len(execs))
		return readAffirmative(as.input())
	}

	as.warnf("     Proceeding (non-interactive). Pass --force to stop without this warning, or wait for the executions to finish.\n")
	return true
}

// formatExecutionAge renders an execution's age compactly (e.g. "45s", "12m30s",
// "1h15m").
func formatExecutionAge(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(time.Second)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
}

// readAffirmative reads one line and reports whether it is an affirmative
// ("y"/"yes", case-insensitive). Anything else — including EOF — is a "no".
func readAffirmative(r io.Reader) bool {
	line, _ := bufio.NewReader(r).ReadString('\n')
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	default:
		return false
	}
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
