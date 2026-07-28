package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var listJSON bool

// NewListCommand creates the list command
func NewListCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List installed AgentField agent node packages",
		Long: `Display all installed AgentField agent node packages with their status.

Shows package name, version, status (running/stopped), and port if running.

Examples:
  af list
  af list --json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if listJSON {
				return runListCommandJSON()
			}
			runListCommand(cmd, args)
			return nil
		},
	}

	cmd.Flags().BoolVar(&listJSON, "json", false, "Emit a machine-readable JSON envelope instead of the table")
	return cmd
}

// runListCommandJSON emits the installed-node registry as a JSON envelope.
func runListCommandJSON() error {
	agentfieldHome := getAgentFieldHomeDir()
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")

	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nodeJSONError("registry_error", fmt.Sprintf("failed to parse registry: %v", err), "Inspect "+registryPath+" for corruption.")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nodeJSONError("registry_error", fmt.Sprintf("failed to read registry: %v", err), "Check permissions on "+registryPath+".")
	}

	names := make([]string, 0, len(registry.Installed))
	for name := range registry.Installed {
		names = append(names, name)
	}
	sort.Strings(names)

	health := resolveNodeHealth(GetServerURL(), registry)

	nodes := make([]map[string]interface{}, 0, len(names))
	for _, name := range names {
		pkg := registry.Installed[name]
		h := health[name]
		node := map[string]interface{}{
			"name":               name,
			"version":            pkg.Version,
			"status":             pkg.Status,
			"description":        pkg.Description,
			"health":             h.Display,
			"health_discrepancy": h.Discrepancy,
		}
		if pkg.Status == "running" && pkg.Runtime.Port != nil {
			node["port"] = *pkg.Runtime.Port
		}
		nodes = append(nodes, node)
	}

	return nodeJSONSuccess(map[string]interface{}{
		"nodes": nodes,
		"total": len(nodes),
	})
}

// controlPlaneNode is the subset of the /api/v1/nodes payload the health
// reconciliation needs.
type controlPlaneNode struct {
	ID            string    `json:"id"`
	HealthStatus  string    `json:"health_status"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
}

// nodeHealth is the reconciled health of one installed node for display.
type nodeHealth struct {
	Display     string
	Discrepancy bool
}

// resolveNodeHealth fetches the control plane's node view and reconciles it with
// each installed node's local registry status, keyed by registry name. When the
// control plane is unreachable every node reports "unknown (control plane
// unreachable)" without erroring — `af list` must never fail just because the
// control plane is down.
func resolveNodeHealth(serverURL string, registry *packages.InstallationRegistry) map[string]nodeHealth {
	cpNodes, reachable := fetchControlPlaneNodes(serverURL)
	out := make(map[string]nodeHealth, len(registry.Installed))
	for name, pkg := range registry.Installed {
		// The control plane keys nodes by manifest node_id, which may differ
		// from the local install name — prefer it, fall back to the name.
		candidates := []string{name}
		if md, err := packages.ParsePackageMetadata(pkg.Path); err == nil && md.AgentNode.NodeID != "" {
			candidates = append([]string{md.AgentNode.NodeID}, candidates...)
		}
		cpNode, found := lookupControlPlaneNode(cpNodes, candidates...)
		display, discrepancy := reconcileHealth(pkg.Status, cpNode.HealthStatus, found, reachable)
		out[name] = nodeHealth{Display: display, Discrepancy: discrepancy}
	}
	return out
}

// reconcileHealth compares a node's local registry status with the control
// plane's reported health, returning the display string and whether they
// disagree. A node the registry calls "running" but the control plane reports
// as inactive/absent (or vice versa) is a discrepancy.
func reconcileHealth(registryStatus, cpHealth string, foundInCP, cpReachable bool) (string, bool) {
	if !cpReachable {
		return "unknown (control plane unreachable)", false
	}
	registryRunning := registryStatus == "running"
	if !foundInCP {
		if registryRunning {
			return "not on control plane (mismatch)", true
		}
		return "—", false
	}
	if cpHealth == "" {
		cpHealth = string(types.HealthStatusUnknown)
	}
	// Active and degraded both mean the node is registered and heartbeating.
	cpAlive := cpHealth == string(types.HealthStatusActive) || cpHealth == string(types.HealthStatusDegraded)
	if registryRunning != cpAlive {
		return cpHealth + " (mismatch)", true
	}
	return cpHealth, false
}

// lookupControlPlaneNode finds a control-plane node matching any of the given id
// candidates, first by exact id then by node-id equivalence (hyphen/underscore
// normalization).
func lookupControlPlaneNode(cpNodes map[string]controlPlaneNode, candidates ...string) (controlPlaneNode, bool) {
	for _, cand := range candidates {
		if cand == "" {
			continue
		}
		if n, ok := cpNodes[cand]; ok {
			return n, true
		}
	}
	for _, cand := range candidates {
		if cand == "" {
			continue
		}
		for id, n := range cpNodes {
			if packages.NodeIDsEquivalent(id, cand) {
				return n, true
			}
		}
	}
	return controlPlaneNode{}, false
}

// fetchControlPlaneNodes returns the control plane's view of every agent node
// (show_all=true includes inactive nodes so a dead-but-registered node is
// visible), keyed by node id. reachable is false when the control plane cannot
// be reached or answers unusably.
func fetchControlPlaneNodes(serverURL string) (map[string]controlPlaneNode, bool) {
	serverURL = strings.TrimRight(strings.TrimSpace(serverURL), "/")
	if serverURL == "" {
		return nil, false
	}

	client := &http.Client{Timeout: 3 * time.Second}
	req, err := http.NewRequest(http.MethodGet, serverURL+"/api/v1/nodes?show_all=true", nil)
	if err != nil {
		return nil, false
	}
	if key := GetAPIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}

	var parsed struct {
		Nodes []controlPlaneNode `json:"nodes"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&parsed); err != nil {
		return nil, false
	}

	nodes := make(map[string]controlPlaneNode, len(parsed.Nodes))
	for _, n := range parsed.Nodes {
		nodes[n.ID] = n
	}
	return nodes, true
}

func runListCommand(cmd *cobra.Command, args []string) {
	agentfieldHome := getAgentFieldHomeDir()
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")

	// Load registry
	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			cmd.PrintErrf("failed to parse registry: %v\n", err)
			return
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		cmd.PrintErrf("failed to read registry: %v\n", err)
		return
	}

	if len(registry.Installed) == 0 {
		fmt.Println(ui.Panel("No agent nodes installed",
			ui.Muted("Install one with:")+"\n  af install <path | git-url | af://registry/<name>>"))
		return
	}

	names := make([]string, 0, len(registry.Installed))
	for name := range registry.Installed {
		names = append(names, name)
	}
	sort.Strings(names)

	// Reconcile the local registry status against the control plane's health so
	// a "running" registry entry backed by a dead node (or vice versa) is
	// visible rather than silently trusted.
	health := resolveNodeHealth(GetServerURL(), registry)
	anyDiscrepancy := false

	rows := make([][]string, 0, len(names))
	for _, name := range names {
		pkg := registry.Installed[name]
		port := "—"
		if pkg.Status == "running" && pkg.Runtime.Port != nil {
			port = fmt.Sprintf("%d", *pkg.Runtime.Port)
		}
		h := health[name]
		if h.Discrepancy {
			anyDiscrepancy = true
		}
		rows = append(rows, []string{
			name,
			"v" + pkg.Version,
			ui.StatusBadge(pkg.Status),
			port,
			h.Display,
			pkg.Description,
		})
	}

	fmt.Println(ui.Table(
		fmt.Sprintf("Installed agent nodes (%d)", len(rows)),
		[]string{"NODE", "VERSION", "STATUS", "PORT", "HEALTH", "DESCRIPTION"},
		rows,
	))
	fmt.Println()
	if anyDiscrepancy {
		fmt.Println(ui.Muted("STATUS is the local registry; HEALTH is the control plane. (mismatch) marks a disagreement — reconcile with `af stop`/`af run`."))
	}
	fmt.Println(ui.Muted("af run <name>  ·  af stop <name>  ·  af logs <name>"))
}
