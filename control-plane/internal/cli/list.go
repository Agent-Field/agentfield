package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// NewListCommand creates the list command
func NewListCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List installed AgentField agent node packages",
		Long: `Display all installed AgentField agent node packages with their status.

Shows package name, version, status (running/stopped), and port if running.

Examples:
  af list`,
		Run: runListCommand,
	}

	return cmd
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

	rows := make([][]string, 0, len(names))
	for _, name := range names {
		pkg := registry.Installed[name]
		port := "—"
		if pkg.Status == "running" && pkg.Runtime.Port != nil {
			port = fmt.Sprintf("%d", *pkg.Runtime.Port)
		}
		rows = append(rows, []string{
			name,
			"v" + pkg.Version,
			ui.StatusBadge(pkg.Status),
			port,
			pkg.Description,
		})
	}

	fmt.Println(ui.Table(
		fmt.Sprintf("Installed agent nodes (%d)", len(rows)),
		[]string{"NODE", "VERSION", "STATUS", "PORT", "DESCRIPTION"},
		rows,
	))
	fmt.Println()
	fmt.Println(ui.Muted("af run <name>  ·  af stop <name>  ·  af logs <name>"))
}
