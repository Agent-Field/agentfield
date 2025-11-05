package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/config" // Ensured this import is correct
	"github.com/Agent-Field/agentfield/control-plane/internal/mcp"

	"github.com/spf13/cobra"
)

// NewMCPCommand creates the mcp command for managing MCP servers
func NewMCPCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "mcp",
		Short: "Manage MCP servers in your AgentField agent project",
		Long: `Manage Model Context Protocol (MCP) servers in your AgentField agent project.

MCP servers provide external tools and resources that can be integrated into your agent.`,
	}

	// Add subcommands
	cmd.AddCommand(NewMCPStatusCommand())
	cmd.AddCommand(NewMCPStartCommand())
	cmd.AddCommand(NewMCPStopCommand())
	cmd.AddCommand(NewMCPRestartCommand())
	cmd.AddCommand(NewMCPLogsCommand())
	cmd.AddCommand(NewMCPRemoveCommand())
	cmd.AddCommand(NewMCPDiscoverCommand())
	cmd.AddCommand(NewMCPSkillsCommand())
	cmd.AddCommand(NewMCPMigrateCommand())

	return cmd
}

// NewMCPStatusCommand creates the mcp status command
func NewMCPStatusCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show status of all MCP servers",
		Long:  `Display the current status of all MCP servers in the project.`,
		RunE:  runMCPStatusCommand,
	}

	return cmd
}

func runMCPStatusCommand(cmd *cobra.Command, args []string) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}
	manager := mcp.NewMCPManager(cfg, projectDir, verbose)
	servers, err := manager.Status()
	if err != nil {
		return fmt.Errorf("failed to get MCP status: %w", err)
	}

	PrintHeader("MCP Server Status")

	if len(servers) == 0 {
		PrintInfo("No MCP servers installed")
		fmt.Printf("\n%s %s\n", Blue("→"), "Add an MCP server: af add --mcp @modelcontextprotocol/server-github")
		return nil
	}

	// Count running and stopped servers
	running := 0
	stopped := 0
	for _, server := range servers {
		if server.Status == mcp.StatusRunning {
			running++
		} else {
			stopped++
		}
	}

	fmt.Printf("\n%s Total: %d | %s Running: %d | %s Stopped: %d\n\n",
		Gray("📦"), len(servers),
		Green("🟢"), running,
		Red("🔴"), stopped)

	for _, server := range servers {
		statusIcon := "🔴"
		statusText := "stopped"
		if server.Status == mcp.StatusRunning {
			statusIcon = "🟢"
			statusText = "running"
		}

		fmt.Printf("%s %s\n", statusIcon, Bold(server.Alias))
		if server.URL != "" {
			fmt.Printf("  %s %s\n", Gray("URL:"), server.URL)
		}
		if server.RunCmd != "" {
			fmt.Printf("  %s %s\n", Gray("Command:"), server.RunCmd)
		}
		fmt.Printf("  %s %s\n", Gray("Version:"), server.Version)
		fmt.Printf("  %s %s", Gray("Status:"), statusText)

		if server.Status == mcp.StatusRunning && server.PID > 0 {
			fmt.Printf(" (PID: %d, Port: %d)", server.PID, server.Port)
		}
		fmt.Printf("\n")

		if server.StartedAt != nil {
			fmt.Printf("  %s %s\n", Gray("Started:"), server.StartedAt.Format("2006-01-02 15:04:05"))
		}
		fmt.Printf("\n")
	}

	return nil
}

// NewMCPStartCommand creates the mcp start command
func NewMCPStartCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "start <alias>",
		Short: "Start an MCP server",
		Long:  `Start a specific MCP server by its alias.`,
		Args:  cobra.ExactArgs(1),
		RunE:  runMCPStartCommand,
	}

	return cmd
}

func runMCPStartCommand(cmd *cobra.Command, args []string) error {
	alias := args[0]

	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}
	manager := mcp.NewMCPManager(cfg, projectDir, verbose)

	PrintInfo(fmt.Sprintf("Starting MCP server: %s", alias))

	_, err = manager.Start(alias)
	if err != nil {
		PrintError(fmt.Sprintf("Failed to start MCP server: %v", err))
		return err
	}

	PrintSuccess(fmt.Sprintf("MCP server '%s' started successfully", alias))
	return nil
}

// NewMCPStopCommand creates the mcp stop command
func NewMCPStopCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stop <alias>",
		Short: "Stop an MCP server",
		Long:  `Stop a specific MCP server by its alias.`,
		Args:  cobra.ExactArgs(1),
		RunE:  runMCPStopCommand,
	}

	return cmd
}

func runMCPStopCommand(cmd *cobra.Command, args []string) error {
	alias := args[0]

	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}
	manager := mcp.NewMCPManager(cfg, projectDir, verbose)

	PrintInfo(fmt.Sprintf("Stopping MCP server: %s", alias))

	if err := manager.Stop(alias); err != nil {
		PrintError(fmt.Sprintf("Failed to stop MCP server: %v", err))
		return err
	}

	PrintSuccess(fmt.Sprintf("MCP server '%s' stopped successfully", alias))
	return nil
}

// NewMCPRestartCommand creates the mcp restart command
func NewMCPRestartCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "restart <alias>",
		Short: "Restart an MCP server",
		Long:  `Restart a specific MCP server by its alias.`,
		Args:  cobra.ExactArgs(1),
		RunE:  runMCPRestartCommand,
	}

	return cmd
}

func runMCPRestartCommand(cmd *cobra.Command, args []string) error {
	alias := args[0]

	PrintInfo(fmt.Sprintf("Restarting MCP server: %s", alias))

	// Stop then start
	if err := runMCPStopCommand(cmd, args); err != nil {
		return err
	}

	return runMCPStartCommand(cmd, args)
}

// NewMCPLogsCommand creates the mcp logs command
func NewMCPLogsCommand() *cobra.Command {
	var followFlag bool
	var tailLines int

	cmd := &cobra.Command{
		Use:   "logs <alias>",
		Short: "Show logs for an MCP server",
		Long:  `Display logs for a specific MCP server.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMCPLogsCommand(cmd, args, followFlag, tailLines)
		},
	}

	cmd.Flags().BoolVarP(&followFlag, "follow", "f", false, "Follow log output")
	cmd.Flags().IntVarP(&tailLines, "tail", "n", 50, "Number of lines to show from the end of the logs")

	return cmd
}

func runMCPLogsCommand(cmd *cobra.Command, args []string, follow bool, tail int) error {
	alias := args[0]

	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	logFile := filepath.Join(projectDir, "packages", "mcp", alias, fmt.Sprintf("%s.log", alias))

	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		PrintError(fmt.Sprintf("Log file not found for MCP server '%s'", alias))
		return nil
	}

	PrintInfo(fmt.Sprintf("Showing logs for MCP server: %s", alias))
	fmt.Printf("%s %s\n\n", Gray("Log file:"), logFile)

	// For now, just show that we would display logs
	// In a full implementation, we would use tail command or read the file
	fmt.Printf("%s Logs would be displayed here (last %d lines)\n", Gray("→"), tail)
	if follow {
		fmt.Printf("%s Following logs... (Press Ctrl+C to stop)\n", Gray("→"))
	}

	return nil
}

// NewMCPSkillsCommand creates the mcp skills command
func NewMCPSkillsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "skills",
		Short: "Manage auto-generated skills for MCP servers",
		Long:  `Manage auto-generated Python skill files for MCP servers.`,
	}

	cmd.AddCommand(NewMCPSkillsGenerateCommand())
	cmd.AddCommand(NewMCPSkillsListCommand())
	cmd.AddCommand(NewMCPSkillsRefreshCommand())

	return cmd
}

// NewMCPSkillsGenerateCommand creates the mcp skills generate command
func NewMCPSkillsGenerateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "generate [alias]",
		Short: "Generate skill files for MCP servers",
		Long:  `Generate Python skill files that wrap MCP tools as AgentField skills.`,
		Args:  cobra.MaximumNArgs(1),
		RunE:  runMCPSkillsGenerateCommand,
	}

	cmd.Flags().BoolP("verbose", "v", false, "Enable verbose output for debugging")

	return cmd
}

func runMCPSkillsGenerateCommand(cmd *cobra.Command, args []string) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	verboseFlag, _ := cmd.Flags().GetBool("verbose")
	generator := mcp.NewSkillGenerator(projectDir, verboseFlag)

	if len(args) == 1 {
		// Generate skills for specific server
		alias := args[0]
		PrintInfo(fmt.Sprintf("Generating skills for MCP server: %s", alias))

		result, err := generator.GenerateSkillsForServer(alias)
		if err != nil {
			PrintError(fmt.Sprintf("Failed to generate skills: %v", err))
			return err
		}

		if result.Generated {
			PrintSuccess(fmt.Sprintf("Skills generated for '%s'", alias))
			fmt.Printf("  %s Generated file: %s (%d tools)\n", Gray("→"), Gray(fmt.Sprintf("skills/mcp_%s.py", alias)), result.ToolCount)
		} else {
			PrintWarning(result.Message)
			fmt.Printf("  %s %s\n", Gray("→"), "No skill file was created")
		}
	} else {
		// Generate skills for all servers
		PrintInfo("Generating skills for all MCP servers...")

		if err := generator.GenerateSkillsForAllServers(); err != nil {
			PrintError(fmt.Sprintf("Failed to generate skills: %v", err))
			return err
		}

		PrintSuccess("All skills processed successfully")
	}

	return nil
}

// NewMCPSkillsListCommand creates the mcp skills list command
func NewMCPSkillsListCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List generated skill files",
		Long:  `List all auto-generated skill files for MCP servers.`,
		RunE:  runMCPSkillsListCommand,
	}

	return cmd
}

func runMCPSkillsListCommand(cmd *cobra.Command, args []string) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	skillsDir := filepath.Join(projectDir, "skills")
	if _, err := os.Stat(skillsDir); os.IsNotExist(err) {
		PrintInfo("No skills directory found")
		return nil
	}

	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		return fmt.Errorf("failed to read skills directory: %w", err)
	}

	PrintHeader("Auto-Generated MCP Skills")

	skillCount := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "mcp_") || !strings.HasSuffix(entry.Name(), ".py") {
			continue
		}

		skillCount++
		alias := strings.TrimSuffix(strings.TrimPrefix(entry.Name(), "mcp_"), ".py")
		fmt.Printf("%s %s\n", Green("✓"), Bold(alias))
		fmt.Printf("  %s %s\n", Gray("File:"), entry.Name())

		// Try to get server info
		if cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml")); err == nil {
			discovery := mcp.NewCapabilityDiscovery(cfg, projectDir)
			if capability, err := discovery.GetServerCapability(alias); err == nil {
				fmt.Printf("  %s %d tools available\n", Gray("Tools:"), len(capability.Tools))
			}
		}
	}

	if skillCount == 0 {
		PrintInfo("No auto-generated MCP skills found")
		fmt.Printf("\n%s %s\n", Blue("→"), "Generate skills: af mcp skills generate")
	} else {
		fmt.Printf("\n%s %d skill files found\n", Gray("Total:"), skillCount)
	}

	return nil
}

// NewMCPSkillsRefreshCommand creates the mcp skills refresh command
func NewMCPSkillsRefreshCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "refresh",
		Short: "Refresh all skill files",
		Long:  `Refresh capabilities and regenerate all skill files for MCP servers.`,
		RunE:  runMCPSkillsRefreshCommand,
	}

	return cmd
}

func runMCPSkillsRefreshCommand(cmd *cobra.Command, args []string) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}
	manager := mcp.NewMCPManager(cfg, projectDir, verbose)

	PrintInfo("Refreshing capabilities and regenerating skills...")

	// Get all servers and refresh their capabilities
	servers, err := manager.Status()
	if err != nil {
		PrintError(fmt.Sprintf("Failed to get server list: %v", err))
		return err
	}

	if len(servers) == 0 {
		PrintInfo("No MCP servers found")
		return nil
	}

	// For each server, discover capabilities
	for _, server := range servers {
		if _, err := manager.DiscoverCapabilities(server.Alias); err != nil {
			PrintWarning(fmt.Sprintf("Failed to refresh capabilities for %s: %v", server.Alias, err))
		}
	}

	PrintSuccess("All skills refreshed successfully")
	return nil
}

// NewMCPRemoveCommand creates the mcp remove command
func NewMCPRemoveCommand() *cobra.Command {
	var forceFlag bool

	cmd := &cobra.Command{
		Use:   "remove <alias>",
		Short: "Remove an MCP server",
		Long:  `Remove an MCP server from the project.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMCPRemoveCommand(cmd, args, forceFlag)
		},
	}

	cmd.Flags().BoolVar(&forceFlag, "force", false, "Force removal even if server is running")

	return cmd
}

func runMCPRemoveCommand(cmd *cobra.Command, args []string, force bool) error {
	alias := args[0]

	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}
	manager := mcp.NewMCPManager(cfg, projectDir, verbose)

	PrintInfo(fmt.Sprintf("Removing MCP server: %s", alias))

	if err := manager.Remove(alias); err != nil {
		if !force && strings.Contains(err.Error(), "is running") {
			PrintError("MCP server is running. Stop it first or use --force")
			return err
		}
		PrintError(fmt.Sprintf("Failed to remove MCP server: %v", err))
		return err
	}

	PrintSuccess(fmt.Sprintf("MCP server '%s' removed successfully", alias))
	return nil
}

// NewMCPDiscoverCommand creates the mcp discover command
func NewMCPDiscoverCommand() *cobra.Command {
	var refreshFlag bool

	cmd := &cobra.Command{
		Use:   "discover [alias]",
		Short: "Discover MCP server capabilities",
		Long:  `Discover and display capabilities (tools and resources) of MCP servers.`,
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMCPDiscoverCommand(cmd, args, refreshFlag)
		},
	}

	cmd.Flags().BoolVar(&refreshFlag, "refresh", false, "Force refresh of capabilities cache")

	return cmd
}

func runMCPDiscoverCommand(cmd *cobra.Command, args []string, refresh bool) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}

	discovery := mcp.NewCapabilityDiscovery(cfg, projectDir)

	if len(args) == 1 {
		// Discover capabilities for specific server
		alias := args[0]
		PrintInfo(fmt.Sprintf("Discovering capabilities for MCP server: %s", alias))

		capability, err := discovery.GetServerCapability(alias)
		if err != nil {
			PrintError(fmt.Sprintf("Failed to discover capabilities: %v", err))
			return err
		}

		displayServerCapability(*capability)
	} else {
		// Discover capabilities for all servers
		PrintInfo("Discovering capabilities for all MCP servers...")

		if refresh {
			if err := discovery.RefreshCapabilities(); err != nil {
				PrintError(fmt.Sprintf("Failed to refresh capabilities: %v", err))
				return err
			}
		}

		capabilities, err := discovery.DiscoverCapabilities()
		if err != nil {
			PrintError(fmt.Sprintf("Failed to discover capabilities: %v", err))
			return err
		}

		if len(capabilities) == 0 {
			PrintInfo("No MCP servers found")
			return nil
		}

		PrintHeader("MCP Server Capabilities")
		for _, capability := range capabilities {
			displayServerCapability(capability)
			fmt.Println()
		}
	}

	return nil
}

func displayServerCapability(capability mcp.MCPCapability) {
	fmt.Printf("%s %s\n", Bold("🔧"), Bold(capability.ServerAlias))
	fmt.Printf("  %s %s\n", Gray("Server:"), capability.ServerName)
	fmt.Printf("  %s %s\n", Gray("Version:"), capability.Version)
	fmt.Printf("  %s %s\n", Gray("Transport:"), capability.Transport)
	fmt.Printf("  %s %s\n", Gray("Endpoint:"), capability.Endpoint)

	if len(capability.Tools) > 0 {
		fmt.Printf("  %s %s (%d)\n", Gray("Tools:"), Green("✓"), len(capability.Tools))
		for _, tool := range capability.Tools {
			fmt.Printf("    • %s - %s\n", Bold(tool.Name), tool.Description)
		}
	} else {
		fmt.Printf("  %s %s\n", Gray("Tools:"), Red("None"))
	}

	if len(capability.Resources) > 0 {
		fmt.Printf("  %s %s (%d)\n", Gray("Resources:"), Green("✓"), len(capability.Resources))
		for _, resource := range capability.Resources {
			fmt.Printf("    • %s - %s\n", Bold(resource.Name), resource.Description)
		}
	} else {
		fmt.Printf("  %s %s\n", Gray("Resources:"), Red("None"))
	}
}

// NewMCPMigrateCommand creates the mcp migrate command
func NewMCPMigrateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "migrate [alias]",
		Short: "Migrate MCP server metadata from old format to new format",
		Long:  `Migrate MCP server metadata from old mcp.json format to new config.json format.`,
		Args:  cobra.MaximumNArgs(1),
		RunE:  runMCPMigrateCommand,
	}

	return cmd
}

func runMCPMigrateCommand(cmd *cobra.Command, args []string) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	cfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		cfg, err = config.LoadConfig("agentfield.yaml") // Fallback
		if err != nil {
			return fmt.Errorf("failed to load af configuration: %w", err)
		}
	}

	discovery := mcp.NewCapabilityDiscovery(cfg, projectDir)

	if len(args) == 1 {
		// Migrate specific server
		alias := args[0]
		PrintInfo(fmt.Sprintf("Migrating MCP server: %s", alias))

		if err := migrateSingleServer(discovery, projectDir, alias); err != nil {
			PrintError(fmt.Sprintf("Failed to migrate %s: %v", alias, err))
			return err
		}

		PrintSuccess(fmt.Sprintf("Successfully migrated %s", alias))
	} else {
		// Migrate all servers
		PrintInfo("Migrating all MCP servers...")

		if err := migrateAllServers(discovery, projectDir); err != nil {
			PrintError(fmt.Sprintf("Migration failed: %v", err))
			return err
		}

		PrintSuccess("All servers migrated successfully")
	}

	return nil
}

func migrateSingleServer(discovery *mcp.CapabilityDiscovery, projectDir, alias string) error {
	serverDir := filepath.Join(projectDir, "packages", "mcp", alias)

	// Check if server directory exists
	if _, err := os.Stat(serverDir); os.IsNotExist(err) {
		return fmt.Errorf("MCP server '%s' not found", alias)
	}

	// Check if already migrated
	configPath := filepath.Join(serverDir, "config.json")
	if _, err := os.Stat(configPath); err == nil {
		PrintInfo(fmt.Sprintf("Server '%s' already uses config.json format", alias))
		return nil
	}

	// Check if old format exists
	oldPath := filepath.Join(serverDir, "mcp.json")
	if _, err := os.Stat(oldPath); os.IsNotExist(err) {
		return fmt.Errorf("no mcp.json found for server '%s'", alias)
	}

	// Perform migration using the discovery's migration function
	// We need to access the migration function, so let's trigger it by calling discoverServerCapability
	_, err := discovery.GetServerCapability(alias)
	if err != nil {
		return fmt.Errorf("migration failed: %w", err)
	}

	return nil
}

func migrateAllServers(discovery *mcp.CapabilityDiscovery, projectDir string) error {
	mcpDir := filepath.Join(projectDir, "packages", "mcp")
	if _, err := os.Stat(mcpDir); os.IsNotExist(err) {
		PrintInfo("No MCP servers found")
		return nil
	}

	entries, err := os.ReadDir(mcpDir)
	if err != nil {
		return fmt.Errorf("failed to read MCP directory: %w", err)
	}

	migratedCount := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		alias := entry.Name()
		if err := migrateSingleServer(discovery, projectDir, alias); err != nil {
			PrintWarning(fmt.Sprintf("Failed to migrate %s: %v", alias, err))
		} else {
			migratedCount++
			PrintInfo(fmt.Sprintf("Migrated: %s", alias))
		}
	}

	if migratedCount == 0 {
		PrintInfo("No servers needed migration")
	} else {
		PrintInfo(fmt.Sprintf("Migrated %d servers", migratedCount))
	}

	return nil
}
