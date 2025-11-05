package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/config"
	"github.com/your-org/agentfield/control-plane/internal/mcp"

	"github.com/spf13/cobra"
)

// MCPAddOptions holds all the flag-based options for the 'add --mcp' command.
type MCPAddOptions struct {
	Source      string   // Positional argument
	Alias       string   // Positional argument or --alias flag
	MCP         bool     // --mcp flag
	Force       bool     // --force flag
	URL         string   // --url flag (for remote MCP servers)
	RunCmd      string   // --run flag (command to run the server)
	SetupCmds   []string // --setup flags (setup commands, repeatable)
	WorkingDir  string   // --working-dir flag
	EnvVars     []string // --env flags (raw "KEY=VALUE")
	Description string   // --description flag
	Tags        []string // --tags flags (repeatable)
	HealthCheck string   // --health-check flag
	Timeout     int      // --timeout flag (in seconds)
	Version     string   // --version flag
}

// Assuming color funcs (Bold, Green, Red, Yellow, Gray, Cyan) and
// status consts (StatusInfo, StatusWarning, StatusError, StatusSuccess)
// are available from the package cli (e.g. defined in utils.go or root.go)
// or imported if they are from a different package.
// Removed local definitions to avoid redeclaration.

// NewAddCommand creates the add command for adding dependencies
func NewAddCommand() *cobra.Command {
	opts := &MCPAddOptions{}

	cmd := &cobra.Command{
		Use:   "add <source> [alias]",
		Short: "Add dependencies to your AgentField agent project",
		Long: `Add dependencies to your AgentField agent project.

Supports adding MCP servers and regular agent packages with advanced configuration options.

Examples:
  # Remote MCP servers (URL-based)
  af add --mcp --url https://github.com/modelcontextprotocol/server-github
  af add --mcp --url https://github.com/ferrislucas/iterm-mcp github-tools

  # Local MCP servers with custom commands
  af add --mcp my-server --run "node server.js --port {{port}}" \
    --setup "npm install" --setup "npm run build"
  
  # Python MCP server with environment variables
  af add --mcp python-server --run "python server.py --port {{port}}" \
    --setup "pip install -r requirements.txt" \
    --env "PYTHONPATH={{server_dir}}" \
    --working-dir "./src"

  # Advanced configuration with health checks
  af add --mcp enterprise-server \
    --url https://github.com/company/mcp-server \
    --run "node dist/server.js --port {{port}} --config {{config_file}}" \
    --setup "npm install" --setup "npm run build" \
    --env "NODE_ENV=production" --env "LOG_LEVEL=info" \
    --health-check "curl -f http://localhost:{{port}}/health" \
    --timeout 60 --description "Enterprise MCP server" \
    --tags "enterprise" --tags "production"

  # Regular agent packages (future)
  af add github.com/agentfield-helpers/email-utils
  af add github.com/openai/prompt-templates

Template Variables:
  {{port}}        - Dynamically assigned port number
  {{config_file}} - Path to server configuration file
  {{data_dir}}    - Server data directory path (Note: mcp-integration2.md uses {{server_dir}})
  {{log_file}}    - Server log file path
  {{server_dir}}  - Server installation directory
  {{alias}}       - Server alias name`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.Source = args[0]
			// If --alias flag is not used, and a second positional arg is present, use it as alias.
			if cmd.Flags().Changed("alias") == false && len(args) > 1 {
				opts.Alias = args[1]
			}
			// verbose flag is typically a persistent flag from the root command.
			// Assuming it's accessible globally as 'verbose' (lowercase) from the cli package (e.g. cli.verbose)
			// or passed via context if that's the pattern. For now, using the package global 'verbose'.
			return runAddCommandWithOptions(opts, verbose) // Changed Verbose to verbose
		},
	}

	cmd.Flags().BoolVar(&opts.MCP, "mcp", false, "Add an MCP server dependency")
	cmd.Flags().StringVar(&opts.Alias, "alias", "", "Custom alias for the dependency")
	cmd.Flags().BoolVar(&opts.Force, "force", false, "Force reinstall if already exists")
	cmd.Flags().StringVar(&opts.URL, "url", "", "URL for remote MCP server")
	cmd.Flags().StringVar(&opts.RunCmd, "run", "", "Command to run the MCP server with template variables")
	cmd.Flags().StringSliceVar(&opts.SetupCmds, "setup", []string{}, "Setup commands to run before starting (repeatable)")
	cmd.Flags().StringVar(&opts.WorkingDir, "working-dir", "", "Working directory for the MCP server")
	cmd.Flags().StringSliceVar(&opts.EnvVars, "env", []string{}, "Environment variables (repeatable, KEY=VALUE format)")
	cmd.Flags().StringVar(&opts.Description, "description", "", "Description of the MCP server")
	cmd.Flags().StringSliceVar(&opts.Tags, "tags", []string{}, "Tags for categorizing the server (repeatable)")
	cmd.Flags().StringVar(&opts.HealthCheck, "health-check", "", "Custom health check command")
	cmd.Flags().IntVar(&opts.Timeout, "timeout", 30, "Timeout for server operations in seconds")
	cmd.Flags().StringVar(&opts.Version, "version", "", "Specific version/tag to install")

	return cmd
}

func runAddCommandWithOptions(opts *MCPAddOptions, verbose bool) error {
	projectDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	if err := validateAgentFieldProject(projectDir); err != nil {
		return err
	}

	if opts.MCP {
		// Create and execute the MCPAddCommand
		mcpCmd, err := NewMCPAddCommand(projectDir, opts, verbose)
		if err != nil {
			// Consider a more user-friendly error format here, possibly using PrintError
			return fmt.Errorf("failed to prepare add MCP command: %w", err)
		}
		return mcpCmd.Execute()
	}

	return fmt.Errorf("only MCP server dependencies are currently supported. Use --mcp flag")
}

func validateAgentFieldProject(projectDir string) error {
	agentfieldYAMLPath := filepath.Join(projectDir, "agentfield.yaml")
	if _, err := os.Stat(agentfieldYAMLPath); os.IsNotExist(err) {
		return fmt.Errorf("not a AgentField project directory (agentfield.yaml not found)")
	}
	return nil
}

// MCPAddCommand encapsulates the logic for adding an MCP server.
type MCPAddCommand struct {
	ProjectDir string
	Opts       *MCPAddOptions
	Verbose    bool
	AppConfig  *config.Config
	Manager    *mcp.MCPManager // Initialized in the builder or Execute
}

// NewMCPAddCommand acts as a builder for MCPAddCommand.
// It performs initial processing and validation.
func NewMCPAddCommand(projectDir string, opts *MCPAddOptions, verboseFlag bool) (*MCPAddCommand, error) {
	// Load application configuration
	appCfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
	if err != nil {
		// Fallback for safety, though agentfield.yaml should exist due to validateAgentFieldProject
		appCfg, err = config.LoadConfig("agentfield.yaml")
		if err != nil {
			return nil, fmt.Errorf("failed to load af configuration: %w. Ensure agentfield.yaml exists", err)
		}
	}

	// Determine final alias
	finalAlias := opts.Alias
	if finalAlias == "" {
		finalAlias = deriveAliasLocally(opts.Source) // Using local helper for now
	}

	// Construct MCPServerConfig (this will be part of the MCPAddCommand or its options)
	// For now, we'll build it inside Execute or pass opts directly.
	// The main point here is to set up the command object.

	// TODO: Perform more comprehensive validation using mcp.ConfigValidator if needed here.
	// For example, validating specific formats of Source, Runtime, etc.
	// The BasicConfigValidator can be instantiated and used.
	// validator := mcp.NewBasicConfigValidator()
	// tempCfgForValidation := mcp.MCPServerConfig{ Alias: finalAlias, Source: opts.Source, ... }
	// validationErrs := validator.Validate(tempCfgForValidation)
	// if len(validationErrs) > 0 {
	//    return nil, fmt.Errorf("MCP configuration validation failed:\n%s", validationErrs.Error())
	// }

	return &MCPAddCommand{
		ProjectDir: projectDir,
		Opts:       opts,
		Verbose:    verboseFlag,
		AppConfig:  appCfg,
		// Manager will be initialized in Execute or if needed earlier.
	}, nil
}

// Execute performs the MCP server addition.
func (cmd *MCPAddCommand) Execute() error {
	fmt.Printf("Adding MCP server: %s\n", Bold(cmd.Opts.Source))

	// Initialize MCPManager here if not done in builder
	cmd.Manager = mcp.NewMCPManager(cmd.AppConfig, cmd.ProjectDir, cmd.Verbose)

	finalAlias := cmd.Opts.Alias // Alias might have been refined by builder if it was more complex
	if finalAlias == "" {
		finalAlias = deriveAliasLocally(cmd.Opts.Source)
	}

	// This check should use the refined alias
	if !cmd.Opts.Force && finalAlias != "" {
		if mcpExists(cmd.ProjectDir, finalAlias) {
			return fmt.Errorf("MCP server with alias '%s' already exists (use --force to reinstall)", finalAlias)
		}
	}

	mcpServerCfg := mcp.MCPServerConfig{
		Alias:       finalAlias,
		Description: cmd.Opts.Description,
		URL:         cmd.Opts.URL,
		RunCmd:      cmd.Opts.RunCmd,
		SetupCmds:   cmd.Opts.SetupCmds,
		WorkingDir:  cmd.Opts.WorkingDir,
		HealthCheck: cmd.Opts.HealthCheck,
		Version:     cmd.Opts.Version,
		Tags:        cmd.Opts.Tags,
		Force:       cmd.Opts.Force,
	}

	// Set timeout if provided
	if cmd.Opts.Timeout > 0 {
		mcpServerCfg.Timeout = time.Duration(cmd.Opts.Timeout) * time.Second
	}

	// Parse environment variables
	if len(cmd.Opts.EnvVars) > 0 {
		mcpServerCfg.Env = make(map[string]string)
		for _, envVar := range cmd.Opts.EnvVars {
			parts := strings.SplitN(envVar, "=", 2)
			if len(parts) == 2 {
				mcpServerCfg.Env[parts[0]] = parts[1]
			} else {
				fmt.Printf("  %s Warning: invalid environment variable format '%s', expected KEY=VALUE\n",
					Yellow(StatusWarning), envVar)
			}
		}
	}

	// TODO: Issue 4 - Re-enable validation with new simplified architecture
	// Temporarily disabled validator to avoid compilation errors
	/*
		// Integrate mcp.ConfigValidator (from Task 1.3)
		validator := mcp.NewBasicConfigValidator()
		validationErrs := validator.Validate(mcpServerCfg)
		if len(validationErrs) > 0 {
		   return fmt.Errorf("MCP configuration validation failed:\n%s", validationErrs.Error())
		}
	*/

	fmt.Printf("  %s Adding MCP server...\n", Blue("→"))

	// Use the new simplified Add method
	if err := cmd.Manager.Add(mcpServerCfg); err != nil {
		fmt.Printf("  %s %s\n", Red(StatusError), err.Error())
		return fmt.Errorf("failed to add MCP server: %w", err)
	}

	fmt.Printf("  %s MCP server added successfully\n", Green(StatusSuccess))
	fmt.Printf("  %s Alias: %s\n", Gray(StatusInfo), Cyan(mcpServerCfg.Alias))
	fmt.Printf("  %s Location: %s\n", Gray(StatusInfo), Gray(filepath.Join("packages", "mcp", mcpServerCfg.Alias)))

	// Show configuration details
	if mcpServerCfg.URL != "" || mcpServerCfg.RunCmd != "" || len(mcpServerCfg.SetupCmds) > 0 || len(mcpServerCfg.Env) > 0 || mcpServerCfg.HealthCheck != "" {
		fmt.Printf("  %s Configuration applied:\n", Gray(StatusInfo))
		if mcpServerCfg.URL != "" {
			fmt.Printf("    URL: %s\n", mcpServerCfg.URL)
		}
		if mcpServerCfg.RunCmd != "" {
			fmt.Printf("    Run command: %s\n", mcpServerCfg.RunCmd)
		}
		if len(mcpServerCfg.SetupCmds) > 0 {
			fmt.Printf("    Setup commands: %v\n", mcpServerCfg.SetupCmds)
		}
		if mcpServerCfg.WorkingDir != "" {
			fmt.Printf("    Working directory: %s\n", mcpServerCfg.WorkingDir)
		}
		if len(mcpServerCfg.Env) > 0 {
			fmt.Printf("    Environment variables: %d set\n", len(mcpServerCfg.Env))
		}
		if mcpServerCfg.HealthCheck != "" {
			fmt.Printf("    Health check: %s\n", mcpServerCfg.HealthCheck)
		}
		if mcpServerCfg.Description != "" {
			fmt.Printf("    Description: %s\n", mcpServerCfg.Description)
		}
		if len(mcpServerCfg.Tags) > 0 {
			fmt.Printf("    Tags: %v\n", mcpServerCfg.Tags)
		}
	}

	fmt.Printf("  %s Capabilities discovery and skill generation handled by manager\n", Gray(StatusInfo))

	fmt.Printf("\n%s %s\n", Blue("→"), Bold("Next steps:"))
	fmt.Printf("  %s Start the MCP server: %s\n", Gray("1."), Cyan(fmt.Sprintf("af mcp start %s", mcpServerCfg.Alias)))
	fmt.Printf("  %s Check status: %s\n", Gray("2."), Cyan("af mcp status"))
	fmt.Printf("  %s Use MCP tools as regular skills: %s\n", Gray("3."), Cyan(fmt.Sprintf("await app.call(\"%s_<tool_name>\", ...)", mcpServerCfg.Alias)))
	fmt.Printf("  %s Generated skill file: %s\n", Gray("4."), Gray(fmt.Sprintf("skills/mcp_%s.py", mcpServerCfg.Alias)))

	return nil
}

// This function is now replaced by MCPAddCommand.Execute()
// func addMCPServer(projectDir string, opts *MCPAddOptions, verbose bool) error {
//	fmt.Printf("Adding MCP server: %s\n", Bold(opts.Source))
//
//	appCfg, err := config.LoadConfig(filepath.Join(projectDir, "agentfield.yaml"))
// The orphaned code block that started with "if err != nil {" and was a remnant
// of the original addMCPServer function body has been removed by this replacement.
// The logic is now consolidated within MCPAddCommand.Execute().

func mcpExists(projectDir, alias string) bool {
	mcpDir := filepath.Join(projectDir, "packages", "mcp", alias)
	_, err := os.Stat(mcpDir)
	return err == nil // True if err is nil (exists), false otherwise
}

// deriveAliasLocally is a placeholder for a more robust alias derivation.
// Ideally, this logic resides in the mcp package or is more comprehensive.
func deriveAliasLocally(source string) string {
	if strings.Contains(source, "@modelcontextprotocol/server-github") {
		return "github"
	}
	if strings.Contains(source, "@modelcontextprotocol/server-memory") {
		return "memory"
	}
	if strings.Contains(source, "@modelcontextprotocol/server-filesystem") {
		return "filesystem"
	}

	// Basic derivation from source string (e.g., github:owner/repo -> repo)
	parts := strings.Split(source, "/")
	namePart := parts[len(parts)-1]

	nameParts := strings.SplitN(namePart, "@", 2) // remove version if any
	namePart = nameParts[0]

	nameParts = strings.SplitN(namePart, ":", 2) // remove scheme if any
	if len(nameParts) > 1 {
		namePart = nameParts[1]
	}

	// Sanitize for use as a directory name (simple version)
	namePart = strings.ReplaceAll(namePart, ".", "_")

	if namePart == "" {
		return "mcp_server" // Default fallback
	}
	return namePart
}
