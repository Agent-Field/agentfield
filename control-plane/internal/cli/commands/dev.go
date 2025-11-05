package commands

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/your-org/agentfield/control-plane/internal/cli/framework"
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
)

type DevCommand struct {
	framework.BaseCommand
	output *framework.OutputFormatter
}

func NewDevCommand(services *framework.ServiceContainer) framework.Command {
	return &DevCommand{
		BaseCommand: framework.BaseCommand{Services: services},
		output:      framework.NewOutputFormatter(false), // Get from context later
	}
}

func (cmd *DevCommand) GetName() string {
	return "dev"
}

func (cmd *DevCommand) GetDescription() string {
	return "Run a AgentField agent package in development mode"
}

func (cmd *DevCommand) BuildCobraCommand() *cobra.Command {
	var port int
	var watch bool
	var verbose bool

	cobraCmd := &cobra.Command{
		Use:   "dev [path]",
		Short: cmd.GetDescription(),
		Long: `Run a AgentField agent package in development mode from the current directory or specified path.

This command is designed for local development and testing. It will:
- Look for agentfield.yaml in the current directory (or specified path)
- Start the agent without requiring installation
- Provide verbose logging for development
- Optionally watch for file changes and auto-restart

Examples:
  af dev                    # Run package in current directory
  af dev ./my-agent         # Run package in specified directory
  af dev --port 8005        # Use specific port
  af dev --watch            # Watch for changes and auto-restart`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cobraCmd *cobra.Command, args []string) error {
			return cmd.execute(args, port, watch, verbose)
		},
	}

	cobraCmd.Flags().IntVarP(&port, "port", "p", 0, "Specific port to use (auto-assigned if not specified)")
	cobraCmd.Flags().BoolVarP(&watch, "watch", "w", false, "Watch for file changes and auto-restart")
	cobraCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")

	return cobraCmd
}

func (cmd *DevCommand) execute(args []string, port int, watch, verbose bool) error {
	// Determine the path to run
	var packagePath string
	if len(args) > 0 {
		packagePath = args[0]
	} else {
		packagePath = "."
	}

	cmd.output.PrintHeader("🚀 AgentField Development Mode")
	cmd.output.PrintInfo(fmt.Sprintf("Package path: %s", packagePath))

	// Create dev options
	options := domain.DevOptions{
		Port:       port,
		WatchFiles: watch,
		Verbose:    verbose,
	}

	// Run in development mode using the service
	err := cmd.Services.DevService.RunInDevMode(packagePath, options)
	if err != nil {
		cmd.output.PrintError(fmt.Sprintf("Failed to run in dev mode: %v", err))
		return err
	}

	return nil
}
