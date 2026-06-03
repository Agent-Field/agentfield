package commands

import (
	"fmt"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/spf13/cobra"
)

// RunCommand implements the run command using the new framework
type RunCommand struct {
	framework.BaseCommand
	output *framework.OutputFormatter
}

// NewRunCommand creates a new run command
func NewRunCommand(services *framework.ServiceContainer) framework.Command {
	return &RunCommand{
		BaseCommand: framework.BaseCommand{Services: services},
		output:      framework.NewOutputFormatter(false), // Will be updated based on flags
	}
}

// GetName returns the command name
func (cmd *RunCommand) GetName() string {
	return "run"
}

// GetDescription returns the command description
func (cmd *RunCommand) GetDescription() string {
	return "Run an installed AgentField agent node package"
}

// BuildCobraCommand builds the Cobra command
func (cmd *RunCommand) BuildCobraCommand() *cobra.Command {
	var port int
	var detach bool
	var verbose bool

	cobraCmd := &cobra.Command{
		Use:   "run <agent-node-name>",
		Short: cmd.GetDescription(),
		Long: `Start an installed AgentField agent node package in the background.

The agent node will be assigned an available port and registered with
the AgentField server if available.

Examples:
  af run email-helper
  af run data-analyzer --port 8005
  af run my-agent --detach=false`,
		Args: cobra.ExactArgs(1),
		RunE: func(cobraCmd *cobra.Command, args []string) error {
			// Update output formatter with verbose setting
			cmd.output.SetVerbose(verbose)
			return cmd.execute(args[0], port, detach, verbose)
		},
	}

	cobraCmd.Flags().IntVarP(&port, "port", "p", 0, "Specific port to use (auto-assigned if not specified)")
	cobraCmd.Flags().BoolVarP(&detach, "detach", "d", true, "Run in background (default: true)")
	cobraCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")

	return cobraCmd
}

// execute performs the actual agent execution
func (cmd *RunCommand) execute(agentName string, port int, detach, verbose bool) error {
	cmd.output.PrintHeader("Running AgentField Agent")
	cmd.output.PrintInfo(fmt.Sprintf("Agent: %s", agentName))

	if verbose {
		cmd.output.PrintVerbose("Using new framework-based run command")
		if port > 0 {
			cmd.output.PrintVerbose(fmt.Sprintf("Requested port: %d", port))
		}
		cmd.output.PrintVerbose(fmt.Sprintf("Detach mode: %t", detach))
	}

	// Create run options
	options := domain.RunOptions{
		Port:   port,
		Detach: detach,
	}

	// Show progress
	cmd.output.PrintProgress("Starting agent...")

	// Use the agent service to run the agent
	runningAgent, err := cmd.Services.AgentService.RunAgent(agentName, options)
	if err != nil {
		cmd.output.PrintError(fmt.Sprintf("Failed to run agent: %v", err))
		return err
	}

	// Display success information
	cmd.output.PrintSuccess(fmt.Sprintf("Agent '%s' started successfully", agentName))
	cmd.output.PrintInfo(fmt.Sprintf("PID: %d", runningAgent.PID))
	cmd.output.PrintInfo(fmt.Sprintf("Port: %d", runningAgent.Port))

	if runningAgent.LogFile != "" {
		cmd.output.PrintInfo(fmt.Sprintf("Logs: %s", runningAgent.LogFile))
	}

	if verbose {
		cmd.output.PrintVerbose(fmt.Sprintf("Status: %s", runningAgent.Status))
		cmd.output.PrintVerbose(fmt.Sprintf("Started at: %s", runningAgent.StartedAt.Format("2006-01-02 15:04:05")))

		// Show running agents
		cmd.output.PrintVerbose("Listing all running agents...")
		agents, err := cmd.Services.AgentService.ListRunningAgents()
		if err != nil {
			cmd.output.PrintWarning(fmt.Sprintf("Could not list running agents: %v", err))
		} else {
			cmd.output.PrintInfo(fmt.Sprintf("Total running agents: %d", len(agents)))
		}
	}

	if detach {
		cmd.output.PrintInfo("Agent is running in the background")
		cmd.output.PrintInfo("Use 'af stop " + agentName + "' to stop the agent")
		cmd.output.PrintInfo("Use 'af logs " + agentName + "' to view logs")
	}

	return nil
}
