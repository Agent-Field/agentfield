package commands

import (
	"fmt"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/spf13/cobra"
)

// InstallCommand implements the install command using the new framework
type InstallCommand struct {
	framework.BaseCommand
	output *framework.OutputFormatter
}

// NewInstallCommand creates a new install command
func NewInstallCommand(services *framework.ServiceContainer) framework.Command {
	return &InstallCommand{
		BaseCommand: framework.BaseCommand{Services: services},
		output:      framework.NewOutputFormatter(false), // Will be updated based on flags
	}
}

// GetName returns the command name
func (cmd *InstallCommand) GetName() string {
	return "install"
}

// GetDescription returns the command description
func (cmd *InstallCommand) GetDescription() string {
	return "Install a AgentField agent node package for local use"
}

// BuildCobraCommand builds the Cobra command
func (cmd *InstallCommand) BuildCobraCommand() *cobra.Command {
	var force bool
	var verbose bool
	var jsonOutput bool

	cobraCmd := &cobra.Command{
		Use:   "install <package-path>",
		Short: cmd.GetDescription(),
		Long: `Install a AgentField agent node package for local use.

The package can be:
- A local directory path
- A GitHub repository URL
- A package name from the AgentField registry

Examples:
  agentfield install ./my-agent
  agentfield install https://github.com/user/agent-repo
  agentfield install agent-name`,
		Args: cobra.ExactArgs(1),
		RunE: func(cobraCmd *cobra.Command, args []string) error {
			if jsonOutput {
				return cmd.executeJSON(args[0], force)
			}
			// Update output formatter with verbose setting
			cmd.output.SetVerbose(verbose)
			return cmd.execute(args[0], force, verbose)
		},
	}

	cobraCmd.Flags().BoolVarP(&force, "force", "f", false, "Force reinstall if package exists")
	cobraCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
	cobraCmd.Flags().BoolVar(&jsonOutput, "json", false, "Emit a machine-readable JSON envelope (diagnostics go to stderr)")

	return cobraCmd
}

// executeJSON installs the package and emits a JSON envelope on stdout.
// Service-layer progress output is redirected to stderr for the duration.
func (cmd *InstallCommand) executeJSON(packagePath string, force bool) error {
	stdout, restore := redirectStdoutToStderr()
	defer restore()

	options := domain.InstallOptions{Force: force}
	if err := cmd.Services.PackageService.InstallPackage(packagePath, options); err != nil {
		printJSONError(stdout, "install_failed", err.Error(), "Check the package source (path, git URL, or registry name); use --force to reinstall.")
		return err
	}

	return printJSONSuccess(stdout, map[string]interface{}{
		"source": packagePath,
		"status": "installed",
	})
}

// execute performs the actual installation
func (cmd *InstallCommand) execute(packagePath string, force, verbose bool) error {
	cmd.output.PrintHeader("Installing AgentField Package")
	cmd.output.PrintInfo(fmt.Sprintf("Package: %s", packagePath))

	if verbose {
		cmd.output.PrintVerbose("Using new framework-based install command")
	}

	// Create install options
	options := domain.InstallOptions{
		Force:   force,
		Verbose: verbose,
	}

	// Show progress
	cmd.output.PrintProgress("Starting installation...")

	// Use the package service to install
	err := cmd.Services.PackageService.InstallPackage(packagePath, options)
	if err != nil {
		cmd.output.PrintError(fmt.Sprintf("Installation failed: %v", err))
		return err
	}

	cmd.output.PrintSuccess("Package installed successfully")

	if verbose {
		// Show installed packages
		cmd.output.PrintVerbose("Listing installed packages...")
		packages, err := cmd.Services.PackageService.ListInstalledPackages()
		if err != nil {
			cmd.output.PrintWarning(fmt.Sprintf("Could not list packages: %v", err))
		} else {
			cmd.output.PrintInfo(fmt.Sprintf("Total installed packages: %d", len(packages)))
		}
	}

	return nil
}
