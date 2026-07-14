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
	var path string

	cobraCmd := &cobra.Command{
		Use:   "install <package-path>",
		Short: cmd.GetDescription(),
		Long: `Install a AgentField agent node package for local use.

The package can be:
- A local directory path
- A GitHub repository URL — append //<subdir> to install a package whose
  agentfield-package.yaml lives in a subdirectory (repos can ship several
  nodes, e.g. a Python root and a Go port), and @<ref> for a branch or tag
- A package name from the AgentField registry

Use --path to install a package that lives in a subdirectory of the source, so a
single repository can ship more than one installable node. The subdirectory must
contain its own agentfield-package.yaml; that subtree becomes the package root.
--path is relative to the source root and may not escape it. It composes with an
@ref pin on a Git URL.

Examples:
  agentfield install ./my-agent
  agentfield install https://github.com/user/agent-repo
  agentfield install https://github.com/user/agent-repo --path go
  agentfield install https://github.com/user/agent-repo@v1.2.3 --path go
  agentfield install https://github.com/user/agent-repo//go@main
  agentfield install agent-name`,
		Args: cobra.ExactArgs(1),
		RunE: func(cobraCmd *cobra.Command, args []string) error {
			// Update output formatter with verbose setting
			cmd.output.SetVerbose(verbose)
			return cmd.execute(args[0], force, verbose, path)
		},
	}

	cobraCmd.Flags().BoolVarP(&force, "force", "f", false, "Force reinstall if package exists")
	cobraCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
	cobraCmd.Flags().StringVar(&path, "path", "", "Install the package from this subdirectory of the source (relative to its root)")

	return cobraCmd
}

// execute performs the actual installation
func (cmd *InstallCommand) execute(packagePath string, force, verbose bool, path string) error {
	cmd.output.PrintHeader("Installing AgentField Package")
	cmd.output.PrintInfo(fmt.Sprintf("Package: %s", packagePath))
	if path != "" {
		cmd.output.PrintInfo(fmt.Sprintf("Subdirectory: %s", path))
	}

	if verbose {
		cmd.output.PrintVerbose("Using new framework-based install command")
	}

	// Create install options
	options := domain.InstallOptions{
		Force:   force,
		Verbose: verbose,
		Path:    path,
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
