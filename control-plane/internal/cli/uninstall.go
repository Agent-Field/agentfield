package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/your-org/haxen/control-plane/internal/packages"
)

var (
	uninstallForce bool
)

// NewUninstallCommand creates the uninstall command
func NewUninstallCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "uninstall <package-name>",
		Short: "Uninstall an agent node package",
		Long: `Uninstall removes an installed agent node package from your system.

This command will:
- Stop the agent node if it's currently running
- Remove the package directory and all its files
- Remove the package from the installation registry
- Clean up any associated logs

Examples:
  haxen uninstall my-agent
  haxen uninstall sentiment-analyzer --force`,
		Args: cobra.ExactArgs(1),
		Run:  runUninstallCommand,
	}

	cmd.Flags().BoolVarP(&uninstallForce, "force", "f", false, "Force uninstall even if agent node is running")

	return cmd
}

func runUninstallCommand(cmd *cobra.Command, args []string) {
	packageName := args[0]

	// Create uninstaller
	uninstaller := &packages.PackageUninstaller{
		HaxenHome: getHaxenHomeDir(),
		Force:     uninstallForce,
	}

	// Uninstall package
	if err := uninstaller.UninstallPackage(packageName); err != nil {
		fmt.Printf("❌ Uninstallation failed: %v\n", err)
		os.Exit(1)
	}
}
