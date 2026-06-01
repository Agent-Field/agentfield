package cli

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"
)

// VersionInfo holds build-time version information
type VersionInfo struct {
	Version string
	Commit  string
	Date    string
}

// NewVersionCommand creates the version command
func NewVersionCommand(versionInfo VersionInfo) *cobra.Command {
	versionCmd := &cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Long:  `Display the version, build date, commit hash, and runtime information for AgentField.`,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("AgentField Control Plane\n")
			fmt.Printf("  Version:    %s\n", versionInfo.Version)
			fmt.Printf("  Commit:     %s\n", versionInfo.Commit)
			fmt.Printf("  Built:      %s\n", versionInfo.Date)
			fmt.Printf("  Go version: %s\n", runtime.Version())
			fmt.Printf("  OS/Arch:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
		},
	}

	return versionCmd
}
