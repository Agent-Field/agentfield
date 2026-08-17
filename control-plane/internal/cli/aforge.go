package cli

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/aforge"
	"github.com/spf13/cobra"
)

func NewAforgeCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "aforge",
		Short: "Manage the aforge coding-harness binary",
	}
	var force bool
	ensureCmd := &cobra.Command{
		Use:   "ensure",
		Short: "Install or repair the pinned aforge coding-harness binary",
		Args:  cobra.NoArgs,
		// Ensure, not EnsureBestEffort: silence is the right default when an
		// install merely offers to provision aforge, but someone who asks for
		// it by name is owed the failure.
		RunE: func(_ *cobra.Command, _ []string) error {
			return aforge.Ensure(aforge.Options{Force: force})
		},
	}
	ensureCmd.Flags().BoolVar(&force, "force", false, "Re-download even when the pinned version is already installed")
	cmd.AddCommand(ensureCmd)
	return cmd
}
