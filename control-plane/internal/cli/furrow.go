package cli

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/furrow"
	"github.com/spf13/cobra"
)

func NewFurrowCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "furrow",
		Short: "Manage the furrow workspace client",
	}
	cmd.AddCommand(&cobra.Command{
		Use:   "ensure",
		Short: "Install or repair the pinned furrow workspace client",
		Args:  cobra.NoArgs,
		// Ensure, not EnsureBestEffort: silence is the right default when an
		// install merely offers to provision furrow, but someone who asks for
		// it by name is owed the failure.
		RunE: func(_ *cobra.Command, _ []string) error {
			return furrow.Ensure(furrow.Options{})
		},
	})
	return cmd
}
