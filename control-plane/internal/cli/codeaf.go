package cli

import (
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
	"github.com/spf13/cobra"
)

func NewCodeafCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "codeaf",
		Short: "Manage the CodeAF coding-harness binary",
	}
	var force bool
	var lockWait time.Duration
	ensureCmd := &cobra.Command{
		Use:   "ensure",
		Short: "Install or repair the pinned CodeAF coding-harness binary",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if codeaf.SkipRequested() {
				_, err := fmt.Fprintln(cmd.OutOrStdout(), "CodeAF provisioning skipped (AGENTFIELD_SKIP_CODEAF or AGENTFIELD_SKIP_AFORGE)")
				return err
			}
			return codeaf.Ensure(codeaf.Options{Force: force, Context: cmd.Context(), LockWait: lockWait, OnDownload: func() {
				_, _ = fmt.Fprintln(cmd.ErrOrStderr(), "Installing the CodeAF harness (~70 MB)…")
			}})
		},
	}
	ensureCmd.Flags().BoolVar(&force, "force", false, "Re-download even when the pinned version is already installed")
	ensureCmd.Flags().DurationVar(&lockWait, "lock-wait", 0, "Maximum wait for another installer (zero waits until canceled)")
	cmd.AddCommand(ensureCmd)
	return cmd
}
