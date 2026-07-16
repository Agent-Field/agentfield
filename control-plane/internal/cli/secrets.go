package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

// NewSecretsCommand returns the `af secrets` command tree for managing the
// encrypted secret store used by agent nodes.
func NewSecretsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "secrets",
		Short: "Manage encrypted secrets for agent nodes",
		Long: `Manage the encrypted secret store under ~/.agentfield.

Secrets are stored encrypted at rest (AES-256-GCM) and are only ever decrypted
into an agent node's process environment at start time. Global secrets are
shared across all nodes; node-scoped secrets override the global value for a
single node.`,
	}

	cmd.AddCommand(newSecretsSetCommand())
	cmd.AddCommand(newSecretsListCommand())
	cmd.AddCommand(newSecretsRemoveCommand())
	return cmd
}

func openSecretStore() (*packages.SecretStore, error) {
	return packages.NewSecretStore(getAgentFieldHomeDir())
}

func newSecretsSetCommand() *cobra.Command {
	var node string
	cmd := &cobra.Command{
		Use:   "set KEY [VALUE]",
		Short: "Store a secret (prompts hidden if VALUE is omitted)",
		Args:  cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			var value string
			if len(args) == 2 {
				value = args[1]
			} else {
				v, err := readHiddenValue(fmt.Sprintf("Enter value for %s", key))
				if err != nil {
					return err
				}
				value = v
			}
			if strings.TrimSpace(value) == "" {
				return fmt.Errorf("value must not be empty")
			}

			store, err := openSecretStore()
			if err != nil {
				return err
			}
			scope := packages.GlobalScope
			if node != "" {
				scope = node
			}
			if err := store.Set(scope, key, value); err != nil {
				return err
			}
			PrintSuccess(fmt.Sprintf("Stored %s in %s scope", key, scope))
			// Secrets decrypt into a node's environment at spawn time only —
			// a running node keeps the env it started with.
			if node != "" {
				PrintInfo(fmt.Sprintf("Applies on next start: af stop %s && af run %s", node, node))
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&node, "node", "", "store as a node-scoped secret instead of global")
	return cmd
}

func newSecretsListCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "ls",
		Aliases: []string{"list"},
		Short:   "List stored secrets (values masked)",
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := openSecretStore()
			if err != nil {
				return err
			}
			refs, err := store.ListAll()
			if err != nil {
				return err
			}
			if len(refs) == 0 {
				fmt.Println(ui.Panel("No secrets stored",
					ui.Muted("Add one with:")+"\n  af secrets set KEY"))
				return nil
			}
			rows := make([][]string, 0, len(refs))
			for _, ref := range refs {
				rows = append(rows, []string{ref.Key, ref.Scope, ui.Muted("••••••••")})
			}
			fmt.Println(ui.Table(
				fmt.Sprintf("Stored secrets (%d)", len(rows)),
				[]string{"KEY", "SCOPE", "VALUE"},
				rows,
			))
			return nil
		},
	}
	return cmd
}

func newSecretsRemoveCommand() *cobra.Command {
	var node string
	cmd := &cobra.Command{
		Use:     "rm KEY",
		Aliases: []string{"remove", "delete"},
		Short:   "Remove a stored secret",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := openSecretStore()
			if err != nil {
				return err
			}
			scope := packages.GlobalScope
			if node != "" {
				scope = node
			}
			if err := store.Delete(scope, args[0]); err != nil {
				return err
			}
			PrintSuccess(fmt.Sprintf("Removed %s from %s scope", args[0], scope))
			return nil
		},
	}
	cmd.Flags().StringVar(&node, "node", "", "remove from a node scope instead of global")
	return cmd
}

// readHiddenValue reads a single line without echo when stdin is a terminal,
// falling back to plain line input otherwise.
func readHiddenValue(prompt string) (string, error) {
	fmt.Printf("%s: ", prompt)
	if term.IsTerminal(int(os.Stdin.Fd())) {
		data, err := term.ReadPassword(int(os.Stdin.Fd()))
		fmt.Println()
		if err != nil {
			return "", fmt.Errorf("failed to read value: %w", err)
		}
		return strings.TrimSpace(string(data)), nil
	}
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil && line == "" {
		return "", fmt.Errorf("failed to read value: %w", err)
	}
	return strings.TrimSpace(line), nil
}
