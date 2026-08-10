package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// NewAuthCommand returns the `af auth` command tree for managing the API key
// the CLI presents to a control plane.
func NewAuthCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "auth",
		Short: "Manage control plane credentials",
		Long: `Store and inspect the API key the CLI sends to a control plane.

A control plane started without an API key needs no credentials at all — local
commands keep working untouched, and nothing is written to disk. When one is
configured (AGENTFIELD_API_KEY on the server, or api.auth.api_key in
~/.agentfield/agentfield.yaml), run "af auth login" once and every later
command authenticates automatically.

Credentials are stored per control plane URL in ~/.agentfield/credentials.json
with 0600 permissions.`,
	}

	cmd.AddCommand(newAuthLoginCommand())
	cmd.AddCommand(newAuthStatusCommand())
	cmd.AddCommand(newAuthLogoutCommand())
	return cmd
}

func newAuthLoginCommand() *cobra.Command {
	var skipVerify bool

	cmd := &cobra.Command{
		Use:   "login",
		Short: "Store an API key for a control plane",
		Long: `Stores an API key for the control plane selected by --server (default:
http://localhost:8080, or $AGENTFIELD_SERVER).

The key is prompted for without echo. Pass --api-key to supply it directly when
scripting. Before saving, the key is checked against the control plane; a
rejected key is not stored. Use --no-verify to store one for a control plane
that is not running yet.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			server := normalizedServerURL()

			key := strings.TrimSpace(apiKey)
			if key == "" {
				value, err := readHiddenValue(fmt.Sprintf("API key for %s", server))
				if err != nil {
					return err
				}
				key = strings.TrimSpace(value)
			}
			if key == "" {
				return fmt.Errorf("api key must not be empty")
			}

			if !skipVerify {
				ctx, cancel := commandContext()
				defer cancel()
				if err := verifyAPIKey(ctx, server, key); err != nil {
					return err
				}
			}

			store, err := credentialStore()
			if err != nil {
				return err
			}
			if err := store.Save(server, key); err != nil {
				return err
			}

			PrintSuccess(fmt.Sprintf("Stored API key %s for %s", maskAPIKey(key), server))
			PrintBullet(fmt.Sprintf("Saved to %s (permissions 0600)", store.Path()))
			return nil
		},
	}

	cmd.Flags().BoolVar(&skipVerify, "no-verify", false, "Store the key without checking it against the control plane")
	return cmd
}

func newAuthStatusCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show which credential the CLI would use",
		Long: `Reports the control plane the CLI is pointed at and the credential it would
send, masked. The full key is never printed.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			server := normalizedServerURL()
			key, source := resolveAPIKeyWithSource()

			PrintHeader("Control plane")
			PrintBullet(server)
			PrintHeader("Credential")

			if key == "" {
				PrintBullet("none — requests are sent unauthenticated")
				PrintBullet(fmt.Sprintf("If this control plane requires a key: af auth login --server %s", server))
				return nil
			}

			PrintBullet(fmt.Sprintf("%s (from %s)", maskAPIKey(key), source))
			if stored := storedAPIKey(server); stored != "" && source != apiKeySourceStored {
				PrintBullet(fmt.Sprintf("A key stored by af auth login (%s) is being overridden by %s", maskAPIKey(stored), source))
			}
			return nil
		},
	}
	return cmd
}

func newAuthLogoutCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logout",
		Short: "Remove the stored API key for a control plane",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			server := normalizedServerURL()

			store, err := credentialStore()
			if err != nil {
				return err
			}
			removed, err := store.Delete(server)
			if err != nil {
				return err
			}
			if !removed {
				PrintInfo(fmt.Sprintf("No stored API key for %s", server))
				return nil
			}

			PrintSuccess(fmt.Sprintf("Removed the stored API key for %s", server))
			// Removing the file entry does not stop an exported key from being
			// sent, so say so rather than letting the user believe otherwise.
			if _, source := resolveAPIKeyWithSource(); source != apiKeySourceNone {
				PrintBullet(fmt.Sprintf("%s still supplies a key for this server", source))
			}
			return nil
		},
	}
	return cmd
}

// normalizedServerURL is the control plane URL the auth commands operate on,
// trimmed so it matches how it is stored and displayed.
func normalizedServerURL() string {
	return strings.TrimRight(strings.TrimSpace(GetServerURL()), "/")
}

// authVerifyPath is the endpoint `af auth login` probes. The node list is
// registered on every control plane (unlike the UI-only routes), is cheap, and
// sits behind the API key middleware when one is configured — so a 401 means
// the key is wrong and anything else means it was accepted.
const authVerifyPath = "/api/v1/nodes"

// verifyAPIKey reports whether the control plane accepts key. Only an outright
// 401 counts as a rejection: a 404 or a 500 says something about the server,
// not about the credential, and should not block the user from saving it.
func verifyAPIKey(ctx context.Context, server, key string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server+authVerifyPath, nil)
	if err != nil {
		return fmt.Errorf("build verification request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "af-cli/auth")
	req.Header.Set("X-API-Key", key)

	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		//nolint:staticcheck // deliberate multi-line user-facing hint
		return fmt.Errorf("could not reach %s to verify the key: %w\nStart the control plane, or re-run with --no-verify to store the key anyway", server, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("%s rejected that API key; nothing was saved", server)
	}
	return nil
}

// authRequiredError is the single place the CLI turns a 401 from the control
// plane into something the user can act on. Every helper that talks to the
// control plane routes through it so the remedy is worded once.
func authRequiredError(serverURL string, body []byte) error {
	server := strings.TrimRight(strings.TrimSpace(serverURL), "/")
	if server == "" {
		server = normalizedServerURL()
	}

	message := fmt.Sprintf("authentication required by %s", server)
	if detail := serverAuthMessage(body); detail != "" {
		message += ": " + detail
	}
	return fmt.Errorf("%s; run: af auth login --server %s", message, server)
}

// serverAuthMessage pulls the explanation out of the control plane's 401 body
// so the reason (bad key vs. remote caller on an unauthenticated server) is not
// lost. A body that is not the expected envelope yields "".
func serverAuthMessage(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var payload struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return ""
	}
	if msg := strings.TrimSpace(payload.Message); msg != "" {
		return msg
	}
	return strings.TrimSpace(payload.Error)
}
