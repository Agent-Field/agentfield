package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// Where GetAPIKey found the credential it returned. Used by `af auth status`
// so the user can tell a stored key from one an env var or flag is shadowing.
const (
	apiKeySourceNone   = ""
	apiKeySourceFlag   = "--api-key flag"
	apiKeySourceEnv    = "AGENTFIELD_API_KEY"
	apiKeySourceStored = "af auth login"
)

// credentialStore opens the credential file backing `af auth`. It deliberately
// does not go through getAgentFieldHomeDir(): that creates directories and
// exits the process on failure, neither of which belongs on a lookup that runs
// before every single control plane request.
func credentialStore() (*packages.CredentialStore, error) {
	home, err := packages.AgentFieldHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve the AgentField home directory: %w", err)
	}
	return packages.NewCredentialStore(home), nil
}

// storedAPIKey returns the key `af auth login` saved for serverURL, or "".
func storedAPIKey(serverURL string) string {
	store, err := credentialStore()
	if err != nil {
		return ""
	}
	return store.Lookup(serverURL)
}

// resolveAPIKeyWithSource is the single credential resolution point for the
// CLI. Precedence, highest first:
//
//	--api-key / -k        explicit, one command
//	AGENTFIELD_API_KEY    explicit, one shell
//	credentials.json      persisted by `af auth login` for this server
//	""                    no key: the default local setup, unauthenticated
func resolveAPIKeyWithSource() (string, string) {
	if apiKey != "" {
		return apiKey, apiKeySourceFlag
	}
	if env := os.Getenv("AGENTFIELD_API_KEY"); env != "" {
		return env, apiKeySourceEnv
	}
	if stored := storedAPIKey(GetServerURL()); stored != "" {
		return stored, apiKeySourceStored
	}
	return "", apiKeySourceNone
}

// maskAPIKey renders a key for display: a recognisable prefix, then the last
// four characters. Short keys are masked completely — there is nothing to
// recognise and every revealed character is a character an onlooker gets for
// free. The full key is never printed anywhere in the CLI.
func maskAPIKey(key string) string {
	key = strings.TrimSpace(key)
	if key == "" {
		return ""
	}
	const revealed = 4
	if len(key) <= revealed*2 {
		return strings.Repeat("•", len(key))
	}

	head := key[:len(key)-revealed]
	tail := key[len(key)-revealed:]
	// Keep a vendor prefix ("af_live_", "sk-") intact when there is one,
	// otherwise show the first three characters.
	if idx := strings.LastIndexAny(head, "_-"); idx > 0 && idx < 12 {
		head = head[:idx+1]
	} else if len(head) > 3 {
		head = head[:3]
	}
	return head + "…" + tail
}
