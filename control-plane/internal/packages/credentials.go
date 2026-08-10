package packages

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Credential layout under the AgentField home directory:
//
//	~/.agentfield/credentials.json   # 0600, one API key per control plane
//
// `af auth login` writes this file so later commands authenticate against a
// control plane that has an API key configured, without the user exporting
// AGENTFIELD_API_KEY in every shell. Unlike the secret store the value is not
// encrypted at rest: it has to be replayed verbatim on every request, so the
// protection is 0600 file permissions — the same posture as ~/.docker/config.json
// or an npm token.
//
// Nothing here creates the file. A machine that never runs `af auth login`
// never grows one, which keeps the default local setup (no API key anywhere)
// exactly as it was.

const (
	credentialsFileName = "credentials.json"
	credentialsFilePerm = 0o600
)

// ServerCredential is the credential set stored for a single control plane.
type ServerCredential struct {
	APIKey string `json:"api_key"`
}

// Credentials is the on-disk shape of credentials.json.
type Credentials struct {
	Servers map[string]ServerCredential `json:"servers"`
}

// CredentialStore reads and writes credentials.json under an AgentField home.
type CredentialStore struct {
	path string
}

// NewCredentialStore returns a store rooted at the given AgentField home.
func NewCredentialStore(agentFieldHome string) *CredentialStore {
	return &CredentialStore{path: filepath.Join(agentFieldHome, credentialsFileName)}
}

// Path returns the credentials file path. The file may not exist.
func (cs *CredentialStore) Path() string {
	return cs.path
}

// NormalizeServerURL canonicalises a control plane URL so it is a stable map
// key: "http://localhost:8080/", "http://localhost:8080" and
// "HTTP://LocalHost:8080" all resolve to the same entry. A URL that cannot be
// parsed degrades to a trimmed, lowercased string rather than being dropped.
func NormalizeServerURL(raw string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(raw), "/")
	if trimmed == "" {
		return ""
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return strings.ToLower(trimmed)
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	parsed.Host = strings.ToLower(parsed.Host)
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}

// Load returns the stored credentials. A missing file is not an error and is
// never created by reading it.
func (cs *CredentialStore) Load() (*Credentials, error) {
	creds := &Credentials{Servers: map[string]ServerCredential{}}

	data, err := os.ReadFile(cs.path)
	if err != nil {
		if os.IsNotExist(err) {
			return creds, nil
		}
		return nil, fmt.Errorf("failed to read credentials: %w", err)
	}
	if strings.TrimSpace(string(data)) == "" {
		return creds, nil
	}
	if err := json.Unmarshal(data, creds); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", cs.path, err)
	}
	if creds.Servers == nil {
		creds.Servers = map[string]ServerCredential{}
	}
	return creds, nil
}

// Lookup returns the API key stored for serverURL, or "" when there is none.
// An unreadable or corrupt file also resolves to "" so a broken credentials
// file degrades to "unauthenticated" instead of breaking every command.
func (cs *CredentialStore) Lookup(serverURL string) string {
	key := NormalizeServerURL(serverURL)
	if key == "" {
		return ""
	}
	creds, err := cs.Load()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(creds.Servers[key].APIKey)
}

// Save stores apiKey for serverURL, replacing any existing entry.
func (cs *CredentialStore) Save(serverURL, apiKey string) error {
	key := NormalizeServerURL(serverURL)
	if key == "" {
		return fmt.Errorf("a control plane URL is required to store a credential")
	}
	if strings.TrimSpace(apiKey) == "" {
		return fmt.Errorf("api key must not be empty")
	}

	creds, err := cs.Load()
	if err != nil {
		return err
	}
	creds.Servers[key] = ServerCredential{APIKey: strings.TrimSpace(apiKey)}
	return cs.write(creds)
}

// Delete removes the entry for serverURL and reports whether one existed.
func (cs *CredentialStore) Delete(serverURL string) (bool, error) {
	creds, err := cs.Load()
	if err != nil {
		return false, err
	}
	key := NormalizeServerURL(serverURL)
	if _, ok := creds.Servers[key]; !ok {
		return false, nil
	}
	delete(creds.Servers, key)
	return true, cs.write(creds)
}

// Servers returns the control plane URLs that have a stored key, sorted.
func (cs *CredentialStore) Servers() ([]string, error) {
	creds, err := cs.Load()
	if err != nil {
		return nil, err
	}
	servers := make([]string, 0, len(creds.Servers))
	for server := range creds.Servers {
		servers = append(servers, server)
	}
	sort.Strings(servers)
	return servers, nil
}

func (cs *CredentialStore) write(creds *Credentials) error {
	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode credentials: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(cs.path), 0o755); err != nil {
		return fmt.Errorf("failed to create credentials directory: %w", err)
	}
	if err := os.WriteFile(cs.path, append(data, '\n'), credentialsFilePerm); err != nil {
		return fmt.Errorf("failed to write credentials: %w", err)
	}
	// WriteFile only applies the mode when it creates the file, so an entry
	// rewritten into a pre-existing, loosely permissioned file would keep the
	// old mode. Chmod every time instead of trusting what was there.
	if err := os.Chmod(cs.path, credentialsFilePerm); err != nil {
		return fmt.Errorf("failed to secure credentials file: %w", err)
	}
	return nil
}

// AgentFieldHomeDir resolves the AgentField home directory (honouring
// AGENTFIELD_HOME) without creating it. Credential lookup happens on every
// command, so it must not have side effects on disk.
func AgentFieldHomeDir() (string, error) {
	if custom := strings.TrimSpace(os.Getenv("AGENTFIELD_HOME")); custom != "" {
		return custom, nil
	}
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	return filepath.Join(homeDir, ".agentfield"), nil
}

// apiKeyOverride carries a key supplied out of band — in practice the CLI's
// --api-key flag, which lives in the cli package and cannot be read from here.
// The CLI sets it once during startup so a child agent process inherits the
// same credential the CLI itself is using.
var apiKeyOverride string

// SetAPIKeyOverride records an out-of-band API key. An empty value clears it.
func SetAPIKeyOverride(key string) {
	apiKeyOverride = strings.TrimSpace(key)
}

// ResolveAPIKey returns the API key to hand to the control plane: the CLI flag
// first, then AGENTFIELD_API_KEY, then whatever `af auth login` stored for the
// resolved server URL. It returns "" on a default local setup with no key
// configured anywhere, which is the common case and stays unauthenticated.
func ResolveAPIKey() string {
	if apiKeyOverride != "" {
		return apiKeyOverride
	}
	if env := strings.TrimSpace(os.Getenv("AGENTFIELD_API_KEY")); env != "" {
		return env
	}
	home, err := AgentFieldHomeDir()
	if err != nil {
		return ""
	}
	return NewCredentialStore(home).Lookup(resolveServerURL())
}
