package packages

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func newTestCredentialStore(t *testing.T) (*CredentialStore, string) {
	t.Helper()
	home := t.TempDir()
	return NewCredentialStore(home), home
}

// Contract: URLs that name the same control plane resolve to the same entry,
// so `af auth login --server http://localhost:8080/` is found by a later
// command that resolved the server as http://localhost:8080.
func TestNormalizeServerURL(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "already canonical", in: "http://localhost:8080", want: "http://localhost:8080"},
		{name: "trailing slash", in: "http://localhost:8080/", want: "http://localhost:8080"},
		{name: "repeated trailing slashes", in: "http://localhost:8080///", want: "http://localhost:8080"},
		{name: "surrounding whitespace", in: "  http://localhost:8080 ", want: "http://localhost:8080"},
		{name: "uppercase scheme and host", in: "HTTP://LocalHost:8080", want: "http://localhost:8080"},
		{name: "path preserved", in: "https://cp.example.com/agentfield/", want: "https://cp.example.com/agentfield"},
		{name: "query and fragment dropped", in: "https://cp.example.com?x=1#frag", want: "https://cp.example.com"},
		{name: "empty", in: "   ", want: ""},
		{name: "unparseable falls back to lowercase", in: "LocalHost:8080", want: "localhost:8080"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := NormalizeServerURL(tc.in); got != tc.want {
				t.Fatalf("NormalizeServerURL(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// Contract: a key stored for a server comes back for any spelling of that
// server's URL, and disappears once deleted.
func TestCredentialStore_SaveLookupDeleteRoundtrip(t *testing.T) {
	store, _ := newTestCredentialStore(t)

	if err := store.Save("http://localhost:8080/", "af_live_secret_value"); err != nil {
		t.Fatalf("Save: %v", err)
	}

	for _, spelling := range []string{
		"http://localhost:8080",
		"http://localhost:8080/",
		"HTTP://LOCALHOST:8080",
	} {
		if got := store.Lookup(spelling); got != "af_live_secret_value" {
			t.Fatalf("Lookup(%q) = %q, want the stored key", spelling, got)
		}
	}

	removed, err := store.Delete("http://localhost:8080")
	if err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if !removed {
		t.Fatal("Delete reported no entry, want removed=true")
	}
	if got := store.Lookup("http://localhost:8080"); got != "" {
		t.Fatalf("Lookup after Delete = %q, want empty", got)
	}

	removed, err = store.Delete("http://localhost:8080")
	if err != nil {
		t.Fatalf("Delete (second): %v", err)
	}
	if removed {
		t.Fatal("Delete reported removed=true for an entry that no longer exists")
	}
}

// Contract: entries for different control planes do not overwrite each other.
func TestCredentialStore_SeparateEntriesPerServer(t *testing.T) {
	store, _ := newTestCredentialStore(t)

	if err := store.Save("http://localhost:8080", "local-key"); err != nil {
		t.Fatalf("Save local: %v", err)
	}
	if err := store.Save("https://cp.example.com", "remote-key"); err != nil {
		t.Fatalf("Save remote: %v", err)
	}

	if got := store.Lookup("http://localhost:8080"); got != "local-key" {
		t.Fatalf("local lookup = %q, want local-key", got)
	}
	if got := store.Lookup("https://cp.example.com"); got != "remote-key" {
		t.Fatalf("remote lookup = %q, want remote-key", got)
	}

	servers, err := store.Servers()
	if err != nil {
		t.Fatalf("Servers: %v", err)
	}
	want := []string{"http://localhost:8080", "https://cp.example.com"}
	if len(servers) != len(want) {
		t.Fatalf("Servers() = %v, want %v", servers, want)
	}
	for i, server := range want {
		if servers[i] != server {
			t.Fatalf("Servers() = %v, want %v", servers, want)
		}
	}
}

// Contract: the credentials file is only readable by its owner. It holds a
// replayable API key in plaintext, so a group- or world-readable file is a
// leak, including when an existing file is rewritten.
func TestCredentialStore_FileIsOwnerReadableOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix file modes are not meaningful on windows")
	}
	store, home := newTestCredentialStore(t)

	if err := store.Save("http://localhost:8080", "first-key"); err != nil {
		t.Fatalf("Save: %v", err)
	}
	path := filepath.Join(home, credentialsFileName)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("credentials file mode = %o, want 600", perm)
	}

	// A file that was loosened by hand is tightened on the next write rather
	// than left as found.
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatalf("Chmod: %v", err)
	}
	if err := store.Save("http://localhost:8080", "second-key"); err != nil {
		t.Fatalf("Save (rewrite): %v", err)
	}
	info, err = os.Stat(path)
	if err != nil {
		t.Fatalf("Stat after rewrite: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("credentials file mode after rewrite = %o, want 600", perm)
	}
}

// Contract: a machine that never ran `af auth login` behaves exactly as before
// — reads succeed, report no credential, and create nothing on disk.
func TestCredentialStore_MissingFileIsNotAnError(t *testing.T) {
	store, home := newTestCredentialStore(t)

	creds, err := store.Load()
	if err != nil {
		t.Fatalf("Load on a missing file: %v", err)
	}
	if len(creds.Servers) != 0 {
		t.Fatalf("Load returned %d servers, want 0", len(creds.Servers))
	}
	if got := store.Lookup("http://localhost:8080"); got != "" {
		t.Fatalf("Lookup = %q, want empty", got)
	}
	if removed, err := store.Delete("http://localhost:8080"); err != nil || removed {
		t.Fatalf("Delete on a missing file = (%v, %v), want (false, nil)", removed, err)
	}

	entries, err := os.ReadDir(home)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("reading credentials created %v; the home directory must stay untouched", entries)
	}
}

// Contract: an empty or corrupt file degrades to "no credential" instead of
// breaking every command that resolves a key.
func TestCredentialStore_UnusableFileResolvesToNoKey(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
	}{
		{name: "empty file", content: "", wantErr: false},
		{name: "whitespace only", content: "  \n", wantErr: false},
		{name: "corrupt json", content: "{not json", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store, home := newTestCredentialStore(t)
			path := filepath.Join(home, credentialsFileName)
			if err := os.WriteFile(path, []byte(tc.content), 0o600); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}

			if _, err := store.Load(); (err != nil) != tc.wantErr {
				t.Fatalf("Load error = %v, wantErr %v", err, tc.wantErr)
			}
			if got := store.Lookup("http://localhost:8080"); got != "" {
				t.Fatalf("Lookup = %q, want empty", got)
			}
		})
	}
}

// Contract: an empty key or an empty server URL is rejected, so a stored entry
// is never a placeholder that silently sends nothing.
func TestCredentialStore_SaveRejectsEmptyInput(t *testing.T) {
	store, _ := newTestCredentialStore(t)

	if err := store.Save("", "key"); err == nil {
		t.Fatal("Save with an empty server URL succeeded, want an error")
	}
	if err := store.Save("http://localhost:8080", "   "); err == nil {
		t.Fatal("Save with an empty key succeeded, want an error")
	}
}

// Contract: the key handed to a spawned agent node follows the same precedence
// as the CLI's own requests — flag, then environment, then stored credential,
// then nothing at all on a default local setup.
func TestResolveAPIKey_Precedence(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")
	t.Setenv("AGENTFIELD_API_KEY", "")
	t.Cleanup(func() { SetAPIKeyOverride("") })

	SetAPIKeyOverride("")
	if got := ResolveAPIKey(); got != "" {
		t.Fatalf("ResolveAPIKey with nothing configured = %q, want empty", got)
	}

	if err := NewCredentialStore(home).Save("http://localhost:8080", "stored-key"); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if got := ResolveAPIKey(); got != "stored-key" {
		t.Fatalf("ResolveAPIKey with a stored credential = %q, want stored-key", got)
	}

	t.Setenv("AGENTFIELD_API_KEY", "env-key")
	if got := ResolveAPIKey(); got != "env-key" {
		t.Fatalf("ResolveAPIKey with AGENTFIELD_API_KEY set = %q, want env-key", got)
	}

	SetAPIKeyOverride("flag-key")
	if got := ResolveAPIKey(); got != "flag-key" {
		t.Fatalf("ResolveAPIKey with a flag override = %q, want flag-key", got)
	}
}

// Contract: a stored credential is scoped to the control plane it was stored
// for; pointing the CLI at a different server does not reuse it.
func TestResolveAPIKey_ScopedToServer(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_API_KEY", "")
	t.Cleanup(func() { SetAPIKeyOverride("") })
	SetAPIKeyOverride("")

	if err := NewCredentialStore(home).Save("http://localhost:8080", "local-key"); err != nil {
		t.Fatalf("Save: %v", err)
	}

	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")
	if got := ResolveAPIKey(); got != "local-key" {
		t.Fatalf("ResolveAPIKey for the stored server = %q, want local-key", got)
	}

	t.Setenv("AGENTFIELD_SERVER", "https://other.example.com")
	if got := ResolveAPIKey(); got != "" {
		t.Fatalf("ResolveAPIKey for a different server = %q, want empty", got)
	}
}

// Contract: AGENTFIELD_HOME steers credential storage, so an isolated test or
// desktop instance never reads or writes the real ~/.agentfield.
func TestAgentFieldHomeDir_HonoursOverride(t *testing.T) {
	custom := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", custom)

	got, err := AgentFieldHomeDir()
	if err != nil {
		t.Fatalf("AgentFieldHomeDir: %v", err)
	}
	if got != custom {
		t.Fatalf("AgentFieldHomeDir = %q, want %q", got, custom)
	}
}
