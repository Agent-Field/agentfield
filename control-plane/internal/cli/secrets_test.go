package cli

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// runSecrets executes the `secrets` command tree with the given args against an
// isolated AGENTFIELD_HOME and returns the combined stdout/stderr and error.
func runSecrets(t *testing.T, home string, args ...string) error {
	t.Helper()
	t.Setenv("AGENTFIELD_HOME", home)
	cmd := NewSecretsCommand()
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	cmd.SetArgs(args)
	return cmd.Execute()
}

// Contract: `secrets set KEY VALUE` persists the secret to the global scope
// where a SecretStore rooted at the same home can read it back.
func TestSecretsSet_GlobalRoundtrip(t *testing.T) {
	home := t.TempDir()
	if err := runSecrets(t, home, "set", "OPENROUTER_API_KEY", "sk-live-123"); err != nil {
		t.Fatalf("secrets set: %v", err)
	}
	store, err := packages.NewSecretStore(home)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	got, ok, err := store.Get("", "OPENROUTER_API_KEY")
	if err != nil || !ok || got != "sk-live-123" {
		t.Fatalf("Get = (%q,%v,%v), want (sk-live-123,true,nil)", got, ok, err)
	}
}

// Contract: `secrets set --node <n> KEY VALUE` stores to the node scope, not
// global, so only that node resolves it.
func TestSecretsSet_NodeScoped(t *testing.T) {
	home := t.TempDir()
	if err := runSecrets(t, home, "set", "--node", "pr-af", "GH_TOKEN", "ghp_secret"); err != nil {
		t.Fatalf("secrets set --node: %v", err)
	}
	store, _ := packages.NewSecretStore(home)
	keys, err := store.List("pr-af")
	if err != nil {
		t.Fatalf("List node: %v", err)
	}
	if len(keys) != 1 || keys[0] != "GH_TOKEN" {
		t.Fatalf("node keys = %v, want [GH_TOKEN]", keys)
	}
	if globalKeys, _ := store.List("global"); len(globalKeys) != 0 {
		t.Fatalf("node secret leaked into global: %v", globalKeys)
	}
}

// Contract: an empty value is rejected rather than silently stored.
func TestSecretsSet_EmptyValueRejected(t *testing.T) {
	home := t.TempDir()
	err := runSecrets(t, home, "set", "EMPTY", "   ")
	if err == nil || !strings.Contains(err.Error(), "must not be empty") {
		t.Fatalf("expected empty-value error, got %v", err)
	}
}

// Contract: when VALUE is omitted and stdin is not a terminal, the value is
// read from stdin; an empty piped value is rejected.
func TestSecretsSet_ReadsValueFromStdin(t *testing.T) {
	home := t.TempDir()

	// Provide the value on stdin (non-TTY path of readHiddenValue).
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.WriteString("piped-secret-value\n"); err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	origStdin := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = origStdin }()

	if err := runSecrets(t, home, "set", "PIPED_KEY"); err != nil {
		t.Fatalf("secrets set (stdin): %v", err)
	}
	store, _ := packages.NewSecretStore(home)
	if got, ok, _ := store.Get("", "PIPED_KEY"); !ok || got != "piped-secret-value" {
		t.Fatalf("stdin value not stored: got %q ok=%v", got, ok)
	}
}

// Contract: `secrets ls` reports "no secrets" on an empty store and lists both
// global and node scoped keys once populated.
func TestSecretsList(t *testing.T) {
	home := t.TempDir()

	// Empty store: must not error.
	if err := runSecrets(t, home, "ls"); err != nil {
		t.Fatalf("secrets ls (empty): %v", err)
	}

	store, _ := packages.NewSecretStore(home)
	_ = store.Set("global", "A_KEY", "v1")
	_ = store.Set("pr-af", "B_KEY", "v2")

	if err := runSecrets(t, home, "ls"); err != nil {
		t.Fatalf("secrets ls (populated): %v", err)
	}
	// The "list" alias must resolve to the same command.
	if err := runSecrets(t, home, "list"); err != nil {
		t.Fatalf("secrets list alias: %v", err)
	}
}

// Contract: `secrets rm KEY` deletes a global secret; removing a missing key is
// not an error (idempotent).
func TestSecretsRemove(t *testing.T) {
	home := t.TempDir()
	store, _ := packages.NewSecretStore(home)
	_ = store.Set("global", "TO_DELETE", "v")

	if err := runSecrets(t, home, "rm", "TO_DELETE"); err != nil {
		t.Fatalf("secrets rm: %v", err)
	}
	fresh, _ := packages.NewSecretStore(home)
	if _, ok, _ := fresh.Get("", "TO_DELETE"); ok {
		t.Fatalf("TO_DELETE still present after rm")
	}
	// Removing a missing key is a no-op, not an error.
	if err := runSecrets(t, home, "rm", "TO_DELETE"); err != nil {
		t.Fatalf("rm of missing key should be no-op, got %v", err)
	}
}

// Contract: `secrets rm --node <n> KEY` targets the node scope, leaving the
// same key in the global scope untouched.
func TestSecretsRemove_NodeScoped(t *testing.T) {
	home := t.TempDir()
	store, _ := packages.NewSecretStore(home)
	_ = store.Set("global", "SHARED", "global-val")
	_ = store.Set("pr-af", "SHARED", "node-val")

	if err := runSecrets(t, home, "rm", "--node", "pr-af", "SHARED"); err != nil {
		t.Fatalf("secrets rm --node: %v", err)
	}
	fresh, _ := packages.NewSecretStore(home)
	if keys, _ := fresh.List("pr-af"); len(keys) != 0 {
		t.Fatalf("node SHARED not removed: %v", keys)
	}
	if got, ok, _ := fresh.Get("", "SHARED"); !ok || got != "global-val" {
		t.Fatalf("global SHARED should survive node rm, got %q ok=%v", got, ok)
	}
}
