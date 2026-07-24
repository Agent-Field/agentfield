package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestStore(t *testing.T) (*SecretStore, string) {
	t.Helper()
	home := t.TempDir()
	store, err := NewSecretStore(home)
	if err != nil {
		t.Fatalf("NewSecretStore: %v", err)
	}
	return store, home
}

// Contract: a secret set in a scope round-trips back out unchanged.
func TestSecretStore_SetGetRoundtrip(t *testing.T) {
	store, _ := newTestStore(t)

	if err := store.Set(globalScope, "OPENROUTER_API_KEY", "sk-abc123"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, ok, err := store.Get("", "OPENROUTER_API_KEY")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !ok || got != "sk-abc123" {
		t.Fatalf("got (%q, %v), want (sk-abc123, true)", got, ok)
	}
}

// Contract: a node-scoped secret overrides the global one for that node,
// while other nodes still see the global value.
func TestSecretStore_NodeOverridesGlobal(t *testing.T) {
	store, _ := newTestStore(t)

	if err := store.Set(globalScope, "MODEL", "global-model"); err != nil {
		t.Fatalf("Set global: %v", err)
	}
	if err := store.Set("pr-af", "MODEL", "node-model"); err != nil {
		t.Fatalf("Set node: %v", err)
	}

	got, _, _ := store.Get("pr-af", "MODEL")
	if got != "node-model" {
		t.Fatalf("pr-af MODEL = %q, want node-model", got)
	}
	got, _, _ = store.Get("sec-af", "MODEL")
	if got != "global-model" {
		t.Fatalf("sec-af MODEL = %q, want global-model (fallback)", got)
	}
}

// Contract: deleting a secret removes it; missing keys are a no-op.
func TestSecretStore_Delete(t *testing.T) {
	store, _ := newTestStore(t)
	_ = store.Set(globalScope, "GH_TOKEN", "ghp_xxx")

	if err := store.Delete(globalScope, "GH_TOKEN"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, ok, _ := store.Get("", "GH_TOKEN"); ok {
		t.Fatalf("GH_TOKEN still present after delete")
	}
	if err := store.Delete(globalScope, "GH_TOKEN"); err != nil {
		t.Fatalf("Delete of missing key should be no-op, got %v", err)
	}
}

// Contract: secret files and the master key are written with 0600 perms,
// and the keyring directory with 0700.
func TestSecretStore_FilePermissions(t *testing.T) {
	store, home := newTestStore(t)
	_ = store.Set(globalScope, "K", "v")

	checks := map[string]os.FileMode{
		filepath.Join(home, "keyring", masterKeyName): 0o600,
		filepath.Join(home, "secrets", "global.enc"):  0o600,
	}
	for path, want := range checks {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
		if info.Mode().Perm() != want {
			t.Errorf("%s perm = %o, want %o", path, info.Mode().Perm(), want)
		}
	}
}

// Contract: the value never appears in plaintext anywhere on disk.
func TestSecretStore_NoPlaintextOnDisk(t *testing.T) {
	store, home := newTestStore(t)
	const secret = "super-secret-value-9f3a"
	_ = store.Set(globalScope, "API_KEY", secret)

	err := filepath.Walk(home, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		data, _ := os.ReadFile(path)
		if strings.Contains(string(data), secret) {
			t.Errorf("plaintext secret found in %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
}

// Contract: a store built from the same home (same keyfile) can decrypt
// secrets written by an earlier instance.
func TestSecretStore_PersistsAcrossInstances(t *testing.T) {
	home := t.TempDir()
	first, err := NewSecretStore(home)
	if err != nil {
		t.Fatalf("first store: %v", err)
	}
	_ = first.Set(globalScope, "TOKEN", "abc")

	second, err := NewSecretStore(home)
	if err != nil {
		t.Fatalf("second store: %v", err)
	}
	if got, ok, _ := second.Get("", "TOKEN"); !ok || got != "abc" {
		t.Fatalf("second store Get = (%q,%v), want (abc,true)", got, ok)
	}
}

// Contract: List returns keys only (never values), sorted; ListAll spans scopes.
func TestSecretStore_ListMasksValues(t *testing.T) {
	store, _ := newTestStore(t)
	_ = store.Set(globalScope, "B_KEY", "v1")
	_ = store.Set(globalScope, "A_KEY", "v2")
	_ = store.Set("pr-af", "N_KEY", "v3")

	keys, err := store.List(globalScope)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(keys) != 2 || keys[0] != "A_KEY" || keys[1] != "B_KEY" {
		t.Fatalf("List = %v, want sorted [A_KEY B_KEY]", keys)
	}

	all, err := store.ListAll()
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("ListAll len = %d, want 3", len(all))
	}
}

func TestMaskSecret(t *testing.T) {
	cases := map[string]string{
		"":              "****",
		"abcd":          "****",
		"sk-abcdef1234": "sk****34",
	}
	for in, want := range cases {
		if got := MaskSecret(in); got != want {
			t.Errorf("MaskSecret(%q) = %q, want %q", in, got, want)
		}
	}
}

// Contract: a wrong key cannot decrypt another store's secrets.
func TestSecretStore_WrongKeyFails(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStore(home)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	_ = store.Set(globalScope, "K", "v")

	// Replace the keyfile with a different key, then try to read.
	if err := os.WriteFile(filepath.Join(home, "keyring", masterKeyName),
		[]byte("00deadbeef00deadbeef00deadbeef00deadbeef00deadbeef00deadbeef0000"), 0o600); err != nil {
		t.Fatalf("rewrite key: %v", err)
	}
	tampered, err := NewSecretStore(home)
	if err != nil {
		t.Fatalf("tampered store: %v", err)
	}
	if _, _, err := tampered.Get("", "K"); err == nil {
		t.Fatalf("expected decryption to fail with wrong key")
	}
}
