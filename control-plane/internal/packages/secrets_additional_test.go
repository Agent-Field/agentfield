package packages

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// fixedProvider supplies a constant passphrase without touching a keyfile, so
// SecretStore error paths can be exercised without the keyring on disk.
type fixedProvider struct {
	pass string
	err  error
}

func (f fixedProvider) Passphrase() (string, error) { return f.pass, f.err }

// skipIfRoot skips permission-dependent tests when running as root, where the
// filesystem mode bits used to force failures are ignored.
func skipIfRoot(t *testing.T) {
	t.Helper()
	if os.Geteuid() == 0 {
		t.Skip("permission-based failure cannot be forced as root")
	}
}

// Contract: an empty master key file is rejected rather than used as a key.
func TestKeyfileProvider_EmptyKeyFileRejected(t *testing.T) {
	home := t.TempDir()
	keyPath := filepath.Join(home, "keyring", "master.key")
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	kp := NewKeyfileProvider(home)
	if _, err := kp.Passphrase(); err == nil {
		t.Fatalf("expected error for empty master key file")
	}
}

// Contract: a keyfile path that is a directory (unreadable as a file) surfaces
// a read error rather than silently regenerating a key.
func TestKeyfileProvider_UnreadableKeyPath(t *testing.T) {
	home := t.TempDir()
	keyPath := filepath.Join(home, "keyring", "master.key")
	if err := os.MkdirAll(keyPath, 0o700); err != nil { // key path is a directory
		t.Fatal(err)
	}
	kp := &KeyfileProvider{Path: keyPath}
	if _, err := kp.Passphrase(); err == nil {
		t.Fatalf("expected read error when key path is a directory")
	}
}

// Contract: the first use of a provider generates and persists a random key
// that a later provider reads back byte-for-byte.
func TestKeyfileProvider_GeneratesAndPersists(t *testing.T) {
	home := t.TempDir()
	kp := NewKeyfileProvider(home)
	first, err := kp.Passphrase()
	if err != nil {
		t.Fatalf("first Passphrase: %v", err)
	}
	if first == "" {
		t.Fatal("generated passphrase is empty")
	}
	second, err := NewKeyfileProvider(home).Passphrase()
	if err != nil {
		t.Fatalf("second Passphrase: %v", err)
	}
	if first != second {
		t.Fatalf("passphrase not stable across reads: %q vs %q", first, second)
	}
}

// Contract: a provider whose key material cannot be created (unwritable keyring
// parent) fails the store construction instead of proceeding without a key.
func TestKeyfileProvider_UnwritableKeyringFails(t *testing.T) {
	skipIfRoot(t)
	parent := t.TempDir()
	if err := os.Chmod(parent, 0o500); err != nil { // read+execute only
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(parent, 0o700) }()

	home := filepath.Join(parent, "home")
	kp := NewKeyfileProvider(home)
	if _, err := kp.Passphrase(); err == nil {
		t.Fatalf("expected key generation to fail under an unwritable parent")
	}
}

// Contract: NewSecretStoreWithProvider propagates a provider error instead of
// returning an unusable store.
func TestNewSecretStore_ProviderError(t *testing.T) {
	sentinel := errors.New("provider boom")
	_, err := NewSecretStoreWithProvider(t.TempDir(), fixedProvider{err: sentinel})
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected provider error to propagate, got %v", err)
	}
}

// Contract: an empty scope name resolves to the global scope, so a value set
// with "" is readable via the global scope and vice versa.
func TestSecretStore_EmptyScopeIsGlobal(t *testing.T) {
	store, err := NewSecretStoreWithProvider(t.TempDir(), fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	if err := store.Set("", "K", "v"); err != nil {
		t.Fatalf("Set empty scope: %v", err)
	}
	keys, err := store.List("global")
	if err != nil {
		t.Fatalf("List global: %v", err)
	}
	if len(keys) != 1 || keys[0] != "K" {
		t.Fatalf("empty-scope write not visible in global scope: %v", keys)
	}
}

// storeWithDirScope returns a store whose given scope file is actually a
// directory, forcing every load of that scope to fail with a read error.
func storeWithDirScope(t *testing.T, scope string) *SecretStore {
	t.Helper()
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	scopePath := filepath.Join(home, "secrets", scope+".enc")
	if err := os.MkdirAll(scopePath, 0o700); err != nil {
		t.Fatal(err)
	}
	return store
}

// Contract: Set/Delete/List/Get surface the underlying read error when a
// scope's backing file cannot be loaded.
func TestSecretStore_LoadErrorsPropagate(t *testing.T) {
	t.Run("Set", func(t *testing.T) {
		s := storeWithDirScope(t, "global")
		if err := s.Set("global", "K", "v"); err == nil {
			t.Fatal("expected Set to fail on unreadable scope")
		}
	})
	t.Run("Delete", func(t *testing.T) {
		s := storeWithDirScope(t, "global")
		if err := s.Delete("global", "K"); err == nil {
			t.Fatal("expected Delete to fail on unreadable scope")
		}
	})
	t.Run("List", func(t *testing.T) {
		s := storeWithDirScope(t, "global")
		if _, err := s.List("global"); err == nil {
			t.Fatal("expected List to fail on unreadable scope")
		}
	})
	t.Run("Get-node-scope", func(t *testing.T) {
		s := storeWithDirScope(t, "pr-af")
		if _, _, err := s.Get("pr-af", "K"); err == nil {
			t.Fatal("expected Get to fail on unreadable node scope")
		}
	})
}

// Contract: writing to a scope fails cleanly when the destination path is not a
// regular file (here, the .enc target is a directory).
func TestSecretStore_SaveWriteError(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	// A fresh scope whose .enc file does not yet exist loads as an empty map,
	// so save() reaches its WriteFile — which then fails because the secrets
	// directory is not writable.
	skipIfRoot(t)
	secretsDir := filepath.Join(home, "secrets")
	if err := os.MkdirAll(secretsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(secretsDir, 0o500); err != nil { // read+execute only
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(secretsDir, 0o700) }()

	if err := store.Set("newscope", "K", "v"); err == nil {
		t.Fatalf("expected Set to fail writing into an unwritable secrets dir")
	}
}

// Contract: ListAll returns nil (no error) when no secrets have ever been
// written, skips non-.enc files, and spans multiple scopes otherwise.
func TestSecretStore_ListAll(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}

	// No secrets dir yet.
	refs, err := store.ListAll()
	if err != nil || refs != nil {
		t.Fatalf("ListAll on empty store = (%v,%v), want (nil,nil)", refs, err)
	}

	_ = store.Set("global", "G", "1")
	_ = store.Set("pr-af", "N", "2")

	// A stray non-.enc file and a subdirectory must be ignored.
	secretsDir := filepath.Join(home, "secrets")
	if err := os.WriteFile(filepath.Join(secretsDir, "README.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(secretsDir, "sub.enc"), 0o700); err != nil {
		t.Fatal(err)
	}

	refs, err = store.ListAll()
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("ListAll = %v, want 2 refs (global/G, pr-af/N)", refs)
	}
	// Sorted by scope then key: global before pr-af.
	if refs[0].Scope != "global" || refs[0].Key != "G" || refs[1].Scope != "pr-af" {
		t.Fatalf("ListAll not sorted by scope/key: %v", refs)
	}
}

// Contract: ListAll surfaces a decrypt error when one scope's .enc file is
// corrupt rather than silently dropping the scope.
func TestSecretStore_ListAllDecryptError(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	_ = store.Set("global", "G", "1")

	// Corrupt a second scope with non-decryptable bytes.
	secretsDir := filepath.Join(home, "secrets")
	if err := os.WriteFile(filepath.Join(secretsDir, "broken.enc"), []byte("not-ciphertext"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ListAll(); err == nil {
		t.Fatalf("expected ListAll to fail on a corrupt scope file")
	}
}

// Contract: ListAll surfaces the directory read error when the secrets path is
// occupied by a regular file instead of a directory.
func TestSecretStore_ListAllReadDirError(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	// Occupy the secrets path with a file.
	if err := os.WriteFile(filepath.Join(home, "secrets"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ListAll(); err == nil {
		t.Fatalf("expected ListAll to fail when secrets path is a file")
	}
}
