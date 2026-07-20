package packages

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
)

// Secrets layout under the AgentField home directory:
//
//	~/.agentfield/
//	  keyring/master.key      # 0600, random 32 bytes — the local decrypt key
//	  secrets/global.enc      # encrypted JSON map, shared across all nodes
//	  secrets/<node>.enc      # encrypted JSON map, scoped to a single node
//
// Values are encrypted at rest with AES-256-GCM (via encryption.EncryptionService)
// and are only ever decrypted into a child process' environment at start time —
// never written back to disk in plaintext.

const (
	globalScope    = "global"
	masterKeyName  = "master.key"
	secretFilePerm = 0o600
	keyDirPerm     = 0o700
)

// GlobalScope is the exported name of the shared (cross-node) secret scope.
const GlobalScope = globalScope

// KeyProvider supplies the passphrase used to derive the at-rest encryption key.
// It is an interface so we can add OS-keychain / passphrase providers later
// without touching the SecretStore.
type KeyProvider interface {
	// Passphrase returns the secret material used to derive the encryption key.
	Passphrase() (string, error)
}

// KeyfileProvider keeps a random 32-byte key in a 0600 file under keyring/.
// It is generated on first use. This protects secrets from casual reading,
// version control, and process listings, while requiring no prompt.
type KeyfileProvider struct {
	Path string // full path to the key file
}

// NewKeyfileProvider returns a KeyfileProvider rooted at the given AgentField home.
func NewKeyfileProvider(agentFieldHome string) *KeyfileProvider {
	return &KeyfileProvider{Path: filepath.Join(agentFieldHome, "keyring", masterKeyName)}
}

// Passphrase returns the hex-encoded key material, creating it on first use.
func (kp *KeyfileProvider) Passphrase() (string, error) {
	if data, err := os.ReadFile(kp.Path); err == nil {
		key := string(data)
		if key == "" {
			return "", fmt.Errorf("master key file %s is empty", kp.Path)
		}
		return key, nil
	} else if !os.IsNotExist(err) {
		return "", fmt.Errorf("failed to read master key: %w", err)
	}

	// Generate a new random key on first use.
	if err := os.MkdirAll(filepath.Dir(kp.Path), keyDirPerm); err != nil {
		return "", fmt.Errorf("failed to create keyring directory: %w", err)
	}
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("failed to generate master key: %w", err)
	}
	key := hex.EncodeToString(raw)
	if err := os.WriteFile(kp.Path, []byte(key), secretFilePerm); err != nil {
		return "", fmt.Errorf("failed to write master key: %w", err)
	}
	return key, nil
}

// SecretStore reads and writes encrypted secret maps scoped globally or per-node.
type SecretStore struct {
	AgentFieldHome string
	provider       KeyProvider
	enc            *encryption.EncryptionService
}

// NewSecretStore builds a SecretStore using a keyfile provider by default.
func NewSecretStore(agentFieldHome string) (*SecretStore, error) {
	return NewSecretStoreWithProvider(agentFieldHome, NewKeyfileProvider(agentFieldHome))
}

// NewSecretStoreWithProvider builds a SecretStore with a custom key provider.
func NewSecretStoreWithProvider(agentFieldHome string, provider KeyProvider) (*SecretStore, error) {
	passphrase, err := provider.Passphrase()
	if err != nil {
		return nil, err
	}
	return &SecretStore{
		AgentFieldHome: agentFieldHome,
		provider:       provider,
		enc:            encryption.NewEncryptionService(passphrase),
	}, nil
}

// scopeFile returns the encrypted file path for a scope ("global" or a node name).
func (s *SecretStore) scopeFile(scope string) string {
	if scope == "" {
		scope = globalScope
	}
	return filepath.Join(s.AgentFieldHome, "secrets", scope+".enc")
}

// load decrypts a scope's secret map. A missing file yields an empty map.
func (s *SecretStore) load(scope string) (map[string]string, error) {
	path := s.scopeFile(scope)
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return map[string]string{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read secrets %s: %w", scope, err)
	}
	plaintext, err := s.enc.DecryptBytes(data)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt secrets %s (wrong key?): %w", scope, err)
	}
	out := map[string]string{}
	if err := json.Unmarshal(plaintext, &out); err != nil {
		return nil, fmt.Errorf("failed to parse secrets %s: %w", scope, err)
	}
	return out, nil
}

// save encrypts and writes a scope's secret map with 0600 permissions.
func (s *SecretStore) save(scope string, values map[string]string) error {
	path := s.scopeFile(scope)
	if err := os.MkdirAll(filepath.Dir(path), keyDirPerm); err != nil {
		return fmt.Errorf("failed to create secrets directory: %w", err)
	}
	plaintext, err := json.Marshal(values)
	if err != nil {
		return fmt.Errorf("failed to marshal secrets %s: %w", scope, err)
	}
	ciphertext, err := s.enc.EncryptBytes(plaintext)
	if err != nil {
		return fmt.Errorf("failed to encrypt secrets %s: %w", scope, err)
	}
	if err := os.WriteFile(path, ciphertext, secretFilePerm); err != nil {
		return fmt.Errorf("failed to write secrets %s: %w", scope, err)
	}
	return nil
}

// Set stores a secret in the given scope (use "global" or a node name).
func (s *SecretStore) Set(scope, key, value string) error {
	values, err := s.load(scope)
	if err != nil {
		return err
	}
	values[key] = value
	return s.save(scope, values)
}

// DeleteScope removes an entire scope's encrypted file — used when a node is
// uninstalled so its node-scoped secrets don't outlive it. A missing file is
// a no-op. The global scope is shared and never deleted this way.
func (s *SecretStore) DeleteScope(scope string) error {
	if scope == "" || scope == globalScope {
		return fmt.Errorf("refusing to delete the %q scope", scope)
	}
	if err := os.Remove(s.scopeFile(scope)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Delete removes a secret from the given scope. Missing keys are a no-op.
func (s *SecretStore) Delete(scope, key string) error {
	values, err := s.load(scope)
	if err != nil {
		return err
	}
	if _, ok := values[key]; !ok {
		return nil
	}
	delete(values, key)
	return s.save(scope, values)
}

// Get resolves a secret for a node, preferring the node scope over global.
// The second return value reports whether the key was found.
func (s *SecretStore) Get(node, key string) (string, bool, error) {
	if node != "" && node != globalScope {
		nodeVals, err := s.load(node)
		if err != nil {
			return "", false, err
		}
		if v, ok := nodeVals[key]; ok {
			return v, true, nil
		}
	}
	globalVals, err := s.load(globalScope)
	if err != nil {
		return "", false, err
	}
	v, ok := globalVals[key]
	return v, ok, nil
}

// SecretRef identifies a stored secret and the scope it lives in.
type SecretRef struct {
	Key   string
	Scope string
}

// List returns the keys present in a scope (sorted). Values are never returned
// here so callers cannot accidentally print them.
func (s *SecretStore) List(scope string) ([]string, error) {
	values, err := s.load(scope)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys, nil
}

// ListAll returns every stored secret reference across global and all node
// scopes, sorted by scope then key. Values are never returned.
func (s *SecretStore) ListAll() ([]SecretRef, error) {
	dir := filepath.Join(s.AgentFieldHome, "secrets")
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read secrets directory: %w", err)
	}
	var refs []SecretRef
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || filepath.Ext(name) != ".enc" {
			continue
		}
		scope := name[:len(name)-len(".enc")]
		keys, err := s.List(scope)
		if err != nil {
			return nil, err
		}
		for _, k := range keys {
			refs = append(refs, SecretRef{Key: k, Scope: scope})
		}
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].Scope != refs[j].Scope {
			return refs[i].Scope < refs[j].Scope
		}
		return refs[i].Key < refs[j].Key
	})
	return refs, nil
}

// MaskSecret returns a display-safe rendering of a secret value.
func MaskSecret(value string) string {
	if len(value) <= 4 {
		return "****"
	}
	return value[:2] + "****" + value[len(value)-2:]
}
