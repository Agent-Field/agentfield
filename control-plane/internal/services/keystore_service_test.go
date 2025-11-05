package services

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/your-org/agentfield/control-plane/internal/config"

	"github.com/stretchr/testify/require"
)

func TestKeystoreServiceLocalLifecycle(t *testing.T) {
	t.Parallel()

	keystoreDir := t.TempDir()
	svc, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	keyID := "agent-secret"
	payload := []byte("super-secret")

	require.NoError(t, svc.StoreKey(keyID, payload))

	storedPath := filepath.Join(keystoreDir, keyID+".key")
	fileContents, err := os.ReadFile(storedPath)
	require.NoError(t, err)
	require.NotEqual(t, payload, fileContents, "data should be encrypted on disk")

	retrieved, err := svc.RetrieveKey(keyID)
	require.NoError(t, err)
	require.Equal(t, payload, retrieved)

	keys, err := svc.ListKeys()
	require.NoError(t, err)
	require.ElementsMatch(t, []string{keyID}, keys)

	encrypted, err := svc.EncryptData([]byte("plaintext"))
	require.NoError(t, err)
	require.False(t, bytes.Equal(encrypted, []byte("plaintext")))

	decrypted, err := svc.DecryptData(encrypted)
	require.NoError(t, err)
	require.Equal(t, []byte("plaintext"), decrypted)

	require.NoError(t, svc.DeleteKey(keyID))
	require.NoError(t, svc.DeleteKey(keyID))

	keys, err = svc.ListKeys()
	require.NoError(t, err)
	require.Empty(t, keys)

	require.NoError(t, svc.BackupKeys())
}

func TestKeystoreServiceRejectsNonLocal(t *testing.T) {
	t.Parallel()

	svc, err := NewKeystoreService(&config.KeystoreConfig{Path: t.TempDir(), Type: "local"})
	require.NoError(t, err)

	svc.config.Type = "remote"
	require.Error(t, svc.StoreKey("id", []byte("data")))
	_, err = svc.RetrieveKey("id")
	require.Error(t, err)
	require.Error(t, svc.DeleteKey("id"))
	_, err = svc.ListKeys()
	require.Error(t, err)
}
