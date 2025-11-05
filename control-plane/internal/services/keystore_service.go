package services

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
)

// KeystoreService handles secure storage and management of cryptographic keys.
type KeystoreService struct {
	config *config.KeystoreConfig
	gcm    cipher.AEAD
}

// NewKeystoreService creates a new keystore service instance.
func NewKeystoreService(cfg *config.KeystoreConfig) (*KeystoreService, error) {
	// For now, use a simple AES-GCM encryption with a fixed key
	// In production, this should use proper key derivation and HSM integration
	key := make([]byte, 32) // 256-bit key
	if _, err := rand.Read(key); err != nil {
		return nil, fmt.Errorf("failed to generate encryption key: %w", err)
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	// Ensure keystore directory exists
	if err := os.MkdirAll(cfg.Path, 0700); err != nil {
		return nil, fmt.Errorf("failed to create keystore directory: %w", err)
	}

	return &KeystoreService{
		config: cfg,
		gcm:    gcm,
	}, nil
}

// StoreKey stores a key securely in the keystore.
func (ks *KeystoreService) StoreKey(keyID string, keyData []byte) error {
	if ks.config.Type != "local" {
		return fmt.Errorf("only local keystore is currently supported")
	}

	// Encrypt the key data
	nonce := make([]byte, ks.gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := ks.gcm.Seal(nonce, nonce, keyData, nil)

	// Store encrypted key to file
	keyPath := filepath.Join(ks.config.Path, keyID+".key")
	if err := os.WriteFile(keyPath, ciphertext, 0600); err != nil {
		return fmt.Errorf("failed to write key file: %w", err)
	}

	return nil
}

// RetrieveKey retrieves a key from the keystore.
func (ks *KeystoreService) RetrieveKey(keyID string) ([]byte, error) {
	if ks.config.Type != "local" {
		return nil, fmt.Errorf("only local keystore is currently supported")
	}

	// Read encrypted key from file
	keyPath := filepath.Join(ks.config.Path, keyID+".key")
	ciphertext, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read key file: %w", err)
	}

	// Extract nonce and decrypt
	nonceSize := ks.gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	plaintext, err := ks.gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt key: %w", err)
	}

	return plaintext, nil
}

// DeleteKey deletes a key from the keystore.
func (ks *KeystoreService) DeleteKey(keyID string) error {
	if ks.config.Type != "local" {
		return fmt.Errorf("only local keystore is currently supported")
	}

	keyPath := filepath.Join(ks.config.Path, keyID+".key")
	if err := os.Remove(keyPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete key file: %w", err)
	}

	return nil
}

// ListKeys lists all keys in the keystore.
func (ks *KeystoreService) ListKeys() ([]string, error) {
	if ks.config.Type != "local" {
		return nil, fmt.Errorf("only local keystore is currently supported")
	}

	entries, err := os.ReadDir(ks.config.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to read keystore directory: %w", err)
	}

	var keys []string
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".key" {
			keyID := entry.Name()[:len(entry.Name())-4] // Remove .key extension
			keys = append(keys, keyID)
		}
	}

	return keys, nil
}

// BackupKeys creates a backup of all keys in the keystore.
func (ks *KeystoreService) BackupKeys() error {
	if !ks.config.BackupEnabled {
		return nil
	}

	// Implementation for key backup
	// This would typically involve creating encrypted backups
	// For now, just return nil
	return nil
}

// EncryptData encrypts arbitrary data using the keystore's encryption.
func (ks *KeystoreService) EncryptData(data []byte) ([]byte, error) {
	nonce := make([]byte, ks.gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := ks.gcm.Seal(nonce, nonce, data, nil)
	return ciphertext, nil
}

// DecryptData decrypts data that was encrypted with EncryptData.
func (ks *KeystoreService) DecryptData(ciphertext []byte) ([]byte, error) {
	nonceSize := ks.gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	plaintext, err := ks.gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return plaintext, nil
}
