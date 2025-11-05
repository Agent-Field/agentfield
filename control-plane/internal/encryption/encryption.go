package encryption

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
)

// EncryptionService provides encryption and decryption for sensitive configuration values
type EncryptionService struct {
	key []byte
}

// NewEncryptionService creates a new encryption service with a derived key
func NewEncryptionService(passphrase string) *EncryptionService {
	// Derive a 32-byte key from the passphrase using SHA-256
	hash := sha256.Sum256([]byte(passphrase))
	return &EncryptionService{
		key: hash[:],
	}
}

// Encrypt encrypts a plaintext string and returns a base64-encoded ciphertext
func (es *EncryptionService) Encrypt(plaintext string) (string, error) {
	if plaintext == "" {
		return "", nil
	}

	// Create AES cipher
	block, err := aes.NewCipher(es.key)
	if err != nil {
		return "", fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("failed to create GCM: %w", err)
	}

	// Generate a random nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt the plaintext
	ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)

	// Return base64-encoded ciphertext
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// Decrypt decrypts a base64-encoded ciphertext and returns the plaintext
func (es *EncryptionService) Decrypt(ciphertext string) (string, error) {
	if ciphertext == "" {
		return "", nil
	}

	// Decode base64
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64: %w", err)
	}

	// Create AES cipher
	block, err := aes.NewCipher(es.key)
	if err != nil {
		return "", fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("failed to create GCM: %w", err)
	}

	// Check minimum length
	nonceSize := gcm.NonceSize()
	if len(data) < nonceSize {
		return "", fmt.Errorf("ciphertext too short")
	}

	// Extract nonce and encrypted data
	nonce, encryptedData := data[:nonceSize], data[nonceSize:]

	// Decrypt
	plaintext, err := gcm.Open(nil, nonce, encryptedData, nil)
	if err != nil {
		return "", fmt.Errorf("failed to decrypt: %w", err)
	}

	return string(plaintext), nil
}

// EncryptConfigurationValues encrypts sensitive values in a configuration map
func (es *EncryptionService) EncryptConfigurationValues(config map[string]interface{}, secretFields []string) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Copy all values
	for key, value := range config {
		result[key] = value
	}

	// Encrypt secret fields
	for _, field := range secretFields {
		if value, exists := result[field]; exists {
			if strValue, ok := value.(string); ok {
				encrypted, err := es.Encrypt(strValue)
				if err != nil {
					return nil, fmt.Errorf("failed to encrypt field '%s': %w", field, err)
				}
				result[field] = encrypted
			}
		}
	}

	return result, nil
}

// DecryptConfigurationValues decrypts sensitive values in a configuration map
func (es *EncryptionService) DecryptConfigurationValues(config map[string]interface{}, secretFields []string) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Copy all values
	for key, value := range config {
		result[key] = value
	}

	// Decrypt secret fields
	for _, field := range secretFields {
		if value, exists := result[field]; exists {
			if strValue, ok := value.(string); ok {
				decrypted, err := es.Decrypt(strValue)
				if err != nil {
					return nil, fmt.Errorf("failed to decrypt field '%s': %w", field, err)
				}
				result[field] = decrypted
			}
		}
	}

	return result, nil
}
