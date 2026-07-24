package encryption

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"strings"

	"golang.org/x/crypto/pbkdf2"
)

const (
	encryptionStringVersion = "v2"
	encryptionBinaryMagic   = "AFENC2"
	encryptionSaltSize      = 16
	encryptionKeySize       = 32
	encryptionPBKDF2Rounds  = 600000
)

// EncryptionService provides encryption and decryption for sensitive configuration values
type EncryptionService struct {
	passphrase []byte
}

// NewEncryptionService creates a new encryption service with a PBKDF2-hardened passphrase.
func NewEncryptionService(passphrase string) *EncryptionService {
	return &EncryptionService{
		passphrase: []byte(passphrase),
	}
}

func (es *EncryptionService) deriveKey(salt []byte) []byte {
	return pbkdf2.Key(es.passphrase, salt, encryptionPBKDF2Rounds, encryptionKeySize, sha256.New)
}

func (es *EncryptionService) encryptRaw(plaintext []byte) ([]byte, error) {
	if len(plaintext) == 0 {
		return nil, nil
	}

	salt := make([]byte, encryptionSaltSize)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}

	block, err := aes.NewCipher(es.deriveKey(salt))
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := gcm.Seal(nil, nonce, plaintext, nil)
	encoded := make([]byte, 0, len(encryptionBinaryMagic)+len(salt)+len(nonce)+len(ciphertext))
	encoded = append(encoded, encryptionBinaryMagic...)
	encoded = append(encoded, salt...)
	encoded = append(encoded, nonce...)
	encoded = append(encoded, ciphertext...)
	return encoded, nil
}

func (es *EncryptionService) decryptRaw(ciphertext []byte) ([]byte, error) {
	if len(ciphertext) == 0 {
		return nil, nil
	}

	if !bytes.HasPrefix(ciphertext, []byte(encryptionBinaryMagic)) {
		return nil, fmt.Errorf("unsupported legacy ciphertext format")
	}

	data := ciphertext[len(encryptionBinaryMagic):]
	if len(data) < encryptionSaltSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	salt, encryptedData := data[:encryptionSaltSize], data[encryptionSaltSize:]

	block, err := aes.NewCipher(es.deriveKey(salt))
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonceSize := gcm.NonceSize()
	if len(encryptedData) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, sealed := encryptedData[:nonceSize], encryptedData[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, sealed, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}

// Encrypt encrypts a plaintext string and returns a versioned, base64-encoded ciphertext.
func (es *EncryptionService) Encrypt(plaintext string) (string, error) {
	if plaintext == "" {
		return "", nil
	}

	encoded, err := es.encryptRaw([]byte(plaintext))
	if err != nil {
		return "", err
	}

	return encryptionStringVersion + ":" + base64.StdEncoding.EncodeToString(encoded), nil
}

// Decrypt decrypts a base64-encoded ciphertext and returns the plaintext
func (es *EncryptionService) Decrypt(ciphertext string) (string, error) {
	if ciphertext == "" {
		return "", nil
	}

	encoded := ciphertext
	if strings.HasPrefix(ciphertext, encryptionStringVersion+":") {
		encoded = strings.TrimPrefix(ciphertext, encryptionStringVersion+":")
	}

	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64: %w", err)
	}

	plaintext, err := es.decryptRaw(data)
	if err != nil {
		return "", err
	}

	return string(plaintext), nil
}

// EncryptBytes encrypts raw bytes and returns the versioned ciphertext bytes.
func (es *EncryptionService) EncryptBytes(plaintext []byte) ([]byte, error) {
	return es.encryptRaw(plaintext)
}

// DecryptBytes decrypts versioned ciphertext bytes and returns the plaintext bytes.
func (es *EncryptionService) DecryptBytes(ciphertext []byte) ([]byte, error) {
	return es.decryptRaw(ciphertext)
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
