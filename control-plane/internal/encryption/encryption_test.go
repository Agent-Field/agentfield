package encryption

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEncryptionService_EncryptDecrypt_Roundtrip(t *testing.T) {
	passphrase := "test-passphrase-123"
	service := NewEncryptionService(passphrase)

	plaintext := "sensitive-data-12345"

	// Encrypt
	ciphertext, err := service.Encrypt(plaintext)
	require.NoError(t, err)
	require.NotEmpty(t, ciphertext)
	require.NotEqual(t, plaintext, ciphertext)

	// Decrypt
	decrypted, err := service.Decrypt(ciphertext)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted)
	require.Contains(t, ciphertext, "v2:")
}

func TestEncryptionService_EncryptDecrypt_EmptyString(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	// Encrypt empty string
	ciphertext, err := service.Encrypt("")
	require.NoError(t, err)
	require.Empty(t, ciphertext)

	// Decrypt empty string
	decrypted, err := service.Decrypt("")
	require.NoError(t, err)
	require.Empty(t, decrypted)
}

func TestEncryptionService_EncryptDecrypt_DifferentNonces(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	plaintext := "same-plaintext"

	// Encrypt same plaintext multiple times
	ciphertext1, err := service.Encrypt(plaintext)
	require.NoError(t, err)

	ciphertext2, err := service.Encrypt(plaintext)
	require.NoError(t, err)

	// Ciphertexts should be different (due to random nonce)
	require.NotEqual(t, ciphertext1, ciphertext2)

	// But both should decrypt to same plaintext
	decrypted1, err := service.Decrypt(ciphertext1)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted1)

	decrypted2, err := service.Decrypt(ciphertext2)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted2)
}

func TestEncryptionService_EncryptDecrypt_WrongKey(t *testing.T) {
	passphrase1 := "passphrase-1"
	service1 := NewEncryptionService(passphrase1)

	passphrase2 := "passphrase-2"
	service2 := NewEncryptionService(passphrase2)

	plaintext := "sensitive-data"

	// Encrypt with service1
	ciphertext, err := service1.Encrypt(plaintext)
	require.NoError(t, err)

	// Try to decrypt with service2 (wrong key)
	decrypted, err := service2.Decrypt(ciphertext)
	require.Error(t, err)
	require.Empty(t, decrypted)
	require.Contains(t, err.Error(), "failed to decrypt")
}

func TestEncryptionService_EncryptDecrypt_InvalidCiphertext(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	// Try to decrypt invalid base64
	_, err := service.Decrypt("invalid-base64!!!")
	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to decode base64")
}

func TestEncryptionService_EncryptDecrypt_CorruptedData(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	plaintext := "test-data"
	ciphertext, err := service.Encrypt(plaintext)
	require.NoError(t, err)

	// Corrupt the ciphertext (change last character)
	corrupted := ciphertext[:len(ciphertext)-1] + "X"

	_, err = service.Decrypt(corrupted)
	require.Error(t, err)
	// Error could be "failed to decode base64" or "failed to decrypt" or "ciphertext too short"
	require.NotEmpty(t, err.Error())
}

func TestEncryptionService_EncryptDecrypt_TooShort(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	// Create a ciphertext that's too short (less than nonce size)
	shortCiphertext := "dGVzdA==" // base64 of "test" (4 bytes, less than GCM nonce size)

	_, err := service.Decrypt(shortCiphertext)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported legacy ciphertext format")
}

func TestEncryptionService_EncryptConfigurationValues(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	config := map[string]interface{}{
		"api_key":    "secret-api-key-123",
		"password":   "secret-password",
		"public_url": "https://example.com", // Should not be encrypted
		"port":       8080,                    // Should not be encrypted
	}

	secretFields := []string{"api_key", "password"}

	encrypted, err := service.EncryptConfigurationValues(config, secretFields)
	require.NoError(t, err)
	require.NotNil(t, encrypted)

	// Verify secret fields are encrypted
	require.NotEqual(t, config["api_key"], encrypted["api_key"])
	require.NotEqual(t, config["password"], encrypted["password"])

	// Verify non-secret fields are unchanged
	require.Equal(t, config["public_url"], encrypted["public_url"])
	require.Equal(t, config["port"], encrypted["port"])

	// Verify encrypted values are strings
	apiKeyStr, ok := encrypted["api_key"].(string)
	require.True(t, ok)
	require.NotEmpty(t, apiKeyStr)
}

func TestEncryptionService_DecryptConfigurationValues(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	// First encrypt
	config := map[string]interface{}{
		"api_key":  "secret-api-key",
		"password": "secret-password",
		"port":     8080,
	}

	secretFields := []string{"api_key", "password"}

	encrypted, err := service.EncryptConfigurationValues(config, secretFields)
	require.NoError(t, err)

	// Then decrypt
	decrypted, err := service.DecryptConfigurationValues(encrypted, secretFields)
	require.NoError(t, err)
	require.NotNil(t, decrypted)

	// Verify secret fields are decrypted
	require.Equal(t, config["api_key"], decrypted["api_key"])
	require.Equal(t, config["password"], decrypted["password"])

	// Verify non-secret fields are unchanged
	require.Equal(t, config["port"], decrypted["port"])
}

func TestEncryptionService_EncryptConfigurationValues_NonStringValue(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	config := map[string]interface{}{
		"api_key": 123, // Not a string
		"port":    8080,
	}

	secretFields := []string{"api_key"}

	encrypted, err := service.EncryptConfigurationValues(config, secretFields)
	require.NoError(t, err)

	// Non-string secret field should be left unchanged
	require.Equal(t, config["api_key"], encrypted["api_key"])
}

func TestEncryptionService_EncryptConfigurationValues_MissingField(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	config := map[string]interface{}{
		"port": 8080,
	}

	secretFields := []string{"api_key"} // Field not in config

	encrypted, err := service.EncryptConfigurationValues(config, secretFields)
	require.NoError(t, err)
	require.Equal(t, config, encrypted)
}

func TestEncryptionService_DecryptConfigurationValues_InvalidCiphertext(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	config := map[string]interface{}{
		"api_key": "invalid-ciphertext!!!",
		"port":    8080,
	}

	secretFields := []string{"api_key"}

	_, err := service.DecryptConfigurationValues(config, secretFields)
	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to decrypt field")
}

func TestEncryptionService_KeyDerivation_Consistency(t *testing.T) {
	passphrase := "same-passphrase"

	service1 := NewEncryptionService(passphrase)
	service2 := NewEncryptionService(passphrase)

	plaintext := "test-data"

	// Encrypt with service1
	ciphertext1, err := service1.Encrypt(plaintext)
	require.NoError(t, err)

	// Decrypt with service2 (same passphrase = same key)
	decrypted, err := service2.Decrypt(ciphertext1)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted)
}

func TestEncryptionService_KeyDerivation_DifferentPassphrases(t *testing.T) {
	service1 := NewEncryptionService("passphrase-1")
	service2 := NewEncryptionService("passphrase-2")

	plaintext := "test-data"

	ciphertext, err := service1.Encrypt(plaintext)
	require.NoError(t, err)

	// Should not decrypt with different passphrase
	_, err = service2.Decrypt(ciphertext)
	require.Error(t, err)
}

func TestEncryptionService_LongPlaintext(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	// Create a long plaintext
	longPlaintext := make([]byte, 10000)
	for i := range longPlaintext {
		longPlaintext[i] = byte(i % 256)
	}

	ciphertext, err := service.Encrypt(string(longPlaintext))
	require.NoError(t, err)

	decrypted, err := service.Decrypt(ciphertext)
	require.NoError(t, err)
	require.Equal(t, string(longPlaintext), decrypted)
}

func TestEncryptionService_EncryptBytesDecryptBytes_Roundtrip(t *testing.T) {
	service := NewEncryptionService("test-passphrase")

	plaintext := []byte("sensitive-bytes")
	ciphertext, err := service.EncryptBytes(plaintext)
	require.NoError(t, err)
	require.NotEmpty(t, ciphertext)
	require.NotEqual(t, plaintext, ciphertext)

	decrypted, err := service.DecryptBytes(ciphertext)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted)
}

func TestEncryptionService_SpecialCharacters(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	plaintext := "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"

	ciphertext, err := service.Encrypt(plaintext)
	require.NoError(t, err)

	decrypted, err := service.Decrypt(ciphertext)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted)
}

func TestEncryptionService_Unicode(t *testing.T) {
	passphrase := "test-passphrase"
	service := NewEncryptionService(passphrase)

	plaintext := "Hello 世界 🌍 测试"

	ciphertext, err := service.Encrypt(plaintext)
	require.NoError(t, err)

	decrypted, err := service.Decrypt(ciphertext)
	require.NoError(t, err)
	require.Equal(t, plaintext, decrypted)
}
