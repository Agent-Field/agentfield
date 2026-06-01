package encryption

import (
	"crypto/rand"
	"errors"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

type errReader struct {
	err error
}

func (r errReader) Read(_ []byte) (int, error) {
	return 0, r.err
}

func TestEncryptionService_EncryptRawAndDecryptRaw_Branches(t *testing.T) {
	service := NewEncryptionService("test-passphrase")

	t.Run("empty bytes roundtrip", func(t *testing.T) {
		ciphertext, err := service.encryptRaw(nil)
		require.NoError(t, err)
		require.Nil(t, ciphertext)

		plaintext, err := service.decryptRaw(nil)
		require.NoError(t, err)
		require.Nil(t, plaintext)
	})

	t.Run("decrypt raw error branches", func(t *testing.T) {
		validCiphertext, err := service.EncryptBytes([]byte("secret"))
		require.NoError(t, err)

		tests := []struct {
			name    string
			input   []byte
			errText string
		}{
			{
				name:    "legacy format rejected",
				input:   []byte("legacy-format"),
				errText: "unsupported legacy ciphertext format",
			},
			{
				name:    "too short after magic",
				input:   []byte(encryptionBinaryMagic + "short"),
				errText: "ciphertext too short",
			},
			{
				name:    "missing nonce bytes",
				input:   append([]byte(encryptionBinaryMagic), make([]byte, encryptionSaltSize)...),
				errText: "ciphertext too short",
			},
			{
				name:    "tampered payload",
				input:   append([]byte(nil), validCiphertext[:len(validCiphertext)-1]...),
				errText: "failed to decrypt",
			},
		}

		tests[3].input[len(tests[3].input)-1] ^= 0xFF

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				plaintext, err := service.decryptRaw(tc.input)
				require.Error(t, err)
				require.Nil(t, plaintext)
				require.Contains(t, err.Error(), tc.errText)
			})
		}
	})
}

func TestEncryptionService_Encrypt_ErrorPaths(t *testing.T) {
	service := NewEncryptionService("test-passphrase")

	tests := []struct {
		name    string
		reader  io.Reader
		errText string
	}{
		{
			name:    "salt generation failure",
			reader:  errReader{err: errors.New("salt boom")},
			errText: "failed to generate salt",
		},
		{
			name:    "nonce generation failure",
			reader: io.MultiReader(
				io.LimitReader(zeroReader{}, encryptionSaltSize),
				errReader{err: errors.New("nonce boom")},
			),
			errText: "failed to generate nonce",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			originalReader := rand.Reader
			t.Cleanup(func() { rand.Reader = originalReader })
			rand.Reader = tc.reader

			ciphertext, err := service.Encrypt("secret")
			require.Error(t, err)
			require.Empty(t, ciphertext)
			require.Contains(t, err.Error(), tc.errText)
		})
	}
}

type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func TestEncryptionService_Decrypt_WithoutVersionPrefix(t *testing.T) {
	service := NewEncryptionService("test-passphrase")

	ciphertext, err := service.Encrypt("secret")
	require.NoError(t, err)

	rawEncoded := ciphertext[len(encryptionStringVersion)+1:]
	plaintext, err := service.Decrypt(rawEncoded)
	require.NoError(t, err)
	require.Equal(t, "secret", plaintext)
}

func TestEncryptionService_ConfigurationValues_ErrorAndBranchCoverage(t *testing.T) {
	service := NewEncryptionService("test-passphrase")

	t.Run("encrypt configuration propagates encrypt errors", func(t *testing.T) {
		originalReader := rand.Reader
		rand.Reader = errReader{err: errors.New("salt boom")}
		t.Cleanup(func() { rand.Reader = originalReader })

		_, err := service.EncryptConfigurationValues(map[string]interface{}{
			"api_key": "secret",
		}, []string{"api_key"})
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to encrypt field 'api_key'")
	})

	t.Run("decrypt configuration leaves non-string and missing fields unchanged", func(t *testing.T) {
		config := map[string]interface{}{
			"api_key": "secret",
			"count":   3,
		}

		encrypted, err := service.EncryptConfigurationValues(config, []string{"api_key"})
		require.NoError(t, err)

		decrypted, err := service.DecryptConfigurationValues(encrypted, []string{"api_key", "count", "missing"})
		require.NoError(t, err)
		require.Equal(t, "secret", decrypted["api_key"])
		require.Equal(t, 3, decrypted["count"])
	})
}
