package services

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type didWebStorageWithErrors struct {
	didWebStorageStub
	storeErr  error
	revokeErr error
}

func (s *didWebStorageWithErrors) StoreDIDDocument(_ context.Context, record *types.DIDDocumentRecord) error {
	if s.storeErr != nil {
		return s.storeErr
	}
	if s.docsByDID == nil {
		s.docsByDID = make(map[string]*types.DIDDocumentRecord)
	}
	s.docsByDID[record.DID] = record
	return nil
}

func (s *didWebStorageWithErrors) RevokeDIDDocument(_ context.Context, did string) error {
	if s.revokeErr != nil {
		return s.revokeErr
	}
	record, ok := s.docsByDID[did]
	if !ok {
		return errors.New("not found")
	}
	now := time.Now().UTC()
	record.RevokedAt = &now
	record.UpdatedAt = now
	return nil
}

func TestDIDWebServiceDocumentLifecycleAndResolution(t *testing.T) {
	t.Run("create resolve by did and agent id", func(t *testing.T) {
		storage := &didWebStorageWithErrors{}
		service := NewDIDWebService("example.com", nil, storage)
		jwk := json.RawMessage(`{"kty":"OKP","crv":"Ed25519","x":"cHVibGlj"}`)

		doc, err := service.CreateDIDDocument(context.Background(), "agent-1", jwk)
		require.NoError(t, err)
		require.Equal(t, "did:web:example.com:agents:agent-1", doc.ID)

		record := storage.docsByDID[doc.ID]
		require.Equal(t, "agent-1", record.AgentID)
		require.JSONEq(t, string(jwk), record.PublicKeyJWK)

		result, err := service.ResolveDID(context.Background(), doc.ID)
		require.NoError(t, err)
		require.Equal(t, doc.ID, result.DIDDocument.ID)
		require.Equal(t, "application/did+ld+json", result.DIDResolutionMetadata.ContentType)
		require.Empty(t, result.DIDResolutionMetadata.Error)
		require.False(t, result.DIDDocumentMetadata.Deactivated)

		byAgentID, err := service.ResolveDIDByAgentID(context.Background(), "agent-1")
		require.NoError(t, err)
		require.Equal(t, doc.ID, byAgentID.DIDDocument.ID)

		require.Equal(t, "example.com", service.GetDomain())
	})

	t.Run("storage failures and malformed payloads return DID resolution metadata", func(t *testing.T) {
		now := time.Now().UTC()
		service := NewDIDWebService("example.com", nil, &didWebStorageWithErrors{
			didWebStorageStub: didWebStorageStub{
				docsByDID: map[string]*types.DIDDocumentRecord{
					"did:web:example.com:agents:revoked": {
						DID:         "did:web:example.com:agents:revoked",
						AgentID:     "revoked",
						DIDDocument: json.RawMessage(`{"id":"did:web:example.com:agents:revoked","verificationMethod":[]}`),
						RevokedAt:   &now,
						CreatedAt:   now,
						UpdatedAt:   now,
					},
					"did:web:example.com:agents:broken": {
						DID:         "did:web:example.com:agents:broken",
						AgentID:     "broken",
						DIDDocument: json.RawMessage(`{not-json`),
						CreatedAt:   now,
						UpdatedAt:   now,
					},
				},
				errByDID: map[string]error{
					"did:web:example.com:agents:missing": errors.New("not found"),
				},
			},
		})

		missing, err := service.ResolveDID(context.Background(), "did:web:example.com:agents:missing")
		require.NoError(t, err)
		require.Equal(t, "notFound", missing.DIDResolutionMetadata.Error)

		revoked, err := service.ResolveDID(context.Background(), "did:web:example.com:agents:revoked")
		require.NoError(t, err)
		require.Equal(t, "deactivated", revoked.DIDResolutionMetadata.Error)
		require.True(t, revoked.DIDDocumentMetadata.Deactivated)

		broken, err := service.ResolveDID(context.Background(), "did:web:example.com:agents:broken")
		require.NoError(t, err)
		require.Equal(t, "invalidDidDocument", broken.DIDResolutionMetadata.Error)
	})
}

func TestDIDWebServiceGetOrCreateAndRevocationPaths(t *testing.T) {
	t.Run("reuses active document and reports malformed existing state", func(t *testing.T) {
		storage := &didWebStorageWithErrors{
			didWebStorageStub: didWebStorageStub{
				docsByDID: map[string]*types.DIDDocumentRecord{
					"did:web:example.com:agents:existing": {
						DID:         "did:web:example.com:agents:existing",
						AgentID:     "existing",
						DIDDocument: json.RawMessage(`{"id":"did:web:example.com:agents:existing","verificationMethod":[]}`),
					},
					"did:web:example.com:agents:malformed": {
						DID:         "did:web:example.com:agents:malformed",
						AgentID:     "malformed",
						DIDDocument: json.RawMessage(`{`),
					},
				},
			},
		}
		service := NewDIDWebService("example.com", nil, storage)

		doc, did, err := service.GetOrCreateDIDDocument(context.Background(), "existing")
		require.NoError(t, err)
		require.Equal(t, "did:web:example.com:agents:existing", did)
		require.Equal(t, did, doc.ID)

		_, _, err = service.GetOrCreateDIDDocument(context.Background(), "malformed")
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to parse existing DID document")
	})

	t.Run("creates a new document when missing or revoked", func(t *testing.T) {
		didService, _, _, _, _ := setupDIDTestEnvironment(t)
		now := time.Now().UTC()
		storage := &didWebStorageWithErrors{
			didWebStorageStub: didWebStorageStub{
				docsByDID: map[string]*types.DIDDocumentRecord{
					"did:web:example.com:agents:revoked": {
						DID:         "did:web:example.com:agents:revoked",
						AgentID:     "revoked",
						DIDDocument: json.RawMessage(`{"id":"did:web:example.com:agents:revoked","verificationMethod":[]}`),
						RevokedAt:   &now,
					},
				},
			},
		}
		service := NewDIDWebService("example.com", didService, storage)

		doc, did, err := service.GetOrCreateDIDDocument(context.Background(), "new-agent")
		require.NoError(t, err)
		require.Equal(t, did, doc.ID)
		require.NotEmpty(t, storage.docsByDID[did].PublicKeyJWK)

		doc, did, err = service.GetOrCreateDIDDocument(context.Background(), "revoked")
		require.NoError(t, err)
		require.Equal(t, did, doc.ID)
		require.Nil(t, storage.docsByDID[did].RevokedAt)
	})

	t.Run("create revoke and revocation checks handle success and errors", func(t *testing.T) {
		service := NewDIDWebService("example.com", nil, &didWebStorageWithErrors{
			didWebStorageStub: didWebStorageStub{
				docsByDID: map[string]*types.DIDDocumentRecord{
					"did:web:example.com:agents:active": {DID: "did:web:example.com:agents:active"},
					"did:web:example.com:agents:revoked": {
						DID:       "did:web:example.com:agents:revoked",
						RevokedAt: func() *time.Time { now := time.Now().UTC(); return &now }(),
					},
				},
				errByDID: map[string]error{
					"did:web:example.com:agents:missing": errors.New("not found"),
					"did:web:example.com:agents:error":   errors.New("database unavailable"),
				},
			},
		})

		require.NoError(t, service.RevokeDID(context.Background(), "did:web:example.com:agents:active"))
		require.True(t, service.IsDIDRevoked(context.Background(), "did:web:example.com:agents:active"))
		require.True(t, service.IsDIDRevoked(context.Background(), "did:web:example.com:agents:revoked"))
		require.False(t, service.IsDIDRevoked(context.Background(), "did:web:example.com:agents:missing"))
		require.True(t, service.IsDIDRevoked(context.Background(), "did:web:example.com:agents:error"))

		errService := NewDIDWebService("example.com", nil, &didWebStorageWithErrors{revokeErr: errors.New("boom")})
		require.ErrorContains(t, errService.RevokeDID(context.Background(), "did:web:example.com:agents:any"), "failed to revoke DID")
	})
}

func TestDIDWebServiceKeyGenerationAndVerification(t *testing.T) {
	t.Run("private key helpers report initialization failures", func(t *testing.T) {
		service := NewDIDWebService("example.com", &DIDService{}, &didWebStorageWithErrors{})

		_, err := service.GetPrivateKeyJWK("agent-1")
		require.ErrorContains(t, err, "failed to get server ID")

		_, err = service.generatePublicKeyJWK("agent-1")
		require.ErrorContains(t, err, "failed to get server ID")
	})

	t.Run("private key helper succeeds with initialized DID service", func(t *testing.T) {
		didService, _, _, _, _ := setupDIDTestEnvironment(t)
		service := NewDIDWebService("example.com", didService, &didWebStorageWithErrors{})

		privateKeyJWK, err := service.GetPrivateKeyJWK("agent-1")
		require.NoError(t, err)
		require.Contains(t, privateKeyJWK, "\"d\":")
	})

	t.Run("verify did:key ownership and error paths", func(t *testing.T) {
		publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
		require.NoError(t, err)

		message := []byte("hello")
		signature := ed25519.Sign(privateKey, message)
		didKey := "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, publicKey...))

		service := NewDIDWebService("example.com", nil, &didWebStorageWithErrors{})
		ok, err := service.VerifyDIDOwnership(context.Background(), didKey, message, signature)
		require.NoError(t, err)
		require.True(t, ok)

		ok, err = service.VerifyDIDOwnership(context.Background(), didKey, []byte("tampered"), signature)
		require.NoError(t, err)
		require.False(t, ok)

		_, err = service.VerifyDIDOwnership(context.Background(), "did:key:zbad", message, signature)
		require.ErrorContains(t, err, "failed to decode did:key public key")
	})

	t.Run("verify did:web ownership and parse errors", func(t *testing.T) {
		publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
		require.NoError(t, err)
		message := []byte("verify web")
		signature := ed25519.Sign(privateKey, message)

		validJWK := json.RawMessage(`{"kty":"OKP","crv":"Ed25519","x":"` + base64.RawURLEncoding.EncodeToString(publicKey) + `"}`)
		validDoc := types.NewDIDWebDocument("did:web:example.com:agents:web-agent", validJWK)
		validBytes, err := json.Marshal(validDoc)
		require.NoError(t, err)

		noMethodBytes, err := json.Marshal(&types.DIDWebDocument{ID: "did:web:example.com:agents:none"})
		require.NoError(t, err)

		storage := &didWebStorageWithErrors{
			didWebStorageStub: didWebStorageStub{
				docsByDID: map[string]*types.DIDDocumentRecord{
					"did:web:example.com:agents:web-agent": {
						DID:         "did:web:example.com:agents:web-agent",
						DIDDocument: validBytes,
					},
					"did:web:example.com:agents:no-method": {
						DID:         "did:web:example.com:agents:no-method",
						DIDDocument: noMethodBytes,
					},
					"did:web:example.com:agents:bad-jwk": {
						DID:         "did:web:example.com:agents:bad-jwk",
						DIDDocument: json.RawMessage(`{"id":"did:web:example.com:agents:bad-jwk","verificationMethod":[{"publicKeyJwk":"not-json"}]}`),
					},
					"did:web:example.com:agents:bad-key": {
						DID:         "did:web:example.com:agents:bad-key",
						DIDDocument: json.RawMessage(`{"id":"did:web:example.com:agents:bad-key","verificationMethod":[{"publicKeyJwk":{"x":"***"}}]}`),
					},
				},
			},
		}
		service := NewDIDWebService("example.com", nil, storage)

		ok, err := service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:web-agent", message, signature)
		require.NoError(t, err)
		require.True(t, ok)

		ok, err = service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:web-agent", []byte("bad"), signature)
		require.NoError(t, err)
		require.False(t, ok)

		_, err = service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:missing", message, signature)
		require.ErrorContains(t, err, "DID not found or deactivated")

		_, err = service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:no-method", message, signature)
		require.ErrorContains(t, err, "no verification method")

		_, err = service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:bad-jwk", message, signature)
		require.ErrorContains(t, err, "failed to parse public key JWK")

		_, err = service.VerifyDIDOwnership(context.Background(), "did:web:example.com:agents:bad-key", message, signature)
		require.ErrorContains(t, err, "failed to decode public key")
	})
}

func TestDIDWebServiceDecodeHelpers(t *testing.T) {
	t.Run("decode did:key public key", func(t *testing.T) {
		publicKey, _, err := ed25519.GenerateKey(rand.Reader)
		require.NoError(t, err)
		didKey := "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, publicKey...))

		decoded, err := decodeDIDKeyPublicKey(didKey)
		require.NoError(t, err)
		require.Equal(t, []byte(publicKey), []byte(decoded))
	})

	tests := []struct {
		name    string
		did     string
		wantErr string
	}{
		{name: "bad prefix", did: "did:web:example.com:agents:1", wantErr: "invalid did:key format"},
		{name: "bad multicodec", did: "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0x00, 0x01}, make([]byte, ed25519.PublicKeySize)...)), wantErr: "unsupported multicodec prefix"},
		{name: "bad length", did: "did:key:z" + base64.RawURLEncoding.EncodeToString(append([]byte{0xed, 0x01}, make([]byte, 8)...)), wantErr: "invalid Ed25519 public key length"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeDIDKeyPublicKey(tt.did)
			require.ErrorContains(t, err, tt.wantErr)
		})
	}

	decoded, err := base64RawURLDecode(base64.RawURLEncoding.EncodeToString([]byte("payload")))
	require.NoError(t, err)
	require.Equal(t, []byte("payload"), decoded)

	_, err = base64RawURLDecode("***")
	require.Error(t, err)
}
