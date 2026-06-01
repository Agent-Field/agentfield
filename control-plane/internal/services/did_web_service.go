package services

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// DIDWebService handles did:web generation, storage, and resolution.
type DIDWebService struct {
	domain     string
	didService *DIDService
	storage    DIDWebStorage
}

// DIDWebStorage defines the storage interface for DID documents.
type DIDWebStorage interface {
	StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error
	GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error)
	GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error)
	RevokeDIDDocument(ctx context.Context, did string) error
	ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error)
}

// NewDIDWebService creates a new did:web service instance.
func NewDIDWebService(domain string, didService *DIDService, storage DIDWebStorage) *DIDWebService {
	return &DIDWebService{
		domain:     domain,
		didService: didService,
		storage:    storage,
	}
}

// GenerateDIDWeb creates a did:web identifier for an agent.
// Format: did:web:{domain}:agents:{agentID}
func (s *DIDWebService) GenerateDIDWeb(agentID string) string {
	// URL-encode the domain (replace : with %3A for port numbers)
	encodedDomain := strings.ReplaceAll(s.domain, ":", "%3A")
	return fmt.Sprintf("did:web:%s:agents:%s", encodedDomain, agentID)
}

// ResolveAgentIDByDID looks up the agent ID for any DID format.
// It first queries the stored DID documents (covers did:web), then falls back
// to the DID service's in-memory registry (covers did:key).
// Returns empty string if the DID is not found.
func (s *DIDWebService) ResolveAgentIDByDID(ctx context.Context, did string) string {
	// Try storage lookup first (works for did:web)
	record, err := s.storage.GetDIDDocument(ctx, did)
	if err == nil && record != nil {
		return record.AgentID
	}
	// Fall back to DID service registry (works for did:key)
	if s.didService != nil {
		return s.didService.ResolveAgentIDByDID(did)
	}
	return ""
}

// ParseDIDWeb extracts the agent ID from a did:web identifier.
// Returns the agent ID or an error if the DID format is invalid.
func (s *DIDWebService) ParseDIDWeb(did string) (string, error) {
	// Expected format: did:web:{domain}:agents:{agentID}
	if !strings.HasPrefix(did, "did:web:") {
		return "", fmt.Errorf("invalid did:web format: must start with 'did:web:'")
	}

	parts := strings.Split(did, ":")
	if len(parts) < 5 {
		return "", fmt.Errorf("invalid did:web format: expected at least 5 parts")
	}

	// Find the "agents" part and extract the agent ID
	for i, part := range parts {
		if part == "agents" && i+1 < len(parts) {
			return parts[i+1], nil
		}
	}

	return "", fmt.Errorf("invalid did:web format: missing 'agents' segment")
}

// CreateDIDDocument creates and stores a DID document for an agent.
func (s *DIDWebService) CreateDIDDocument(ctx context.Context, agentID string, publicKeyJWK json.RawMessage) (*types.DIDWebDocument, error) {
	// Generate the did:web identifier
	did := s.GenerateDIDWeb(agentID)

	// Create the DID document
	didDoc := types.NewDIDWebDocument(did, publicKeyJWK)

	// Serialize the document for storage
	docBytes, err := json.Marshal(didDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal DID document: %w", err)
	}

	// Create the storage record
	record := &types.DIDDocumentRecord{
		DID:          did,
		AgentID:      agentID,
		DIDDocument:  docBytes,
		PublicKeyJWK: string(publicKeyJWK),
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	// Store the record
	if err := s.storage.StoreDIDDocument(ctx, record); err != nil {
		return nil, fmt.Errorf("failed to store DID document: %w", err)
	}

	logger.Logger.Info().
		Str("did", did).
		Str("agent_id", agentID).
		Msg("Created DID document for agent")

	return didDoc, nil
}

// ResolveDID resolves a did:web identifier to its DID document.
// Returns the DID document or an error if not found or revoked.
func (s *DIDWebService) ResolveDID(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
	// Get the DID document record
	record, err := s.storage.GetDIDDocument(ctx, did)
	if err != nil {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{
				Error: "notFound",
			},
		}, nil
	}

	// Check if revoked
	if record.IsRevoked() {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{
				Error: "deactivated",
			},
			DIDDocumentMetadata: types.DIDDocumentMetadata{
				Deactivated: true,
			},
		}, nil
	}

	// Parse the stored DID document
	var didDoc types.DIDWebDocument
	if err := json.Unmarshal(record.DIDDocument, &didDoc); err != nil {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{
				Error: "invalidDidDocument",
			},
		}, nil
	}

	return &types.DIDResolutionResult{
		DIDDocument: &didDoc,
		DIDResolutionMetadata: types.DIDResolutionMetadata{
			ContentType: "application/did+ld+json",
		},
		DIDDocumentMetadata: types.DIDDocumentMetadata{
			Created: record.CreatedAt.Format(time.RFC3339),
			Updated: record.UpdatedAt.Format(time.RFC3339),
		},
	}, nil
}

// ResolveDIDByAgentID resolves a DID document by agent ID.
func (s *DIDWebService) ResolveDIDByAgentID(ctx context.Context, agentID string) (*types.DIDResolutionResult, error) {
	did := s.GenerateDIDWeb(agentID)
	return s.ResolveDID(ctx, did)
}

// RevokeDID revokes a did:web identifier, making it invalid.
func (s *DIDWebService) RevokeDID(ctx context.Context, did string) error {
	if err := s.storage.RevokeDIDDocument(ctx, did); err != nil {
		return fmt.Errorf("failed to revoke DID: %w", err)
	}

	logger.Logger.Info().
		Str("did", did).
		Msg("Revoked DID document")

	return nil
}

// IsDIDRevoked checks if a DID has been revoked.
// Returns true if revoked, false if active or not found.
// On storage errors (other than not-found), returns true to fail closed.
func (s *DIDWebService) IsDIDRevoked(ctx context.Context, did string) bool {
	record, err := s.storage.GetDIDDocument(ctx, did)
	if err != nil {
		// Check if this is a "not found" error vs a real storage failure.
		// Not found means the DID was never registered — treat as not revoked.
		// Any other error (DB timeout, connection failure) — fail closed.
		if strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "no rows") {
			return false
		}
		logger.Logger.Warn().Err(err).Str("did", did).Msg("Storage error checking DID revocation, failing closed")
		return true
	}
	return record.IsRevoked()
}

// GetOrCreateDIDDocument gets an existing DID document or creates a new one.
// This is useful when registering agents - we want to reuse existing DIDs if the agent
// has the same ID, or create new ones for new agents.
func (s *DIDWebService) GetOrCreateDIDDocument(ctx context.Context, agentID string) (*types.DIDWebDocument, string, error) {
	// Try to get existing DID document
	did := s.GenerateDIDWeb(agentID)
	record, err := s.storage.GetDIDDocument(ctx, did)
	if err == nil && !record.IsRevoked() {
		// Parse and return existing document
		var didDoc types.DIDWebDocument
		if err := json.Unmarshal(record.DIDDocument, &didDoc); err != nil {
			return nil, "", fmt.Errorf("failed to parse existing DID document: %w", err)
		}
		return &didDoc, did, nil
	}

	// Generate new key pair for the agent
	publicKeyJWK, err := s.generatePublicKeyJWK(agentID)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate public key: %w", err)
	}

	// Create new DID document
	didDoc, err := s.CreateDIDDocument(ctx, agentID, publicKeyJWK)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create DID document: %w", err)
	}

	return didDoc, did, nil
}

// generatePublicKeyJWK generates a new Ed25519 public key JWK for an agent.
// This uses the DID service's key derivation to ensure deterministic keys.
func (s *DIDWebService) generatePublicKeyJWK(agentID string) (json.RawMessage, error) {
	// Get the registry to access the master seed
	serverID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get server ID: %w", err)
	}

	registry, err := s.didService.registry.GetRegistry(serverID)
	if err != nil {
		return nil, fmt.Errorf("failed to get registry: %w", err)
	}

	if registry == nil {
		return nil, fmt.Errorf("DID registry not initialized")
	}

	// Generate derivation path for this agent
	// Use the agent ID to create a unique path
	derivationPath := fmt.Sprintf("m/44'/web'/%s'", agentID)

	// Derive the public key JWK
	publicKeyJWK, err := s.didService.regeneratePublicKeyJWK(registry.MasterSeed, derivationPath)
	if err != nil {
		return nil, fmt.Errorf("failed to generate public key JWK: %w", err)
	}

	return json.RawMessage(publicKeyJWK), nil
}

// GetPrivateKeyJWK retrieves the private key JWK for signing operations.
// This should only be used by the control plane for signing VCs.
func (s *DIDWebService) GetPrivateKeyJWK(agentID string) (string, error) {
	// Get the registry to access the master seed
	serverID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		return "", fmt.Errorf("failed to get server ID: %w", err)
	}

	registry, err := s.didService.registry.GetRegistry(serverID)
	if err != nil {
		return "", fmt.Errorf("failed to get registry: %w", err)
	}

	if registry == nil {
		return "", fmt.Errorf("DID registry not initialized")
	}

	// Generate derivation path for this agent
	derivationPath := fmt.Sprintf("m/44'/web'/%s'", agentID)

	// Derive the private key JWK
	privateKeyJWK, err := s.didService.regeneratePrivateKeyJWK(registry.MasterSeed, derivationPath)
	if err != nil {
		return "", fmt.Errorf("failed to generate private key JWK: %w", err)
	}

	return privateKeyJWK, nil
}

// GetDomain returns the configured domain for did:web identifiers.
func (s *DIDWebService) GetDomain() string {
	return s.domain
}

// VerifyDIDOwnership verifies that a signature was created by the private key
// corresponding to a did:web identifier.
func (s *DIDWebService) VerifyDIDOwnership(ctx context.Context, did string, message []byte, signature []byte) (bool, error) {
	// Handle did:key self-resolution: public key is encoded directly in the DID.
	if strings.HasPrefix(did, "did:key:z") {
		pubKey, err := decodeDIDKeyPublicKey(did)
		if err != nil {
			return false, fmt.Errorf("failed to decode did:key public key: %w", err)
		}
		return ed25519.Verify(pubKey, message, signature), nil
	}

	// Resolve did:web (or other methods) via stored DID documents.
	result, err := s.ResolveDID(ctx, did)
	if err != nil {
		return false, fmt.Errorf("failed to resolve DID: %w", err)
	}

	if result.DIDDocument == nil {
		return false, fmt.Errorf("DID not found or deactivated")
	}

	if len(result.DIDDocument.VerificationMethod) == 0 {
		return false, fmt.Errorf("no verification method in DID document")
	}

	// Get the public key from the verification method
	vm := result.DIDDocument.VerificationMethod[0]

	// Parse the JWK to extract the public key
	var jwk struct {
		X string `json:"x"`
	}
	if err := json.Unmarshal(vm.PublicKeyJwk, &jwk); err != nil {
		return false, fmt.Errorf("failed to parse public key JWK: %w", err)
	}

	// Decode the public key
	publicKeyBytes, err := base64RawURLDecode(jwk.X)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	// Verify the signature
	publicKey := ed25519.PublicKey(publicKeyBytes)
	return ed25519.Verify(publicKey, message, signature), nil
}

// decodeDIDKeyPublicKey extracts the Ed25519 public key from a did:key identifier.
// Format: did:key:z<base64url(0xed01 + 32-byte-public-key)>
func decodeDIDKeyPublicKey(did string) (ed25519.PublicKey, error) {
	const prefix = "did:key:z"
	if !strings.HasPrefix(did, prefix) {
		return nil, fmt.Errorf("invalid did:key format")
	}

	encoded := did[len(prefix):]
	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("failed to base64url decode did:key: %w", err)
	}

	// Verify multicodec prefix (0xed, 0x01 for Ed25519)
	if len(decoded) < 2 || decoded[0] != 0xed || decoded[1] != 0x01 {
		return nil, fmt.Errorf("unsupported multicodec prefix in did:key")
	}

	pubKeyBytes := decoded[2:]
	if len(pubKeyBytes) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("invalid Ed25519 public key length: got %d, want %d", len(pubKeyBytes), ed25519.PublicKeySize)
	}

	return ed25519.PublicKey(pubKeyBytes), nil
}

// base64RawURLDecode decodes a base64 raw URL encoded string.
func base64RawURLDecode(s string) ([]byte, error) {
	return base64.RawURLEncoding.DecodeString(s)
}
