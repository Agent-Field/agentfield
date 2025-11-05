package services

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"time"

	"github.com/your-org/haxen/control-plane/internal/config"
	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/types"
)

// DIDService handles DID generation, management, and resolution.
type DIDService struct {
	config        *config.DIDConfig
	keystore      *KeystoreService
	registry      *DIDRegistry
	haxenServerID string
}

// NewDIDService creates a new DID service instance.
func NewDIDService(cfg *config.DIDConfig, keystore *KeystoreService, registry *DIDRegistry) *DIDService {
	return &DIDService{
		config:        cfg,
		keystore:      keystore,
		registry:      registry,
		haxenServerID: "", // Will be set during initialization
	}
}

// Initialize initializes the DID service and creates haxen server master seed if needed.
func (s *DIDService) Initialize(haxenServerID string) error {
	if !s.config.Enabled {
		return nil
	}

	// Store the haxen server ID for dynamic resolution
	s.haxenServerID = haxenServerID

	// Check if haxen server already has a DID registry
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return fmt.Errorf("failed to check existing registry: %w", err)
	}

	if registry == nil {
		// Create new haxen server registry
		masterSeed := make([]byte, 32)
		if _, err := rand.Read(masterSeed); err != nil {
			return fmt.Errorf("failed to generate master seed: %w", err)
		}

		// Generate root DID from master seed
		rootDID, err := s.generateDIDFromSeed(masterSeed, "m/44'/0'")
		if err != nil {
			return fmt.Errorf("failed to generate root DID: %w", err)
		}

		// Create and store registry
		registry = &types.DIDRegistry{
			HaxenServerID:   haxenServerID,
			MasterSeed:      masterSeed,
			RootDID:         rootDID,
			AgentNodes:      make(map[string]types.AgentDIDInfo),
			TotalDIDs:       1,
			CreatedAt:       time.Now(),
			LastKeyRotation: time.Now(),
		}

		if err := s.registry.StoreRegistry(registry); err != nil {
			return fmt.Errorf("failed to store DID registry: %w", err)
		}

	}

	return nil
}

// GetHaxenServerID returns the haxen server ID for this DID service instance.
// This method provides dynamic haxen server ID resolution instead of hardcoded "default".
func (s *DIDService) GetHaxenServerID() (string, error) {
	if s.haxenServerID == "" {
		return "", fmt.Errorf("haxen server ID not initialized - call Initialize() first")
	}
	return s.haxenServerID, nil
}

// getHaxenServerID is an internal helper that returns the haxen server ID.
func (s *DIDService) getHaxenServerID() (string, error) {
	return s.GetHaxenServerID()
}

// validateHaxenServerRegistry ensures that the haxen server registry exists before operations.
func (s *DIDService) validateHaxenServerRegistry() error {
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return err
	}

	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return fmt.Errorf("failed to get haxen server registry: %w", err)
	}

	if registry == nil {
		return fmt.Errorf("haxen server registry not found for ID: %s - ensure Initialize() was called", haxenServerID)
	}

	return nil
}

// GetRegistry retrieves a DID registry for a haxen server.
func (s *DIDService) GetRegistry(haxenServerID string) (*types.DIDRegistry, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.registry.GetRegistry(haxenServerID)
}

// RegisterAgent generates DIDs for an agent node and all its components.
// Enhanced to support partial registration for existing agents.
func (s *DIDService) RegisterAgent(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	if !s.config.Enabled {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("haxen server registry validation failed: %v", err),
		}, nil
	}

	// Check if agent already exists
	existingAgent, err := s.GetExistingAgentDID(req.AgentNodeID)
	if err != nil && err.Error() != fmt.Sprintf("agent not found: %s", req.AgentNodeID) {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to check existing agent: %v", err),
		}, nil
	}

	if existingAgent != nil {
		// Perform differential analysis
		newReasonerIDs := extractReasonerIDs(req.Reasoners)
		newSkillIDs := extractSkillIDs(req.Skills)

		diffResult, err := s.PerformDifferentialAnalysis(req.AgentNodeID, newReasonerIDs, newSkillIDs)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("differential analysis failed: %v", err),
			}, nil
		}

		if !diffResult.RequiresUpdate {
			// No changes needed, return existing identity package
			identityPackage := s.buildExistingIdentityPackage(existingAgent)
			return &types.DIDRegistrationResponse{
				Success:         true,
				Message:         "No changes detected, registration skipped",
				IdentityPackage: identityPackage,
			}, nil
		}

		// Handle partial registration
		return s.handlePartialRegistration(req, diffResult)
	}

	// Handle new registration (existing logic)
	return s.handleNewRegistration(req)
}

// handleNewRegistration handles registration for new agents (original logic).
func (s *DIDService) handleNewRegistration(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get haxen server ID: %v", err),
		}, nil
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get DID registry: %v", err),
		}, nil
	}

	// Generate haxen server hash for derivation path
	haxenServerHash := s.hashHaxenServerID(registry.HaxenServerID)

	// Get next agent index
	agentIndex := len(registry.AgentNodes)

	// Generate agent DID
	agentPath := fmt.Sprintf("m/44'/%d'/%d'", haxenServerHash, agentIndex)
	agentDID, agentPrivKey, agentPubKey, err := s.generateDIDWithKeys(registry.MasterSeed, agentPath)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to generate agent DID: %v", err),
		}, nil
	}

	// Generate reasoner DIDs
	reasonerDIDs := make(map[string]types.DIDIdentity)
	reasonerInfos := make(map[string]types.ReasonerDIDInfo)

	logger.Logger.Debug().Msgf("🔍 DEBUG: Calling did_manager.register_agent() with %d reasoners and %d skills", len(req.Reasoners), len(req.Skills))

	validReasonerIndex := 0
	for i, reasoner := range req.Reasoners {
		// Skip reasoners with empty IDs to prevent malformed DIDs
		if reasoner.ID == "" {
			logger.Logger.Warn().Msgf("⚠️ Skipping reasoner at index %d with empty ID", i)
			continue
		}

		reasonerPath := fmt.Sprintf("m/44'/%d'/%d'/0'/%d'", haxenServerHash, agentIndex, validReasonerIndex)
		reasonerDID, reasonerPrivKey, reasonerPubKey, err := s.generateDIDWithKeys(registry.MasterSeed, reasonerPath)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate reasoner DID for %s: %v", reasoner.ID, err),
			}, nil
		}

		reasonerDIDs[reasoner.ID] = types.DIDIdentity{
			DID:            reasonerDID,
			PrivateKeyJWK:  reasonerPrivKey,
			PublicKeyJWK:   reasonerPubKey,
			DerivationPath: reasonerPath,
			ComponentType:  "reasoner",
			FunctionName:   reasoner.ID,
		}

		reasonerInfos[reasoner.ID] = types.ReasonerDIDInfo{
			DID:            reasonerDID,
			FunctionName:   reasoner.ID,
			PublicKeyJWK:   json.RawMessage(reasonerPubKey),
			DerivationPath: reasonerPath,
			Capabilities:   []string{}, // TODO: Extract from reasoner definition
			ExposureLevel:  "internal",
			CreatedAt:      time.Now(),
		}

		validReasonerIndex++
		logger.Logger.Debug().Msgf("🔍 Created DID for reasoner %s: %s", reasoner.ID, reasonerDID)
	}

	logger.Logger.Debug().Msgf("🔍 Successfully created %d reasoner DIDs out of %d total reasoners", len(reasonerDIDs), len(req.Reasoners))

	// Generate skill DIDs
	skillDIDs := make(map[string]types.DIDIdentity)
	skillInfos := make(map[string]types.SkillDIDInfo)

	validSkillIndex := 0
	for i, skill := range req.Skills {
		// Skip skills with empty IDs to prevent malformed DIDs
		if skill.ID == "" {
			logger.Logger.Warn().Msgf("⚠️ Skipping skill at index %d with empty ID", i)
			continue
		}

		skillPath := fmt.Sprintf("m/44'/%d'/%d'/1'/%d'", haxenServerHash, agentIndex, validSkillIndex)
		skillDID, skillPrivKey, skillPubKey, err := s.generateDIDWithKeys(registry.MasterSeed, skillPath)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate skill DID for %s: %v", skill.ID, err),
			}, nil
		}

		skillDIDs[skill.ID] = types.DIDIdentity{
			DID:            skillDID,
			PrivateKeyJWK:  skillPrivKey,
			PublicKeyJWK:   skillPubKey,
			DerivationPath: skillPath,
			ComponentType:  "skill",
			FunctionName:   skill.ID,
		}

		skillInfos[skill.ID] = types.SkillDIDInfo{
			DID:            skillDID,
			FunctionName:   skill.ID,
			PublicKeyJWK:   json.RawMessage(skillPubKey),
			DerivationPath: skillPath,
			Tags:           skill.Tags,
			ExposureLevel:  "internal",
			CreatedAt:      time.Now(),
		}

		validSkillIndex++
		logger.Logger.Debug().Msgf("🔍 Created DID for skill %s: %s", skill.ID, skillDID)
	}

	logger.Logger.Debug().Msgf("🔍 Successfully created %d skill DIDs out of %d total skills", len(skillDIDs), len(req.Skills))

	// Create agent DID info
	agentDIDInfo := types.AgentDIDInfo{
		DID:            agentDID,
		AgentNodeID:    req.AgentNodeID,
		PublicKeyJWK:   json.RawMessage(agentPubKey),
		DerivationPath: agentPath,
		Reasoners:      reasonerInfos,
		Skills:         skillInfos,
		Status:         types.AgentDIDStatusActive,
		RegisteredAt:   time.Now(),
	}

	// Update registry
	registry.AgentNodes[req.AgentNodeID] = agentDIDInfo
	registry.TotalDIDs += 1 + len(req.Reasoners) + len(req.Skills)

	// Store updated registry
	if err := s.registry.StoreRegistry(registry); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store updated registry: %v", err),
		}, nil
	}

	// Create identity package
	identityPackage := types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:            agentDID,
			PrivateKeyJWK:  agentPrivKey,
			PublicKeyJWK:   agentPubKey,
			DerivationPath: agentPath,
			ComponentType:  "agent",
		},
		ReasonerDIDs:  reasonerDIDs,
		SkillDIDs:     skillDIDs,
		HaxenServerID: registry.HaxenServerID,
	}

	// Debug log the response structure
	reasonerDIDKeys := make([]string, 0, len(reasonerDIDs))
	for key := range reasonerDIDs {
		reasonerDIDKeys = append(reasonerDIDKeys, key)
	}
	logger.Logger.Debug().Msgf("🔍 DEBUG: DID registration response: {'reasoner_dids': %v, 'skill_dids': %d}", reasonerDIDKeys, len(skillDIDs))

	return &types.DIDRegistrationResponse{
		Success:         true,
		IdentityPackage: identityPackage,
		Message:         fmt.Sprintf("Successfully registered agent %s with %d reasoners and %d skills", req.AgentNodeID, len(reasonerDIDs), len(skillDIDs)),
	}, nil
}

// ResolveDID resolves a DID to its public key and metadata.
func (s *DIDService) ResolveDID(did string) (*types.DIDIdentity, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return nil, fmt.Errorf("haxen server registry validation failed: %w", err)
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get haxen server ID: %w", err)
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get DID registry: %w", err)
	}

	// Check if this is the haxen server root DID
	if registry.RootDID == did {
		// Regenerate private key for root DID using root derivation path
		privateKeyJWK, err := s.regeneratePrivateKeyJWK(registry.MasterSeed, "m/44'/0'")
		if err != nil {
			return nil, fmt.Errorf("failed to regenerate private key for root DID %s: %w", did, err)
		}

		// Generate public key JWK for consistency
		publicKeyJWK, err := s.regeneratePublicKeyJWK(registry.MasterSeed, "m/44'/0'")
		if err != nil {
			return nil, fmt.Errorf("failed to regenerate public key for root DID %s: %w", did, err)
		}

		return &types.DIDIdentity{
			DID:            registry.RootDID,
			PrivateKeyJWK:  privateKeyJWK,
			PublicKeyJWK:   publicKeyJWK,
			DerivationPath: "m/44'/0'",
			ComponentType:  "haxen_server",
		}, nil
	}

	// Search through all agent nodes and their components
	for _, agentInfo := range registry.AgentNodes {
		if agentInfo.DID == did {
			// Regenerate private key from master seed and derivation path
			privateKeyJWK, err := s.regeneratePrivateKeyJWK(registry.MasterSeed, agentInfo.DerivationPath)
			if err != nil {
				return nil, fmt.Errorf("failed to regenerate private key for agent DID %s: %w", did, err)
			}

			return &types.DIDIdentity{
				DID:            agentInfo.DID,
				PrivateKeyJWK:  privateKeyJWK,
				PublicKeyJWK:   string(agentInfo.PublicKeyJWK),
				DerivationPath: agentInfo.DerivationPath,
				ComponentType:  "agent",
			}, nil
		}

		// Check reasoners
		for _, reasonerInfo := range agentInfo.Reasoners {
			if reasonerInfo.DID == did {
				// Regenerate private key from master seed and derivation path
				privateKeyJWK, err := s.regeneratePrivateKeyJWK(registry.MasterSeed, reasonerInfo.DerivationPath)
				if err != nil {
					return nil, fmt.Errorf("failed to regenerate private key for reasoner DID %s: %w", did, err)
				}

				return &types.DIDIdentity{
					DID:            reasonerInfo.DID,
					PrivateKeyJWK:  privateKeyJWK,
					PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
					DerivationPath: reasonerInfo.DerivationPath,
					ComponentType:  "reasoner",
					FunctionName:   reasonerInfo.FunctionName,
				}, nil
			}
		}

		// Check skills
		for _, skillInfo := range agentInfo.Skills {
			if skillInfo.DID == did {
				// Regenerate private key from master seed and derivation path
				privateKeyJWK, err := s.regeneratePrivateKeyJWK(registry.MasterSeed, skillInfo.DerivationPath)
				if err != nil {
					return nil, fmt.Errorf("failed to regenerate private key for skill DID %s: %w", did, err)
				}

				return &types.DIDIdentity{
					DID:            skillInfo.DID,
					PrivateKeyJWK:  privateKeyJWK,
					PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
					DerivationPath: skillInfo.DerivationPath,
					ComponentType:  "skill",
					FunctionName:   skillInfo.FunctionName,
				}, nil
			}
		}
	}

	return nil, fmt.Errorf("DID not found: %s", did)
}

// generateDIDWithKeys generates a DID with private and public keys from master seed and derivation path.
func (s *DIDService) generateDIDWithKeys(masterSeed []byte, derivationPath string) (string, string, string, error) {
	// Derive private key using simplified BIP32-style derivation
	privateKey, err := s.derivePrivateKey(masterSeed, derivationPath)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to derive private key: %w", err)
	}

	// Generate Ed25519 key pair
	publicKey := privateKey.Public().(ed25519.PublicKey)

	// Generate DID:key
	did := s.generateDIDKey(publicKey)

	// Convert keys to JWK format
	privateKeyJWK, err := s.ed25519PrivateKeyToJWK(privateKey)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to convert private key to JWK: %w", err)
	}

	publicKeyJWK, err := s.ed25519PublicKeyToJWK(publicKey)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to convert public key to JWK: %w", err)
	}

	return did, privateKeyJWK, publicKeyJWK, nil
}

// generateDIDFromSeed generates a DID from master seed and derivation path.
func (s *DIDService) generateDIDFromSeed(masterSeed []byte, derivationPath string) (string, error) {
	privateKey, err := s.derivePrivateKey(masterSeed, derivationPath)
	if err != nil {
		return "", fmt.Errorf("failed to derive private key: %w", err)
	}

	publicKey := privateKey.Public().(ed25519.PublicKey)
	return s.generateDIDKey(publicKey), nil
}

// derivePrivateKey derives a private key from master seed using simplified BIP32-style derivation.
func (s *DIDService) derivePrivateKey(masterSeed []byte, derivationPath string) (ed25519.PrivateKey, error) {
	// Simplified derivation: hash master seed with derivation path
	h := sha256.New()
	h.Write(masterSeed)
	h.Write([]byte(derivationPath))
	derivedSeed := h.Sum(nil)

	// Generate Ed25519 private key from derived seed
	privateKey := ed25519.NewKeyFromSeed(derivedSeed)
	return privateKey, nil
}

// generateDIDKey generates a DID:key from an Ed25519 public key.
func (s *DIDService) generateDIDKey(publicKey ed25519.PublicKey) string {
	// DID:key format: did:key:z + base58(multicodec + public key)
	// For Ed25519, multicodec prefix is 0xed01
	multicodecKey := append([]byte{0xed, 0x01}, publicKey...)

	// Use base64 encoding for simplicity (in production, use base58)
	encoded := base64.RawURLEncoding.EncodeToString(multicodecKey)
	return fmt.Sprintf("did:key:z%s", encoded)
}

// ed25519PrivateKeyToJWK converts an Ed25519 private key to JWK format.
func (s *DIDService) ed25519PrivateKeyToJWK(privateKey ed25519.PrivateKey) (string, error) {
	publicKey := privateKey.Public().(ed25519.PublicKey)

	jwk := map[string]interface{}{
		"kty": "OKP",
		"crv": "Ed25519",
		"x":   base64.RawURLEncoding.EncodeToString(publicKey),
		"d":   base64.RawURLEncoding.EncodeToString(privateKey.Seed()),
		"use": "sig",
		"alg": "EdDSA",
	}

	jwkBytes, err := json.Marshal(jwk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal JWK: %w", err)
	}

	return string(jwkBytes), nil
}

// ed25519PublicKeyToJWK converts an Ed25519 public key to JWK format.
func (s *DIDService) ed25519PublicKeyToJWK(publicKey ed25519.PublicKey) (string, error) {
	jwk := map[string]interface{}{
		"kty": "OKP",
		"crv": "Ed25519",
		"x":   base64.RawURLEncoding.EncodeToString(publicKey),
		"use": "sig",
		"alg": "EdDSA",
	}

	jwkBytes, err := json.Marshal(jwk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal JWK: %w", err)
	}

	return string(jwkBytes), nil
}

// hashHaxenServerID creates a deterministic hash of haxen server ID for derivation paths.
func (s *DIDService) hashHaxenServerID(haxenServerID string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(haxenServerID))
	return h.Sum32() % (1 << 31) // Ensure it fits in BIP32 hardened derivation
}

// regeneratePrivateKeyJWK regenerates a private key JWK from master seed and derivation path.
func (s *DIDService) regeneratePrivateKeyJWK(masterSeed []byte, derivationPath string) (string, error) {
	// Derive private key using the same method as during generation
	privateKey, err := s.derivePrivateKey(masterSeed, derivationPath)
	if err != nil {
		return "", fmt.Errorf("failed to derive private key: %w", err)
	}

	// Convert to JWK format
	privateKeyJWK, err := s.ed25519PrivateKeyToJWK(privateKey)
	if err != nil {
		return "", fmt.Errorf("failed to convert private key to JWK: %w", err)
	}

	return privateKeyJWK, nil
}

// regeneratePublicKeyJWK regenerates a public key JWK from master seed and derivation path.
func (s *DIDService) regeneratePublicKeyJWK(masterSeed []byte, derivationPath string) (string, error) {
	// Derive private key using the same method as during generation
	privateKey, err := s.derivePrivateKey(masterSeed, derivationPath)
	if err != nil {
		return "", fmt.Errorf("failed to derive private key: %w", err)
	}

	// Get public key from private key
	publicKey := privateKey.Public().(ed25519.PublicKey)

	// Convert to JWK format
	publicKeyJWK, err := s.ed25519PublicKeyToJWK(publicKey)
	if err != nil {
		return "", fmt.Errorf("failed to convert public key to JWK: %w", err)
	}

	return publicKeyJWK, nil
}

// ListAllAgentDIDs returns all registered agent DIDs from the registry.
func (s *DIDService) ListAllAgentDIDs() ([]string, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return nil, fmt.Errorf("haxen server registry validation failed: %w", err)
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get haxen server ID: %w", err)
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get DID registry: %w", err)
	}

	var agentDIDs []string
	for _, agentInfo := range registry.AgentNodes {
		agentDIDs = append(agentDIDs, agentInfo.DID)
	}

	return agentDIDs, nil
}

// BackfillExistingNodes registers existing nodes that don't have DIDs
func (s *DIDService) BackfillExistingNodes(ctx context.Context, storageProvider storage.StorageProvider) error {
	if !s.config.Enabled {
		logger.Logger.Debug().Msg("🔍 DID system disabled, skipping backfill")
		return nil
	}

	logger.Logger.Debug().Msg("🔍 Starting DID backfill for existing nodes...")

	// Get all registered nodes
	nodes, err := storageProvider.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return fmt.Errorf("failed to list agents: %w", err)
	}

	if len(nodes) == 0 {
		logger.Logger.Debug().Msg("🔍 No existing nodes found, backfill complete")
		return nil
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return fmt.Errorf("haxen server registry validation failed: %w", err)
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return fmt.Errorf("failed to get haxen server ID: %w", err)
	}

	// Get current DID registry using dynamic ID
	registry, err := s.GetRegistry(haxenServerID)
	if err != nil {
		return fmt.Errorf("failed to get DID registry: %w", err)
	}

	backfillCount := 0
	skippedCount := 0

	for _, node := range nodes {
		// Check if node already has DID
		if registry != nil {
			if _, exists := registry.AgentNodes[node.ID]; exists {
				logger.Logger.Debug().Msgf("🔍 Node %s already has DID, skipping", node.ID)
				skippedCount++
				continue // Already has DID
			}
		}

		// Register node with DID system
		didReq := &types.DIDRegistrationRequest{
			AgentNodeID: node.ID,
			Reasoners:   node.Reasoners,
			Skills:      node.Skills,
		}

		didResponse, err := s.RegisterAgent(didReq)
		if err != nil {
			logger.Logger.Warn().Err(err).Msgf("⚠️ Failed to backfill DID for node %s", node.ID)
		} else if !didResponse.Success {
			logger.Logger.Warn().Msgf("⚠️ DID backfill unsuccessful for node %s: %s", node.ID, didResponse.Error)
		} else {
			logger.Logger.Debug().Msgf("✅ Backfilled DID for node %s: %s", node.ID, didResponse.IdentityPackage.AgentDID.DID)
			backfillCount++
		}
	}

	logger.Logger.Debug().Msgf("🎉 DID backfill completed: %d nodes processed, %d new DIDs created, %d nodes already had DIDs",
		len(nodes), backfillCount, skippedCount)
	return nil
}

// GetExistingAgentDID retrieves existing DID information for an agent node.
func (s *DIDService) GetExistingAgentDID(agentNodeID string) (*types.AgentDIDInfo, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return nil, fmt.Errorf("haxen server registry validation failed: %w", err)
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get haxen server ID: %w", err)
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get DID registry: %w", err)
	}

	agentInfo, exists := registry.AgentNodes[agentNodeID]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", agentNodeID)
	}

	return &agentInfo, nil
}

// PerformDifferentialAnalysis compares existing vs new reasoners/skills to determine what needs to be updated.
func (s *DIDService) PerformDifferentialAnalysis(agentNodeID string, newReasonerIDs, newSkillIDs []string) (*types.DifferentialAnalysisResult, error) {
	existingAgent, err := s.GetExistingAgentDID(agentNodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get existing agent: %w", err)
	}

	// Extract existing IDs
	existingReasonerIDs := make([]string, 0, len(existingAgent.Reasoners))
	for id := range existingAgent.Reasoners {
		existingReasonerIDs = append(existingReasonerIDs, id)
	}

	existingSkillIDs := make([]string, 0, len(existingAgent.Skills))
	for id := range existingAgent.Skills {
		existingSkillIDs = append(existingSkillIDs, id)
	}

	// Perform set operations
	result := &types.DifferentialAnalysisResult{
		NewReasonerIDs:     setDifference(newReasonerIDs, existingReasonerIDs),
		RemovedReasonerIDs: setDifference(existingReasonerIDs, newReasonerIDs),
		UpdatedReasonerIDs: setIntersection(newReasonerIDs, existingReasonerIDs),
		NewSkillIDs:        setDifference(newSkillIDs, existingSkillIDs),
		RemovedSkillIDs:    setDifference(existingSkillIDs, newSkillIDs),
		UpdatedSkillIDs:    setIntersection(newSkillIDs, existingSkillIDs),
	}

	result.RequiresUpdate = len(result.NewReasonerIDs) > 0 ||
		len(result.RemovedReasonerIDs) > 0 ||
		len(result.NewSkillIDs) > 0 ||
		len(result.RemovedSkillIDs) > 0

	logger.Logger.Debug().Msgf("🔍 Differential analysis for agent %s: new_reasoners=%d, removed_reasoners=%d, new_skills=%d, removed_skills=%d, requires_update=%v",
		agentNodeID, len(result.NewReasonerIDs), len(result.RemovedReasonerIDs), len(result.NewSkillIDs), len(result.RemovedSkillIDs), result.RequiresUpdate)

	return result, nil
}

// setDifference returns elements in slice a that are not in slice b.
func setDifference(a, b []string) []string {
	bMap := make(map[string]bool)
	for _, item := range b {
		bMap[item] = true
	}

	var result []string
	for _, item := range a {
		if !bMap[item] {
			result = append(result, item)
		}
	}
	return result
}

// setIntersection returns elements that are in both slice a and slice b.
func setIntersection(a, b []string) []string {
	bMap := make(map[string]bool)
	for _, item := range b {
		bMap[item] = true
	}

	var result []string
	for _, item := range a {
		if bMap[item] {
			result = append(result, item)
		}
	}
	return result
}

// extractReasonerIDs extracts reasoner IDs from reasoner definitions.
func extractReasonerIDs(reasoners []types.ReasonerDefinition) []string {
	ids := make([]string, 0, len(reasoners))
	for _, reasoner := range reasoners {
		if reasoner.ID != "" {
			ids = append(ids, reasoner.ID)
		}
	}
	return ids
}

// extractSkillIDs extracts skill IDs from skill definitions.
func extractSkillIDs(skills []types.SkillDefinition) []string {
	ids := make([]string, 0, len(skills))
	for _, skill := range skills {
		if skill.ID != "" {
			ids = append(ids, skill.ID)
		}
	}
	return ids
}

// findReasonerByID finds a reasoner definition by ID.
func (s *DIDService) findReasonerByID(reasoners []types.ReasonerDefinition, id string) *types.ReasonerDefinition {
	for _, reasoner := range reasoners {
		if reasoner.ID == id {
			return &reasoner
		}
	}
	return nil
}

// findSkillByID finds a skill definition by ID.
func (s *DIDService) findSkillByID(skills []types.SkillDefinition, id string) *types.SkillDefinition {
	for _, skill := range skills {
		if skill.ID == id {
			return &skill
		}
	}
	return nil
}

// generateReasonerPath generates a derivation path for a reasoner.
func (s *DIDService) generateReasonerPath(agentNodeID, reasonerID string) string {
	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get haxen server ID for reasoner path generation")
		return ""
	}

	// Get registry to find agent index
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get registry for reasoner path generation")
		return ""
	}

	// Generate haxen server hash for derivation path
	haxenServerHash := s.hashHaxenServerID(registry.HaxenServerID)

	// Find agent index (this is a simplified approach - in production you might want to store this)
	agentIndex := 0
	for nodeID := range registry.AgentNodes {
		if nodeID == agentNodeID {
			break
		}
		agentIndex++
	}

	// Count existing reasoners to get next index
	existingAgent := registry.AgentNodes[agentNodeID]
	reasonerIndex := len(existingAgent.Reasoners)

	return fmt.Sprintf("m/44'/%d'/%d'/0'/%d'", haxenServerHash, agentIndex, reasonerIndex)
}

// generateSkillPath generates a derivation path for a skill.
func (s *DIDService) generateSkillPath(agentNodeID, skillID string) string {
	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get haxen server ID for skill path generation")
		return ""
	}

	// Get registry to find agent index
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get registry for skill path generation")
		return ""
	}

	// Generate haxen server hash for derivation path
	haxenServerHash := s.hashHaxenServerID(registry.HaxenServerID)

	// Find agent index (this is a simplified approach - in production you might want to store this)
	agentIndex := 0
	for nodeID := range registry.AgentNodes {
		if nodeID == agentNodeID {
			break
		}
		agentIndex++
	}

	// Count existing skills to get next index
	existingAgent := registry.AgentNodes[agentNodeID]
	skillIndex := len(existingAgent.Skills)

	return fmt.Sprintf("m/44'/%d'/%d'/1'/%d'", haxenServerHash, agentIndex, skillIndex)
}

// buildExistingIdentityPackage builds an identity package from existing agent DID info.
func (s *DIDService) buildExistingIdentityPackage(existingAgent *types.AgentDIDInfo) types.DIDIdentityPackage {
	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get haxen server ID for identity package")
		haxenServerID = "unknown"
	}

	// Build reasoner DIDs map
	reasonerDIDs := make(map[string]types.DIDIdentity)
	for id, reasonerInfo := range existingAgent.Reasoners {
		reasonerDIDs[id] = types.DIDIdentity{
			DID:            reasonerInfo.DID,
			PrivateKeyJWK:  "", // Don't include private keys in existing package
			PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
			DerivationPath: reasonerInfo.DerivationPath,
			ComponentType:  "reasoner",
			FunctionName:   reasonerInfo.FunctionName,
		}
	}

	// Build skill DIDs map
	skillDIDs := make(map[string]types.DIDIdentity)
	for id, skillInfo := range existingAgent.Skills {
		skillDIDs[id] = types.DIDIdentity{
			DID:            skillInfo.DID,
			PrivateKeyJWK:  "", // Don't include private keys in existing package
			PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
			DerivationPath: skillInfo.DerivationPath,
			ComponentType:  "skill",
			FunctionName:   skillInfo.FunctionName,
		}
	}

	return types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:            existingAgent.DID,
			PrivateKeyJWK:  "", // Don't include private keys in existing package
			PublicKeyJWK:   string(existingAgent.PublicKeyJWK),
			DerivationPath: existingAgent.DerivationPath,
			ComponentType:  "agent",
		},
		ReasonerDIDs:  reasonerDIDs,
		SkillDIDs:     skillDIDs,
		HaxenServerID: haxenServerID,
	}
}

// handlePartialRegistration handles partial registration for existing agents.
func (s *DIDService) handlePartialRegistration(req *types.DIDRegistrationRequest, diffResult *types.DifferentialAnalysisResult) (*types.DIDRegistrationResponse, error) {
	// Handle deregistration of removed components first
	if len(diffResult.RemovedReasonerIDs) > 0 || len(diffResult.RemovedSkillIDs) > 0 {
		deregReq := &types.ComponentDeregistrationRequest{
			AgentNodeID:         req.AgentNodeID,
			ReasonerIDsToRemove: diffResult.RemovedReasonerIDs,
			SkillIDsToRemove:    diffResult.RemovedSkillIDs,
		}

		deregResponse, err := s.DeregisterComponents(deregReq)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("component deregistration failed: %v", err),
			}, nil
		}

		if !deregResponse.Success {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("component deregistration failed: %s", deregResponse.Error),
			}, nil
		}

		logger.Logger.Debug().Msgf("✅ Deregistered %d components for agent %s", deregResponse.RemovedCount, req.AgentNodeID)
	}

	// Handle partial registration of new components
	if len(diffResult.NewReasonerIDs) > 0 || len(diffResult.NewSkillIDs) > 0 {
		partialReq := &types.PartialDIDRegistrationRequest{
			AgentNodeID:        req.AgentNodeID,
			NewReasonerIDs:     diffResult.NewReasonerIDs,
			NewSkillIDs:        diffResult.NewSkillIDs,
			UpdatedReasonerIDs: diffResult.UpdatedReasonerIDs,
			UpdatedSkillIDs:    diffResult.UpdatedSkillIDs,
			AllReasoners:       req.Reasoners,
			AllSkills:          req.Skills,
		}

		return s.PartialRegisterAgent(partialReq)
	}

	// If we reach here, only removals were needed
	existingAgent, err := s.GetExistingAgentDID(req.AgentNodeID)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get updated agent info: %v", err),
		}, nil
	}

	identityPackage := s.buildExistingIdentityPackage(existingAgent)
	return &types.DIDRegistrationResponse{
		Success:         true,
		Message:         fmt.Sprintf("Registration updated: removed %d reasoners, %d skills", len(diffResult.RemovedReasonerIDs), len(diffResult.RemovedSkillIDs)),
		IdentityPackage: identityPackage,
	}, nil
}

// PartialRegisterAgent registers only new components for an existing agent.
func (s *DIDService) PartialRegisterAgent(req *types.PartialDIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	if !s.config.Enabled {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("haxen server registry validation failed: %v", err),
		}, nil
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get haxen server ID: %v", err),
		}, nil
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get DID registry: %v", err),
		}, nil
	}

	existingAgent, exists := registry.AgentNodes[req.AgentNodeID]
	if !exists {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("agent %s not found", req.AgentNodeID),
		}, nil
	}

	// Generate DIDs for new reasoners only
	newReasonerDIDs := make(map[string]types.DIDIdentity)
	newReasonerInfos := make(map[string]types.ReasonerDIDInfo)

	for _, reasonerID := range req.NewReasonerIDs {
		reasoner := s.findReasonerByID(req.AllReasoners, reasonerID)
		if reasoner == nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("reasoner %s not found in request", reasonerID),
			}, nil
		}

		// Generate DID for new reasoner
		reasonerPath := s.generateReasonerPath(req.AgentNodeID, reasonerID)
		if reasonerPath == "" {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate derivation path for reasoner %s", reasonerID),
			}, nil
		}

		reasonerDID, privKey, pubKey, err := s.generateDIDWithKeys(registry.MasterSeed, reasonerPath)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate DID for reasoner %s: %v", reasonerID, err),
			}, nil
		}

		newReasonerDIDs[reasonerID] = types.DIDIdentity{
			DID:            reasonerDID,
			PrivateKeyJWK:  privKey,
			PublicKeyJWK:   pubKey,
			DerivationPath: reasonerPath,
			ComponentType:  "reasoner",
			FunctionName:   reasonerID,
		}

		newReasonerInfos[reasonerID] = types.ReasonerDIDInfo{
			DID:            reasonerDID,
			FunctionName:   reasonerID,
			PublicKeyJWK:   json.RawMessage(pubKey),
			DerivationPath: reasonerPath,
			Capabilities:   []string{}, // Default empty capabilities
			ExposureLevel:  "internal",
			CreatedAt:      time.Now(),
		}

		logger.Logger.Debug().Msgf("🔍 Generated new DID for reasoner %s: %s", reasonerID, reasonerDID)
	}

	// Generate DIDs for new skills
	newSkillDIDs := make(map[string]types.DIDIdentity)
	newSkillInfos := make(map[string]types.SkillDIDInfo)

	for _, skillID := range req.NewSkillIDs {
		skill := s.findSkillByID(req.AllSkills, skillID)
		if skill == nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("skill %s not found in request", skillID),
			}, nil
		}

		// Generate DID for new skill
		skillPath := s.generateSkillPath(req.AgentNodeID, skillID)
		if skillPath == "" {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate derivation path for skill %s", skillID),
			}, nil
		}

		skillDID, privKey, pubKey, err := s.generateDIDWithKeys(registry.MasterSeed, skillPath)
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to generate DID for skill %s: %v", skillID, err),
			}, nil
		}

		newSkillDIDs[skillID] = types.DIDIdentity{
			DID:            skillDID,
			PrivateKeyJWK:  privKey,
			PublicKeyJWK:   pubKey,
			DerivationPath: skillPath,
			ComponentType:  "skill",
			FunctionName:   skillID,
		}

		newSkillInfos[skillID] = types.SkillDIDInfo{
			DID:            skillDID,
			FunctionName:   skillID,
			PublicKeyJWK:   json.RawMessage(pubKey),
			DerivationPath: skillPath,
			Tags:           skill.Tags,
			ExposureLevel:  "internal",
			CreatedAt:      time.Now(),
		}

		logger.Logger.Debug().Msgf("🔍 Generated new DID for skill %s: %s", skillID, skillDID)
	}

	// Update existing agent info with new components
	for id, info := range newReasonerInfos {
		existingAgent.Reasoners[id] = info
	}
	for id, info := range newSkillInfos {
		existingAgent.Skills[id] = info
	}

	// Update registry
	registry.AgentNodes[req.AgentNodeID] = existingAgent
	registry.TotalDIDs += len(newReasonerDIDs) + len(newSkillDIDs)

	if err := s.registry.StoreRegistry(registry); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store updated registry: %v", err),
		}, nil
	}

	// Build response with only new DIDs
	identityPackage := types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:            existingAgent.DID,
			PrivateKeyJWK:  "", // Don't regenerate existing agent key
			PublicKeyJWK:   string(existingAgent.PublicKeyJWK),
			DerivationPath: existingAgent.DerivationPath,
			ComponentType:  "agent",
		},
		ReasonerDIDs:  newReasonerDIDs,
		SkillDIDs:     newSkillDIDs,
		HaxenServerID: registry.HaxenServerID,
	}

	logger.Logger.Debug().Msgf("✅ Partial registration successful for agent %s: %d new reasoners, %d new skills",
		req.AgentNodeID, len(newReasonerDIDs), len(newSkillDIDs))

	return &types.DIDRegistrationResponse{
		Success:         true,
		IdentityPackage: identityPackage,
		Message:         fmt.Sprintf("Partial registration successful: %d new reasoners, %d new skills", len(newReasonerDIDs), len(newSkillDIDs)),
	}, nil
}

// DeregisterComponents removes specific components from an agent's DID registry.
func (s *DIDService) DeregisterComponents(req *types.ComponentDeregistrationRequest) (*types.ComponentDeregistrationResponse, error) {
	if !s.config.Enabled {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate haxen server registry exists
	if err := s.validateHaxenServerRegistry(); err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("haxen server registry validation failed: %v", err),
		}, nil
	}

	// Get haxen server ID dynamically
	haxenServerID, err := s.getHaxenServerID()
	if err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get haxen server ID: %v", err),
		}, nil
	}

	// Get haxen server registry using dynamic ID
	registry, err := s.registry.GetRegistry(haxenServerID)
	if err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get DID registry: %v", err),
		}, nil
	}

	existingAgent, exists := registry.AgentNodes[req.AgentNodeID]
	if !exists {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("agent %s not found", req.AgentNodeID),
		}, nil
	}

	removedCount := 0

	// Remove reasoners
	for _, reasonerID := range req.ReasonerIDsToRemove {
		if _, exists := existingAgent.Reasoners[reasonerID]; exists {
			delete(existingAgent.Reasoners, reasonerID)
			removedCount++
			logger.Logger.Debug().Msgf("🗑️ Removed reasoner DID: %s from agent %s", reasonerID, req.AgentNodeID)
		} else {
			logger.Logger.Warn().Msgf("⚠️ Reasoner %s not found in agent %s, skipping removal", reasonerID, req.AgentNodeID)
		}
	}

	// Remove skills
	for _, skillID := range req.SkillIDsToRemove {
		if _, exists := existingAgent.Skills[skillID]; exists {
			delete(existingAgent.Skills, skillID)
			removedCount++
			logger.Logger.Debug().Msgf("🗑️ Removed skill DID: %s from agent %s", skillID, req.AgentNodeID)
		} else {
			logger.Logger.Warn().Msgf("⚠️ Skill %s not found in agent %s, skipping removal", skillID, req.AgentNodeID)
		}
	}

	// Update registry
	registry.AgentNodes[req.AgentNodeID] = existingAgent
	registry.TotalDIDs -= removedCount

	if err := s.registry.StoreRegistry(registry); err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store updated registry: %v", err),
		}, nil
	}

	logger.Logger.Debug().Msgf("✅ Component deregistration successful for agent %s: removed %d components",
		req.AgentNodeID, removedCount)

	return &types.ComponentDeregistrationResponse{
		Success:      true,
		RemovedCount: removedCount,
		Message:      fmt.Sprintf("Successfully removed %d components", removedCount),
	}, nil
}
