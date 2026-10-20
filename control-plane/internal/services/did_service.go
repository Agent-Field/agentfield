package services

import (
	"context"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5/pgconn"
	"golang.org/x/crypto/hkdf"
)

// DIDService handles DID generation, management, and resolution.
type DIDService struct {
	// mu serializes allocation within this service instance. Other replicas
	// are coordinated by the storage conflict check and retry.
	mu                 sync.Mutex
	config             *config.DIDConfig
	keystore           *KeystoreService
	registry           *DIDRegistry
	agentfieldServerID string
}

// NewDIDService creates a new DID service instance.
func NewDIDService(cfg *config.DIDConfig, keystore *KeystoreService, registry *DIDRegistry) *DIDService {
	return &DIDService{
		config:             cfg,
		keystore:           keystore,
		registry:           registry,
		agentfieldServerID: "", // Will be set during initialization
	}
}

// Initialize initializes the DID service and creates af server master seed if needed.
func (s *DIDService) Initialize(agentfieldServerID string) error {
	if !s.config.Enabled {
		return nil
	}

	// Store the af server ID for dynamic resolution
	s.agentfieldServerID = agentfieldServerID

	// Refuse to run on a registry whose stored key material is inconsistent,
	// rather than signing with keys that do not match the published DIDs.
	if err := s.registry.IntegrityError(agentfieldServerID); err != nil {
		return fmt.Errorf("DID registry for af server %s failed integrity check: %w", agentfieldServerID, err)
	}

	// Check if af server already has a DID registry
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return fmt.Errorf("failed to check existing registry: %w", err)
	}

	if registry == nil {
		// Create new af server registry
		masterSeed := make([]byte, 32)
		if _, err := rand.Read(masterSeed); err != nil {
			return fmt.Errorf("failed to generate master seed: %w", err)
		}

		// Generate root DID from master seed
		rootDID, err := s.generateDIDFromSeed(masterSeed, rootDerivationPath)
		if err != nil {
			return fmt.Errorf("failed to generate root DID: %w", err)
		}

		// Create and store registry
		registry = &types.DIDRegistry{
			AgentFieldServerID: agentfieldServerID,
			MasterSeed:         masterSeed,
			RootDID:            rootDID,
			AgentNodes:         make(map[string]types.AgentDIDInfo),
			TotalDIDs:          1,
			CreatedAt:          time.Now(),
			LastKeyRotation:    time.Now(),
		}

		// Another control plane sharing this database may create the same
		// server ID at the same time; CreateRegistry keeps whichever identity
		// was stored first instead of overwriting it.
		if _, err := s.registry.CreateRegistry(registry); err != nil {
			return fmt.Errorf("failed to store DID registry: %w", err)
		}
	}

	return nil
}

// GetAgentFieldServerID returns the af server ID for this DID service instance.
// This method provides dynamic af server ID resolution instead of hardcoded "default".
func (s *DIDService) GetAgentFieldServerID() (string, error) {
	if s.agentfieldServerID == "" {
		return "", fmt.Errorf("af server ID not initialized - call Initialize() first")
	}
	return s.agentfieldServerID, nil
}

// getAgentFieldServerID is an internal helper that returns the af server ID.
func (s *DIDService) getAgentFieldServerID() (string, error) {
	return s.GetAgentFieldServerID()
}

// GetControlPlaneIssuerDID returns the root DID (did:key format) for the
// control plane, suitable for signing VCs. This DID is resolvable via
// ResolveDID(), unlike the did:web URI returned by GenerateDIDWeb().
func (s *DIDService) GetControlPlaneIssuerDID() (string, error) {
	if !s.config.Enabled {
		return "", fmt.Errorf("DID system is disabled")
	}
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return "", err
	}
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return "", fmt.Errorf("failed to get DID registry: %w", err)
	}
	if registry.RootDID == "" {
		return "", fmt.Errorf("root DID not initialized")
	}
	return registry.RootDID, nil
}

// validateAgentFieldServerRegistry ensures that the af server registry exists before operations.
func (s *DIDService) validateAgentFieldServerRegistry() error {
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return err
	}

	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return fmt.Errorf("failed to get af server registry: %w", err)
	}

	if registry == nil {
		return fmt.Errorf("af server registry not found for ID: %s - ensure Initialize() was called", agentfieldServerID)
	}

	return nil
}

// GetRegistry retrieves a DID registry for a af server.
func (s *DIDService) GetRegistry(agentfieldServerID string) (*types.DIDRegistry, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.registry.GetRegistry(agentfieldServerID)
}

// RegisterAgent generates DIDs for an agent node and all its components.
// Enhanced to support partial registration for existing agents.
func (s *DIDService) RegisterAgent(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), didRegistrationTimeout)
	defer cancel()
	return retryDIDAllocation(ctx, func() (*types.DIDRegistrationResponse, error) {
		return s.registerAgentOnce(ctx, req)
	})
}

func (s *DIDService) registerAgentOnce(ctx context.Context, req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {

	if !s.config.Enabled {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("af server registry validation failed: %v", err),
		}, nil
	}

	// Check if agent already exists
	existingAgent, err := s.getExistingAgentDID(ctx, req.AgentNodeID)
	if err != nil && err.Error() != fmt.Sprintf("agent not found: %s", req.AgentNodeID) {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to check existing agent: %v", err),
		}, nil
	}

	if existingAgent != nil {
		if reason := s.quarantinedIdentityReason(existingAgent); reason != "" {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   quarantinedAgentError(req.AgentNodeID, reason),
			}, nil
		}

		// Perform differential analysis
		newReasonerIDs := extractReasonerIDs(req.Reasoners)
		newSkillIDs := extractSkillIDs(req.Skills)

		diffResult, err := s.performDifferentialAnalysis(ctx, req.AgentNodeID, newReasonerIDs, newSkillIDs)
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
		return s.handlePartialRegistrationContext(ctx, req, diffResult)
	}

	// Handle new registration (existing logic)
	return s.handleNewRegistration(ctx, req)
}

// handleNewRegistration handles registration for new agents (original logic).
func (s *DIDService) handleNewRegistration(ctx context.Context, req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	return s.handleNewRegistrationOnce(ctx, req)
}

// Each attempt reads the issued identities again before deriving a path. The
// per-service mutex does not serialize other control-plane replicas.
const maxDIDAllocationAttempts = 64

// Lock/busy failures have already waited in the database; retry those only once.
const maxDIDContentionRetries = 1

// Bound this registration's work after acquiring mu, excluding time in the queue.
// Tests may shorten the deadline without waiting 30 seconds per queued request.
var didRegistrationTimeout = 30 * time.Second

func retryDIDAllocation(ctx context.Context, attempt func() (*types.DIDRegistrationResponse, error)) (*types.DIDRegistrationResponse, error) {
	var lastErr error
	contentionErrors, attempts := 0, 0
	backoff := 5 * time.Millisecond
	for i := 0; i < maxDIDAllocationAttempts && ctx.Err() == nil; i++ {
		response, err := attempt()
		attempts++
		if err == nil && response != nil && response.Success {
			return response, nil
		}
		if ctx.Err() != nil {
			// Keep the previous database failure if cancellation interrupted a retry.
			if lastErr == nil {
				lastErr = err
				if lastErr == nil && response != nil {
					lastErr = errors.New(response.Error)
				}
			}
			break
		}
		if !errors.Is(err, storage.ErrDIDAllocationConflict) && !errors.Is(err, storage.ErrDIDNodeAlreadyRegistered) && !storage.IsTransientDIDTransactionError(err) {
			return response, err
		}
		lastErr = err
		var jitter [1]byte
		_, _ = rand.Read(jitter[:])
		delay := time.Duration(1+int(jitter[0])%3) * time.Millisecond
		if storage.IsTransientDIDTransactionError(err) {
			var pgErr *pgconn.PgError
			if errors.As(err, &pgErr) && (pgErr.Code == "40001" || pgErr.Code == "40P01") {
				// Equal jitter spreads replicas across [backoff/2, backoff], capped at 250ms.
				delay = backoff/2 + time.Duration(jitter[0])*backoff/512
				backoff = min(2*backoff, 250*time.Millisecond)
			} else {
				contentionErrors++
				if contentionErrors > maxDIDContentionRetries {
					break
				}
			}
		}
		if ctx.Err() != nil {
			break
		}
		if i+1 < maxDIDAllocationAttempts {
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
			case <-timer.C:
			}
			timer.Stop()
		}
	}
	if lastErr == nil {
		lastErr = ctx.Err()
	}
	message := fmt.Sprintf("DID allocation failed after %d attempts: %v", attempts, lastErr)
	if errors.Is(lastErr, storage.ErrDIDAllocationConflict) || errors.Is(lastErr, storage.ErrDIDNodeAlreadyRegistered) {
		message = fmt.Sprintf("DID allocation kept colliding after %d attempts: %v", attempts, lastErr)
	}
	if ctx.Err() != nil {
		message += fmt.Sprintf(" (registration deadline: %v)", ctx.Err())
	}
	return &types.DIDRegistrationResponse{Success: false, Error: message}, nil
}

func (s *DIDService) handleNewRegistrationOnce(ctx context.Context, req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get af server ID: %v", err),
		}, nil
	}

	// Get af server registry using dynamic ID
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get DID registry: %v", err),
		}, nil
	}

	// Allocate paths that were never issued from this database, by any agent
	// of any server, so a new identity can never be handed an existing key.
	used, err := s.registry.usedIdentities(ctx)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to read issued DIDs: %v", err),
		}, nil
	}
	agentPath, err := nextAgentPathFor(registry, used)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to allocate agent derivation path: %v", err),
		}, nil
	}
	paths := newComponentPathAllocator(registry.MasterSeed, agentPath, used)
	agentDID, agentPrivKey, agentPubKey, err := s.generateDIDWithKeys(registry.MasterSeed, agentPath)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to generate agent DID: %v", err),
		}, nil
	}

	// Derive the agent's X25519 keyAgreement (encryption) keypair from the same
	// master seed and derivation path (distinct HKDF salt => independent key).
	// New agents start at rotation epoch 0.
	agentX25519Epoch := 0
	agentX25519PubKey, agentX25519PrivKey, err := s.regenerateX25519KeyPairJWKAtEpoch(registry.MasterSeed, agentPath, agentX25519Epoch)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to generate agent X25519 keyAgreement key: %v", err),
		}, nil
	}

	// Generate reasoner DIDs
	reasonerDIDs := make(map[string]types.DIDIdentity)
	reasonerInfos := make(map[string]types.ReasonerDIDInfo)

	logger.Logger.Debug().Msgf("🔍 DEBUG: Calling did_manager.register_agent() with %d reasoners and %d skills", len(req.Reasoners), len(req.Skills))

	for i, reasoner := range req.Reasoners {
		// Skip reasoners with empty IDs to prevent malformed DIDs
		if reasoner.ID == "" {
			logger.Logger.Warn().Msgf("⚠️ Skipping reasoner at index %d with empty ID", i)
			continue
		}

		reasonerPath, err := paths.allocate("reasoner")
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to allocate derivation path for reasoner %s: %v", reasoner.ID, err),
			}, nil
		}
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

		logger.Logger.Debug().Msgf("🔍 Created DID for reasoner %s: %s", reasoner.ID, reasonerDID)
	}

	logger.Logger.Debug().Msgf("🔍 Successfully created %d reasoner DIDs out of %d total reasoners", len(reasonerDIDs), len(req.Reasoners))

	// Generate skill DIDs
	skillDIDs := make(map[string]types.DIDIdentity)
	skillInfos := make(map[string]types.SkillDIDInfo)

	for i, skill := range req.Skills {
		// Skip skills with empty IDs to prevent malformed DIDs
		if skill.ID == "" {
			logger.Logger.Warn().Msgf("⚠️ Skipping skill at index %d with empty ID", i)
			continue
		}

		skillPath, err := paths.allocate("skill")
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to allocate derivation path for skill %s: %v", skill.ID, err),
			}, nil
		}
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

		logger.Logger.Debug().Msgf("🔍 Created DID for skill %s: %s", skill.ID, skillDID)
	}

	logger.Logger.Debug().Msgf("🔍 Successfully created %d skill DIDs out of %d total skills", len(skillDIDs), len(req.Skills))

	// Create agent DID info
	agentDIDInfo := types.AgentDIDInfo{
		DID:                agentDID,
		AgentNodeID:        req.AgentNodeID,
		PublicKeyJWK:       json.RawMessage(agentPubKey),
		X25519PublicKeyJWK: json.RawMessage(agentX25519PubKey),
		X25519Epoch:        agentX25519Epoch,
		DerivationPath:     agentPath,
		Reasoners:          reasonerInfos,
		Skills:             skillInfos,
		Status:             types.AgentDIDStatusActive,
		RegisteredAt:       time.Now(),
	}

	// Update registry
	err = s.registry.updateAgents(ctx, agentfieldServerID, func(live *types.DIDRegistry) ([]string, error) {
		if _, exists := live.AgentNodes[req.AgentNodeID]; exists {
			return nil, fmt.Errorf("agent %s was registered concurrently: %w", req.AgentNodeID, storage.ErrDIDNodeAlreadyRegistered)
		}
		live.AgentNodes[req.AgentNodeID] = agentDIDInfo
		live.TotalDIDs += 1 + len(reasonerInfos) + len(skillInfos)
		return []string{req.AgentNodeID}, nil
	})
	if err != nil {
		if errors.Is(err, storage.ErrDIDAllocationConflict) || errors.Is(err, storage.ErrDIDNodeAlreadyRegistered) || storage.IsTransientDIDTransactionError(err) {
			return nil, err
		}
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store updated registry: %v", err),
		}, nil
	}

	// Create identity package
	identityPackage := types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:                 agentDID,
			PrivateKeyJWK:       agentPrivKey,
			PublicKeyJWK:        agentPubKey,
			X25519PublicKeyJWK:  agentX25519PubKey,
			X25519PrivateKeyJWK: agentX25519PrivKey,
			X25519Epoch:         agentX25519Epoch,
			DerivationPath:      agentPath,
			ComponentType:       "agent",
		},
		ReasonerDIDs:       reasonerDIDs,
		SkillDIDs:          skillDIDs,
		AgentFieldServerID: registry.AgentFieldServerID,
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

// ResolveDID resolves a DID to its public key and metadata. DIDs issued by an
// earlier registry on the same database stay resolvable, so credentials issued
// before a server identity change remain verifiable. Quarantined identities
// are never resolved.
func (s *DIDService) ResolveDID(did string) (*types.DIDIdentity, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return nil, fmt.Errorf("af server registry validation failed: %w", err)
	}

	if reason := s.registry.QuarantineReason(did); reason != "" {
		return nil, fmt.Errorf("DID %s is not usable: %s", did, reason)
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get af server ID: %w", err)
	}

	// Get af server registry using dynamic ID
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get DID registry: %w", err)
	}

	if identity, found, err := s.resolveInRegistry(registry, did, true); found || err != nil {
		return identity, err
	}

	// Fall back to other registries stored in the same database, in a stable
	// order, skipping any that failed the integrity check.
	others, err := s.registry.ListRegistries()
	if err != nil {
		return nil, fmt.Errorf("failed to list DID registries: %w", err)
	}
	sort.Slice(others, func(i, j int) bool { return others[i].AgentFieldServerID < others[j].AgentFieldServerID })
	for _, other := range others {
		if other.AgentFieldServerID == agentfieldServerID || s.registry.IntegrityError(other.AgentFieldServerID) != nil {
			continue
		}
		if identity, found, err := s.resolveInRegistry(other, did, false); found || err != nil {
			return identity, err
		}
	}

	return nil, fmt.Errorf("DID not found: %s", did)
}

// resolveInRegistry resolves did within one registry. found is false when the
// registry does not contain the DID.
//
// includePrivate must only be true for the registry this control plane owns.
// Resolving a DID that belongs to another registry returns public key material
// only: those identities are still verifiable, but nothing here may sign on
// their behalf.
func (s *DIDService) resolveInRegistry(registry *types.DIDRegistry, did string, includePrivate bool) (*types.DIDIdentity, bool, error) {
	if registry == nil {
		return nil, false, nil
	}

	// privateJWK returns the derived private key, or "" when this registry is
	// not the active one.
	privateJWK := func(derivationPath, kind string) (string, error) {
		if !includePrivate {
			return "", nil
		}
		jwk, err := s.regeneratePrivateKeyJWK(registry.MasterSeed, derivationPath)
		if err != nil {
			return "", fmt.Errorf("failed to regenerate private key for %s DID %s: %w", kind, did, err)
		}
		return jwk, nil
	}

	// Check if this is the af server root DID
	if registry.RootDID == did {
		privateKeyJWK, err := privateJWK(rootDerivationPath, "root")
		if err != nil {
			return nil, true, err
		}
		publicKeyJWK, err := s.regeneratePublicKeyJWK(registry.MasterSeed, rootDerivationPath)
		if err != nil {
			return nil, true, fmt.Errorf("failed to regenerate public key for root DID %s: %w", did, err)
		}
		return &types.DIDIdentity{
			DID:            registry.RootDID,
			PrivateKeyJWK:  privateKeyJWK,
			PublicKeyJWK:   publicKeyJWK,
			DerivationPath: rootDerivationPath,
			ComponentType:  "agentfield_server",
		}, true, nil
	}

	for _, agentInfo := range registry.AgentNodes {
		if agentInfo.DID == did {
			privateKeyJWK, err := privateJWK(agentInfo.DerivationPath, "agent")
			if err != nil {
				return nil, true, err
			}

			// Regenerate the X25519 keyAgreement keypair at the agent's CURRENT
			// rotation epoch so resolution exposes the public encryption key and
			// re-derives the matching private key for the owner.
			x25519PubKey, x25519PrivKey, err := s.regenerateX25519KeyPairJWKAtEpoch(registry.MasterSeed, agentInfo.DerivationPath, agentInfo.X25519Epoch)
			if err != nil {
				return nil, true, fmt.Errorf("failed to regenerate X25519 keyAgreement key for agent DID %s: %w", did, err)
			}
			if !includePrivate {
				x25519PrivKey = ""
			}
			// Prefer the stored public key when present (it is kept in lockstep
			// with the epoch on rotation); fall back to the freshly derived one.
			if len(agentInfo.X25519PublicKeyJWK) > 0 {
				x25519PubKey = string(agentInfo.X25519PublicKeyJWK)
			}

			return &types.DIDIdentity{
				DID:                 agentInfo.DID,
				PrivateKeyJWK:       privateKeyJWK,
				PublicKeyJWK:        string(agentInfo.PublicKeyJWK),
				X25519PublicKeyJWK:  x25519PubKey,
				X25519PrivateKeyJWK: x25519PrivKey,
				X25519Epoch:         agentInfo.X25519Epoch,
				DerivationPath:      agentInfo.DerivationPath,
				ComponentType:       "agent",
			}, true, nil
		}

		for _, reasonerInfo := range agentInfo.Reasoners {
			if reasonerInfo.DID == did {
				privateKeyJWK, err := privateJWK(reasonerInfo.DerivationPath, "reasoner")
				if err != nil {
					return nil, true, err
				}
				return &types.DIDIdentity{
					DID:            reasonerInfo.DID,
					PrivateKeyJWK:  privateKeyJWK,
					PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
					DerivationPath: reasonerInfo.DerivationPath,
					ComponentType:  "reasoner",
					FunctionName:   reasonerInfo.FunctionName,
				}, true, nil
			}
		}

		for _, skillInfo := range agentInfo.Skills {
			if skillInfo.DID == did {
				privateKeyJWK, err := privateJWK(skillInfo.DerivationPath, "skill")
				if err != nil {
					return nil, true, err
				}
				return &types.DIDIdentity{
					DID:            skillInfo.DID,
					PrivateKeyJWK:  privateKeyJWK,
					PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
					DerivationPath: skillInfo.DerivationPath,
					ComponentType:  "skill",
					FunctionName:   skillInfo.FunctionName,
				}, true, nil
			}
		}
	}

	return nil, false, nil
}

// ResolveAgentIDByDID looks up the agent node ID for any DID (including did:key)
// by searching the in-memory DID registry. Returns empty string if not found.
func (s *DIDService) ResolveAgentIDByDID(did string) string {
	if !s.config.Enabled {
		return ""
	}
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return ""
	}
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return ""
	}
	for _, agentInfo := range registry.AgentNodes {
		if agentInfo.DID == did {
			return agentInfo.AgentNodeID
		}
	}
	return ""
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

// derivePrivateKey derives a private key from master seed using HKDF (RFC 5869).
// Uses domain-separated derivation with SHA-256, the derivation path as info,
// and a fixed salt for domain separation.
func (s *DIDService) derivePrivateKey(masterSeed []byte, derivationPath string) (ed25519.PrivateKey, error) {
	return deriveEd25519Key(masterSeed, derivationPath)
}

// generateDIDKey generates a DID:key from an Ed25519 public key.
func (s *DIDService) generateDIDKey(publicKey ed25519.PublicKey) string {
	return didKeyFromPublicKey(publicKey)
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

// hashAgentFieldServerID creates a deterministic hash of af server ID for derivation paths.
func (s *DIDService) hashAgentFieldServerID(agentfieldServerID string) uint32 {
	return serverPathSegment(agentfieldServerID)
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

// deriveX25519PrivateKeyAtEpoch derives an X25519 keyAgreement private key from
// the master seed using HKDF (RFC 5869). It uses a DISTINCT salt from the
// Ed25519 signing-key derivation so the encryption key is cryptographically
// independent of the signing key. The rotation epoch is folded into the HKDF
// `info` (`<derivationPath>/enc/<epoch>`) so each epoch yields a fresh,
// independent keypair — rotating the epoch retires the prior key entirely.
func (s *DIDService) deriveX25519PrivateKeyAtEpoch(masterSeed []byte, derivationPath string, epoch int) (*ecdh.PrivateKey, error) {
	salt := []byte("agentfield-did-keyagreement-v1")
	info := []byte(derivationPath + "/enc/" + strconv.Itoa(epoch))

	hkdfReader := hkdf.New(sha256.New, masterSeed, salt, info)
	derivedSeed := make([]byte, 32)
	if _, err := io.ReadFull(hkdfReader, derivedSeed); err != nil {
		return nil, fmt.Errorf("HKDF X25519 key derivation failed: %w", err)
	}

	privateKey, err := ecdh.X25519().NewPrivateKey(derivedSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to create X25519 private key: %w", err)
	}
	return privateKey, nil
}

// x25519PublicKeyToJWK converts an X25519 public key to JWK format (RFC 8037).
func (s *DIDService) x25519PublicKeyToJWK(pub *ecdh.PublicKey) (string, error) {
	jwk := map[string]interface{}{
		"kty": "OKP",
		"crv": "X25519",
		"x":   base64.RawURLEncoding.EncodeToString(pub.Bytes()),
		"use": "enc",
		"alg": "ECDH-ES",
	}

	jwkBytes, err := json.Marshal(jwk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal X25519 public JWK: %w", err)
	}

	return string(jwkBytes), nil
}

// x25519PrivateKeyToJWK converts an X25519 private key to JWK format (RFC 8037),
// including the private `d` component.
func (s *DIDService) x25519PrivateKeyToJWK(priv *ecdh.PrivateKey) (string, error) {
	jwk := map[string]interface{}{
		"kty": "OKP",
		"crv": "X25519",
		"x":   base64.RawURLEncoding.EncodeToString(priv.PublicKey().Bytes()),
		"d":   base64.RawURLEncoding.EncodeToString(priv.Bytes()),
		"use": "enc",
		"alg": "ECDH-ES",
	}

	jwkBytes, err := json.Marshal(jwk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal X25519 private JWK: %w", err)
	}

	return string(jwkBytes), nil
}

// regenerateX25519KeyPairJWK derives the X25519 keyAgreement keypair from the
// master seed and derivation path at the default rotation epoch (0) and returns
// both the public and private JWKs.
func (s *DIDService) regenerateX25519KeyPairJWK(masterSeed []byte, derivationPath string) (pubJWK string, privJWK string, err error) {
	return s.regenerateX25519KeyPairJWKAtEpoch(masterSeed, derivationPath, 0)
}

// regenerateX25519KeyPairJWKAtEpoch derives the X25519 keyAgreement keypair from
// the master seed and derivation path at the given rotation epoch and returns
// both the public and private JWKs.
func (s *DIDService) regenerateX25519KeyPairJWKAtEpoch(masterSeed []byte, derivationPath string, epoch int) (pubJWK string, privJWK string, err error) {
	priv, err := s.deriveX25519PrivateKeyAtEpoch(masterSeed, derivationPath, epoch)
	if err != nil {
		return "", "", fmt.Errorf("failed to derive X25519 private key: %w", err)
	}

	pubJWK, err = s.x25519PublicKeyToJWK(priv.PublicKey())
	if err != nil {
		return "", "", fmt.Errorf("failed to convert X25519 public key to JWK: %w", err)
	}

	privJWK, err = s.x25519PrivateKeyToJWK(priv)
	if err != nil {
		return "", "", fmt.Errorf("failed to convert X25519 private key to JWK: %w", err)
	}

	return pubJWK, privJWK, nil
}

// RotateAgentX25519Key rotates the X25519 keyAgreement (encryption) key of the
// agent-node identified by did. It increments the agent's stored rotation epoch,
// re-derives the keypair at the new epoch, updates the stored public key, and
// persists the registry. The new public key and epoch are returned so callers
// can re-publish the encryption key.
//
// After rotation, ResolveDID returns the NEW keypair; a payload encrypted to the
// OLD public key can no longer be decrypted with the re-derived private key — the
// old key is retired (the derivation no longer produces it at the new epoch).
//
// Only agent-node DIDs are supported. Reasoner/skill/root DIDs do not carry an
// independent keyAgreement key and return a clear error.
func (s *DIDService) RotateAgentX25519Key(did string) (newPubJWK string, newEpoch int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if did == "" {
		return "", 0, fmt.Errorf("did is required")
	}

	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return "", 0, fmt.Errorf("af server registry validation failed: %w", err)
	}

	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return "", 0, fmt.Errorf("failed to get af server ID: %w", err)
	}

	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		return "", 0, fmt.Errorf("failed to get DID registry: %w", err)
	}
	if registry == nil {
		return "", 0, fmt.Errorf("DID registry not found for af server %s", agentfieldServerID)
	}

	// The root af-server DID and component (reasoner/skill) DIDs do not own a
	// rotatable keyAgreement key — reject them with a clear error.
	if did == registry.RootDID {
		return "", 0, fmt.Errorf("cannot rotate keyAgreement key for af server root DID %s: not an agent-node DID", did)
	}

	// Locate the agent node owning this DID.
	var (
		agentNodeID string
		agentInfo   types.AgentDIDInfo
		found       bool
	)
	for nodeID, info := range registry.AgentNodes {
		if info.DID == did {
			agentNodeID, agentInfo, found = nodeID, info, true
			break
		}
		// Surface a precise error for component DIDs rather than a generic
		// "not found" so callers know rotation is unsupported for them.
		for _, reasonerInfo := range info.Reasoners {
			if reasonerInfo.DID == did {
				return "", 0, fmt.Errorf("cannot rotate keyAgreement key for reasoner DID %s: rotation is only supported for agent-node DIDs", did)
			}
		}
		for _, skillInfo := range info.Skills {
			if skillInfo.DID == did {
				return "", 0, fmt.Errorf("cannot rotate keyAgreement key for skill DID %s: rotation is only supported for agent-node DIDs", did)
			}
		}
	}
	if !found {
		return "", 0, fmt.Errorf("agent-node DID %s not found in registry", did)
	}
	if reason := s.registry.QuarantineReason(did); reason != "" {
		return "", 0, fmt.Errorf("cannot rotate keyAgreement key for DID %s: %s", did, reason)
	}

	// Hold the registry mutex through commit and publication so instances
	// sharing this registry cannot publish a smaller epoch over a newer one.
	s.registry.mu.Lock()
	defer s.registry.mu.Unlock()
	live := s.registry.registries[agentfieldServerID]
	agentInfo = live.AgentNodes[agentNodeID]
	rotator, ok := s.registry.storageProvider.(interface {
		AdvanceAgentX25519Epoch(context.Context, string, string, int) (int, error)
	})
	if !ok {
		return "", 0, fmt.Errorf("storage does not support durable X25519 rotation")
	}
	newEpoch, err = rotator.AdvanceAgentX25519Epoch(context.Background(), agentfieldServerID, did, agentInfo.X25519Epoch)
	if err != nil {
		return "", 0, fmt.Errorf("failed to commit X25519 rotation: %w", err)
	}
	pubJWK, _, err := s.regenerateX25519KeyPairJWKAtEpoch(registry.MasterSeed, agentInfo.DerivationPath, newEpoch)
	if err != nil {
		return "", 0, fmt.Errorf("failed to re-derive X25519 keyAgreement key at epoch %d for agent DID %s: %w", newEpoch, did, err)
	}

	// Persist the new epoch + public key.
	agentInfo.X25519Epoch = newEpoch
	agentInfo.X25519PublicKeyJWK = json.RawMessage(pubJWK)
	next := *live
	next.AgentNodes = make(map[string]types.AgentDIDInfo, len(live.AgentNodes))
	for nodeID, info := range live.AgentNodes {
		next.AgentNodes[nodeID] = info
	}
	next.AgentNodes[agentNodeID] = agentInfo
	next.LastKeyRotation = time.Now()
	s.registry.registries[agentfieldServerID] = &next

	logger.Logger.Info().
		Str("did", did).
		Str("agent_node_id", agentNodeID).
		Int("epoch", newEpoch).
		Msg("Rotated agent X25519 keyAgreement key")

	return pubJWK, newEpoch, nil
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

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return nil, fmt.Errorf("af server registry validation failed: %w", err)
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get af server ID: %w", err)
	}

	// Get af server registry using dynamic ID
	registry, err := s.registry.GetRegistry(agentfieldServerID)
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

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return fmt.Errorf("af server registry validation failed: %w", err)
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return fmt.Errorf("failed to get af server ID: %w", err)
	}

	// Get current DID registry using dynamic ID
	registry, err := s.GetRegistry(agentfieldServerID)
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
	return s.getExistingAgentDID(context.Background(), agentNodeID)
}

func (s *DIDService) getExistingAgentDID(ctx context.Context, agentNodeID string) (*types.AgentDIDInfo, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return nil, fmt.Errorf("af server registry validation failed: %w", err)
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get af server ID: %w", err)
	}

	stored, err := s.registry.readThroughAgent(ctx, agentfieldServerID, agentNodeID)
	if err != nil {
		return nil, err
	}
	if stored == nil {
		return nil, fmt.Errorf("agent not found: %s", agentNodeID)
	}
	return stored, nil
}

// PerformDifferentialAnalysis compares existing vs new reasoners/skills to determine what needs to be updated.
func (s *DIDService) PerformDifferentialAnalysis(agentNodeID string, newReasonerIDs, newSkillIDs []string) (*types.DifferentialAnalysisResult, error) {
	return s.performDifferentialAnalysis(context.Background(), agentNodeID, newReasonerIDs, newSkillIDs)
}

func (s *DIDService) performDifferentialAnalysis(ctx context.Context, agentNodeID string, newReasonerIDs, newSkillIDs []string) (*types.DifferentialAnalysisResult, error) {
	existingAgent, err := s.getExistingAgentDID(ctx, agentNodeID)
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

// quarantinedAgentError explains why an agent cannot be given keys and how an
// operator can issue it a new identity.
func quarantinedAgentError(agentNodeID, reason string) string {
	return fmt.Sprintf("refusing to issue keys for agent %s: %s. If this identity cannot be recovered, "+
		"set status = 'revoked' on its agent_dids row and restart the control plane to issue a new identity "+
		"(credentials signed by the old DID will no longer verify)", agentNodeID, reason)
}

// quarantinedIdentityReason returns why any identity belonging to the agent
// is unusable, or "" when all of them verified.
func (s *DIDService) quarantinedIdentityReason(agent *types.AgentDIDInfo) string {
	if reason := s.registry.QuarantineReason(agent.DID); reason != "" {
		return reason
	}
	for _, reasoner := range agent.Reasoners {
		if reason := s.registry.QuarantineReason(reasoner.DID); reason != "" {
			return reason
		}
	}
	for _, skill := range agent.Skills {
		if reason := s.registry.QuarantineReason(skill.DID); reason != "" {
			return reason
		}
	}
	return ""
}

// buildExistingIdentityPackage builds an identity package from existing agent DID info.
func (s *DIDService) buildExistingIdentityPackage(existingAgent *types.AgentDIDInfo) types.DIDIdentityPackage {
	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get af server ID for identity package")
		agentfieldServerID = "unknown"
	}

	// Retrieve master seed to re-derive private keys for the requesting agent.
	// Agents need their private keys to sign cross-agent requests (DID auth).
	var masterSeed []byte
	registry, err := s.registry.GetRegistry(agentfieldServerID)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get registry for key re-derivation")
	} else {
		masterSeed = registry.MasterSeed
	}

	// Helper to re-derive private key JWK from master seed and derivation path.
	rederivePrivKey := func(derivationPath string) string {
		if masterSeed == nil || derivationPath == "" {
			return ""
		}
		_, privKeyJWK, _, err := s.generateDIDWithKeys(masterSeed, derivationPath)
		if err != nil {
			logger.Logger.Error().Err(err).Str("path", derivationPath).Msg("Failed to re-derive private key")
			return ""
		}
		return privKeyJWK
	}

	// Build reasoner DIDs map
	reasonerDIDs := make(map[string]types.DIDIdentity)
	for id, reasonerInfo := range existingAgent.Reasoners {
		reasonerDIDs[id] = types.DIDIdentity{
			DID:            reasonerInfo.DID,
			PrivateKeyJWK:  rederivePrivKey(reasonerInfo.DerivationPath),
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
			PrivateKeyJWK:  rederivePrivKey(skillInfo.DerivationPath),
			PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
			DerivationPath: skillInfo.DerivationPath,
			ComponentType:  "skill",
			FunctionName:   skillInfo.FunctionName,
		}
	}

	// Re-derive the agent's X25519 keyAgreement keypair so re-registering agents
	// still receive their encryption keys. Best-effort: empty if the seed is
	// unavailable, mirroring the Ed25519 re-derivation above.
	// Re-derive at the agent's CURRENT rotation epoch so re-registration preserves
	// (never resets) any prior keyAgreement rotation.
	var agentX25519PubKey, agentX25519PrivKey string
	if masterSeed != nil && existingAgent.DerivationPath != "" {
		pubJWK, privJWK, err := s.regenerateX25519KeyPairJWKAtEpoch(masterSeed, existingAgent.DerivationPath, existingAgent.X25519Epoch)
		if err != nil {
			logger.Logger.Error().Err(err).Str("path", existingAgent.DerivationPath).Msg("Failed to re-derive X25519 keyAgreement key")
		} else {
			agentX25519PubKey, agentX25519PrivKey = pubJWK, privJWK
		}
	}
	if len(existingAgent.X25519PublicKeyJWK) > 0 {
		agentX25519PubKey = string(existingAgent.X25519PublicKeyJWK)
	}

	return types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:                 existingAgent.DID,
			PrivateKeyJWK:       rederivePrivKey(existingAgent.DerivationPath),
			PublicKeyJWK:        string(existingAgent.PublicKeyJWK),
			X25519PublicKeyJWK:  agentX25519PubKey,
			X25519PrivateKeyJWK: agentX25519PrivKey,
			X25519Epoch:         existingAgent.X25519Epoch,
			DerivationPath:      existingAgent.DerivationPath,
			ComponentType:       "agent",
		},
		ReasonerDIDs:       reasonerDIDs,
		SkillDIDs:          skillDIDs,
		AgentFieldServerID: agentfieldServerID,
	}
}

// handlePartialRegistration handles partial registration for existing agents.
func (s *DIDService) handlePartialRegistration(req *types.DIDRegistrationRequest, diffResult *types.DifferentialAnalysisResult) (*types.DIDRegistrationResponse, error) {
	return s.handlePartialRegistrationContext(context.Background(), req, diffResult)
}

func (s *DIDService) handlePartialRegistrationContext(ctx context.Context, req *types.DIDRegistrationRequest, diffResult *types.DifferentialAnalysisResult) (*types.DIDRegistrationResponse, error) {
	// Handle deregistration of removed components first
	if len(diffResult.RemovedReasonerIDs) > 0 || len(diffResult.RemovedSkillIDs) > 0 {
		deregReq := &types.ComponentDeregistrationRequest{
			AgentNodeID:         req.AgentNodeID,
			ReasonerIDsToRemove: diffResult.RemovedReasonerIDs,
			SkillIDsToRemove:    diffResult.RemovedSkillIDs,
		}

		deregResponse, err := s.deregisterComponents(ctx, deregReq)
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

		return s.partialRegisterAgentOnce(ctx, partialReq)
	}

	// If we reach here, only removals were needed
	existingAgent, err := s.getExistingAgentDID(ctx, req.AgentNodeID)
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
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), didRegistrationTimeout)
	defer cancel()
	return retryDIDAllocation(ctx, func() (*types.DIDRegistrationResponse, error) {
		return s.partialRegisterAgentOnce(ctx, req)
	})
}

func (s *DIDService) partialRegisterAgentOnce(ctx context.Context, req *types.PartialDIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	if !s.config.Enabled {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("af server registry validation failed: %v", err),
		}, nil
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get af server ID: %v", err),
		}, nil
	}
	// Refresh before allocating so direct partial registrations see other replicas.
	if _, err := s.registry.readThroughAgent(ctx, agentfieldServerID, req.AgentNodeID); err != nil {
		return &types.DIDRegistrationResponse{Success: false, Error: fmt.Sprintf("failed to refresh agent: %v", err)}, nil
	}

	// Get af server registry using dynamic ID
	registry, err := s.registry.GetRegistry(agentfieldServerID)
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

	if reason := s.quarantinedIdentityReason(&existingAgent); reason != "" {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   quarantinedAgentError(req.AgentNodeID, reason),
		}, nil
	}
	if !identityMatchesPath(registry.MasterSeed, existingAgent.DerivationPath, existingAgent.DID) {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("refusing to extend agent %s: its derivation path does not reproduce its DID", req.AgentNodeID),
		}, nil
	}
	// A competing registration may already have issued some requested components.
	var missingReasoners, missingSkills []string
	for _, id := range req.NewReasonerIDs {
		if _, exists := existingAgent.Reasoners[id]; !exists {
			missingReasoners = append(missingReasoners, id)
		}
	}
	for _, id := range req.NewSkillIDs {
		if _, exists := existingAgent.Skills[id]; !exists {
			missingSkills = append(missingSkills, id)
		}
	}
	if len(missingReasoners)+len(missingSkills) == 0 {
		// Another replica issued the requested components. Keep the partial
		// response limited to those components and omit the agent signing key.
		identityPackage := s.buildExistingIdentityPackage(&existingAgent)
		identityPackage.AgentDID.PrivateKeyJWK = ""
		issuedReasoners := make(map[string]types.DIDIdentity, len(req.NewReasonerIDs))
		for _, id := range req.NewReasonerIDs {
			issuedReasoners[id] = identityPackage.ReasonerDIDs[id]
		}
		issuedSkills := make(map[string]types.DIDIdentity, len(req.NewSkillIDs))
		for _, id := range req.NewSkillIDs {
			issuedSkills[id] = identityPackage.SkillDIDs[id]
		}
		identityPackage.ReasonerDIDs = issuedReasoners
		identityPackage.SkillDIDs = issuedSkills
		return &types.DIDRegistrationResponse{
			Success: true, IdentityPackage: identityPackage,
			Message: "Partial registration successful: 0 new reasoners, 0 new skills",
		}, nil
	}
	existingAgent = cloneAgentInfo(existingAgent)
	used, err := s.registry.usedIdentities(ctx)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to read issued DIDs: %v", err),
		}, nil
	}
	paths := newComponentPathAllocator(registry.MasterSeed, existingAgent.DerivationPath, used)

	// Generate DIDs for new reasoners only
	newReasonerDIDs := make(map[string]types.DIDIdentity)
	newReasonerInfos := make(map[string]types.ReasonerDIDInfo)

	for _, reasonerID := range missingReasoners {
		reasoner := s.findReasonerByID(req.AllReasoners, reasonerID)
		if reasoner == nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("reasoner %s not found in request", reasonerID),
			}, nil
		}

		// Generate DID for new reasoner
		reasonerPath, err := paths.allocate("reasoner")
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to allocate derivation path for reasoner %s: %v", reasonerID, err),
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

	for _, skillID := range missingSkills {
		skill := s.findSkillByID(req.AllSkills, skillID)
		if skill == nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("skill %s not found in request", skillID),
			}, nil
		}

		// Generate DID for new skill
		skillPath, err := paths.allocate("skill")
		if err != nil {
			return &types.DIDRegistrationResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to allocate derivation path for skill %s: %v", skillID, err),
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

	// Derive the agent's X25519 keyAgreement keypair from the same master seed +
	// derivation path. Backfill the stored public key for agents registered before
	// keyAgreement support so resolution and the returned package stay consistent.
	agentX25519PubKey, agentX25519PrivKey, err := s.regenerateX25519KeyPairJWKAtEpoch(registry.MasterSeed, existingAgent.DerivationPath, existingAgent.X25519Epoch)
	if err != nil {
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to generate agent X25519 keyAgreement key: %v", err),
		}, nil
	}
	if len(existingAgent.X25519PublicKeyJWK) == 0 {
		existingAgent.X25519PublicKeyJWK = json.RawMessage(agentX25519PubKey)
	} else {
		agentX25519PubKey = string(existingAgent.X25519PublicKeyJWK)
	}

	// Update registry
	err = s.registry.updateAgents(ctx, agentfieldServerID, func(live *types.DIDRegistry) ([]string, error) {
		live.AgentNodes[req.AgentNodeID] = existingAgent
		live.TotalDIDs += len(newReasonerDIDs) + len(newSkillDIDs)
		return []string{req.AgentNodeID}, nil
	})
	if err != nil {
		if errors.Is(err, storage.ErrDIDAllocationConflict) || errors.Is(err, storage.ErrDIDNodeAlreadyRegistered) || storage.IsTransientDIDTransactionError(err) {
			return nil, err
		}
		return &types.DIDRegistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store updated registry: %v", err),
		}, nil
	}

	// Build response with only new DIDs
	identityPackage := types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:                 existingAgent.DID,
			PrivateKeyJWK:       "", // Don't regenerate existing agent signing key
			PublicKeyJWK:        string(existingAgent.PublicKeyJWK),
			X25519PublicKeyJWK:  agentX25519PubKey,
			X25519PrivateKeyJWK: agentX25519PrivKey,
			X25519Epoch:         existingAgent.X25519Epoch,
			DerivationPath:      existingAgent.DerivationPath,
			ComponentType:       "agent",
		},
		ReasonerDIDs:       newReasonerDIDs,
		SkillDIDs:          newSkillDIDs,
		AgentFieldServerID: registry.AgentFieldServerID,
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
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deregisterComponents(context.Background(), req)
}

// deregisterComponents is DeregisterComponents for callers holding s.mu.
func (s *DIDService) deregisterComponents(ctx context.Context, req *types.ComponentDeregistrationRequest) (*types.ComponentDeregistrationResponse, error) {
	if !s.config.Enabled {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   "DID system is disabled",
		}, nil
	}

	// Validate af server registry exists
	if err := s.validateAgentFieldServerRegistry(); err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("af server registry validation failed: %v", err),
		}, nil
	}

	// Get af server ID dynamically
	agentfieldServerID, err := s.getAgentFieldServerID()
	if err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to get af server ID: %v", err),
		}, nil
	}

	// Get af server registry using dynamic ID
	registry, err := s.registry.GetRegistry(agentfieldServerID)
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

	existingAgent = cloneAgentInfo(existingAgent)
	removedCount := 0
	var removedDIDs []string
	seenDIDs := make(map[string]bool)
	addRemovedDID := func(did string) {
		if !seenDIDs[did] {
			seenDIDs[did] = true
			removedDIDs = append(removedDIDs, did)
		}
	}
	// Values record whether a requested name was found in memory or storage.
	removedNames := map[string]map[string]bool{"reasoner": {}, "skill": {}}
	for _, name := range req.ReasonerIDsToRemove {
		removedNames["reasoner"][name] = false
	}
	for _, name := range req.SkillIDsToRemove {
		removedNames["skill"][name] = false
	}

	// Remove reasoners
	for _, reasonerID := range req.ReasonerIDsToRemove {
		if reasonerInfo, exists := existingAgent.Reasoners[reasonerID]; exists {
			delete(existingAgent.Reasoners, reasonerID)
			addRemovedDID(reasonerInfo.DID)
			removedNames["reasoner"][reasonerID] = true
			removedCount++
			logger.Logger.Debug().Msgf("🗑️ Removed reasoner DID: %s from agent %s", reasonerID, req.AgentNodeID)
		}
	}

	// Remove skills
	for _, skillID := range req.SkillIDsToRemove {
		if skillInfo, exists := existingAgent.Skills[skillID]; exists {
			delete(existingAgent.Skills, skillID)
			addRemovedDID(skillInfo.DID)
			removedNames["skill"][skillID] = true
			removedCount++
			logger.Logger.Debug().Msgf("🗑️ Removed skill DID: %s from agent %s", skillID, req.AgentNodeID)
		}
	}
	memoryRemovedCount := removedCount

	// Older writers allowed multiple active rows per name. Explicit removal
	// retires every such row, even when the name is already absent from memory.
	if len(removedNames["reasoner"]) > 0 || len(removedNames["skill"]) > 0 {
		rows, err := s.registry.storageProvider.ListComponentDIDs(ctx, existingAgent.DID)
		if err != nil {
			return &types.ComponentDeregistrationResponse{Error: fmt.Sprintf("failed to list removed component DIDs: %v", err)}, nil
		}
		for _, row := range rows {
			if found, requested := removedNames[row.ComponentType][row.ComponentName]; requested {
				addRemovedDID(row.ComponentDID)
				if !found {
					removedCount++
					removedNames[row.ComponentType][row.ComponentName] = true
				}
			}
		}
	}

	// Retire durably before publishing the removal. A failed retirement must
	// leave the live registry unchanged so a retry can still see the components.
	if err := s.registry.retireComponents(ctx, agentfieldServerID, req.AgentNodeID, removedDIDs); err != nil {
		return &types.ComponentDeregistrationResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to retire removed component DIDs: %v", err),
		}, nil
	}

	// Persist and publish the remaining components using the same context.
	// If persistence fails, still remove the durably retired DIDs from memory
	// so this service cannot resolve their private keys or sign with them.
	err = s.registry.updateAgents(ctx, agentfieldServerID, func(live *types.DIDRegistry) ([]string, error) {
		live.AgentNodes[req.AgentNodeID] = existingAgent
		live.TotalDIDs -= memoryRemovedCount
		return []string{req.AgentNodeID}, nil
	})
	if err != nil {
		s.registry.removeRetiredComponentsFromMemory(agentfieldServerID, req.AgentNodeID, seenDIDs)
		logger.Logger.Error().Err(err).Str("agent_node_id", req.AgentNodeID).
			Msg("Failed to publish component removal after durable retirement")
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
