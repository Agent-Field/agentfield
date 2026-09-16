package services

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"sync"

	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// DIDRegistry manages the storage and retrieval of DID registries using database-only operations.
type DIDRegistry struct {
	mu                sync.RWMutex
	registries        map[string]*types.DIDRegistry
	quarantined       map[string]string
	brokenServers     map[string]error
	storageProvider   storage.StorageProvider
	encryptionService *encryption.EncryptionService
}

// NewDIDRegistryWithStorage creates a new DID registry instance with database storage.
func NewDIDRegistryWithStorage(storageProvider storage.StorageProvider) *DIDRegistry {
	return &DIDRegistry{
		registries:      make(map[string]*types.DIDRegistry),
		quarantined:     make(map[string]string),
		brokenServers:   make(map[string]error),
		storageProvider: storageProvider,
	}
}

// SetEncryptionService sets the encryption service for encrypting master seeds at rest.
func (r *DIDRegistry) SetEncryptionService(svc *encryption.EncryptionService) {
	r.encryptionService = svc
}

// Initialize initializes the DID registry storage.
func (r *DIDRegistry) Initialize() error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}

	// Load existing registries from database
	return r.loadRegistriesFromDatabase()
}

// GetRegistry retrieves a DID registry for a af server.
// Returns (nil, nil) if registry doesn't exist, (nil, error) for actual errors.
func (r *DIDRegistry) GetRegistry(agentfieldServerID string) (*types.DIDRegistry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		// Return nil, nil for "not found" to distinguish from actual errors
		return nil, nil
	}

	return registry, nil
}

// StoreRegistry stores a DID registry for a af server.
func (r *DIDRegistry) StoreRegistry(registry *types.DIDRegistry) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Store in memory
	r.registries[registry.AgentFieldServerID] = registry

	// Persist to database
	return r.saveRegistryToDatabase(registry)
}

// ListRegistries lists all af server registries.
func (r *DIDRegistry) ListRegistries() ([]*types.DIDRegistry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registries := make([]*types.DIDRegistry, 0, len(r.registries))
	for _, registry := range r.registries {
		registries = append(registries, registry)
	}

	return registries, nil
}

// DeleteRegistry deletes a DID registry for a af server.
func (r *DIDRegistry) DeleteRegistry(agentfieldServerID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Remove from memory
	delete(r.registries, agentfieldServerID)

	// TODO: Add database deletion method to storage interface
	// For now, we'll just remove from memory
	return nil
}

// UpdateAgentStatus updates the status of an agent DID.
func (r *DIDRegistry) UpdateAgentStatus(agentfieldServerID, agentNodeID string, status types.AgentDIDStatus) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		return fmt.Errorf("registry not found for af server: %s", agentfieldServerID)
	}

	agentInfo, exists := registry.AgentNodes[agentNodeID]
	if !exists {
		return fmt.Errorf("agent not found: %s", agentNodeID)
	}

	agentInfo.Status = status
	registry.AgentNodes[agentNodeID] = agentInfo

	// Persist changes to database
	return r.saveRegistryToDatabase(registry)
}

// FindDIDByComponent finds a DID by component type and function name.
func (r *DIDRegistry) FindDIDByComponent(agentfieldServerID, componentType, functionName string) (*types.DIDIdentity, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		return nil, fmt.Errorf("registry not found for af server: %s", agentfieldServerID)
	}

	// Search through all agent nodes
	for _, agentInfo := range registry.AgentNodes {
		switch componentType {
		case "agent":
			if agentInfo.AgentNodeID == functionName {
				return &types.DIDIdentity{
					DID:            agentInfo.DID,
					PublicKeyJWK:   string(agentInfo.PublicKeyJWK),
					DerivationPath: agentInfo.DerivationPath,
					ComponentType:  "agent",
				}, nil
			}
		case "reasoner":
			for _, reasonerInfo := range agentInfo.Reasoners {
				if reasonerInfo.FunctionName == functionName {
					return &types.DIDIdentity{
						DID:            reasonerInfo.DID,
						PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
						DerivationPath: reasonerInfo.DerivationPath,
						ComponentType:  "reasoner",
						FunctionName:   reasonerInfo.FunctionName,
					}, nil
				}
			}
		case "skill":
			for _, skillInfo := range agentInfo.Skills {
				if skillInfo.FunctionName == functionName {
					return &types.DIDIdentity{
						DID:            skillInfo.DID,
						PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
						DerivationPath: skillInfo.DerivationPath,
						ComponentType:  "skill",
						FunctionName:   skillInfo.FunctionName,
					}, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("DID not found for component: %s/%s", componentType, functionName)
}

// GetAgentDIDs retrieves all DIDs for a specific agent node.
func (r *DIDRegistry) GetAgentDIDs(agentfieldServerID, agentNodeID string) (*types.DIDIdentityPackage, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		return nil, fmt.Errorf("registry not found for af server: %s", agentfieldServerID)
	}

	agentInfo, exists := registry.AgentNodes[agentNodeID]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", agentNodeID)
	}

	// Build identity package (without private keys for security)
	reasonerDIDs := make(map[string]types.DIDIdentity)
	for id, reasonerInfo := range agentInfo.Reasoners {
		reasonerDIDs[id] = types.DIDIdentity{
			DID:            reasonerInfo.DID,
			PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
			DerivationPath: reasonerInfo.DerivationPath,
			ComponentType:  "reasoner",
			FunctionName:   reasonerInfo.FunctionName,
		}
	}

	skillDIDs := make(map[string]types.DIDIdentity)
	for id, skillInfo := range agentInfo.Skills {
		skillDIDs[id] = types.DIDIdentity{
			DID:            skillInfo.DID,
			PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
			DerivationPath: skillInfo.DerivationPath,
			ComponentType:  "skill",
			FunctionName:   skillInfo.FunctionName,
		}
	}

	return &types.DIDIdentityPackage{
		AgentDID: types.DIDIdentity{
			DID:            agentInfo.DID,
			PublicKeyJWK:   string(agentInfo.PublicKeyJWK),
			DerivationPath: agentInfo.DerivationPath,
			ComponentType:  "agent",
		},
		ReasonerDIDs:       reasonerDIDs,
		SkillDIDs:          skillDIDs,
		AgentFieldServerID: agentfieldServerID,
	}, nil
}

// loadRegistriesFromDatabase loads all registries from the database. Each
// agent is loaded only into the registry of the server that owns it, with its
// stored derivation paths and public keys. Every identity is then verified
// against the master seed: stale paths from older versions are recovered and
// written back, and identities that cannot be verified are quarantined.
func (r *DIDRegistry) loadRegistriesFromDatabase() error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}

	ctx := context.Background()
	agentfieldServerDIDs, err := r.storageProvider.ListAgentFieldServerDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to list af server DIDs: %w", err)
	}

	agentDIDs, err := r.storageProvider.ListAgentDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to list agent DIDs: %w", err)
	}
	agentsByServer := make(map[string][]*types.AgentDIDInfo)
	for _, agentDIDInfo := range agentDIDs {
		if agentDIDInfo == nil {
			continue
		}
		agentsByServer[agentDIDInfo.AgentFieldServerID] = append(agentsByServer[agentDIDInfo.AgentFieldServerID], agentDIDInfo)
	}

	registries := make(map[string]*types.DIDRegistry, len(agentfieldServerDIDs))
	quarantined := make(map[string]string)
	brokenServers := make(map[string]error)
	var toRepair []*types.DIDRegistry

	for _, agentfieldServerDIDInfo := range agentfieldServerDIDs {
		serverID := agentfieldServerDIDInfo.AgentFieldServerID

		// Decrypt master seed if encryption is configured
		masterSeed := agentfieldServerDIDInfo.MasterSeed
		if r.encryptionService != nil {
			decrypted, err := r.encryptionService.DecryptBytes(masterSeed)
			if err != nil {
				// Backward compatibility: if decryption fails, the seed may be stored
				// as plaintext from before encryption was configured. The root DID
				// check below rejects the registry if these bytes are not the seed.
				log.Printf("Warning: master seed decryption failed for %s (may be plaintext from before encryption was enabled), using raw bytes",
					serverID)
				decrypted = masterSeed
			}
			masterSeed = decrypted
		}

		registry := &types.DIDRegistry{
			AgentFieldServerID: serverID,
			RootDID:            agentfieldServerDIDInfo.RootDID,
			MasterSeed:         masterSeed,
			AgentNodes:         make(map[string]types.AgentDIDInfo),
			TotalDIDs:          0,
			CreatedAt:          agentfieldServerDIDInfo.CreatedAt,
			LastKeyRotation:    agentfieldServerDIDInfo.LastKeyRotation,
		}

		rows := agentsByServer[serverID]
		delete(agentsByServer, serverID)
		sort.SliceStable(rows, func(i, j int) bool {
			if !rows[i].RegisteredAt.Equal(rows[j].RegisteredAt) {
				return rows[i].RegisteredAt.Before(rows[j].RegisteredAt)
			}
			return rows[i].DID < rows[j].DID
		})

		for _, agentDIDInfo := range rows {
			if existing, dup := registry.AgentNodes[agentDIDInfo.AgentNodeID]; dup {
				// Keep the earliest identity deterministically; never let row
				// order decide which key an agent gets.
				log.Printf("Warning: agent %s has more than one DID under af server %s; keeping %s and ignoring %s",
					agentDIDInfo.AgentNodeID, serverID, existing.DID, agentDIDInfo.DID)
				continue
			}

			agentInfo := types.AgentDIDInfo{
				DID:                agentDIDInfo.DID,
				AgentNodeID:        agentDIDInfo.AgentNodeID,
				AgentFieldServerID: serverID,
				PublicKeyJWK:       agentDIDInfo.PublicKeyJWK,
				X25519Epoch:        agentDIDInfo.X25519Epoch,
				DerivationPath:     agentDIDInfo.DerivationPath,
				Status:             agentDIDInfo.Status,
				RegisteredAt:       agentDIDInfo.RegisteredAt,
				Reasoners:          make(map[string]types.ReasonerDIDInfo),
				Skills:             make(map[string]types.SkillDIDInfo),
			}

			componentDIDs, err := r.storageProvider.ListComponentDIDs(ctx, agentDIDInfo.DID)
			if err != nil {
				return fmt.Errorf("failed to list component DIDs for agent %s: %w", agentDIDInfo.AgentNodeID, err)
			}

			for _, componentDID := range componentDIDs {
				var publicKeyJWK json.RawMessage
				if componentDID.PublicKeyJWK != "" {
					publicKeyJWK = json.RawMessage(componentDID.PublicKeyJWK)
				}
				switch componentDID.ComponentType {
				case "reasoner":
					agentInfo.Reasoners[componentDID.ComponentName] = types.ReasonerDIDInfo{
						DID:            componentDID.ComponentDID,
						FunctionName:   componentDID.ComponentName,
						PublicKeyJWK:   publicKeyJWK,
						DerivationPath: componentDID.DerivationPath,
						Capabilities:   []string{},
						ExposureLevel:  "private",
						CreatedAt:      componentDID.CreatedAt,
					}
				case "skill":
					agentInfo.Skills[componentDID.ComponentName] = types.SkillDIDInfo{
						DID:            componentDID.ComponentDID,
						FunctionName:   componentDID.ComponentName,
						PublicKeyJWK:   publicKeyJWK,
						DerivationPath: componentDID.DerivationPath,
						Tags:           []string{},
						ExposureLevel:  "private",
						CreatedAt:      componentDID.CreatedAt,
					}
				}
			}

			registry.AgentNodes[agentInfo.AgentNodeID] = agentInfo
			registry.TotalDIDs++
		}

		repaired, registryQuarantine, rootErr := verifyRegistryIdentities(registry)
		for did, reason := range registryQuarantine {
			quarantined[did] = reason
		}
		if rootErr != nil {
			log.Printf("Error: %v", rootErr)
			brokenServers[serverID] = rootErr
		} else if repaired > 0 {
			log.Printf("Recovered %d DID derivation paths or public keys for af server %s; persisting corrections", repaired, serverID)
			toRepair = append(toRepair, registry)
		}

		registries[serverID] = registry
	}

	for serverID, orphans := range agentsByServer {
		log.Printf("Warning: %d agent DIDs reference af server %s, which has no DID registry; they were not loaded", len(orphans), serverID)
	}

	r.mu.Lock()
	r.registries = registries
	r.quarantined = quarantined
	r.brokenServers = brokenServers
	for _, registry := range toRepair {
		// The recovered paths are already correct in memory; a failed write is
		// retried on the next registry save, so it must not block startup.
		if err := r.saveRegistryToDatabase(registry); err != nil {
			log.Printf("Warning: failed to persist recovered DID paths for af server %s: %v", registry.AgentFieldServerID, err)
		}
	}
	r.mu.Unlock()

	return nil
}

// saveRegistryToDatabase saves a registry to the database. Callers must hold
// r.mu. Every identity is written with its exact derivation path and public
// key. Quarantined identities are left untouched in storage.
func (r *DIDRegistry) saveRegistryToDatabase(registry *types.DIDRegistry) error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}
	if err := r.brokenServers[registry.AgentFieldServerID]; err != nil {
		return fmt.Errorf("refusing to write DID registry for af server %s: %w", registry.AgentFieldServerID, err)
	}

	ctx := context.Background()

	// Encrypt master seed before storing if encryption is configured
	seedToStore := registry.MasterSeed
	if r.encryptionService != nil {
		encrypted, err := r.encryptionService.EncryptBytes(registry.MasterSeed)
		if err != nil {
			return fmt.Errorf("failed to encrypt master seed: %w", err)
		}
		seedToStore = encrypted
	}

	// Store af server DID information
	err := r.storageProvider.StoreAgentFieldServerDID(
		ctx,
		registry.AgentFieldServerID,
		registry.RootDID,
		seedToStore,
		registry.CreatedAt,
		registry.LastKeyRotation,
	)
	if err != nil {
		return fmt.Errorf("failed to store af server DID: %w", err)
	}

	nodeIDs := make([]string, 0, len(registry.AgentNodes))
	for nodeID := range registry.AgentNodes {
		nodeIDs = append(nodeIDs, nodeID)
	}
	sort.Strings(nodeIDs)

	for _, nodeID := range nodeIDs {
		agentInfo := registry.AgentNodes[nodeID]
		if _, bad := r.quarantined[agentInfo.DID]; bad {
			continue
		}

		components := make([]storage.ComponentDIDRequest, 0, len(agentInfo.Reasoners)+len(agentInfo.Skills))
		for _, reasonerInfo := range agentInfo.Reasoners {
			if _, bad := r.quarantined[reasonerInfo.DID]; bad {
				continue
			}
			components = append(components, storage.ComponentDIDRequest{
				ComponentDID:   reasonerInfo.DID,
				ComponentType:  "reasoner",
				ComponentName:  reasonerInfo.FunctionName,
				PublicKeyJWK:   string(reasonerInfo.PublicKeyJWK),
				DerivationPath: reasonerInfo.DerivationPath,
			})
		}
		for _, skillInfo := range agentInfo.Skills {
			if _, bad := r.quarantined[skillInfo.DID]; bad {
				continue
			}
			components = append(components, storage.ComponentDIDRequest{
				ComponentDID:   skillInfo.DID,
				ComponentType:  "skill",
				ComponentName:  skillInfo.FunctionName,
				PublicKeyJWK:   string(skillInfo.PublicKeyJWK),
				DerivationPath: skillInfo.DerivationPath,
			})
		}

		err := r.storageProvider.UpsertAgentDIDWithComponents(ctx, storage.AgentDIDRecord{
			AgentNodeID:        agentInfo.AgentNodeID,
			DID:                agentInfo.DID,
			AgentFieldServerID: registry.AgentFieldServerID,
			PublicKeyJWK:       string(agentInfo.PublicKeyJWK),
			DerivationPath:     agentInfo.DerivationPath,
			X25519Epoch:        agentInfo.X25519Epoch,
			Status:             string(agentInfo.Status),
			Components:         components,
		})
		if err != nil {
			return fmt.Errorf("failed to store agent DID %s with components: %w", agentInfo.AgentNodeID, err)
		}
	}

	return nil
}

// QuarantineReason reports why a DID must not be used, or "" when it is sound.
func (r *DIDRegistry) QuarantineReason(did string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.quarantined[did]
}

// IntegrityError returns a non-nil error when the stored registry for the
// server cannot be trusted, for example when its root DID does not match the
// decrypted master seed.
func (r *DIDRegistry) IntegrityError(agentfieldServerID string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.brokenServers[agentfieldServerID]
}

// RetireComponents marks removed components of an agent as retired in storage,
// so they are not loaded back into the registry on the next start. The rows are
// kept so their derivation paths are never allocated again.
func (r *DIDRegistry) RetireComponents(agentfieldServerID, agentNodeID string, componentDIDs []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}
	if len(componentDIDs) == 0 {
		return nil
	}
	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		return fmt.Errorf("registry not found for af server: %s", agentfieldServerID)
	}
	agentInfo, exists := registry.AgentNodes[agentNodeID]
	if !exists {
		return fmt.Errorf("agent not found: %s", agentNodeID)
	}
	if err := r.storageProvider.RetireComponentDIDs(context.Background(), agentInfo.DID, componentDIDs); err != nil {
		return fmt.Errorf("failed to retire component DIDs for agent %s: %w", agentNodeID, err)
	}
	return nil
}

// persistedComponentPaths returns stored derivation paths for an agent's
// components, active and retired, so a new component never reuses the key of a
// component that was removed, even across restarts.
func (r *DIDRegistry) persistedComponentPaths(agentDID string) []string {
	if r.storageProvider == nil || agentDID == "" {
		return nil
	}
	components, err := r.storageProvider.ListAllComponentDIDs(context.Background(), agentDID)
	if err != nil {
		log.Printf("Warning: failed to list stored component DIDs for %s: %v", agentDID, err)
		return nil
	}
	paths := make([]string, 0, len(components))
	for _, component := range components {
		if component != nil && component.DerivationPath != "" {
			paths = append(paths, component.DerivationPath)
		}
	}
	return paths
}
