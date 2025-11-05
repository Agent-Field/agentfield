package services

import (
	"context"
	"fmt"
	"sync"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// DIDRegistry manages the storage and retrieval of DID registries using database-only operations.
type DIDRegistry struct {
	mu              sync.RWMutex
	registries      map[string]*types.DIDRegistry
	storageProvider storage.StorageProvider
}

// NewDIDRegistryWithStorage creates a new DID registry instance with database storage.
func NewDIDRegistryWithStorage(storageProvider storage.StorageProvider) *DIDRegistry {
	return &DIDRegistry{
		registries:      make(map[string]*types.DIDRegistry),
		storageProvider: storageProvider,
	}
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

// loadRegistriesFromDatabase loads all registries from the database.
func (r *DIDRegistry) loadRegistriesFromDatabase() error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}

	ctx := context.Background()
	// Load af server DID information
	agentfieldServerDIDs, err := r.storageProvider.ListAgentFieldServerDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to list af server DIDs: %w", err)
	}

	// Create registries for each af server
	for _, agentfieldServerDIDInfo := range agentfieldServerDIDs {
		registry := &types.DIDRegistry{
			AgentFieldServerID: agentfieldServerDIDInfo.AgentFieldServerID,
			RootDID:            agentfieldServerDIDInfo.RootDID,
			MasterSeed:         agentfieldServerDIDInfo.MasterSeed,
			AgentNodes:         make(map[string]types.AgentDIDInfo),
			TotalDIDs:          0,
			CreatedAt:          agentfieldServerDIDInfo.CreatedAt,
			LastKeyRotation:    agentfieldServerDIDInfo.LastKeyRotation,
		}

		// Load agent DIDs for this af server
		agentDIDs, err := r.storageProvider.ListAgentDIDs(ctx)
		if err != nil {
			return fmt.Errorf("failed to list agent DIDs: %w", err)
		}

		for _, agentDIDInfo := range agentDIDs {
			// Filter agents for this af server (assuming we can match by some criteria)
			// For now, we'll add all agents to the default af server
			// TODO: Add af server filtering when the storage interface supports it

			agentInfo := types.AgentDIDInfo{
				DID:                agentDIDInfo.DID,
				AgentNodeID:        agentDIDInfo.AgentNodeID,
				AgentFieldServerID: agentfieldServerDIDInfo.AgentFieldServerID,
				PublicKeyJWK:       agentDIDInfo.PublicKeyJWK,
				DerivationPath:     agentDIDInfo.DerivationPath,
				Status:             agentDIDInfo.Status,
				RegisteredAt:       agentDIDInfo.RegisteredAt,
				Reasoners:          make(map[string]types.ReasonerDIDInfo),
				Skills:             make(map[string]types.SkillDIDInfo),
			}

			// Load component DIDs for this agent
			componentDIDs, err := r.storageProvider.ListComponentDIDs(ctx, agentDIDInfo.DID)
			if err != nil {
				return fmt.Errorf("failed to list component DIDs for agent %s: %w", agentDIDInfo.AgentNodeID, err)
			}

			for _, componentDID := range componentDIDs {
				switch componentDID.ComponentType {
				case "reasoner":
					reasonerInfo := types.ReasonerDIDInfo{
						DID:            componentDID.ComponentDID,
						FunctionName:   componentDID.ComponentName,
						DerivationPath: fmt.Sprintf("m/44'/0'/0'/%d", componentDID.DerivationIndex),
						Capabilities:   []string{}, // TODO: Load from database
						ExposureLevel:  "private",  // TODO: Load from database
						CreatedAt:      componentDID.CreatedAt,
					}
					agentInfo.Reasoners[componentDID.ComponentName] = reasonerInfo

				case "skill":
					skillInfo := types.SkillDIDInfo{
						DID:            componentDID.ComponentDID,
						FunctionName:   componentDID.ComponentName,
						DerivationPath: fmt.Sprintf("m/44'/0'/0'/%d", componentDID.DerivationIndex),
						Tags:           []string{}, // TODO: Load from database
						ExposureLevel:  "private",  // TODO: Load from database
						CreatedAt:      componentDID.CreatedAt,
					}
					agentInfo.Skills[componentDID.ComponentName] = skillInfo
				}
			}

			registry.AgentNodes[agentInfo.AgentNodeID] = agentInfo
			registry.TotalDIDs++
		}

		r.registries[agentfieldServerDIDInfo.AgentFieldServerID] = registry
	}

	return nil
}

// saveRegistryToDatabase saves a registry to the database.
func (r *DIDRegistry) saveRegistryToDatabase(registry *types.DIDRegistry) error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}

	ctx := context.Background()
	// Store af server DID information
	err := r.storageProvider.StoreAgentFieldServerDID(
		ctx,
		registry.AgentFieldServerID,
		registry.RootDID,
		registry.MasterSeed,
		registry.CreatedAt,
		registry.LastKeyRotation,
	)
	if err != nil {
		return fmt.Errorf("failed to store af server DID: %w", err)
	}

	// Store each agent DID and its components using transaction-safe method
	for _, agentInfo := range registry.AgentNodes {
		// Extract derivation index from path (simplified)
		derivationIndex := 0 // TODO: Parse from agentInfo.DerivationPath

		// Prepare component DIDs for batch storage
		var components []storage.ComponentDIDRequest

		// Add reasoner DIDs
		for _, reasonerInfo := range agentInfo.Reasoners {
			reasonerDerivationIndex := 0 // TODO: Parse from reasonerInfo.DerivationPath
			components = append(components, storage.ComponentDIDRequest{
				ComponentDID:    reasonerInfo.DID,
				ComponentType:   "reasoner",
				ComponentName:   reasonerInfo.FunctionName,
				PublicKeyJWK:    string(reasonerInfo.PublicKeyJWK),
				DerivationIndex: reasonerDerivationIndex,
			})
		}

		// Add skill DIDs
		for _, skillInfo := range agentInfo.Skills {
			skillDerivationIndex := 0 // TODO: Parse from skillInfo.DerivationPath
			components = append(components, storage.ComponentDIDRequest{
				ComponentDID:    skillInfo.DID,
				ComponentType:   "skill",
				ComponentName:   skillInfo.FunctionName,
				PublicKeyJWK:    string(skillInfo.PublicKeyJWK),
				DerivationIndex: skillDerivationIndex,
			})
		}

		// Use the enhanced storage method with transaction safety
		err := r.storageProvider.StoreAgentDIDWithComponents(
			ctx,
			agentInfo.AgentNodeID,
			agentInfo.DID,
			registry.AgentFieldServerID, // Use af server ID instead of root DID
			string(agentInfo.PublicKeyJWK),
			derivationIndex,
			components,
		)
		if err != nil {
			// Enhanced error handling for different constraint types
			if validationErr, ok := err.(*storage.ValidationError); ok {
				return fmt.Errorf("validation failed for agent %s: %w", agentInfo.AgentNodeID, validationErr)
			}
			if fkErr, ok := err.(*storage.ForeignKeyConstraintError); ok {
				return fmt.Errorf("foreign key constraint violation for agent %s: %w", agentInfo.AgentNodeID, fkErr)
			}
			if dupErr, ok := err.(*storage.DuplicateDIDError); ok {
				return fmt.Errorf("duplicate DID detected for agent %s: %w", agentInfo.AgentNodeID, dupErr)
			}
			return fmt.Errorf("failed to store agent DID %s with components: %w", agentInfo.AgentNodeID, err)
		}
	}

	return nil
}
