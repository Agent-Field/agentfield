package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
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
//
// The result is a snapshot. Published agent maps are never modified in place
// (see UpdateAgents), so it is safe to read without further locking, and
// changing it has no effect on the registry.
func (r *DIDRegistry) GetRegistry(agentfieldServerID string) (*types.DIDRegistry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registry, exists := r.registries[agentfieldServerID]
	if !exists {
		// Return nil, nil for "not found" to distinguish from actual errors
		return nil, nil
	}

	snapshot := *registry
	return &snapshot, nil
}

// StoreRegistry stores a DID registry for a af server and writes all of its
// identities. The in-memory registry is replaced only when every write
// succeeded.
func (r *DIDRegistry) StoreRegistry(registry *types.DIDRegistry) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.saveRegistryToDatabase(context.Background(), registry, nil); err != nil {
		return err
	}
	r.registries[registry.AgentFieldServerID] = registry
	return nil
}

// CreateRegistry persists a newly generated registry without ever replacing
// one that already exists in storage. When another control plane created the
// same server ID first, its stored identity is adopted instead, after checking
// that its root DID matches its master seed.
func (r *DIDRegistry) CreateRegistry(registry *types.DIDRegistry) (*types.DIDRegistry, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	serverID := registry.AgentFieldServerID
	if existing, ok := r.registries[serverID]; ok {
		snapshot := *existing
		return &snapshot, nil
	}
	if err := r.refuseNewRegistryWithOrphanedAgents(serverID); err != nil {
		return nil, err
	}
	if err := r.storeServerRow(context.Background(), registry); err != nil {
		return nil, err
	}

	stored, err := r.storageProvider.GetAgentFieldServerDID(context.Background(), serverID)
	if err != nil {
		return nil, fmt.Errorf("failed to read back af server DID %s: %w", serverID, err)
	}
	if stored == nil {
		return nil, fmt.Errorf("af server DID %s was not stored", serverID)
	}
	if stored.RootDID != registry.RootDID {
		seed := r.decryptSeed(serverID, stored.MasterSeed)
		if !identityMatchesPath(seed, rootDerivationPath, stored.RootDID) {
			err := fmt.Errorf("stored root DID %s for af server %s does not match its master seed; see docs/ENVIRONMENT_VARIABLES.md, Lost-root recovery", stored.RootDID, serverID)
			r.brokenServers[serverID] = err
			return nil, err
		}
		log.Printf("DID registry for af server %s was created by another control plane; adopting root DID %s", serverID, stored.RootDID)
		registry.RootDID = stored.RootDID
		registry.MasterSeed = seed
		registry.CreatedAt = stored.CreatedAt
		registry.LastKeyRotation = stored.LastKeyRotation
	}

	r.registries[serverID] = registry
	snapshot := *registry
	return &snapshot, nil
}

// Check before persisting a newly minted seed. Reading agents before
// registries prevents observing agents without their registry row.
func (r *DIDRegistry) refuseNewRegistryWithOrphanedAgents(serverID string) error {
	ctx := context.Background()
	agents, err := r.storageProvider.ListAgentDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to check existing agent identities: %w", err)
	}
	servers, err := r.storageProvider.ListAgentFieldServerDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to check existing DID registries: %w", err)
	}
	knownServers := make(map[string]bool, len(servers))
	for _, server := range servers {
		knownServers[server.AgentFieldServerID] = true
	}
	if knownServers[serverID] {
		return nil // Another replica already created the registry to adopt.
	}
	orphans := make(map[string][]string)
	for _, agent := range agents {
		if agent != nil && agent.Status != types.AgentDIDStatusRevoked && !knownServers[agent.AgentFieldServerID] {
			orphans[agent.AgentFieldServerID] = append(orphans[agent.AgentFieldServerID], agent.AgentNodeID)
		}
	}
	if len(orphans) == 0 {
		return nil
	}
	ids := make([]string, 0, len(orphans))
	for id := range orphans {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var details, sqlLines []string
	for _, id := range ids {
		nodes := orphans[id]
		sort.Strings(nodes)
		listed := nodes
		if len(listed) > 10 {
			listed = listed[:10]
		}
		summary := strings.Join(listed, ", ")
		if len(nodes) > len(listed) {
			summary += fmt.Sprintf(", and %d more", len(nodes)-len(listed))
		}
		details = append(details, fmt.Sprintf("%d orphaned agents of server %q (%s)", len(nodes), id, summary))
		sqlLines = append(sqlLines, fmt.Sprintf("UPDATE agent_dids SET status = 'revoked' WHERE agentfield_server_id = '%s';", strings.ReplaceAll(id, "'", "''")))
	}
	return fmt.Errorf("DID registry is missing while %s exist. To recover, restore the did_registry row or the whole database from a backup (the master seed exists only there). If that identity is permanently lost, revoke its agents and restart; they get new identities when they next register:\n%s", strings.Join(details, "; "), strings.Join(sqlLines, "\n"))
}

// UpdateAgents applies a change to a server's agents. fn receives a private
// copy of the registry whose AgentNodes map may be modified freely; inner
// Reasoners and Skills maps are shared and must be replaced, not modified (see
// cloneAgentInfo). fn returns the node IDs it changed. Only those agents are
// written, and the change is published in memory only when every write
// succeeded, so a failed save leaves the registry exactly as it was.
func (r *DIDRegistry) UpdateAgents(agentfieldServerID string, fn func(registry *types.DIDRegistry) ([]string, error)) error {
	return r.updateAgents(context.Background(), agentfieldServerID, fn)
}

func (r *DIDRegistry) updateAgents(ctx context.Context, agentfieldServerID string, fn func(registry *types.DIDRegistry) ([]string, error)) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	current, exists := r.registries[agentfieldServerID]
	if !exists {
		return fmt.Errorf("registry not found for af server: %s", agentfieldServerID)
	}
	if err := r.brokenServers[agentfieldServerID]; err != nil {
		return fmt.Errorf("refusing to modify DID registry for af server %s: %w", agentfieldServerID, err)
	}

	next := *current
	next.AgentNodes = make(map[string]types.AgentDIDInfo, len(current.AgentNodes)+1)
	for nodeID, agent := range current.AgentNodes {
		next.AgentNodes[nodeID] = agent
	}

	changed, err := fn(&next)
	if err != nil {
		return err
	}
	if len(changed) == 0 {
		return nil
	}
	if err := r.saveRegistryToDatabase(ctx, &next, changed); err != nil {
		return err
	}
	r.registries[agentfieldServerID] = &next
	return nil
}

// removeRetiredComponentsFromMemory publishes durable retirements when saving
// the remaining components failed. It never writes to storage.
func (r *DIDRegistry) removeRetiredComponentsFromMemory(agentfieldServerID, agentNodeID string, retiredDIDs map[string]bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	current := r.registries[agentfieldServerID]
	if current == nil {
		return
	}
	agent, exists := current.AgentNodes[agentNodeID]
	if !exists {
		return
	}
	agent = cloneAgentInfo(agent)
	removedCount := 0
	for name, reasoner := range agent.Reasoners {
		if retiredDIDs[reasoner.DID] {
			delete(agent.Reasoners, name)
			removedCount++
		}
	}
	for name, skill := range agent.Skills {
		if retiredDIDs[skill.DID] {
			delete(agent.Skills, name)
			removedCount++
		}
	}
	if removedCount == 0 {
		return
	}

	next := *current
	next.AgentNodes = make(map[string]types.AgentDIDInfo, len(current.AgentNodes))
	for nodeID, info := range current.AgentNodes {
		next.AgentNodes[nodeID] = info
	}
	next.AgentNodes[agentNodeID] = agent
	next.TotalDIDs -= removedCount
	r.registries[agentfieldServerID] = &next
}

// cloneAgentInfo returns a copy of agent whose component maps can be modified
// without affecting published registry snapshots.
func cloneAgentInfo(agent types.AgentDIDInfo) types.AgentDIDInfo {
	reasoners := make(map[string]types.ReasonerDIDInfo, len(agent.Reasoners))
	for id, reasoner := range agent.Reasoners {
		reasoners[id] = reasoner
	}
	skills := make(map[string]types.SkillDIDInfo, len(agent.Skills))
	for id, skill := range agent.Skills {
		skills[id] = skill
	}
	agent.Reasoners = reasoners
	agent.Skills = skills
	return agent
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

// unverifiableStoredIdentityError stops registration without an allocation retry.
// Startup can repair legacy paths, but this read-only path cannot.
type unverifiableStoredIdentityError struct {
	nodeID, componentType, componentName string
}

func (e *unverifiableStoredIdentityError) Error() string {
	identity := "agent"
	if e.componentType != "" {
		identity = fmt.Sprintf("%s %s", e.componentType, e.componentName)
	}
	return fmt.Sprintf("stored identity for node %s (%s) was written without a verifiable derivation path; restart this control plane to recover and persist derivation paths written by older versions; if it still cannot be recovered, revoke the agent (or retire the component) as docs/ENVIRONMENT_VARIABLES.md describes", e.nodeID, identity)
}

// readThroughAgent adds only verified identities absent from this replica's
// memory (D1.1-D1.3). It retries retirement of startup-quarantined components,
// but never repairs derivation paths or replaces a local identity.
func (r *DIDRegistry) readThroughAgent(ctx context.Context, serverID, nodeID string) (*types.AgentDIDInfo, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	current := r.registries[serverID]
	if current == nil {
		return nil, fmt.Errorf("registry not found for af server: %s", serverID)
	}
	agent, inMemory := current.AgentNodes[nodeID]
	if !inMemory {
		// D1.2: use the same earliest active row selected by startup loading.
		stored, err := r.storageProvider.GetActiveAgentDID(ctx, serverID, nodeID)
		if err != nil {
			return nil, fmt.Errorf("failed to look up stored agent %s: %w", nodeID, err)
		}
		if stored == nil {
			return nil, nil
		}
		if !identityMatchesPath(current.MasterSeed, stored.DerivationPath, stored.DID) {
			return nil, &unverifiableStoredIdentityError{nodeID: nodeID}
		}
		agent = *stored
		agent.Reasoners = make(map[string]types.ReasonerDIDInfo)
		agent.Skills = make(map[string]types.SkillDIDInfo)
		normalizePublicKeyJWK(&agent.PublicKeyJWK, agent.DID)
	}

	if r.quarantined[agent.DID] != "" {
		// An unrecoverable agent requires operator revocation. Its children
		// are quarantined with it, not pending component retirement.
		return &agent, nil
	}

	// D1.1: components are read by the agent DID already held in memory.
	components, err := r.storageProvider.ListComponentDIDs(ctx, agent.DID)
	if err != nil {
		return nil, fmt.Errorf("failed to load components for agent %s: %w", nodeID, err)
	}
	// Retire only quarantined DIDs before importing surviving same-name rows.
	for _, component := range components {
		if r.quarantined[component.ComponentDID] == "" {
			continue
		}
		if err := r.storageProvider.RetireComponentDIDs(ctx, agent.DID, []string{component.ComponentDID}); err != nil {
			return nil, fmt.Errorf("cannot retire unverifiable component DID %s: %w; stop the control plane and run: UPDATE component_dids SET status = 'retired' WHERE did = '%s'; (see docs/ENVIRONMENT_VARIABLES.md, Component retirement recovery)", component.ComponentDID, err, strings.ReplaceAll(component.ComponentDID, "'", "''"))
		}
	}
	var additions []string
	changed := !inMemory
	// Storage returns newest first; startup's last assignment keeps the
	// earliest active component when old databases contain duplicate names.
	for i := len(components) - 1; i >= 0; i-- {
		component := components[i]
		if r.quarantined[component.ComponentDID] != "" {
			continue
		}
		switch component.ComponentType {
		case "reasoner":
			if _, exists := agent.Reasoners[component.ComponentName]; exists {
				continue
			}
		case "skill":
			if _, exists := agent.Skills[component.ComponentName]; exists {
				continue
			}
		default:
			continue
		}
		if !identityMatchesPath(current.MasterSeed, component.DerivationPath, component.ComponentDID) {
			return nil, &unverifiableStoredIdentityError{nodeID: nodeID, componentType: component.ComponentType, componentName: component.ComponentName}
		}
		if !changed {
			agent = cloneAgentInfo(agent)
			changed = true
		}
		publicKey := json.RawMessage(component.PublicKeyJWK)
		normalizePublicKeyJWK(&publicKey, component.ComponentDID)
		switch component.ComponentType {
		case "reasoner":
			agent.Reasoners[component.ComponentName] = types.ReasonerDIDInfo{DID: component.ComponentDID, FunctionName: component.ComponentName, PublicKeyJWK: publicKey, DerivationPath: component.DerivationPath, Capabilities: []string{}, ExposureLevel: "private", CreatedAt: component.CreatedAt}
		case "skill":
			agent.Skills[component.ComponentName] = types.SkillDIDInfo{DID: component.ComponentDID, FunctionName: component.ComponentName, PublicKeyJWK: publicKey, DerivationPath: component.DerivationPath, Tags: []string{}, ExposureLevel: "private", CreatedAt: component.CreatedAt}
		}
		additions = append(additions, component.ComponentDID)
	}
	if changed {
		next := *current
		next.AgentNodes = make(map[string]types.AgentDIDInfo, len(current.AgentNodes)+1)
		for id, identity := range current.AgentNodes {
			next.AgentNodes[id] = identity
		}
		next.AgentNodes[nodeID] = agent
		next.TotalDIDs += len(additions)
		if !inMemory {
			next.TotalDIDs++
		}
		r.registries[serverID] = &next
		if !inMemory {
			delete(r.quarantined, agent.DID)
		}
		for _, did := range additions {
			delete(r.quarantined, did)
		}
	}
	return &agent, nil
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
	return r.UpdateAgents(agentfieldServerID, func(registry *types.DIDRegistry) ([]string, error) {
		agentInfo, exists := registry.AgentNodes[agentNodeID]
		if !exists {
			return nil, fmt.Errorf("agent not found: %s", agentNodeID)
		}
		agentInfo.Status = status
		registry.AgentNodes[agentNodeID] = agentInfo
		return []string{agentNodeID}, nil
	})
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
	agentDIDs, err := r.storageProvider.ListAgentDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to list agent DIDs: %w", err)
	}

	agentfieldServerDIDs, err := r.storageProvider.ListAgentFieldServerDIDs(ctx)
	if err != nil {
		return fmt.Errorf("failed to list af server DIDs: %w", err)
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

		masterSeed := r.decryptSeed(serverID, agentfieldServerDIDInfo.MasterSeed)

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

		componentSnapshots := make(map[string][]*types.ComponentDIDInfo)
		for _, agentDIDInfo := range rows {
			if agentDIDInfo.Status == types.AgentDIDStatusRevoked {
				// An operator revoked this identity. It stays in storage so its
				// DID and derivation path are never issued again, but the node
				// gets a new identity the next time it registers.
				log.Printf("Agent %s DID %s under af server %s is revoked; it will not be loaded",
					agentDIDInfo.AgentNodeID, agentDIDInfo.DID, serverID)
				continue
			}
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

			componentSnapshots[agentDIDInfo.DID] = componentDIDs
			for i := len(componentDIDs) - 1; i >= 0; i-- {
				adoptStoredComponent(&agentInfo, componentDIDs[i])
			}

			registry.AgentNodes[agentInfo.AgentNodeID] = agentInfo
			registry.TotalDIDs++
		}

		repairedTotal := 0
		for {
			repaired, registryQuarantine, retire, rootErr := verifyRegistryIdentities(registry, len(agentDIDs))
			repairedTotal += repaired
			for did, reason := range registryQuarantine {
				quarantined[did] = reason
			}
			if rootErr != nil {
				log.Printf("Error: %v", rootErr)
				brokenServers[serverID] = rootErr
				break
			}
			for agentDID, componentDIDs := range retire {
				// Failed retirement is retried by read-through before issuing keys.
				if err := r.storageProvider.RetireComponentDIDs(ctx, agentDID, componentDIDs); err != nil {
					log.Printf("Warning: failed to retire unverifiable component DIDs of agent %s: %v", agentDID, err)
				}
			}
			// Quarantine applies to a DID, not its name. Verify the next stored
			// candidate before adopting it, including recovery of legacy paths.
			added := false
			for nodeID, agent := range registry.AgentNodes {
				if quarantined[agent.DID] != "" {
					continue
				}
				rows := componentSnapshots[agent.DID]
				for i := len(rows) - 1; i >= 0; i-- {
					if quarantined[rows[i].ComponentDID] == "" && adoptStoredComponent(&agent, rows[i]) {
						added = true
					}
				}
				registry.AgentNodes[nodeID] = agent
			}
			if !added {
				break
			}
		}
		if repairedTotal > 0 && brokenServers[serverID] == nil {
			log.Printf("Recovered %d DID derivation paths or public keys for af server %s; persisting corrections", repairedTotal, serverID)
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
		if err := r.saveRegistryToDatabase(context.Background(), registry, nil); err != nil {
			log.Printf("Warning: failed to persist recovered DID paths for af server %s: %v", registry.AgentFieldServerID, err)
		}
	}
	r.mu.Unlock()

	return nil
}

// adoptStoredComponent fills an absent name; callers visit oldest candidates first.
func adoptStoredComponent(agentInfo *types.AgentDIDInfo, componentDID *types.ComponentDIDInfo) bool {
	switch componentDID.ComponentType {
	case "reasoner":
		if _, exists := agentInfo.Reasoners[componentDID.ComponentName]; exists {
			return false
		}
	case "skill":
		if _, exists := agentInfo.Skills[componentDID.ComponentName]; exists {
			return false
		}
	default:
		return false
	}

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
	return true
}

// decryptSeed decrypts a stored master seed when encryption is configured.
func (r *DIDRegistry) decryptSeed(serverID string, stored []byte) []byte {
	if r.encryptionService == nil {
		return stored
	}
	decrypted, err := r.encryptionService.DecryptBytes(stored)
	if err != nil {
		// Backward compatibility: if decryption fails, the seed may be stored
		// as plaintext from before encryption was configured. The root DID
		// check rejects the registry if these bytes are not the seed.
		log.Printf("Warning: master seed decryption failed for %s (may be plaintext from before encryption was enabled), using raw bytes",
			serverID)
		return stored
	}
	return decrypted
}

// storeServerRow writes the af server DID row. Storage never replaces an
// existing row's key material with a different identity.
func (r *DIDRegistry) storeServerRow(ctx context.Context, registry *types.DIDRegistry) error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}

	// Encrypt master seed before storing if encryption is configured
	seedToStore := registry.MasterSeed
	if r.encryptionService != nil {
		encrypted, err := r.encryptionService.EncryptBytes(registry.MasterSeed)
		if err != nil {
			return fmt.Errorf("failed to encrypt master seed: %w", err)
		}
		seedToStore = encrypted
	}

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
	return nil
}

// saveRegistryToDatabase saves a registry to the database. Callers must hold
// r.mu. Only the agents in nodeIDs are written, or all agents when nodeIDs is
// nil. Every identity is written with its exact derivation path and public
// key. Quarantined identities are left untouched in storage. A failure for one
// agent does not stop the others from being written; all failures are
// returned together.
func (r *DIDRegistry) saveRegistryToDatabase(ctx context.Context, registry *types.DIDRegistry, nodeIDs []string) error {
	if r.storageProvider == nil {
		return fmt.Errorf("storage provider not available")
	}
	if err := r.brokenServers[registry.AgentFieldServerID]; err != nil {
		return fmt.Errorf("refusing to write DID registry for af server %s: %w", registry.AgentFieldServerID, err)
	}

	if err := r.storeServerRow(ctx, registry); err != nil {
		return err
	}

	if nodeIDs == nil {
		nodeIDs = make([]string, 0, len(registry.AgentNodes))
		for nodeID := range registry.AgentNodes {
			nodeIDs = append(nodeIDs, nodeID)
		}
	}
	nodeIDs = append([]string(nil), nodeIDs...)
	sort.Strings(nodeIDs)

	var errs []error
	for _, nodeID := range nodeIDs {
		agentInfo, exists := registry.AgentNodes[nodeID]
		if !exists {
			continue
		}
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
			errs = append(errs, fmt.Errorf("failed to store agent DID %s with components: %w", agentInfo.AgentNodeID, err))
		}
	}

	return errors.Join(errs...)
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
	return r.retireComponents(context.Background(), agentfieldServerID, agentNodeID, componentDIDs)
}

func (r *DIDRegistry) retireComponents(ctx context.Context, agentfieldServerID, agentNodeID string, componentDIDs []string) error {
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
	if err := r.storageProvider.RetireComponentDIDs(ctx, agentInfo.DID, componentDIDs); err != nil {
		return fmt.Errorf("failed to retire component DIDs for agent %s: %w", agentNodeID, err)
	}
	return nil
}

// usedIdentities collects every DID and derivation path that was ever issued
// from this database: all agent and component rows of every server and status,
// plus everything in memory and in quarantine. A storage error is returned, so
// allocation never proceeds on an incomplete view.
func (r *DIDRegistry) usedIdentities(ctx context.Context) (*usedIdentities, error) {
	if r.storageProvider == nil {
		return nil, fmt.Errorf("storage provider not available")
	}
	used := newUsedIdentities()

	agents, err := r.storageProvider.ListAgentDIDs(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list stored agent DIDs: %w", err)
	}
	for _, agent := range agents {
		if agent != nil {
			used.add(agent.DID, agent.DerivationPath)
		}
	}
	components, err := r.storageProvider.ListAllComponentDIDs(ctx, "")
	if err != nil {
		return nil, fmt.Errorf("failed to list stored component DIDs: %w", err)
	}
	for _, component := range components {
		if component != nil {
			used.add(component.ComponentDID, component.DerivationPath)
		}
	}

	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, registry := range r.registries {
		used.addRegistry(registry)
	}
	for did := range r.quarantined {
		used.add(did, "")
	}
	return used, nil
}
