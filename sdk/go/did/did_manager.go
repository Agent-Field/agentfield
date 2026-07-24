package did

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// Manager handles DID registration with the control plane and stores
// the resulting identity package (agent DID, per-reasoner DIDs, per-skill DIDs).
type Manager struct {
	client      *Client
	identityPkg *DIDIdentityPackage
	mu          sync.RWMutex
	logger      *log.Logger
}

// NewManager creates a DID manager backed by the given client.
func NewManager(client *Client, logger *log.Logger) *Manager {
	if logger == nil {
		logger = log.Default()
	}
	return &Manager{
		client: client,
		logger: logger,
	}
}

// RegisterAgent registers the agent and its functions with the control plane's
// DID service. On success, the identity package (containing agent DID, private
// key, and per-function DIDs) is stored locally.
func (m *Manager) RegisterAgent(ctx context.Context, nodeID string, reasonerNames, skillNames []string) error {
	reasoners := make([]FunctionDef, len(reasonerNames))
	for i, name := range reasonerNames {
		reasoners[i] = FunctionDef{ID: name}
	}
	skills := make([]FunctionDef, len(skillNames))
	for i, name := range skillNames {
		skills[i] = FunctionDef{ID: name}
	}

	resp, err := m.client.RegisterAgent(ctx, RegistrationRequest{
		AgentNodeID: nodeID,
		Reasoners:   reasoners,
		Skills:      skills,
	})
	if err != nil {
		return err
	}

	m.mu.Lock()
	m.identityPkg = &resp.IdentityPackage
	m.mu.Unlock()

	m.logger.Printf("DID registered: %s", resp.IdentityPackage.AgentDID.DID)
	return nil
}

// IsRegistered returns true if DID registration has completed successfully.
func (m *Manager) IsRegistered() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.identityPkg != nil && m.identityPkg.AgentDID.DID != ""
}

// GetAgentDID returns the agent's DID, or empty string if not registered.
func (m *Manager) GetAgentDID() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.identityPkg == nil {
		return ""
	}
	return m.identityPkg.AgentDID.DID
}

// GetAgentPrivateKeyJWK returns the agent's private key in JWK format,
// or empty string if not registered.
func (m *Manager) GetAgentPrivateKeyJWK() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.identityPkg == nil {
		return ""
	}
	return m.identityPkg.AgentDID.PrivateKeyJWK
}

// GetFunctionDID resolves the DID for a specific reasoner or skill by name.
// Falls back to the agent-level DID if no function-specific DID is found.
func (m *Manager) GetFunctionDID(name string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.identityPkg == nil {
		return ""
	}
	if id, ok := m.identityPkg.ReasonerDIDs[name]; ok {
		return id.DID
	}
	if id, ok := m.identityPkg.SkillDIDs[name]; ok {
		return id.DID
	}
	return m.identityPkg.AgentDID.DID
}

// GetIdentityPackage returns the full identity package, or nil if not registered.
func (m *Manager) GetIdentityPackage() *DIDIdentityPackage {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.identityPkg
}

// SetIdentityFromCredentials initializes the manager with pre-existing DID credentials
// (for agents that already have DID/PrivateKeyJWK configured). This allows the VC
// generator and DID context propagation to work without calling RegisterAgent.
func (m *Manager) SetIdentityFromCredentials(agentDID, privateKeyJWK string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.identityPkg = &DIDIdentityPackage{
		AgentDID: DIDIdentity{
			DID:           agentDID,
			PrivateKeyJWK: privateKeyJWK,
			ComponentType: "agent",
		},
		ReasonerDIDs: make(map[string]DIDIdentity),
		SkillDIDs:    make(map[string]DIDIdentity),
	}
	m.logger.Printf("DID credentials set: %s", fmt.Sprintf("%.40s...", agentDID))
}
