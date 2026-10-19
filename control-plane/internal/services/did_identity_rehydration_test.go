// Identity-continuity regression tests, contributed with the original report.
//
// These tests assert the invariants an identity/provenance system must hold across
// a control-plane restart. The failure messages name the violated invariant.
//
// Only synthetic, disposable state is used (t.TempDir). No model or provider is
// invoked. No secret material is printed -- only the public "x" coordinate of a
// JWK is compared, and only to show that two public keys differ.
package services

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type safeJWK struct {
	X string `json:"x"`
}

// publicX extracts ONLY the public x coordinate of a JWK. For a private Ed25519
// JWK the "x" member is the public key, so comparing x values reveals nothing
// secret while still proving whether a private key matches a retained public key.
func publicX(t *testing.T, label, value string) string {
	t.Helper()
	if strings.TrimSpace(value) == "" {
		t.Fatalf("%s: JWK is empty", label)
	}
	var jwk safeJWK
	if err := json.Unmarshal([]byte(value), &jwk); err != nil {
		t.Fatalf("%s: parse JWK: %v", label, err)
	}
	if jwk.X == "" {
		t.Fatalf("%s: JWK has no public x coordinate", label)
	}
	return jwk.X
}

func openReproStorage(t *testing.T, dir string) storage.StorageProvider {
	t.Helper()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(dir, "agentfield.db"),
			KVStorePath:  filepath.Join(dir, "agentfield.bolt"),
		},
	}
	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := provider.Initialize(context.Background(), cfg); err != nil {
		t.Fatalf("initialize storage: %v", err)
	}
	return provider
}

func openReproService(t *testing.T, provider storage.StorageProvider, serverID, keyDir string) (*DIDService, *DIDRegistry) {
	t.Helper()
	registry := NewDIDRegistryWithStorage(provider)
	if err := registry.Initialize(); err != nil {
		t.Fatalf("initialize registry: %v", err)
	}
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keyDir, Type: "local"})
	if err != nil {
		t.Fatalf("initialize keystore: %v", err)
	}
	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keyDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)
	if err := service.Initialize(serverID); err != nil {
		t.Fatalf("initialize DID service: %v", err)
	}
	return service, registry
}

// TestReproIdentityRehydrationMustPreserveKeysAndPaths covers proposed regression
// invariants T2, T3 and T8: after a full control-plane restart against the SAME
// state and the SAME logical home, an existing node must get back the same DIDs
// AND private keys that correspond to its retained public keys.
func TestReproIdentityRehydrationMustPreserveKeysAndPaths(t *testing.T) {
	stateDir := t.TempDir()
	keyDir := filepath.Join(stateDir, "keys")
	serverID := "synthetic-stable-server"
	ctx := context.Background()

	// --- Generation A: first boot, first registration -----------------------
	providerA := openReproStorage(t, stateDir)
	serviceA, _ := openReproService(t, providerA, serverID, keyDir)
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "synthetic-node",
		Reasoners:   []types.ReasonerDefinition{{ID: "echo"}},
	}
	created, err := serviceA.RegisterAgent(req)
	if err != nil || !created.Success {
		t.Fatalf("initial registration failed: response=%+v err=%v", created, err)
	}
	originalAgent := created.IdentityPackage.AgentDID
	originalReasoner := created.IdentityPackage.ReasonerDIDs["echo"]
	t.Logf("generation A: agent DID        = %s", originalAgent.DID)
	t.Logf("generation A: agent path       = %s", originalAgent.DerivationPath)
	t.Logf("generation A: reasoner DID     = %s", originalReasoner.DID)
	t.Logf("generation A: reasoner path    = %s", originalReasoner.DerivationPath)

	// What actually reached persistent storage.
	storedAgents, err := providerA.ListAgentDIDs(ctx)
	if err != nil || len(storedAgents) != 1 {
		t.Fatalf("expected one stored agent: count=%d err=%v", len(storedAgents), err)
	}
	storedComponents, err := providerA.ListComponentDIDs(ctx, originalAgent.DID)
	if err != nil || len(storedComponents) != 1 {
		t.Fatalf("expected one stored component: count=%d err=%v", len(storedComponents), err)
	}
	storedAgentPath := storedAgents[0].DerivationPath
	storedComponentPath := storedComponents[0].DerivationPath
	t.Logf("persisted:    agent path       = %s", storedAgentPath)
	t.Logf("persisted:    reasoner path    = %s", storedComponentPath)

	if err := providerA.Close(ctx); err != nil {
		t.Fatalf("close first storage: %v", err)
	}

	// --- Generation B: full restart, same state, same logical home -----------
	providerB := openReproStorage(t, stateDir)
	defer providerB.Close(ctx)
	serviceB, _ := openReproService(t, providerB, serverID, keyDir)
	rehydrated, err := serviceB.RegisterAgent(req)
	if err != nil || !rehydrated.Success {
		t.Fatalf("re-registration failed: response=%+v err=%v", rehydrated, err)
	}
	reloadedAgent := rehydrated.IdentityPackage.AgentDID
	reloadedReasoner := rehydrated.IdentityPackage.ReasonerDIDs["echo"]
	t.Logf("generation B: agent path       = %s", reloadedAgent.DerivationPath)
	t.Logf("generation B: reasoner path    = %s", reloadedReasoner.DerivationPath)

	failures := make([]string, 0, 6)

	// T8 (persistence half): the path used to derive the key must survive a save.
	if storedAgentPath != originalAgent.DerivationPath {
		failures = append(failures, fmt.Sprintf(
			"agent derivation path was not persisted verbatim: generated=%q persisted=%q",
			originalAgent.DerivationPath, storedAgentPath))
	}
	if storedComponentPath != originalReasoner.DerivationPath {
		failures = append(failures, fmt.Sprintf(
			"component derivation path was not persisted verbatim: generated=%q persisted=%q",
			originalReasoner.DerivationPath, storedComponentPath))
	}

	// T2: an existing node reconnecting keeps its DID.
	if reloadedAgent.DID != originalAgent.DID {
		failures = append(failures, fmt.Sprintf(
			"agent DID changed across restart: before=%s after=%s", originalAgent.DID, reloadedAgent.DID))
	}
	// T8 (reload half): the rederived private key must match the retained public key.
	if got, want := publicX(t, "reloaded agent private JWK", reloadedAgent.PrivateKeyJWK),
		publicX(t, "original agent public JWK", originalAgent.PublicKeyJWK); got != want {
		failures = append(failures, fmt.Sprintf(
			"rehydrated agent private key does not correspond to the retained public JWK "+
				"(public x from rederived key=%s, retained public x=%s)", got, want))
	}

	// T3: an existing component reconnecting keeps its DID.
	if reloadedReasoner.DID != originalReasoner.DID {
		failures = append(failures, fmt.Sprintf(
			"reasoner DID changed across restart: before=%s after=%s",
			originalReasoner.DID, reloadedReasoner.DID))
	}
	// Component public-key metadata must survive reload.
	if strings.TrimSpace(reloadedReasoner.PublicKeyJWK) == "" {
		failures = append(failures,
			"rehydrated reasoner public JWK is empty: component public-key metadata was not reloaded")
	} else if got, want := publicX(t, "reloaded reasoner private JWK", reloadedReasoner.PrivateKeyJWK),
		publicX(t, "original reasoner public JWK", originalReasoner.PublicKeyJWK); got != want {
		failures = append(failures, fmt.Sprintf(
			"rehydrated reasoner private key does not correspond to the retained public JWK "+
				"(public x from rederived key=%s, retained public x=%s)", got, want))
	}

	if len(failures) > 0 {
		t.Fatalf("identity continuity violated after full control-plane reload:\n- %s",
			strings.Join(failures, "\n- "))
	}
}

// TestReproRegistryReloadMustScopeAgentsToOwningServer covers the registry-scoping
// half of the defect: DIDRegistry.loadRegistriesFromDatabase calls
// StorageProvider.ListAgentDIDs(ctx), which is unfiltered, and inserts every agent
// row into EVERY server registry while overwriting AgentFieldServerID. An agent
// owned by server-a therefore appears inside server-b's registry, where ResolveDID
// will rederive its keys from server-b's unrelated master seed.
func TestReproRegistryReloadMustScopeAgentsToOwningServer(t *testing.T) {
	stateDir := t.TempDir()
	keyDir := filepath.Join(stateDir, "keys")
	ctx := context.Background()

	providerA := openReproStorage(t, stateDir)
	serviceA, _ := openReproService(t, providerA, "server-a", keyDir)
	created, err := serviceA.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node-a"})
	if err != nil || !created.Success {
		t.Fatalf("server-a registration failed: response=%+v err=%v", created, err)
	}
	t.Logf("node-a DID (owned by server-a) = %s", created.IdentityPackage.AgentDID.DID)

	// A second logical server on the same state -- exactly what a changed
	// AGENTFIELD_HOME produces, since the server ID is SHA256(abs home)[:16].
	if _, _ = openReproService(t, providerA, "server-b", keyDir); false {
		t.Fatal("unreachable")
	}
	if err := providerA.Close(ctx); err != nil {
		t.Fatalf("close first storage: %v", err)
	}

	// Restart: reload both registries from the database.
	providerReload := openReproStorage(t, stateDir)
	defer providerReload.Close(ctx)
	registryReload := NewDIDRegistryWithStorage(providerReload)
	if err := registryReload.Initialize(); err != nil {
		t.Fatalf("reload registry: %v", err)
	}
	registryB, err := registryReload.GetRegistry("server-b")
	if err != nil || registryB == nil {
		t.Fatalf("load server-b registry: registry=%v err=%v", registryB, err)
	}
	if leaked, ok := registryB.AgentNodes["node-a"]; ok {
		t.Fatalf("registry scoping violated: node-a is owned by server-a but was loaded "+
			"into server-b's registry (leaked DID=%s, AgentFieldServerID rewritten to %q). "+
			"ResolveDID would rederive this DID's keys from server-b's master seed.",
			leaked.DID, leaked.AgentFieldServerID)
	}
}
