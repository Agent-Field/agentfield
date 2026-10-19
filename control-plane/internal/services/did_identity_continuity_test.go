package services

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// continuityEnv is one on-disk state directory that can be opened repeatedly to
// simulate full control-plane restarts.
type continuityEnv struct {
	t        *testing.T
	stateDir string
	keyDir   string
	provider storage.StorageProvider
}

func newContinuityEnv(t *testing.T) *continuityEnv {
	t.Helper()
	stateDir := t.TempDir()
	return &continuityEnv{t: t, stateDir: stateDir, keyDir: filepath.Join(stateDir, "keys")}
}

// restart closes any open storage and reopens it from disk.
func (e *continuityEnv) restart() storage.StorageProvider {
	e.t.Helper()
	if e.provider != nil {
		require.NoError(e.t, e.provider.Close(context.Background()))
	}
	e.provider = openReproStorage(e.t, e.stateDir)
	e.t.Cleanup(func() {
		if e.provider != nil {
			_ = e.provider.Close(context.Background())
			e.provider = nil
		}
	})
	return e.provider
}

func (e *continuityEnv) registry() *DIDRegistry {
	e.t.Helper()
	registry := NewDIDRegistryWithStorage(e.provider)
	require.NoError(e.t, registry.Initialize())
	return registry
}

func (e *continuityEnv) service(registry *DIDRegistry, serverID string) *DIDService {
	e.t.Helper()
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: e.keyDir, Type: "local"})
	require.NoError(e.t, err)
	cfg := &config.DIDConfig{
		Enabled:  true,
		Keystore: config.KeystoreConfig{Path: e.keyDir, Type: "local"},
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: true,
			PersistExecutionVC:    true,
			HashSensitiveData:     true,
		},
	}
	svc := NewDIDService(cfg, ks, registry)
	require.NoError(e.t, svc.Initialize(serverID))
	return svc
}

func register(t *testing.T, svc *DIDService, nodeID string, reasoners, skills []string) *types.DIDRegistrationResponse {
	t.Helper()
	req := &types.DIDRegistrationRequest{AgentNodeID: nodeID}
	for _, id := range reasoners {
		req.Reasoners = append(req.Reasoners, types.ReasonerDefinition{ID: id})
	}
	for _, id := range skills {
		req.Skills = append(req.Skills, types.SkillDefinition{ID: id})
	}
	resp, err := svc.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success, resp.Error)
	return resp
}

// requireKeyMatchesDID asserts a private JWK is the key encoded in the DID.
func requireKeyMatchesDID(t *testing.T, did, privateKeyJWK string) {
	t.Helper()
	require.NotEmpty(t, privateKeyJWK, "private key for %s is empty", did)
	var jwk struct {
		X string `json:"x"`
	}
	require.NoError(t, json.Unmarshal([]byte(privateKeyJWK), &jwk))
	want, err := publicKeyJWKFromDIDKey(did)
	require.NoError(t, err)
	var wantJWK struct {
		X string `json:"x"`
	}
	require.NoError(t, json.Unmarshal([]byte(want), &wantJWK))
	require.Equal(t, wantJWK.X, jwk.X, "private key does not belong to %s", did)
}

func requirePackageConsistent(t *testing.T, pkg types.DIDIdentityPackage) {
	t.Helper()
	requireKeyMatchesDID(t, pkg.AgentDID.DID, pkg.AgentDID.PrivateKeyJWK)
	for _, identity := range pkg.ReasonerDIDs {
		requireKeyMatchesDID(t, identity.DID, identity.PrivateKeyJWK)
		require.NotEmpty(t, identity.PublicKeyJWK)
	}
	for _, identity := range pkg.SkillDIDs {
		requireKeyMatchesDID(t, identity.DID, identity.PrivateKeyJWK)
		require.NotEmpty(t, identity.PublicKeyJWK)
	}
}

func TestIdentityContinuity_RestartPreservesEverything(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	first := register(t, svc, "node-1", []string{"r1", "r2"}, []string{"s1"})
	second := register(t, svc, "node-2", []string{"r1"}, nil)
	rootBefore, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	for i := 0; i < 2; i++ {
		env.restart()
		svc = env.service(env.registry(), "server-a")

		rootAfter, err := svc.GetControlPlaneIssuerDID()
		require.NoError(t, err)
		require.Equal(t, rootBefore, rootAfter)

		again := register(t, svc, "node-1", []string{"r1", "r2"}, []string{"s1"})
		require.Equal(t, first.IdentityPackage.AgentDID.DID, again.IdentityPackage.AgentDID.DID)
		require.Equal(t, first.IdentityPackage.AgentDID.DerivationPath, again.IdentityPackage.AgentDID.DerivationPath)
		require.Equal(t, first.IdentityPackage.AgentDID.X25519PublicKeyJWK, again.IdentityPackage.AgentDID.X25519PublicKeyJWK)
		for id, identity := range first.IdentityPackage.ReasonerDIDs {
			require.Equal(t, identity.DID, again.IdentityPackage.ReasonerDIDs[id].DID)
			require.Equal(t, identity.DerivationPath, again.IdentityPackage.ReasonerDIDs[id].DerivationPath)
		}
		require.Equal(t, first.IdentityPackage.SkillDIDs["s1"].DID, again.IdentityPackage.SkillDIDs["s1"].DID)
		requirePackageConsistent(t, again.IdentityPackage)

		again2 := register(t, svc, "node-2", []string{"r1"}, nil)
		require.Equal(t, second.IdentityPackage.AgentDID.DID, again2.IdentityPackage.AgentDID.DID)
		requirePackageConsistent(t, again2.IdentityPackage)

		servers, err := env.provider.ListAgentFieldServerDIDs(context.Background())
		require.NoError(t, err)
		require.Len(t, servers, 1)
		agents, err := env.provider.ListAgentDIDs(context.Background())
		require.NoError(t, err)
		require.Len(t, agents, 2)
	}
}

func TestIdentityContinuity_LegacyPlaceholderRowsAreRecovered(t *testing.T) {
	ctx := context.Background()

	// Build real identities in a scratch database.
	scratch := newContinuityEnv(t)
	scratch.restart()
	scratchSvc := scratch.service(scratch.registry(), "legacy-server")
	a := register(t, scratchSvc, "node-a", []string{"r1", "r2"}, []string{"s1"})
	b := register(t, scratchSvc, "node-b", []string{"r1"}, []string{"s1", "s2"})
	scratchRegistry, err := scratchSvc.GetRegistry("legacy-server")
	require.NoError(t, err)

	misplacedPath := componentDerivationPath(agentDerivationPath(serverPathSegment("legacy-server"), 1), 0, 5)
	misplacedDID, err := didForPath(scratchRegistry.MasterSeed, misplacedPath)
	require.NoError(t, err)

	// Write them the way earlier versions did: placeholder paths and no
	// component public keys.
	legacy := newContinuityEnv(t)
	legacy.restart()
	now := time.Now()
	require.NoError(t, legacy.provider.StoreAgentFieldServerDID(ctx, "legacy-server", scratchRegistry.RootDID, scratchRegistry.MasterSeed, now, now))
	for _, resp := range []*types.DIDRegistrationResponse{a, b} {
		var components []storage.ComponentDIDRequest
		for name, identity := range resp.IdentityPackage.ReasonerDIDs {
			components = append(components, storage.ComponentDIDRequest{ComponentDID: identity.DID, ComponentType: "reasoner", ComponentName: name, PublicKeyJWK: identity.PublicKeyJWK})
		}
		for name, identity := range resp.IdentityPackage.SkillDIDs {
			components = append(components, storage.ComponentDIDRequest{ComponentDID: identity.DID, ComponentType: "skill", ComponentName: name, PublicKeyJWK: identity.PublicKeyJWK})
		}
		nodeID := "node-a"
		if resp == b {
			nodeID = "node-b"
		}
		if resp == a {
			// Older partial registrations could derive a component under another
			// agent's index. Recovery must still find its real path.
			components = append(components, storage.ComponentDIDRequest{
				ComponentDID:  misplacedDID,
				ComponentType: "reasoner",
				ComponentName: "r-misplaced",
				PublicKeyJWK:  "",
			})
		}
		require.NoError(t, legacy.provider.StoreAgentDIDWithComponents(ctx, nodeID, resp.IdentityPackage.AgentDID.DID, "legacy-server", resp.IdentityPackage.AgentDID.PublicKeyJWK, 0, components))
	}

	legacy.restart()
	svc := legacy.service(legacy.registry(), "legacy-server")
	for nodeID, want := range map[string]*types.DIDRegistrationResponse{"node-a": a, "node-b": b} {
		var reasoners, skills []string
		for id := range want.IdentityPackage.ReasonerDIDs {
			reasoners = append(reasoners, id)
		}
		for id := range want.IdentityPackage.SkillDIDs {
			skills = append(skills, id)
		}
		if nodeID == "node-a" {
			reasoners = append(reasoners, "r-misplaced")
		}
		got := register(t, svc, nodeID, reasoners, skills)
		if nodeID == "node-a" {
			require.Equal(t, misplacedDID, got.IdentityPackage.ReasonerDIDs["r-misplaced"].DID)
			require.Equal(t, misplacedPath, got.IdentityPackage.ReasonerDIDs["r-misplaced"].DerivationPath)
		}
		require.Equal(t, want.IdentityPackage.AgentDID.DID, got.IdentityPackage.AgentDID.DID)
		require.Equal(t, want.IdentityPackage.AgentDID.DerivationPath, got.IdentityPackage.AgentDID.DerivationPath)
		requirePackageConsistent(t, got.IdentityPackage)
	}

	// The recovered paths were written back to storage.
	agents, err := legacy.provider.ListAgentDIDs(ctx)
	require.NoError(t, err)
	for _, agent := range agents {
		want := a
		if agent.AgentNodeID == "node-b" {
			want = b
		}
		require.Equal(t, want.IdentityPackage.AgentDID.DerivationPath, agent.DerivationPath)
		components, err := legacy.provider.ListComponentDIDs(ctx, agent.DID)
		require.NoError(t, err)
		for _, component := range components {
			require.NotEmpty(t, component.PublicKeyJWK)
			require.True(t, identityMatchesPath(scratchRegistry.MasterSeed, component.DerivationPath, component.ComponentDID))
		}
	}

	// node-b owns agent index 1, where node-a's misplaced component also lives.
	// New components of node-b must skip that recovered path.
	var bReasoners, bSkills []string
	for id := range b.IdentityPackage.ReasonerDIDs {
		bReasoners = append(bReasoners, id)
	}
	for id := range b.IdentityPackage.SkillDIDs {
		bSkills = append(bSkills, id)
	}
	for i := 0; i < 6; i++ {
		bReasoners = append(bReasoners, fmt.Sprintf("new-%d", i))
	}
	extended := register(t, svc, "node-b", bReasoners, bSkills)
	seen := map[string]bool{misplacedDID: true}
	for i := 0; i < 6; i++ {
		identity := extended.IdentityPackage.ReasonerDIDs[fmt.Sprintf("new-%d", i)]
		require.NotEqual(t, misplacedPath, identity.DerivationPath)
		require.False(t, seen[identity.DID], "component DID %s was issued twice", identity.DID)
		seen[identity.DID] = true
	}

	legacy.restart()
	svc = legacy.service(legacy.registry(), "legacy-server")
	again := register(t, svc, "node-b", bReasoners, bSkills)
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("new-%d", i)
		require.Equal(t, extended.IdentityPackage.ReasonerDIDs[id].DID, again.IdentityPackage.ReasonerDIDs[id].DID)
	}
	requirePackageConsistent(t, again.IdentityPackage)
}

func TestIdentityContinuity_UnverifiableIdentityFailsClosed(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	register(t, svc, "good-node", []string{"r1"}, nil)

	// A row whose DID was not derived from this server's seed.
	foreignPub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	foreignDID := didKeyFromPublicKey(foreignPub)
	foreignJWK, err := ed25519PublicKeyJWK(foreignPub)
	require.NoError(t, err)
	require.NoError(t, env.provider.StoreAgentDIDWithComponents(ctx, "bad-node", foreignDID, "server-a", foreignJWK, 0, nil))

	env.restart()
	registry := env.registry()
	require.NotEmpty(t, registry.QuarantineReason(foreignDID))
	svc = env.service(registry, "server-a")

	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "bad-node"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "refusing to issue keys")
	require.Empty(t, resp.IdentityPackage.AgentDID.PrivateKeyJWK)

	_, err = svc.ResolveDID(foreignDID)
	require.Error(t, err)

	// Healthy identities on the same server keep working.
	good := register(t, svc, "good-node", []string{"r1"}, nil)
	requirePackageConsistent(t, good.IdentityPackage)
}

func TestIdentityContinuity_WrongMasterSeedRefusesToStart(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	register(t, svc, "node-1", nil, nil)
	root, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	wrongSeed := make([]byte, 32)
	_, err = rand.Read(wrongSeed)
	require.NoError(t, err)
	now := time.Now()
	require.NoError(t, env.provider.StoreAgentFieldServerDID(ctx, "server-a", root, wrongSeed, now, now))

	env.restart()
	registry := env.registry()
	require.Error(t, registry.IntegrityError("server-a"))

	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: env.keyDir, Type: "local"})
	require.NoError(t, err)
	err = NewDIDService(&config.DIDConfig{Enabled: true}, ks, registry).Initialize("server-a")
	require.Error(t, err)
	require.Contains(t, err.Error(), "integrity check")
}

func TestSelectAgentFieldServerID(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()

	id, err := SelectAgentFieldServerID(env.registry(), "", "home-hash")
	require.NoError(t, err)
	require.Equal(t, "home-hash", id, "no registries: the deterministic home-path ID names the new one")

	id, err = SelectAgentFieldServerID(env.registry(), "pinned", "home-hash")
	require.NoError(t, err)
	require.Equal(t, "pinned", id, "a pinned ID may name the first registry")

	env.service(env.registry(), "original")

	id, err = SelectAgentFieldServerID(env.registry(), "", "home-hash")
	require.NoError(t, err)
	require.Equal(t, "original", id, "single registry is adopted after a home change")

	id, err = SelectAgentFieldServerID(env.registry(), "original", "home-hash")
	require.NoError(t, err)
	require.Equal(t, "original", id, "a pinned ID that exists is used")

	_, err = SelectAgentFieldServerID(env.registry(), "orignal", "home-hash")
	require.Error(t, err, "a pinned ID with no stored registry must not create a new identity")
	require.Contains(t, err.Error(), "original")

	env.service(env.registry(), "second")

	id, err = SelectAgentFieldServerID(env.registry(), "", "second")
	require.NoError(t, err)
	require.Equal(t, "second", id, "registry matching the home path is used")

	_, err = SelectAgentFieldServerID(env.registry(), "", "home-hash")
	require.Error(t, err, "ambiguous choice fails closed")
	require.Contains(t, err.Error(), "AGENTFIELD_DID_SERVER_ID")
}

// Replicas that share a home path and database must converge on one server
// identity on first boot, and later boots must keep it.
func TestIdentityContinuity_NewInstallServerIDIsStable(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()

	homeID := LegacyAgentFieldServerID("/srv/agentfield", nil)

	replicaA := env.registry()
	replicaB := env.registry()
	idA, err := SelectAgentFieldServerID(replicaA, "", homeID)
	require.NoError(t, err)
	idB, err := SelectAgentFieldServerID(replicaB, "", homeID)
	require.NoError(t, err)
	require.Equal(t, homeID, idA)
	require.Equal(t, idA, idB, "replicas booting together pick the same ID")

	svc := env.service(replicaA, idA)
	before := register(t, svc, "node-1", []string{"r1"}, []string{"s1"})
	rootBefore, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	env.restart()
	registry := env.registry()
	secondID, err := SelectAgentFieldServerID(registry, "", homeID)
	require.NoError(t, err)
	require.Equal(t, idA, secondID)

	svc = env.service(registry, secondID)
	rootAfter, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	require.Equal(t, rootBefore, rootAfter)

	after := register(t, svc, "node-1", []string{"r1"}, []string{"s1"})
	require.Equal(t, before.IdentityPackage.AgentDID.DID, after.IdentityPackage.AgentDID.DID)
	require.Equal(t, before.IdentityPackage.ReasonerDIDs["r1"].DID, after.IdentityPackage.ReasonerDIDs["r1"].DID)
	requirePackageConsistent(t, after.IdentityPackage)

	servers, err := env.provider.ListAgentFieldServerDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, servers, 1)
}

func TestIdentityContinuity_HomeChangeKeepsIdentity(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()

	oldHomeID := LegacyAgentFieldServerID("/home/nonroot/.agentfield", nil)
	svc := env.service(env.registry(), oldHomeID)
	before := register(t, svc, "node-1", []string{"r1"}, []string{"s1"})
	rootBefore, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	env.restart()
	registry := env.registry()
	newHomeID := LegacyAgentFieldServerID("/data", nil)
	require.NotEqual(t, oldHomeID, newHomeID)
	selected, err := SelectAgentFieldServerID(registry, "", newHomeID)
	require.NoError(t, err)
	require.Equal(t, oldHomeID, selected)

	svc = env.service(registry, selected)
	rootAfter, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	require.Equal(t, rootBefore, rootAfter)

	after := register(t, svc, "node-1", []string{"r1"}, []string{"s1"})
	require.Equal(t, before.IdentityPackage.AgentDID.DID, after.IdentityPackage.AgentDID.DID)
	require.Equal(t, before.IdentityPackage.ReasonerDIDs["r1"].DID, after.IdentityPackage.ReasonerDIDs["r1"].DID)
	requirePackageConsistent(t, after.IdentityPackage)

	servers, err := env.provider.ListAgentFieldServerDIDs(ctx)
	require.NoError(t, err)
	require.Len(t, servers, 1)
}

// requirePublicOnly asserts an identity resolved from an inactive registry
// carries its public keys but no private key material.
func requirePublicOnly(t *testing.T, identity *types.DIDIdentity) {
	t.Helper()
	require.Empty(t, identity.PrivateKeyJWK, "inactive registry leaked a private key for %s", identity.DID)
	require.Empty(t, identity.X25519PrivateKeyJWK, "inactive registry leaked an X25519 private key for %s", identity.DID)
	want, err := publicKeyJWKFromDIDKey(identity.DID)
	require.NoError(t, err)
	require.JSONEq(t, want, identity.PublicKeyJWK)
	require.NotEmpty(t, identity.DerivationPath)
}

func TestIdentityContinuity_OtherRegistryDIDsStayResolvable(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	fromA := register(t, svcA, "node-a", []string{"r1"}, []string{"s1"})
	rootA, err := svcA.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	vcServiceA := NewVCService(svcA.config, svcA, env.provider)
	require.NoError(t, vcServiceA.Initialize())
	signedByA, err := vcServiceA.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID:  "exec-a",
		WorkflowID:   "workflow-a",
		SessionID:    "session-a",
		CallerDID:    fromA.IdentityPackage.AgentDID.DID,
		AgentNodeDID: fromA.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}, []byte(`{"in":1}`), []byte(`{"out":1}`), "succeeded", nil, 10)
	require.NoError(t, err)
	svcB := env.service(env.registry(), "server-b")
	fromB := register(t, svcB, "node-b", nil, nil)

	env.restart()
	registry := env.registry()
	svcB = env.service(registry, "server-b")

	identity, err := svcB.ResolveDID(fromA.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	requirePublicOnly(t, identity)
	require.NotEmpty(t, identity.X25519PublicKeyJWK)

	reasoner, err := svcB.ResolveDID(fromA.IdentityPackage.ReasonerDIDs["r1"].DID)
	require.NoError(t, err)
	requirePublicOnly(t, reasoner)
	require.Equal(t, "reasoner", reasoner.ComponentType)

	skill, err := svcB.ResolveDID(fromA.IdentityPackage.SkillDIDs["s1"].DID)
	require.NoError(t, err)
	requirePublicOnly(t, skill)
	require.Equal(t, "skill", skill.ComponentType)

	root, err := svcB.ResolveDID(rootA)
	require.NoError(t, err)
	requirePublicOnly(t, root)

	own, err := svcB.ResolveDID(fromB.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	requireKeyMatchesDID(t, own.DID, own.PrivateKeyJWK)
	require.NotEmpty(t, own.X25519PrivateKeyJWK)

	// Credentials issued by the other registry still verify, but the active
	// server cannot sign as one of its identities.
	vcServiceB := NewVCService(svcB.config, svcB, env.provider)
	require.NoError(t, vcServiceB.Initialize())
	result, err := vcServiceB.VerifyVC(signedByA.VCDocument)
	require.NoError(t, err)
	require.True(t, result.Valid, "%s %s", result.Message, result.Error)

	_, err = vcServiceB.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID:  "exec-b",
		WorkflowID:   "workflow-b",
		SessionID:    "session-b",
		CallerDID:    fromA.IdentityPackage.AgentDID.DID,
		AgentNodeDID: fromA.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}, []byte(`{"in":1}`), []byte(`{"out":1}`), "succeeded", nil, 10)
	require.Error(t, err)
	require.Contains(t, err.Error(), "cannot sign")
	require.Contains(t, err.Error(), "inactive registry")

	regA, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.NotContains(t, regA.AgentNodes, "node-b")
	regB, err := registry.GetRegistry("server-b")
	require.NoError(t, err)
	require.NotContains(t, regB.AgentNodes, "node-a")
}

func TestIdentityContinuity_X25519RotationSurvivesRestart(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	resp := register(t, svc, "node-1", nil, nil)
	did := resp.IdentityPackage.AgentDID.DID

	rotatedPub, epoch, err := svc.RotateAgentX25519Key(did)
	require.NoError(t, err)
	require.Equal(t, 1, epoch)

	env.restart()
	svc = env.service(env.registry(), "server-a")
	identity, err := svc.ResolveDID(did)
	require.NoError(t, err)
	require.Equal(t, 1, identity.X25519Epoch)
	require.JSONEq(t, rotatedPub, identity.X25519PublicKeyJWK)
}

func TestIdentityContinuity_ComponentKeysAreNeverReused(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	first := register(t, svc, "node-1", []string{"r1"}, nil)
	agentPath := first.IdentityPackage.AgentDID.DerivationPath

	// Every DID and reasoner-branch index ever issued to node-1.
	issued := map[string]bool{}
	maxReasonerIndex := -1
	record := func(resp *types.DIDRegistrationResponse) {
		for _, identity := range resp.IdentityPackage.ReasonerDIDs {
			issued[identity.DID] = true
			index, ok := componentIndexFromPath(identity.DerivationPath, agentPath, 0)
			require.True(t, ok, "reasoner path %s is not under agent path %s", identity.DerivationPath, agentPath)
			if index > maxReasonerIndex {
				maxReasonerIndex = index
			}
		}
		for _, identity := range resp.IdentityPackage.SkillDIDs {
			issued[identity.DID] = true
		}
	}
	record(first)

	// Two components added in one partial registration get distinct keys.
	added := register(t, svc, "node-1", []string{"r1", "r2", "r3"}, nil)
	require.NotEqual(t, added.IdentityPackage.ReasonerDIDs["r2"].DID, added.IdentityPackage.ReasonerDIDs["r3"].DID)
	record(added)
	retired := added.IdentityPackage.ReasonerDIDs["r3"].DID

	// Remove r3, then add r4: r4 must not inherit r3's key.
	register(t, svc, "node-1", []string{"r1", "r2"}, nil)
	withR4 := register(t, svc, "node-1", []string{"r1", "r2", "r4"}, nil)
	require.NotEqual(t, retired, withR4.IdentityPackage.ReasonerDIDs["r4"].DID)
	record(withR4)

	// Components added after the first registration survive a restart.
	env.restart()
	svc = env.service(env.registry(), "server-a")
	again := register(t, svc, "node-1", []string{"r1", "r2", "r4"}, nil)
	require.Equal(t, withR4.IdentityPackage.ReasonerDIDs["r4"].DID, again.IdentityPackage.ReasonerDIDs["r4"].DID)
	requirePackageConsistent(t, again.IdentityPackage)
	record(again)

	// After a restart the retired r3 is no longer in memory; only its retired
	// row remembers the index. A new component must still get a fresh key.
	withR5 := register(t, svc, "node-1", []string{"r1", "r2", "r4", "r5"}, nil)
	r5 := withR5.IdentityPackage.ReasonerDIDs["r5"]
	require.NotEqual(t, retired, r5.DID)
	require.False(t, issued[r5.DID], "r5 was given a DID that was already issued: %s", r5.DID)
	r5Index, ok := componentIndexFromPath(r5.DerivationPath, agentPath, 0)
	require.True(t, ok)
	require.Greater(t, r5Index, maxReasonerIndex, "r5 reused a reasoner derivation index")
	requireKeyMatchesDID(t, r5.DID, r5.PrivateKeyJWK)

	// A new agent never receives an existing agent's key.
	other := register(t, svc, "node-2", nil, nil)
	require.NotEqual(t, again.IdentityPackage.AgentDID.DID, other.IdentityPackage.AgentDID.DID)
}

func TestIdentityContinuity_RemovedComponentsDoNotResurrect(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	full := register(t, svc, "node-1", []string{"r1", "r2"}, []string{"s1"})
	agentDID := full.IdentityPackage.AgentDID.DID
	r1DID := full.IdentityPackage.ReasonerDIDs["r1"].DID
	r2DID := full.IdentityPackage.ReasonerDIDs["r2"].DID
	s1DID := full.IdentityPackage.SkillDIDs["s1"].DID

	// Re-register with fewer components: r2 and s1 are removed.
	register(t, svc, "node-1", []string{"r1"}, nil)

	env.restart()
	registry := env.registry()
	svc = env.service(registry, "server-a")

	reloaded, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.NotNil(t, reloaded)
	node, ok := reloaded.AgentNodes["node-1"]
	require.True(t, ok)
	require.Contains(t, node.Reasoners, "r1")
	require.NotContains(t, node.Reasoners, "r2", "removed reasoner was reloaded after restart")
	require.NotContains(t, node.Skills, "s1", "removed skill was reloaded after restart")

	_, err = svc.ResolveDID(r2DID)
	require.Error(t, err, "a removed reasoner DID must not resolve to keys")
	_, err = svc.ResolveDID(s1DID)
	require.Error(t, err, "a removed skill DID must not resolve to keys")
	_, err = registry.FindDIDByComponent("server-a", "reasoner", "r2")
	require.Error(t, err)
	resolved, err := svc.ResolveDID(r1DID)
	require.NoError(t, err)
	require.Equal(t, r1DID, resolved.DID)

	active, err := env.provider.ListComponentDIDs(ctx, agentDID)
	require.NoError(t, err)
	require.Len(t, active, 1)
	require.Equal(t, r1DID, active[0].ComponentDID)
	require.Equal(t, storage.ComponentDIDStatusActive, active[0].Status)

	all, err := env.provider.ListAllComponentDIDs(ctx, agentDID)
	require.NoError(t, err)
	statuses := map[string]string{}
	for _, component := range all {
		statuses[component.ComponentDID] = component.Status
	}
	require.Equal(t, map[string]string{
		r1DID: storage.ComponentDIDStatusActive,
		r2DID: storage.ComponentDIDStatusRetired,
		s1DID: storage.ComponentDIDStatusRetired,
	}, statuses)

	// Re-adding a removed name issues a new identity and leaves the old row retired.
	readded := register(t, svc, "node-1", []string{"r1", "r2"}, nil)
	require.NotEqual(t, r2DID, readded.IdentityPackage.ReasonerDIDs["r2"].DID)
	env.restart()
	registry = env.registry()
	reloaded, err = registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, readded.IdentityPackage.ReasonerDIDs["r2"].DID, reloaded.AgentNodes["node-1"].Reasoners["r2"].DID)
}

func TestIdentityContinuity_VCsVerifyAcrossRestart(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	resp := register(t, svc, "node-1", []string{"r1"}, nil)

	generate := func(svc *DIDService, executionID, callerDID string) json.RawMessage {
		vcService := NewVCService(svc.config, svc, env.provider)
		require.NoError(t, vcService.Initialize())
		vc, err := vcService.GenerateExecutionVC(&types.ExecutionContext{
			ExecutionID:  executionID,
			WorkflowID:   "workflow-1",
			SessionID:    "session-1",
			CallerDID:    callerDID,
			AgentNodeDID: resp.IdentityPackage.AgentDID.DID,
			Timestamp:    time.Now(),
		}, []byte(`{"in":1}`), []byte(`{"out":1}`), "succeeded", nil, 10)
		require.NoError(t, err)
		require.NotNil(t, vc)
		return vc.VCDocument
	}
	agentDID := resp.IdentityPackage.AgentDID.DID
	reasonerDID := resp.IdentityPackage.ReasonerDIDs["r1"].DID
	docs := map[string]json.RawMessage{
		"agent-signed before restart":    generate(svc, "exec-agent-before", agentDID),
		"reasoner-signed before restart": generate(svc, "exec-reasoner-before", reasonerDID),
	}

	env.restart()
	svc = env.service(env.registry(), "server-a")
	docs["agent-signed after restart"] = generate(svc, "exec-agent-after", agentDID)
	docs["reasoner-signed after restart"] = generate(svc, "exec-reasoner-after", reasonerDID)

	vcService := NewVCService(svc.config, svc, env.provider)
	require.NoError(t, vcService.Initialize())
	for name, doc := range docs {
		result, err := vcService.VerifyVC(doc)
		require.NoError(t, err)
		require.True(t, result.Valid, "%s: %s %s", name, result.Message, result.Error)
	}
}

func TestVCChainBundleReportsUnresolvedDIDs(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	resp := register(t, svc, "node-1", nil, nil)

	orphanPub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	orphanDID := didKeyFromPublicKey(orphanPub)

	vcService := NewVCService(svc.config, svc, env.provider)
	bundle, unresolved := vcService.collectDIDResolutionBundle([]types.ExecutionVC{{
		IssuerDID:  resp.IdentityPackage.AgentDID.DID,
		VCDocument: json.RawMessage(`{"issuer":"` + resp.IdentityPackage.AgentDID.DID + `"}`),
		CallerDID:  orphanDID,
		TargetDID:  "did:web:example.com:agents:unknown",
	}}, &types.WorkflowVC{IssuerDID: resp.IdentityPackage.AgentDID.DID})

	require.Equal(t, "bundled", bundle[resp.IdentityPackage.AgentDID.DID].ResolvedFrom)
	require.Equal(t, "did_key", bundle[orphanDID].ResolvedFrom)
	wantJWK, err := ed25519PublicKeyJWK(orphanPub)
	require.NoError(t, err)
	require.JSONEq(t, wantJWK, string(bundle[orphanDID].PublicKeyJWK))
	require.Equal(t, []string{"did:web:example.com:agents:unknown"}, unresolved)
}

func TestUpsertAgentDIDNeverMovesRowsBetweenServers(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	resp := register(t, svcA, "node-1", []string{"r1"}, nil)
	env.service(env.registry(), "server-b")

	err := env.provider.UpsertAgentDIDWithComponents(ctx, storage.AgentDIDRecord{
		AgentNodeID:        "node-1",
		DID:                resp.IdentityPackage.AgentDID.DID,
		AgentFieldServerID: "server-b",
		PublicKeyJWK:       resp.IdentityPackage.AgentDID.PublicKeyJWK,
		DerivationPath:     "m/44'/1'/0'",
	})
	require.Error(t, err)

	agents, err := env.provider.ListAgentDIDs(ctx)
	require.NoError(t, err)
	require.Len(t, agents, 1)
	require.Equal(t, "server-a", agents[0].AgentFieldServerID)
	require.Equal(t, resp.IdentityPackage.AgentDID.DerivationPath, agents[0].DerivationPath)
}

func TestSignVCRequiresPrivateKey(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	did := didKeyFromPublicKey(pub)
	publicJWK, err := ed25519PublicKeyJWK(pub)
	require.NoError(t, err)

	vcService := &VCService{}
	identity := &types.DIDIdentity{DID: did, PublicKeyJWK: publicJWK}

	_, err = vcService.signVC(&types.VCDocument{}, identity)
	require.EqualError(t, err, "DID "+did+" cannot sign: no private key available (identity belongs to an inactive registry)")

	_, err = vcService.signWorkflowVC(&types.WorkflowVCDocument{}, identity)
	require.ErrorContains(t, err, "cannot sign")

	_, err = vcService.signCanonical([]byte("{}"), identity)
	require.ErrorContains(t, err, "cannot sign")
}
