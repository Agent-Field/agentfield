package services

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// faultyDIDStorage injects DID storage failures.
type faultyDIDStorage struct {
	storage.StorageProvider

	mu             sync.Mutex
	failListAll    bool
	failUpsertNode string
}

func (f *faultyDIDStorage) ListAllComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	f.mu.Lock()
	fail := f.failListAll
	f.mu.Unlock()
	if fail {
		return nil, errors.New("injected component listing failure")
	}
	return f.StorageProvider.ListAllComponentDIDs(ctx, agentDID)
}

func (f *faultyDIDStorage) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	f.mu.Lock()
	fail := f.failUpsertNode == record.AgentNodeID
	f.mu.Unlock()
	if fail {
		return errors.New("injected upsert failure")
	}
	return f.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

func foreignDIDKey(t *testing.T) (string, string) {
	t.Helper()
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	jwk, err := ed25519PublicKeyJWK(pub)
	require.NoError(t, err)
	return didKeyFromPublicKey(pub), jwk
}

func TestIdentityHardening_ConcurrentRegistrationsGetDistinctIdentities(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")

	const nodes = 20
	responses := make([]*types.DIDRegistrationResponse, nodes)
	var wg sync.WaitGroup
	for i := 0; i < nodes; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{
				AgentNodeID: fmt.Sprintf("node-%02d", i),
				Reasoners:   []types.ReasonerDefinition{{ID: "r1"}, {ID: "r2"}},
				Skills:      []types.SkillDefinition{{ID: "s1"}},
			})
			if err == nil && resp.Success {
				responses[i] = resp
			}
			// Readers run alongside writers.
			_, _ = svc.GetExistingAgentDID("node-00")
			_, _ = svc.ListAllAgentDIDs()
		}(i)
	}
	wg.Wait()

	seen := make(map[string]string)
	record := func(did, owner string) {
		prev, dup := seen[did]
		require.False(t, dup, "DID %s issued to both %s and %s", did, prev, owner)
		seen[did] = owner
	}
	for i, resp := range responses {
		require.NotNil(t, resp, "registration %d failed", i)
		requirePackageConsistent(t, resp.IdentityPackage)
		record(resp.IdentityPackage.AgentDID.DID, fmt.Sprintf("node-%02d", i))
		for id, identity := range resp.IdentityPackage.ReasonerDIDs {
			record(identity.DID, fmt.Sprintf("node-%02d/%s", i, id))
		}
		for id, identity := range resp.IdentityPackage.SkillDIDs {
			record(identity.DID, fmt.Sprintf("node-%02d/%s", i, id))
		}
	}

	env.restart()
	svc = env.service(env.registry(), "server-a")
	for i, want := range responses {
		got := register(t, svc, fmt.Sprintf("node-%02d", i), []string{"r1", "r2"}, []string{"s1"})
		require.Equal(t, want.IdentityPackage.AgentDID.DID, got.IdentityPackage.AgentDID.DID)
		require.Equal(t, want.IdentityPackage.ReasonerDIDs["r2"].DID, got.IdentityPackage.ReasonerDIDs["r2"].DID)
	}
}

// Earlier versions counted agents of every server when picking an agent index,
// so a server's agent can sit far above the number of agents it owns.
func TestIdentityHardening_LegacyAgentIndexCountedAcrossServers(t *testing.T) {
	ctx := context.Background()
	scratch := newContinuityEnv(t)
	scratch.restart()
	scratchSvc := scratch.service(scratch.registry(), "legacy-server")
	scratchRegistry, err := scratchSvc.GetRegistry("legacy-server")
	require.NoError(t, err)

	const agentIndex = 90
	realPath := agentDerivationPath(serverPathSegment("legacy-server"), agentIndex)
	realDID, err := didForPath(scratchRegistry.MasterSeed, realPath)
	require.NoError(t, err)
	realJWK, err := publicKeyJWKFromDIDKey(realDID)
	require.NoError(t, err)

	legacy := newContinuityEnv(t)
	legacy.restart()
	now := time.Now()
	require.NoError(t, legacy.provider.StoreAgentFieldServerDID(ctx, "legacy-server", scratchRegistry.RootDID, scratchRegistry.MasterSeed, now, now))
	otherRoot, _ := foreignDIDKey(t)
	require.NoError(t, legacy.provider.StoreAgentFieldServerDID(ctx, "other-server", otherRoot, []byte("other-seed"), now, now))
	for i := 0; i < agentIndex; i++ {
		did, jwk := foreignDIDKey(t)
		require.NoError(t, legacy.provider.StoreAgentDIDWithComponents(ctx, fmt.Sprintf("other-%d", i), did, "other-server", jwk, i, nil))
	}
	require.NoError(t, legacy.provider.StoreAgentDIDWithComponents(ctx, "node-x", realDID, "legacy-server", realJWK, agentIndex, nil))

	legacy.restart()
	registry := legacy.registry()
	require.Empty(t, registry.QuarantineReason(realDID))
	svc := legacy.service(registry, "legacy-server")
	got := register(t, svc, "node-x", nil, nil)
	require.Equal(t, realDID, got.IdentityPackage.AgentDID.DID)
	require.Equal(t, realPath, got.IdentityPackage.AgentDID.DerivationPath)
	requirePackageConsistent(t, got.IdentityPackage)
}

func TestIdentityHardening_UnverifiableComponentIsRetired(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	before := register(t, svc, "good-node", []string{"r1"}, nil)
	agentDID := before.IdentityPackage.AgentDID.DID

	badDID, _ := foreignDIDKey(t)
	require.NoError(t, env.provider.StoreComponentDID(ctx, "r-bad", badDID, agentDID, "reasoner", "r-bad", 0))

	env.restart()
	registry := env.registry()
	require.NotEmpty(t, registry.QuarantineReason(badDID))
	svc = env.service(registry, "server-a")

	register(t, svc, "good-node", []string{"r1", "r-bad"}, nil)
	after := register(t, svc, "good-node", []string{"r1", "r-bad"}, nil)
	require.Equal(t, agentDID, after.IdentityPackage.AgentDID.DID, "the agent keeps its identity")
	require.Equal(t, before.IdentityPackage.ReasonerDIDs["r1"].DID, after.IdentityPackage.ReasonerDIDs["r1"].DID)
	require.NotEqual(t, badDID, after.IdentityPackage.ReasonerDIDs["r-bad"].DID, "the component gets a fresh identity")
	requirePackageConsistent(t, after.IdentityPackage)

	_, err := svc.ResolveDID(badDID)
	require.Error(t, err)

	active, err := env.provider.ListComponentDIDs(ctx, agentDID)
	require.NoError(t, err)
	for _, component := range active {
		require.NotEqual(t, badDID, component.ComponentDID)
	}
}

func TestIdentityHardening_RevokedAgentGetsNewIdentity(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	register(t, svc, "good-node", nil, nil)

	foreignDID, foreignJWK := foreignDIDKey(t)
	require.NoError(t, env.provider.StoreAgentDIDWithComponents(ctx, "bad-node", foreignDID, "server-a", foreignJWK, 0, nil))

	env.restart()
	svc = env.service(env.registry(), "server-a")
	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "bad-node"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "revoked", "the error tells the operator how to recover")

	// Operator recovery: revoke the unrecoverable identity.
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(ctx, storage.AgentDIDRecord{
		AgentNodeID:        "bad-node",
		DID:                foreignDID,
		AgentFieldServerID: "server-a",
		PublicKeyJWK:       foreignJWK,
		DerivationPath:     "unrecoverable",
		Status:             string(types.AgentDIDStatusRevoked),
	}))

	env.restart()
	svc = env.service(env.registry(), "server-a")
	fresh := register(t, svc, "bad-node", []string{"r1"}, nil)
	require.NotEqual(t, foreignDID, fresh.IdentityPackage.AgentDID.DID)
	requirePackageConsistent(t, fresh.IdentityPackage)

	_, err = svc.ResolveDID(foreignDID)
	require.Error(t, err, "a revoked identity is not resolvable as the agent")

	env.restart()
	svc = env.service(env.registry(), "server-a")
	again := register(t, svc, "bad-node", []string{"r1"}, nil)
	require.Equal(t, fresh.IdentityPackage.AgentDID.DID, again.IdentityPackage.AgentDID.DID)
	require.Equal(t, fresh.IdentityPackage.ReasonerDIDs["r1"].DID, again.IdentityPackage.ReasonerDIDs["r1"].DID)
}

func TestIdentityHardening_ReplicasCreatingOneServerConverge(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()

	registryA := env.registry()
	registryB := env.registry()
	svcA := env.service(registryA, "shared")
	svcB := env.service(registryB, "shared")

	rootA, err := svcA.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	rootB, err := svcB.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	require.Equal(t, rootA, rootB, "the second replica adopts the stored identity instead of overwriting it")

	fromB := register(t, svcB, "node-1", []string{"r1"}, nil)
	requirePackageConsistent(t, fromB.IdentityPackage)

	env.restart()
	svc := env.service(env.registry(), "shared")
	root, err := svc.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	require.Equal(t, rootA, root)
	again := register(t, svc, "node-1", []string{"r1"}, nil)
	require.Equal(t, fromB.IdentityPackage.AgentDID.DID, again.IdentityPackage.AgentDID.DID)
}

func TestIdentityHardening_StorageReadErrorAbortsAllocation(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	faulty := &faultyDIDStorage{StorageProvider: env.provider}
	registry := NewDIDRegistryWithStorage(faulty)
	require.NoError(t, registry.Initialize())
	svc := env.service(registry, "server-a")
	register(t, svc, "node-1", []string{"r1"}, nil)

	faulty.mu.Lock()
	faulty.failListAll = true
	faulty.mu.Unlock()

	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node-2"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "failed to read issued DIDs")

	resp, err = svc.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "node-1",
		Reasoners:   []types.ReasonerDefinition{{ID: "r1"}, {ID: "r2"}},
	})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "failed to read issued DIDs")
}

func TestIdentityHardening_FailedSaveLeavesRegistryUnchanged(t *testing.T) {
	ctx := context.Background()
	env := newContinuityEnv(t)
	env.restart()
	faulty := &faultyDIDStorage{StorageProvider: env.provider, failUpsertNode: "node-bad"}
	registry := NewDIDRegistryWithStorage(faulty)
	require.NoError(t, registry.Initialize())
	svc := env.service(registry, "server-a")

	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node-bad"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	_, err = svc.GetExistingAgentDID("node-bad")
	require.Error(t, err, "a failed save must not leave the agent in memory")

	first := register(t, svc, "node-a", []string{"r1"}, nil)
	register(t, svc, "node-z", nil, nil)

	// A partial registration whose save fails is rolled back too.
	faulty.mu.Lock()
	faulty.failUpsertNode = "node-a"
	faulty.mu.Unlock()
	resp, err = svc.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "node-a",
		Reasoners:   []types.ReasonerDefinition{{ID: "r1"}, {ID: "r2"}},
	})
	require.NoError(t, err)
	require.False(t, resp.Success)
	existing, err := svc.GetExistingAgentDID("node-a")
	require.NoError(t, err)
	require.NotContains(t, existing.Reasoners, "r2")

	// Saving every agent continues past one agent's failure.
	snapshot, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	nodes := make(map[string]types.AgentDIDInfo, len(snapshot.AgentNodes))
	for nodeID, agent := range snapshot.AgentNodes {
		agent.Status = types.AgentDIDStatusInactive
		nodes[nodeID] = agent
	}
	snapshot.AgentNodes = nodes
	err = registry.StoreRegistry(snapshot)
	require.Error(t, err)
	require.Contains(t, err.Error(), "node-a")

	stored, err := env.provider.ListAgentDIDs(ctx)
	require.NoError(t, err)
	statuses := make(map[string]types.AgentDIDStatus)
	for _, agent := range stored {
		statuses[agent.AgentNodeID] = agent.Status
	}
	require.Equal(t, types.AgentDIDStatusInactive, statuses["node-z"], "agents after the failing one are still written")
	require.Equal(t, types.AgentDIDStatusActive, statuses["node-a"])

	current, err := svc.GetExistingAgentDID("node-a")
	require.NoError(t, err)
	require.Equal(t, first.IdentityPackage.AgentDID.DID, current.DID)
}

type c5RetireFailure struct {
	storage.StorageProvider
	failures int
}

func (s *c5RetireFailure) RetireComponentDIDs(ctx context.Context, agent string, dids []string) error {
	if s.failures != 0 {
		s.failures--
		return errors.New("injected retirement write failure")
	}
	return s.StorageProvider.RetireComponentDIDs(ctx, agent, dids)
}

func TestC5F5DIDQuarantineRetirementRetry(t *testing.T) {
	for _, persistent := range []bool{false, true} {
		t.Run(fmt.Sprintf("persistent=%v", persistent), func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			svc := env.service(env.registry(), "server-a")
			agent := register(t, svc, "node", nil, nil).IdentityPackage.AgentDID
			foreign, foreignJWK := foreignDIDKey(t)
			require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
				AgentNodeID: "node", DID: agent.DID, AgentFieldServerID: "server-a", PublicKeyJWK: agent.PublicKeyJWK, DerivationPath: agent.DerivationPath,
				Components: []storage.ComponentDIDRequest{{ComponentDID: foreign, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: foreignJWK, DerivationPath: "legacy"}},
			}))
			faulty := &c5RetireFailure{StorageProvider: env.provider, failures: 1}
			if persistent {
				faulty.failures = -1
			}
			reg := NewDIDRegistryWithStorage(faulty)
			require.NoError(t, reg.Initialize())
			svc = env.service(reg, "server-a")
			identity, err := svc.ResolveDID(foreign)
			require.Error(t, err)
			require.Nil(t, identity)
			resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
			require.NoError(t, err)
			if persistent {
				require.False(t, resp.Success)
				require.Empty(t, resp.IdentityPackage)
				require.Contains(t, resp.Error, foreign)
				require.Contains(t, resp.Error, "UPDATE component_dids SET status = 'retired' WHERE did = '"+foreign+"';")
				require.Contains(t, resp.Error, "injected retirement write failure")
			} else {
				require.True(t, resp.Success, resp.Error)
				require.NotEqual(t, foreign, resp.IdentityPackage.ReasonerDIDs["r"].DID)
				rows, err := env.provider.ListAllComponentDIDs(context.Background(), agent.DID)
				require.NoError(t, err)
				for _, row := range rows {
					if row.ComponentDID == foreign {
						require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
					}
				}
			}
			identity, err = svc.ResolveDID(foreign)
			require.Error(t, err)
			require.Nil(t, identity)
		})
	}
}

func TestC7F7DIDLegacyRecoveryDerivationCount(t *testing.T) {
	reg := &types.DIDRegistry{AgentFieldServerID: "server", MasterSeed: []byte("synthetic"), AgentNodes: map[string]types.AgentDIDInfo{}}
	for i := 0; i < 100; i++ {
		agent := types.AgentDIDInfo{Reasoners: map[string]types.ReasonerDIDInfo{}}
		count := 1
		if i == 0 {
			count = 100
		}
		for j := 0; j < count; j++ {
			agent.Reasoners[fmt.Sprint(j)] = types.ReasonerDIDInfo{}
		}
		reg.AgentNodes[fmt.Sprint(i)] = agent
	}
	ix := newLegacyPathIndex(reg, 100)
	for i := 0; i < 100; i++ {
		agentPath := agentDerivationPath(ix.serverSegment, i)
		did, err := didForPath(reg.MasterSeed, agentPath)
		require.NoError(t, err)
		path, ok := ix.agentPath(did)
		require.True(t, ok)
		require.Equal(t, agentPath, path)
		count := 1
		if i == 0 {
			count = 100
		}
		for j := 0; j < count; j++ {
			want := componentDerivationPath(agentPath, 0, j)
			did, err := didForPath(reg.MasterSeed, want)
			require.NoError(t, err)
			path, ok := ix.componentPath(did, i, true)
			require.True(t, ok)
			require.Equal(t, want, path)
		}
	}
	require.Less(t, ix.derivations, 600, "previous eager component search alone derived 41,800 candidates")
	t.Logf("100 agents and 199 components recovered with %d candidate derivations", ix.derivations)
	// Repeated lookups only use cached candidates.
	before := ix.derivations
	want := componentDerivationPath(agentDerivationPath(ix.serverSegment, 0), 0, 99)
	did, err := didForPath(reg.MasterSeed, want)
	require.NoError(t, err)
	path, ok := ix.componentPath(did, 0, true)
	require.True(t, ok)
	require.Equal(t, want, path)
	require.Equal(t, before, ix.derivations)
}

func TestC7F7DIDLegacyRecoveryPreservesSearchSpace(t *testing.T) {
	reg := &types.DIDRegistry{AgentFieldServerID: "server", MasterSeed: []byte("synthetic"), AgentNodes: map[string]types.AgentDIDInfo{}}
	ix := newLegacyPathIndex(reg, 1)
	want := componentDerivationPath(agentDerivationPath(ix.serverSegment, ix.maxAgentIndex), 1, ix.maxComponentIx)
	did, err := didForPath(reg.MasterSeed, want)
	require.NoError(t, err)
	path, ok := ix.componentPath(did, 0, true)
	require.True(t, ok)
	require.Equal(t, want, path)
	missing, _ := foreignDIDKey(t)
	_, ok = ix.componentPath(missing, 0, true)
	require.False(t, ok)
}

func TestC5F5DIDUnrecoverableAgentDoesNotRetireChildren(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	reg, err := svc.GetRegistry("server-a")
	require.NoError(t, err)
	foreign, publicKey := foreignDIDKey(t)
	path := componentDerivationPath(agentDerivationPath(serverPathSegment("server-a"), 4), 0, 0)
	component, _, componentPublicKey, err := svc.generateDIDWithKeys(reg.MasterSeed, path)
	require.NoError(t, err)
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "bad-node", DID: foreign, AgentFieldServerID: "server-a", PublicKeyJWK: publicKey, DerivationPath: "unrecoverable",
		Components: []storage.ComponentDIDRequest{{ComponentDID: component, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: componentPublicKey, DerivationPath: path}},
	}))
	spy := &readOnlyDIDSpy{StorageProvider: env.provider}
	reloaded := NewDIDRegistryWithStorage(spy)
	require.NoError(t, reloaded.Initialize())
	svc = env.service(reloaded, "server-a")
	spy.writes.Store(0)
	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "bad-node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "revoked")
	require.Zero(t, spy.writes.Load())
	rows, err := env.provider.ListComponentDIDs(context.Background(), foreign)
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, component, rows[0].ComponentDID)
}
