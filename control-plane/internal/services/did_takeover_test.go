package services

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib"
	_ "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/require"
)

func distinctDIDRows(t *testing.T, provider storage.StorageProvider, agentsWant, componentsWant int) {
	t.Helper()
	agents, err := provider.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, agents, agentsWant)
	components, err := provider.ListAllComponentDIDs(context.Background(), "")
	require.NoError(t, err)
	require.Len(t, components, componentsWant)
	dids, paths := map[string]bool{}, map[string]bool{}
	add := func(did, path string) {
		require.False(t, dids[did], "duplicate DID %s", did)
		require.False(t, paths[path], "duplicate path %s", path)
		dids[did], paths[path] = true, true
	}
	for _, agent := range agents {
		add(agent.DID, agent.DerivationPath)
	}
	for _, component := range components {
		add(component.ComponentDID, component.DerivationPath)
	}
}

func runCrossInstanceRegistrations(t *testing.T, first, second storage.StorageProvider, sharedRegistry bool, priorAgents, priorComponents int) []*types.DIDRegistrationResponse {
	t.Helper()
	keyDir := filepath.Join(t.TempDir(), "keys")
	svcA, regA := openReproService(t, first, "server-a", keyDir)
	var svcB *DIDService
	if sharedRegistry {
		// C1.1: two services sharing one registry is also a supported shape.
		svcB = NewDIDService(svcA.config, svcA.keystore, regA)
		require.NoError(t, svcB.Initialize("server-a"))
	} else {
		// C1.1: replicas have separate service and registry state.
		svcB, _ = openReproService(t, second, "server-a", keyDir)
	}

	const count = 40
	responses := make([]*types.DIDRegistrationResponse, count)
	failures := make([]error, count)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range responses {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			svc := svcA
			if i%2 != 0 {
				svc = svcB
			}
			responses[i], failures[i] = svc.RegisterAgent(&types.DIDRegistrationRequest{
				AgentNodeID: fmt.Sprintf("node-%02d", i),
				Reasoners:   []types.ReasonerDefinition{{ID: "reasoner"}},
				Skills:      []types.SkillDefinition{{ID: "skill"}},
			})
		}(i)
	}
	close(start)
	wg.Wait()
	seenDIDs, seenPaths := map[string]bool{}, map[string]bool{}
	for i, resp := range responses {
		require.NoError(t, failures[i], "node %d", i)
		require.NotNil(t, resp, "node %d", i)
		require.True(t, resp.Success, "node %d: %s", i, resp.Error)
		// C1.2: responses have unique agent, reasoner, and skill identities.
		requirePackageConsistent(t, resp.IdentityPackage)
		for _, identity := range []types.DIDIdentity{resp.IdentityPackage.AgentDID, resp.IdentityPackage.ReasonerDIDs["reasoner"], resp.IdentityPackage.SkillDIDs["skill"]} {
			require.False(t, seenDIDs[identity.DID], "duplicate response DID %s", identity.DID)
			require.False(t, seenPaths[identity.DerivationPath], "duplicate response path %s", identity.DerivationPath)
			seenDIDs[identity.DID], seenPaths[identity.DerivationPath] = true, true
		}
	}
	// C1.2: count independently from durable rows, not just responses.
	distinctDIDRows(t, first, count+priorAgents, count*2+priorComponents)
	// C1.3: a fresh registry loads every identity and re-derives its key.
	fresh, freshRegistry := openReproService(t, first, "server-a", keyDir)
	reg, err := freshRegistry.GetRegistry("server-a")
	require.NoError(t, err)
	require.Len(t, reg.AgentNodes, count+priorAgents)
	for _, resp := range responses {
		for _, identity := range []types.DIDIdentity{resp.IdentityPackage.AgentDID, resp.IdentityPackage.ReasonerDIDs["reasoner"], resp.IdentityPackage.SkillDIDs["skill"]} {
			require.Empty(t, freshRegistry.QuarantineReason(identity.DID))
			loaded, err := fresh.ResolveDID(identity.DID)
			require.NoError(t, err)
			require.Equal(t, identity.DerivationPath, loaded.DerivationPath)
			requireKeyMatchesDID(t, loaded.DID, loaded.PrivateKeyJWK)
		}
	}
	return responses
}

// C1.1-C1.3 and C1.6: independent SQLite registries share one LocalStorage,
// because opening its BoltDB file twice takes an exclusive lock.
func TestDIDTakeover_SQLiteCrossInstanceAllocation(t *testing.T) {
	for _, shared := range []bool{false, true} {
		name := "separate-registries"
		if shared {
			name = "shared-registry"
		}
		t.Run(name, func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			runCrossInstanceRegistrations(t, env.provider, env.provider, shared, 0, 0)
		})
	}
}

func postgresTakeoverStores(t *testing.T) (storage.StorageProvider, storage.StorageProvider, string) {
	t.Helper()
	raw := os.Getenv("POSTGRES_TEST_URL")
	if raw == "" {
		t.Skip("POSTGRES_TEST_URL is unset")
	}
	u, err := url.Parse(raw)
	require.NoError(t, err)
	admin, err := sql.Open("pgx", raw)
	require.NoError(t, err)
	defer admin.Close()
	name := fmt.Sprintf("fix2_did_takeover_%d", time.Now().UnixNano())
	_, err = admin.Exec("CREATE DATABASE " + name)
	require.NoError(t, err)
	t.Cleanup(func() {
		db, openErr := sql.Open("pgx", raw)
		if openErr == nil {
			defer db.Close()
			_, _ = db.Exec("DROP DATABASE " + name + " WITH (FORCE)")
		}
	})
	u.Path = "/" + name
	vectorOff := false
	cfg := storage.StorageConfig{Mode: "postgres", Postgres: storage.PostgresStorageConfig{DSN: u.String()}, Vector: storage.VectorStoreConfig{Enabled: &vectorOff}}
	open := func() storage.StorageProvider {
		provider := storage.NewPostgresStorage(storage.PostgresStorageConfig{})
		require.NoError(t, provider.Initialize(context.Background(), cfg))
		t.Cleanup(func() { require.NoError(t, provider.Close(context.Background())) })
		return provider
	}
	return open(), open(), u.String()
}

// C1.1-C1.3: two Postgres connection pools share a database but not memory.
func TestDIDTakeover_PostgresCrossInstanceAllocation(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	runCrossInstanceRegistrations(t, first, second, false, 0, 0)
}

type collisionStorage struct {
	storage.StorageProvider
	attempts atomic.Int32
	failure  error
}

func (s *collisionStorage) UpsertAgentDIDWithComponents(context.Context, storage.AgentDIDRecord) error {
	s.attempts.Add(1)
	return s.failure
}

// C1.4-C1.5: only allocation conflicts retry, and attempts are bounded.
func TestDIDTakeover_AllocationRetriesAreBounded(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int32
	}{
		{"collision", fmt.Errorf("injected: %w", storage.ErrDIDAllocationConflict), maxDIDAllocationAttempts},
		{"closed-store", errors.New("database is closed"), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			wrapped := &collisionStorage{StorageProvider: env.provider, failure: tc.err}
			svc, registry := openReproService(t, wrapped, "server-a", env.keyDir)
			result := make(chan *types.DIDRegistrationResponse, 1)
			go func() {
				resp, _ := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node"})
				result <- resp
			}()
			select {
			case resp := <-result:
				require.NotNil(t, resp)
				require.False(t, resp.Success)
				if tc.want > 1 {
					require.Contains(t, resp.Error, "allocation kept colliding")
				} else {
					require.Contains(t, resp.Error, "database is closed")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("registration deadlocked or exceeded its retry bound")
			}
			require.Equal(t, tc.want, wrapped.attempts.Load())
			reg, err := registry.GetRegistry("server-a")
			require.NoError(t, err)
			require.Empty(t, reg.AgentNodes)
		})
	}
}

// C1.6: retry support does not change the ordinary allocation sequence.
func TestDIDTakeover_SingleInstancePaths(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	for i := 0; i < 2; i++ {
		resp := register(t, svc, fmt.Sprintf("node-%d", i), []string{"r"}, []string{"s"})
		require.Equal(t, agentDerivationPath(serverPathSegment("server-a"), i), resp.IdentityPackage.AgentDID.DerivationPath)
	}
}

// C1.7: independent registries add components under the same existing agent.
func runCrossInstancePartialAllocation(t *testing.T, first, second storage.StorageProvider) {
	t.Helper()
	keyDir := filepath.Join(t.TempDir(), "keys")
	svcA, _ := openReproService(t, first, "server-a", keyDir)
	initial := register(t, svcA, "node", nil, nil)
	svcB, _ := openReproService(t, second, "server-a", keyDir)
	const count = 40
	responses := make([]*types.DIDRegistrationResponse, count)
	failures := make([]error, count)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			svc := svcA
			if i%2 != 0 {
				svc = svcB
			}
			id := fmt.Sprintf("r-%02d", i)
			sk := fmt.Sprintf("s-%02d", i)
			responses[i], failures[i] = svc.PartialRegisterAgent(&types.PartialDIDRegistrationRequest{
				AgentNodeID: "node", NewReasonerIDs: []string{id}, NewSkillIDs: []string{sk},
				AllReasoners: []types.ReasonerDefinition{{ID: id}}, AllSkills: []types.SkillDefinition{{ID: sk}},
			})
		}(i)
	}
	close(start)
	wg.Wait()
	seen := map[string]bool{initial.IdentityPackage.AgentDID.DID: true}
	for i, resp := range responses {
		require.NoError(t, failures[i])
		require.NotNil(t, resp)
		require.True(t, resp.Success, "request %d: %s", i, resp.Error)
		for _, identities := range []map[string]types.DIDIdentity{resp.IdentityPackage.ReasonerDIDs, resp.IdentityPackage.SkillDIDs} {
			for _, identity := range identities {
				require.False(t, seen[identity.DID], "duplicate component DID")
				seen[identity.DID] = true
			}
		}
	}
	components, err := first.ListAllComponentDIDs(context.Background(), initial.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, components, count*2)
	fresh := NewDIDRegistryWithStorage(first)
	require.NoError(t, fresh.Initialize())
	reg, err := fresh.GetRegistry("server-a")
	require.NoError(t, err)
	require.Len(t, reg.AgentNodes["node"].Reasoners, count)
	require.Len(t, reg.AgentNodes["node"].Skills, count)
}

func TestDIDTakeover_SQLiteCrossInstancePartialAllocation(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	runCrossInstancePartialAllocation(t, env.provider, env.provider)
}

// C1.7: partial registration also retries component collisions on Postgres.
func TestDIDTakeover_PostgresCrossInstancePartialAllocation(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	runCrossInstancePartialAllocation(t, first, second)
}

// C2.1-C2.4: surviving agents block a new seed for unchanged, moved, and pinned homes.
func TestDIDTakeover_MissingRegistryRefusesAndRecovers(t *testing.T) {
	for _, selected := range []string{"server-a", "moved-home", "pinned-id"} {
		t.Run(selected, func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			old := env.service(env.registry(), "server-a")
			before := register(t, old, "node-old", []string{"r"}, nil)
			require.NoError(t, env.provider.Close(context.Background()))
			env.provider = nil
			db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
			require.NoError(t, err)
			_, err = db.Exec("DELETE FROM did_registry WHERE agentfield_server_id = ?", "server-a")
			require.NoError(t, err)
			require.NoError(t, db.Close())
			env.restart()
			registry := env.registry()
			// Both startup doors initialize the DID service through this registry path.
			configuredID := ""
			if selected == "pinned-id" {
				configuredID = selected
			}
			serverID, err := SelectAgentFieldServerID(registry, configuredID, selected)
			require.NoError(t, err)
			require.Equal(t, selected, serverID)
			keystore, err := NewKeystoreService(&config.KeystoreConfig{Path: env.keyDir, Type: "local"})
			require.NoError(t, err)
			startup := NewDIDService(&config.DIDConfig{Enabled: true}, keystore, registry)
			initErr := startup.Initialize(serverID)
			require.Error(t, initErr)
			message := initErr.Error()
			for _, required := range []string{"registry is missing", "server-a", "1 orphaned agents", "node-old", "restore", "master seed", "UPDATE agent_dids SET status = 'revoked' WHERE agentfield_server_id = 'server-a';"} {
				require.Contains(t, strings.ToLower(message), strings.ToLower(required))
			}
			servers, err := env.provider.ListAgentFieldServerDIDs(context.Background())
			require.NoError(t, err)
			require.Empty(t, servers)
			agents, err := env.provider.ListAgentDIDs(context.Background())
			require.NoError(t, err)
			require.Len(t, agents, 1)
			require.Equal(t, before.IdentityPackage.AgentDID.DID, agents[0].DID)
			require.Equal(t, types.AgentDIDStatusActive, agents[0].Status)

			// C2.4: the documented SQL explicitly acknowledges lost key material.
			require.NoError(t, env.provider.Close(context.Background()))
			env.provider = nil
			db, err = sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
			require.NoError(t, err)
			_, err = db.Exec("UPDATE agent_dids SET status = 'revoked' WHERE agentfield_server_id = 'server-a';")
			require.NoError(t, err)
			require.NoError(t, db.Close())
			env.restart()
			svc := env.service(env.registry(), selected)
			after := register(t, svc, "node-old", []string{"r"}, nil)
			require.NotEqual(t, before.IdentityPackage.AgentDID.DID, after.IdentityPackage.AgentDID.DID)
			requirePackageConsistent(t, after.IdentityPackage)
			env.restart()
			restarted := env.service(env.registry(), selected)
			require.Equal(t, after.IdentityPackage.AgentDID.DID, register(t, restarted, "node-old", []string{"r"}, nil).IdentityPackage.AgentDID.DID)
		})
	}
}

// C2.1: report every orphaned server and cap the displayed node list.
func TestDIDTakeover_MissingRegistryListsServersAndCapsNodes(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	for i := 0; i < 12; i++ {
		register(t, svcA, fmt.Sprintf("node-%02d", i), nil, nil)
	}
	svcB := env.service(env.registry(), "server-b")
	register(t, svcB, "other-node", nil, nil)
	require.NoError(t, env.provider.Close(context.Background()))
	env.provider = nil
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	_, err = db.Exec("DELETE FROM did_registry")
	require.NoError(t, err)
	require.NoError(t, db.Close())
	env.restart()
	registry := env.registry()
	keystore, err := NewKeystoreService(&config.KeystoreConfig{Path: env.keyDir, Type: "local"})
	require.NoError(t, err)
	err = NewDIDService(&config.DIDConfig{Enabled: true}, keystore, registry).Initialize("new-server")
	require.Error(t, err)
	for _, part := range []string{"server-a", "server-b", "12 orphaned agents", "1 orphaned agents", "node-00", "node-09", "and 2 more", "other-node"} {
		require.Contains(t, err.Error(), part)
	}
	require.NotContains(t, err.Error(), "node-10")
}

// C2.3-C2.5: an empty database starts, while another server's orphans do not
// block a server whose own registry is still present.
func TestDIDTakeover_OtherServerOrphansDoNotBlock(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	register(t, svcA, "node-a", nil, nil)
	env.service(env.registry(), "server-b")
	require.NoError(t, env.provider.Close(context.Background()))
	env.provider = nil
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	_, err = db.Exec("DELETE FROM did_registry WHERE agentfield_server_id = ?", "server-a")
	require.NoError(t, err)
	require.NoError(t, db.Close())
	env.restart()
	svcB := env.service(env.registry(), "server-b")
	register(t, svcB, "node-b", nil, nil)
}

// C2.1, C2.4, and C2.7: the common creation path refuses orphaned identities
// on Postgres as well, and the documented revoke restores startup.
func TestDIDTakeover_PostgresMissingRegistryRefusesAndRecovers(t *testing.T) {
	first, second, dsn := postgresTakeoverStores(t)
	keyDir := filepath.Join(t.TempDir(), "keys")
	svc, _ := openReproService(t, first, "server-a", keyDir)
	before := register(t, svc, "node", []string{"r"}, nil)
	db, err := sql.Open("pgx", dsn)
	require.NoError(t, err)
	defer db.Close()
	// Simulate a damaged backup that retained agent rows but lost the registry.
	rows, err := db.Query(`SELECT conname FROM pg_constraint WHERE conrelid = 'agent_dids'::regclass AND confrelid = 'did_registry'::regclass`)
	require.NoError(t, err)
	var constraints []string
	for rows.Next() {
		var name string
		require.NoError(t, rows.Scan(&name))
		constraints = append(constraints, name)
	}
	require.NoError(t, rows.Err())
	require.NoError(t, rows.Close())
	for _, name := range constraints {
		_, err = db.Exec(`ALTER TABLE agent_dids DROP CONSTRAINT "` + strings.ReplaceAll(name, `"`, `""`) + `"`)
		require.NoError(t, err)
	}
	_, err = db.Exec("DELETE FROM did_registry WHERE agentfield_server_id = $1", "server-a")
	require.NoError(t, err)
	registry := NewDIDRegistryWithStorage(second)
	require.NoError(t, registry.Initialize())
	keystore, err := NewKeystoreService(&config.KeystoreConfig{Path: keyDir, Type: "local"})
	require.NoError(t, err)
	newService := func(r *DIDRegistry) *DIDService { return NewDIDService(svc.config, keystore, r) }
	err = newService(registry).Initialize("server-a")
	require.ErrorContains(t, err, "registry is missing")
	require.ErrorContains(t, err, "node")
	servers, err := second.ListAgentFieldServerDIDs(context.Background())
	require.NoError(t, err)
	require.Empty(t, servers)
	_, err = db.Exec("UPDATE agent_dids SET status = 'revoked' WHERE agentfield_server_id = 'server-a';")
	require.NoError(t, err)
	registry = NewDIDRegistryWithStorage(second)
	require.NoError(t, registry.Initialize())
	recovered := newService(registry)
	require.NoError(t, recovered.Initialize("server-a"))
	after := register(t, recovered, "node", []string{"r"}, nil)
	require.NotEqual(t, before.IdentityPackage.AgentDID.DID, after.IdentityPackage.AgentDID.DID)
	registry = NewDIDRegistryWithStorage(second)
	require.NoError(t, registry.Initialize())
	restarted := newService(registry)
	require.NoError(t, restarted.Initialize("server-a"))
	require.Equal(t, after.IdentityPackage.AgentDID.DID, register(t, restarted, "node", []string{"r"}, nil).IdentityPackage.AgentDID.DID)
}

// C1.2-C1.3 upgrade path: main stored placeholder paths for agent and
// component rows. Recovery must precede allocation and preserve old VCs.
func TestDIDTakeover_UpgradeThenCrossInstanceAllocation(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	legacy := register(t, svc, "legacy-node", []string{"legacy-r"}, []string{"legacy-s"})
	legacyAgent := legacy.IdentityPackage.AgentDID
	legacyReasoner := legacy.IdentityPackage.ReasonerDIDs["legacy-r"]
	legacySkill := legacy.IdentityPackage.SkillDIDs["legacy-s"]
	vcService := NewVCService(svc.config, svc, env.provider)
	require.NoError(t, vcService.Initialize())
	oldVC, err := vcService.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID: "legacy-execution", WorkflowID: "legacy-workflow", SessionID: "legacy-session",
		CallerDID: legacyReasoner.DID, AgentNodeDID: legacyAgent.DID, Timestamp: time.Now(),
	}, []byte(`{"in":1}`), []byte(`{"out":1}`), "succeeded", nil, 10)
	require.NoError(t, err)

	const placeholder = "m/44'/0'/0'/0"
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "legacy-node", DID: legacyAgent.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: legacyAgent.PublicKeyJWK, DerivationPath: placeholder,
		Components: []storage.ComponentDIDRequest{
			{ComponentDID: legacyReasoner.DID, ComponentType: "reasoner", ComponentName: "legacy-r", PublicKeyJWK: legacyReasoner.PublicKeyJWK, DerivationPath: placeholder},
			{ComponentDID: legacySkill.DID, ComponentType: "skill", ComponentName: "legacy-s", PublicKeyJWK: legacySkill.PublicKeyJWK, DerivationPath: placeholder},
		},
	}))
	env.restart()
	recovered := env.registry()
	loaded, err := recovered.GetRegistry("server-a")
	require.NoError(t, err)
	agent := loaded.AgentNodes["legacy-node"]
	require.Equal(t, legacyAgent.DerivationPath, agent.DerivationPath)
	require.Equal(t, legacyReasoner.DerivationPath, agent.Reasoners["legacy-r"].DerivationPath)
	require.Equal(t, legacySkill.DerivationPath, agent.Skills["legacy-s"].DerivationPath)
	for _, did := range []string{legacyAgent.DID, legacyReasoner.DID, legacySkill.DID} {
		require.Empty(t, recovered.QuarantineReason(did))
	}
	recoveredService := env.service(recovered, "server-a")
	vcService = NewVCService(recoveredService.config, recoveredService, env.provider)
	require.NoError(t, vcService.Initialize())
	verification, err := vcService.VerifyVC(oldVC.VCDocument)
	require.NoError(t, err)
	require.True(t, verification.Valid, "%s %s", verification.Message, verification.Error)

	responses := runCrossInstanceRegistrations(t, env.provider, env.provider, false, 1, 2)
	oldDIDs := map[string]bool{legacyAgent.DID: true, legacyReasoner.DID: true, legacySkill.DID: true}
	oldPaths := map[string]bool{legacyAgent.DerivationPath: true, legacyReasoner.DerivationPath: true, legacySkill.DerivationPath: true}
	for _, resp := range responses {
		for _, identity := range []types.DIDIdentity{resp.IdentityPackage.AgentDID, resp.IdentityPackage.ReasonerDIDs["reasoner"], resp.IdentityPackage.SkillDIDs["skill"]} {
			require.False(t, oldDIDs[identity.DID], "new DID reuses a legacy DID")
			require.False(t, oldPaths[identity.DerivationPath], "new path reuses a recovered legacy path")
		}
	}
}

// C1.2-C1.3 upgrade path on Postgres: recovered legacy paths remain reserved
// while independent replicas register new nodes.
func TestDIDTakeover_PostgresUpgradeThenCrossInstanceAllocation(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	keyDir := filepath.Join(t.TempDir(), "keys")
	svc, _ := openReproService(t, first, "server-a", keyDir)
	svc.config.VCRequirements = config.VCRequirements{RequireVCForExecution: true, PersistExecutionVC: true, HashSensitiveData: true}
	legacy := register(t, svc, "legacy-node", []string{"r"}, []string{"s"})
	agent := legacy.IdentityPackage.AgentDID
	reasoner := legacy.IdentityPackage.ReasonerDIDs["r"]
	skill := legacy.IdentityPackage.SkillDIDs["s"]
	vcService := NewVCService(svc.config, svc, first)
	require.NoError(t, vcService.Initialize())
	oldVC, err := vcService.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID: "legacy-execution", WorkflowID: "legacy-workflow", SessionID: "legacy-session",
		CallerDID: reasoner.DID, AgentNodeDID: agent.DID, Timestamp: time.Now(),
	}, []byte(`{"in":1}`), []byte(`{"out":1}`), "succeeded", nil, 10)
	require.NoError(t, err)
	require.NotNil(t, oldVC)
	const placeholder = "m/44'/0'/0'/0"
	require.NoError(t, first.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "legacy-node", DID: agent.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: agent.PublicKeyJWK, DerivationPath: placeholder,
		Components: []storage.ComponentDIDRequest{
			{ComponentDID: reasoner.DID, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: reasoner.PublicKeyJWK, DerivationPath: placeholder},
			{ComponentDID: skill.DID, ComponentType: "skill", ComponentName: "s", PublicKeyJWK: skill.PublicKeyJWK, DerivationPath: placeholder},
		},
	}))
	fresh, registry := openReproService(t, second, "server-a", keyDir)
	loaded, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, agent.DerivationPath, loaded.AgentNodes["legacy-node"].DerivationPath)
	require.Equal(t, reasoner.DerivationPath, loaded.AgentNodes["legacy-node"].Reasoners["r"].DerivationPath)
	require.Equal(t, skill.DerivationPath, loaded.AgentNodes["legacy-node"].Skills["s"].DerivationPath)
	for _, did := range []string{agent.DID, reasoner.DID, skill.DID} {
		require.Empty(t, registry.QuarantineReason(did))
	}
	vcService = NewVCService(fresh.config, fresh, second)
	require.NoError(t, vcService.Initialize())
	verification, err := vcService.VerifyVC(oldVC.VCDocument)
	require.NoError(t, err)
	require.True(t, verification.Valid, "%s %s", verification.Message, verification.Error)
	responses := runCrossInstanceRegistrations(t, first, second, false, 1, 2)
	oldDIDs := map[string]bool{agent.DID: true, reasoner.DID: true, skill.DID: true}
	oldPaths := map[string]bool{agent.DerivationPath: true, reasoner.DerivationPath: true, skill.DerivationPath: true}
	for _, resp := range responses {
		for _, identity := range []types.DIDIdentity{resp.IdentityPackage.AgentDID, resp.IdentityPackage.ReasonerDIDs["reasoner"], resp.IdentityPackage.SkillDIDs["skill"]} {
			require.False(t, oldDIDs[identity.DID])
			require.False(t, oldPaths[identity.DerivationPath])
		}
	}
}

// S1: a replica that missed another replica's registration must return the
// stored identity for the same node instead of minting a second one.
func TestDIDTakeover_ReplicaReadThroughPreservesNodeIdentity(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	svcB := env.service(env.registry(), "server-a")
	first := register(t, svcA, "node", []string{"r"}, []string{"s"})
	second := register(t, svcB, "node", []string{"r"}, []string{"s"})
	require.Equal(t, first.IdentityPackage.AgentDID.DID, second.IdentityPackage.AgentDID.DID)
	require.Equal(t, first.IdentityPackage.ReasonerDIDs["r"].DID, second.IdentityPackage.ReasonerDIDs["r"].DID)
	agents, err := env.provider.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, agents, 1)
}

// C3.1 and C3.2: a replica refreshes one node without writing any identity.
type readOnlyDIDSpy struct {
	storage.StorageProvider
	writes atomic.Int32
}

func (s *readOnlyDIDSpy) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	s.writes.Add(1)
	return s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

func (s *readOnlyDIDSpy) RetireComponentDIDs(ctx context.Context, did string, ids []string) error {
	s.writes.Add(1)
	return s.StorageProvider.RetireComponentDIDs(ctx, did, ids)
}

func TestDIDTakeover_ReadThroughIsTargetedAndReadOnly(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	x := register(t, svcA, "node-x", []string{"r"}, nil)
	spy := &readOnlyDIDSpy{StorageProvider: env.provider}
	registryB := NewDIDRegistryWithStorage(spy)
	require.NoError(t, registryB.Initialize())
	svcB := env.service(registryB, "server-a")
	y := register(t, svcA, "node-y", nil, nil)
	// C3.1: B holds X in memory while A revokes X in durable storage.
	require.NoError(t, svcA.registry.UpdateAgentStatus("server-a", "node-x", types.AgentDIDStatusRevoked))
	got, err := svcB.GetExistingAgentDID("node-y")
	require.NoError(t, err)
	require.Equal(t, y.IdentityPackage.AgentDID.DID, got.DID)
	require.EqualValues(t, 0, spy.writes.Load())
	rows, err := env.provider.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	for _, row := range rows {
		if row.DID == x.IdentityPackage.AgentDID.DID {
			require.Equal(t, types.AgentDIDStatusRevoked, row.Status)
		}
	}
	loaded, err := registryB.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, len(loaded.AgentNodes), loaded.TotalDIDs)
}

// C3.2: a replica must merge a component issued by another replica before diffing.
func TestDIDTakeover_ReplicaMergesStoredComponent(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	first := register(t, svcA, "node", nil, nil)
	svcB := env.service(env.registry(), "server-a")
	issued := register(t, svcA, "node", []string{"r"}, nil)
	again := register(t, svcB, "node", []string{"r"}, nil)
	require.Equal(t, first.IdentityPackage.AgentDID.DID, again.IdentityPackage.AgentDID.DID)
	require.Equal(t, issued.IdentityPackage.ReasonerDIDs["r"].DID, again.IdentityPackage.ReasonerDIDs["r"].DID)
	rows, err := env.provider.ListComponentDIDs(context.Background(), first.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 1)
}

// C3.3: database serialization admits one active row when two replicas issue
// the same new node or component at the same instant.
func runSameIdentityRaces(t *testing.T, first, second storage.StorageProvider) {
	t.Helper()
	keyDir := filepath.Join(t.TempDir(), "keys")
	svcA, _ := openReproService(t, first, "server-a", keyDir)
	svcB, _ := openReproService(t, second, "server-a", keyDir)
	pair := func(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, *types.DIDRegistrationResponse) {
		t.Helper()
		start := make(chan struct{})
		var wg sync.WaitGroup
		var responses [2]*types.DIDRegistrationResponse
		var errs [2]error
		for i, svc := range []*DIDService{svcA, svcB} {
			wg.Add(1)
			go func(i int, svc *DIDService) {
				defer wg.Done()
				<-start
				responses[i], errs[i] = svc.RegisterAgent(req)
			}(i, svc)
		}
		close(start)
		wg.Wait()
		for i := range responses {
			require.NoError(t, errs[i])
			require.NotNil(t, responses[i])
			require.True(t, responses[i].Success, responses[i].Error)
		}
		return responses[0], responses[1]
	}
	for i := 0; i < 20; i++ {
		node := fmt.Sprintf("same-node-%02d", i)
		a, b := pair(&types.DIDRegistrationRequest{AgentNodeID: node})
		require.Equal(t, a.IdentityPackage.AgentDID.DID, b.IdentityPackage.AgentDID.DID)
		rows, err := first.ListAgentDIDs(context.Background())
		require.NoError(t, err)
		active := 0
		for _, row := range rows {
			if row.AgentNodeID == node && row.AgentFieldServerID == "server-a" && row.Status != types.AgentDIDStatusRevoked {
				active++
			}
		}
		require.Equal(t, 1, active)
	}
	register(t, svcA, "component-node", nil, nil)
	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("reasoner-%02d", i)
		reasoners := make([]types.ReasonerDefinition, i+1)
		for j := range reasoners {
			reasoners[j] = types.ReasonerDefinition{ID: fmt.Sprintf("reasoner-%02d", j)}
		}
		a, b := pair(&types.DIDRegistrationRequest{AgentNodeID: "component-node", Reasoners: reasoners})
		require.Equal(t, a.IdentityPackage.ReasonerDIDs[name].DID, b.IdentityPackage.ReasonerDIDs[name].DID)
		rows, err := first.ListComponentDIDs(context.Background(), a.IdentityPackage.AgentDID.DID)
		require.NoError(t, err)
		active := 0
		for _, row := range rows {
			if row.ComponentType == "reasoner" && row.ComponentName == name {
				active++
			}
		}
		require.Equal(t, 1, active)
		// G4: when the other replica already issued this requested component,
		// direct partial registration returns only that component and no agent key.
		partial, err := svcB.PartialRegisterAgent(&types.PartialDIDRegistrationRequest{
			AgentNodeID: "component-node", NewReasonerIDs: []string{name}, AllReasoners: reasoners,
		})
		require.NoError(t, err)
		require.True(t, partial.Success, partial.Error)
		require.Empty(t, partial.IdentityPackage.AgentDID.PrivateKeyJWK)
		require.Equal(t, a.IdentityPackage.AgentDID.DID, partial.IdentityPackage.AgentDID.DID)
		require.Equal(t, map[string]types.DIDIdentity{name: partial.IdentityPackage.ReasonerDIDs[name]}, partial.IdentityPackage.ReasonerDIDs)
		require.Equal(t, a.IdentityPackage.ReasonerDIDs[name].DID, partial.IdentityPackage.ReasonerDIDs[name].DID)
		require.Empty(t, partial.IdentityPackage.SkillDIDs)
	}
}

// G2.a-G2.c: a stale replica adding a component after the other replica
// revoked, retired and rotated cannot undo any of those durable decisions.
func runStaleReplicaSavePreservesIdentityState(t *testing.T, first, second storage.StorageProvider) {
	t.Helper()
	keyDir := filepath.Join(t.TempDir(), "keys")
	svcA, registryA := openReproService(t, first, "server-a", keyDir)
	initial := register(t, svcA, "node", []string{"old"}, nil)
	svcB, _ := openReproService(t, second, "server-a", keyDir)
	agentDID := initial.IdentityPackage.AgentDID.DID
	oldDID := initial.IdentityPackage.ReasonerDIDs["old"].DID
	_, epoch, err := svcA.RotateAgentX25519Key(agentDID)
	require.NoError(t, err)
	require.Equal(t, 1, epoch)
	require.NoError(t, registryA.UpdateAgentStatus("server-a", "node", types.AgentDIDStatusRevoked))
	require.NoError(t, first.RetireComponentDIDs(context.Background(), agentDID, []string{oldDID}))
	added := register(t, svcB, "node", []string{"old", "new"}, nil)
	require.Equal(t, agentDID, added.IdentityPackage.AgentDID.DID)
	rows, err := first.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, types.AgentDIDStatusRevoked, rows[0].Status)
	require.Equal(t, 1, rows[0].X25519Epoch)
	components, err := first.ListAllComponentDIDs(context.Background(), agentDID)
	require.NoError(t, err)
	statuses := make(map[string]string, len(components))
	for _, component := range components {
		statuses[component.ComponentDID] = component.Status
	}
	require.Equal(t, "retired", statuses[oldDID])
	require.Equal(t, "active", statuses[added.IdentityPackage.ReasonerDIDs["new"].DID])
}

func TestDIDTakeover_SQLiteStaleReplicaSavePreservesIdentityState(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	runStaleReplicaSavePreservesIdentityState(t, env.provider, env.provider)
}

func TestDIDTakeover_PostgresStaleReplicaSavePreservesIdentityState(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	runStaleReplicaSavePreservesIdentityState(t, first, second)
}

// G5: read-through counts an imported agent and each added component without
// replacing the running total with the number of agent nodes.
func TestDIDTakeover_ReadThroughAddsToTotalDIDs(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	svcB, registryB := openReproService(t, env.provider, "server-a", env.keyDir)
	first := register(t, svcA, "node", []string{"r1"}, nil)
	got, err := svcB.GetExistingAgentDID("node")
	require.NoError(t, err)
	require.Equal(t, first.IdentityPackage.AgentDID.DID, got.DID)
	loaded, err := registryB.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, 2, loaded.TotalDIDs)
	register(t, svcA, "node", []string{"r1", "r2"}, nil)
	_, err = svcB.GetExistingAgentDID("node")
	require.NoError(t, err)
	loaded, err = registryB.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, 3, loaded.TotalDIDs)
}

func TestDIDTakeover_SQLiteSameIdentityRaces(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	runSameIdentityRaces(t, env.provider, env.provider)
}

func TestDIDTakeover_PostgresSameIdentityRaces(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	runSameIdentityRaces(t, first, second)
}

// C3.1: reading Y must not repair or otherwise update X after another replica
// rotated X's epoch, even if X has an old placeholder path in storage.
func TestDIDTakeover_ReadThroughDoesNotTouchOtherNode(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	x := register(t, svcA, "node-x", nil, nil)
	spy := &readOnlyDIDSpy{StorageProvider: env.provider}
	registryB := NewDIDRegistryWithStorage(spy)
	require.NoError(t, registryB.Initialize())
	svcB := env.service(registryB, "server-a")
	y := register(t, svcA, "node-y", nil, nil)
	const placeholder = "m/44'/0'/0'/0"
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "node-x", DID: x.IdentityPackage.AgentDID.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: x.IdentityPackage.AgentDID.PublicKeyJWK, DerivationPath: placeholder, X25519Epoch: 2,
	}))
	got, err := svcB.GetExistingAgentDID("node-y")
	require.NoError(t, err)
	require.Equal(t, y.IdentityPackage.AgentDID.DID, got.DID)
	require.EqualValues(t, 0, spy.writes.Load())
	stored, err := env.provider.GetActiveAgentDID(context.Background(), "server-a", "node-x")
	require.NoError(t, err)
	require.Equal(t, placeholder, stored.DerivationPath)
	require.EqualValues(t, 2, stored.X25519Epoch)
}

type countedReadThroughStorage struct {
	storage.StorageProvider
	agentReads     atomic.Int32
	componentReads atomic.Int32
}

func (s *countedReadThroughStorage) GetActiveAgentDID(ctx context.Context, serverID, nodeID string) (*types.AgentDIDInfo, error) {
	s.agentReads.Add(1)
	return s.StorageProvider.GetActiveAgentDID(ctx, serverID, nodeID)
}

func (s *countedReadThroughStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	s.componentReads.Add(1)
	return s.StorageProvider.ListComponentDIDs(ctx, agentDID)
}

// D1.1: an in-memory path recovered at startup remains usable even if the
// attempted repair write failed and storage still holds legacy placeholders.
func TestDIDTakeover_FailedPathRepairKeepsMemoryIdentity(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	first := register(t, env.service(env.registry(), "server-a"), "node", []string{"r"}, []string{"s"})
	agent := first.IdentityPackage.AgentDID
	reasoner := first.IdentityPackage.ReasonerDIDs["r"]
	skill := first.IdentityPackage.SkillDIDs["s"]
	const placeholder = "m/44'/0'/0'/0"
	writePlaceholders := func() {
		t.Helper()
		require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
			AgentNodeID: "node", DID: agent.DID, AgentFieldServerID: "server-a",
			PublicKeyJWK: agent.PublicKeyJWK, DerivationPath: placeholder,
			Components: []storage.ComponentDIDRequest{
				{ComponentDID: reasoner.DID, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: reasoner.PublicKeyJWK, DerivationPath: placeholder},
				{ComponentDID: skill.DID, ComponentType: "skill", ComponentName: "s", PublicKeyJWK: skill.PublicKeyJWK, DerivationPath: placeholder},
			},
		}))
	}
	writePlaceholders()
	recovered := env.registry()
	before, err := recovered.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, agent.DerivationPath, before.AgentNodes["node"].DerivationPath)
	writePlaceholders() // Simulate the failed startup repair write.
	again := register(t, env.service(recovered, "server-a"), "node", []string{"r"}, []string{"s"})
	require.Equal(t, agent.DID, again.IdentityPackage.AgentDID.DID)
	require.Equal(t, reasoner.DID, again.IdentityPackage.ReasonerDIDs["r"].DID)
	after, err := recovered.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, before.AgentNodes["node"], after.AgentNodes["node"])
	require.Equal(t, len(after.AgentNodes), after.TotalDIDs)
	require.Empty(t, recovered.QuarantineReason(agent.DID))
	require.Empty(t, recovered.QuarantineReason(reasoner.DID))
	require.Empty(t, recovered.QuarantineReason(skill.DID))
}

// D1.1: read-through leaves all metadata and paths held in memory untouched.
func TestDIDTakeover_ReadThroughPreservesMemoryMetadata(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	registry := env.registry()
	svc := env.service(registry, "server-a")
	register(t, svc, "node", []string{"r"}, []string{"s"})
	require.NoError(t, registry.UpdateAgents("server-a", func(reg *types.DIDRegistry) ([]string, error) {
		agent := cloneAgentInfo(reg.AgentNodes["node"])
		reasoner := agent.Reasoners["r"]
		reasoner.Capabilities = []string{"review"}
		reasoner.ExposureLevel = "public"
		agent.Reasoners["r"] = reasoner
		skill := agent.Skills["s"]
		skill.Tags = []string{"trusted"}
		skill.ExposureLevel = "public"
		agent.Skills["s"] = skill
		reg.AgentNodes["node"] = agent
		return []string{"node"}, nil
	}))
	before, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	register(t, svc, "node", []string{"r"}, []string{"s"})
	after, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, before.AgentNodes["node"], after.AgentNodes["node"])
	require.Equal(t, []string{"review"}, after.AgentNodes["node"].Reasoners["r"].Capabilities)
	require.Equal(t, []string{"trusted"}, after.AgentNodes["node"].Skills["s"].Tags)
}

// D1.1: an operator's revoke in storage takes effect after the documented
// restart; registration cannot delete or re-key an identity already in memory.
func TestDIDTakeover_StorageRevokeDoesNotReplaceMemoryIdentity(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	registry := env.registry()
	svc := env.service(registry, "server-a")
	first := register(t, svc, "node", nil, nil)
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	defer db.Close()
	_, err = db.Exec("UPDATE agent_dids SET status = 'revoked' WHERE did = ?", first.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	again := register(t, svc, "node", nil, nil)
	require.Equal(t, first.IdentityPackage.AgentDID.DID, again.IdentityPackage.AgentDID.DID)
	loaded, err := registry.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, first.IdentityPackage.AgentDID.DID, loaded.AgentNodes["node"].DID)
	require.Equal(t, types.AgentDIDStatusActive, loaded.AgentNodes["node"].Status)
	rows, err := env.provider.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, types.AgentDIDStatusRevoked, rows[0].Status)
}

// D1.2-D1.3: a node absent from memory imports only verifiable storage rows,
// and an invalid agent fails on the first read with operator recovery steps.
func TestDIDTakeover_ReadThroughRefusesUnverifiedAgent(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	spy := &countedReadThroughStorage{StorageProvider: env.provider}
	svcB, registryB := openReproService(t, spy, "server-a", env.keyDir)
	svcA := env.service(env.registry(), "server-a")
	x := register(t, svcA, "node-x", nil, nil)
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "node-x", DID: x.IdentityPackage.AgentDID.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: x.IdentityPackage.AgentDID.PublicKeyJWK, DerivationPath: "m/invalid",
	}))
	spy.agentReads.Store(0)
	resp, err := svcB.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node-x"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "node-x (agent) was written without a verifiable derivation path")
	require.Contains(t, resp.Error, "restart this control plane")
	require.Contains(t, resp.Error, "docs/ENVIRONMENT_VARIABLES.md")
	require.NotContains(t, resp.Error, "allocation kept colliding")
	require.EqualValues(t, 1, spy.agentReads.Load())
	require.Empty(t, registryB.QuarantineReason(x.IdentityPackage.AgentDID.DID))
	stored, err := env.provider.GetActiveAgentDID(context.Background(), "server-a", "node-x")
	require.NoError(t, err)
	require.Equal(t, "m/invalid", stored.DerivationPath)
}

// D1.3: a component added by another replica under an old placeholder path
// fails on the first read instead of exhausting the allocation retry budget.
func TestDIDTakeover_ReadThroughRefusesUnverifiedComponent(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svcA := env.service(env.registry(), "server-a")
	first := register(t, svcA, "node", nil, nil)
	spy := &countedReadThroughStorage{StorageProvider: env.provider}
	svcB, registryB := openReproService(t, spy, "server-a", env.keyDir)
	issued := register(t, svcA, "node", []string{"r"}, nil)
	reasoner := issued.IdentityPackage.ReasonerDIDs["r"]
	require.NoError(t, env.provider.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "node", DID: first.IdentityPackage.AgentDID.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: first.IdentityPackage.AgentDID.PublicKeyJWK, DerivationPath: first.IdentityPackage.AgentDID.DerivationPath,
		Components: []storage.ComponentDIDRequest{{ComponentDID: reasoner.DID, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: reasoner.PublicKeyJWK, DerivationPath: "m/44'/0'/0'/0"}},
	}))
	spy.componentReads.Store(0)
	resp, err := svcB.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "node (reasoner r) was written without a verifiable derivation path")
	require.Contains(t, resp.Error, "retire the component")
	require.NotContains(t, resp.Error, "allocation kept colliding")
	require.EqualValues(t, 1, spy.componentReads.Load())
	loaded, err := registryB.GetRegistry("server-a")
	require.NoError(t, err)
	require.Empty(t, loaded.AgentNodes["node"].Reasoners)
	require.Empty(t, registryB.QuarantineReason(reasoner.DID))
}

// D1.3: the same first-read refusal applies across independent Postgres pools.
func TestDIDTakeover_PostgresReadThroughRefusesUnverifiedComponent(t *testing.T) {
	first, second, _ := postgresTakeoverStores(t)
	keyDir := filepath.Join(t.TempDir(), "keys")
	svcA, _ := openReproService(t, first, "server-a", keyDir)
	agent := register(t, svcA, "node", nil, nil).IdentityPackage.AgentDID
	spy := &countedReadThroughStorage{StorageProvider: second}
	svcB, registryB := openReproService(t, spy, "server-a", keyDir)
	reasoner := register(t, svcA, "node", []string{"r"}, nil).IdentityPackage.ReasonerDIDs["r"]
	require.NoError(t, first.UpsertAgentDIDWithComponents(context.Background(), storage.AgentDIDRecord{
		AgentNodeID: "node", DID: agent.DID, AgentFieldServerID: "server-a",
		PublicKeyJWK: agent.PublicKeyJWK, DerivationPath: agent.DerivationPath,
		Components: []storage.ComponentDIDRequest{{ComponentDID: reasoner.DID, ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: reasoner.PublicKeyJWK, DerivationPath: "m/44'/0'/0'/0"}},
	}))
	spy.componentReads.Store(0)
	resp, err := svcB.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "node (reasoner r) was written without a verifiable derivation path")
	require.Contains(t, resp.Error, "restart this control plane")
	require.EqualValues(t, 1, spy.componentReads.Load())
	loaded, err := registryB.GetRegistry("server-a")
	require.NoError(t, err)
	require.Empty(t, loaded.AgentNodes["node"].Reasoners)
}

type transientOnceStorage struct {
	storage.StorageProvider
	failure  error
	attempts atomic.Int32
}

func (s *transientOnceStorage) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	if s.attempts.Add(1) == 1 {
		return s.failure
	}
	return s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

// C3.4: a wrapped transient error from BeginTx, in-transaction work, or Commit
// retries registration from a fresh allocation; unrelated failures do not.
func TestDIDTakeover_TransientTransactionErrorsRetryWholeRegistration(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"postgres-serialization", &pgconn.PgError{Code: "40001"}},
		{"postgres-deadlock", &pgconn.PgError{Code: "40P01"}},
		{"sqlite-begin", errors.New("failed to begin transaction: database is locked")},
		{"sqlite-inside", errors.New("failed to check active DID: database table is locked")},
		{"sqlite-commit", errors.New("failed to commit transaction: SQLITE_BUSY")},
		{"sqlite-locked-code", errors.New("failed to commit transaction: SQLITE_LOCKED")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			wrapped := &transientOnceStorage{StorageProvider: env.provider, failure: tc.err}
			svc, _ := openReproService(t, wrapped, "server-a", env.keyDir)
			resp := register(t, svc, "node", nil, nil)
			require.EqualValues(t, 2, wrapped.attempts.Load())
			rows, err := env.provider.ListAgentDIDs(context.Background())
			require.NoError(t, err)
			require.Len(t, rows, 1)
			require.Equal(t, resp.IdentityPackage.AgentDID.DID, rows[0].DID)
		})
	}
}

// C2/F2: a stale replica must allocate from the committed epoch, on real storage.
func runC2DurableX25519Rotation(t *testing.T, first, second storage.StorageProvider) {
	t.Helper()
	keyDir := filepath.Join(t.TempDir(), "keys")
	a, _ := openReproService(t, first, "server-a", keyDir)
	did := register(t, a, "node", nil, nil).IdentityPackage.AgentDID.DID
	b, _ := openReproService(t, second, "server-a", keyDir)
	pub1, epoch1, err := a.RotateAgentX25519Key(did)
	require.NoError(t, err)
	require.Equal(t, 1, epoch1)
	pub2, epoch2, err := b.RotateAgentX25519Key(did)
	require.NoError(t, err)
	require.Equal(t, 2, epoch2)
	require.NotEqual(t, pub1, pub2)
	rows, err := first.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, 2, rows[0].X25519Epoch)
	fresh, _ := openReproService(t, first, "server-a", keyDir)
	identity, err := fresh.ResolveDID(did)
	require.NoError(t, err)
	require.Equal(t, 2, identity.X25519Epoch)
	require.Equal(t, pub2, identity.X25519PublicKeyJWK)
	// The original instance is now stale too; it must advance beyond 2.
	_, epoch3, err := a.RotateAgentX25519Key(did)
	require.NoError(t, err)
	require.Equal(t, 3, epoch3)
}

func TestC2F2X25519DurableRotationSQLite(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	// Independent real SQLite pools, sharing one file.
	second := storage.NewLocalStorage(storage.LocalStorageConfig{})
	vectorOff := false
	require.NoError(t, second.Initialize(context.Background(), storage.StorageConfig{
		Mode: "local", Local: storage.LocalStorageConfig{DatabasePath: filepath.Join(env.stateDir, "agentfield.db"), KVStorePath: filepath.Join(t.TempDir(), "second.bolt")},
		Vector: storage.VectorStoreConfig{Enabled: &vectorOff},
	}))
	t.Cleanup(func() { require.NoError(t, second.Close(context.Background())) })
	runC2DurableX25519Rotation(t, env.provider, second)
}

func TestC2F2X25519DurableRotationPostgres(t *testing.T) {
	a, b, _ := postgresTakeoverStores(t)
	runC2DurableX25519Rotation(t, a, b)
}

type c3AllocationBarrier struct {
	storage.StorageProvider
	arrived sync.WaitGroup
	calls   atomic.Int32
}

func (s *c3AllocationBarrier) ListAllComponentDIDs(ctx context.Context, did string) ([]*types.ComponentDIDInfo, error) {
	rows, err := s.StorageProvider.ListAllComponentDIDs(ctx, did)
	if s.calls.Add(1) <= 2 {
		s.arrived.Done()
		s.arrived.Wait()
	}
	return rows, err
}

func runC3SharedRegistryRace(t *testing.T, provider storage.StorageProvider) {
	t.Helper()
	for round := 0; round < 20; round++ {
		wrapped := &c3AllocationBarrier{StorageProvider: provider}
		wrapped.arrived.Add(2)
		a, reg := openReproService(t, wrapped, "server-a", filepath.Join(t.TempDir(), "keys"))
		b := NewDIDService(a.config, a.keystore, reg)
		require.NoError(t, b.Initialize("server-a"))
		var wg sync.WaitGroup
		responses := make([]*types.DIDRegistrationResponse, 2)
		errs := make([]error, 2)
		node := fmt.Sprintf("shared-%d", round)
		for i, svc := range []*DIDService{a, b} {
			wg.Add(1)
			go func(i int, svc *DIDService) {
				defer wg.Done()
				responses[i], errs[i] = svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: node, Reasoners: []types.ReasonerDefinition{{ID: "r"}}})
			}(i, svc)
		}
		wg.Wait()
		for i, resp := range responses {
			require.NoError(t, errs[i])
			require.NotNil(t, resp)
			require.True(t, resp.Success, resp.Error)
		}
		require.Equal(t, responses[0].IdentityPackage.AgentDID.DID, responses[1].IdentityPackage.AgentDID.DID)
		require.Equal(t, responses[0].IdentityPackage.ReasonerDIDs["r"].DID, responses[1].IdentityPackage.ReasonerDIDs["r"].DID)
	}
	rows, err := provider.ListAgentDIDs(context.Background())
	require.NoError(t, err)
	require.Len(t, rows, 20)
	for _, row := range rows {
		require.Equal(t, types.AgentDIDStatusActive, row.Status)
	}
	components, err := provider.ListComponentDIDs(context.Background(), "")
	require.NoError(t, err)
	require.Len(t, components, 20)
}

func TestC3F3DIDSharedRegistrySameNodeSQLite(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	runC3SharedRegistryRace(t, env.provider)
}

func TestC3F3DIDSharedRegistrySameNodePostgres(t *testing.T) {
	a, _, _ := postgresTakeoverStores(t)
	runC3SharedRegistryRace(t, a)
}

func TestC4F4DIDSQLiteContentionBudget(t *testing.T) {
	t.Setenv("AGENTFIELD_SQLITE_BUSY_TIMEOUT_MS", "5")
	env := newContinuityEnv(t)
	env.restart()
	// Count attempts at the first durable write in normal registration.
	spy := &c4ServerWriteSpy{StorageProvider: env.provider}
	svc, _ := openReproService(t, spy, "server-a", env.keyDir)
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	defer db.Close()
	tx, err := db.Begin()
	require.NoError(t, err)
	defer tx.Rollback()
	_, err = tx.Exec("UPDATE did_registry SET total_dids=total_dids+1")
	require.NoError(t, err)
	spy.writes.Store(0)
	start := time.Now()
	resp, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "blocked"})
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "database is locked")
	require.EqualValues(t, maxDIDContentionRetries+1, spy.writes.Load())
	require.Less(t, time.Since(start), 2*time.Second)
	require.NoError(t, tx.Rollback())
	register(t, svc, "blocked", nil, nil)
}

func TestC6F6DIDDuplicateComponentsRetireTogether(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	defer db.Close()
	runC6DuplicateComponents(t, env.provider, db, false)
}

func TestC6F6DIDDuplicateComponentsRetireTogetherPostgres(t *testing.T) {
	first, _, dsn := postgresTakeoverStores(t)
	db, err := sql.Open("pgx", dsn)
	require.NoError(t, err)
	defer db.Close()
	runC6DuplicateComponents(t, first, db, true)
}

func runC6DuplicateComponents(t *testing.T, provider storage.StorageProvider, db *sql.DB, postgres bool) {
	t.Helper()
	svc, _ := openReproService(t, provider, "server-a", filepath.Join(t.TempDir(), "keys"))
	first := register(t, svc, "node", []string{"r"}, nil)
	reg, err := svc.GetRegistry("server-a")
	require.NoError(t, err)
	path := componentDerivationPath(first.IdentityPackage.AgentDID.DerivationPath, 0, 1)
	did, _, pub, err := svc.generateDIDWithKeys(reg.MasterSeed, path)
	require.NoError(t, err)
	// A legal main-schema row: only DID, not component name, was unique.
	query := `INSERT INTO component_dids(did,agent_did,component_type,function_name,public_key_jwk,derivation_path,status,created_at) VALUES(?,?,?,?,?,?,'active',?)`
	if postgres {
		query = `INSERT INTO component_dids(did,agent_did,component_type,function_name,public_key_jwk,derivation_path,status,created_at) VALUES($1,$2,$3,$4,$5,$6,'active',$7)`
	}
	_, err = db.Exec(query, did, first.IdentityPackage.AgentDID.DID, "reasoner", "r", pub, path, time.Now().Add(time.Second))
	require.NoError(t, err)
	reg2 := NewDIDRegistryWithStorage(provider)
	require.NoError(t, reg2.Initialize())
	svc = NewDIDService(svc.config, svc.keystore, reg2)
	require.NoError(t, svc.Initialize("server-a"))
	resp, err := svc.DeregisterComponents(&types.ComponentDeregistrationRequest{AgentNodeID: "node", ReasonerIDsToRemove: []string{"r"}})
	require.NoError(t, err)
	require.True(t, resp.Success, resp.Error)
	agent, err := svc.GetExistingAgentDID("node")
	require.NoError(t, err)
	require.NotContains(t, agent.Reasoners, "r")
	for _, old := range []string{did, first.IdentityPackage.ReasonerDIDs["r"].DID} {
		identity, err := svc.ResolveDID(old)
		require.Error(t, err)
		require.Nil(t, identity)
	}
	rows, err := provider.ListAllComponentDIDs(context.Background(), first.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 2)
	for _, row := range rows {
		require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
	}
	again := register(t, svc, "node", []string{"r"}, nil)
	require.NotEqual(t, did, again.IdentityPackage.ReasonerDIDs["r"].DID)
	require.NotEqual(t, first.IdentityPackage.ReasonerDIDs["r"].DID, again.IdentityPackage.ReasonerDIDs["r"].DID)
}

type c4ServerWriteSpy struct {
	storage.StorageProvider
	writes atomic.Int32
}

func (s *c4ServerWriteSpy) StoreAgentFieldServerDID(ctx context.Context, serverID, rootDID string, seed []byte, created, rotated time.Time) error {
	s.writes.Add(1)
	return s.StorageProvider.StoreAgentFieldServerDID(ctx, serverID, rootDID, seed, created, rotated)
}

func TestC5F5DIDQuarantineRetiresSnapshotDuplicates(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	first := register(t, svc, "node", []string{"r"}, nil)
	foreign, publicKey := foreignDIDKey(t)
	db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
	require.NoError(t, err)
	defer db.Close()
	// Give both rows explicit timestamps: older schema defaults can leave the
	// original row NULL, which would otherwise sort after our old duplicate.
	_, err = db.Exec("UPDATE component_dids SET created_at = ? WHERE did = ?", time.Now(), first.IdentityPackage.ReasonerDIDs["r"].DID)
	require.NoError(t, err)
	// The oldest duplicate is the unrecoverable row chosen during startup.
	_, err = db.Exec(`INSERT INTO component_dids(did,agent_did,component_type,function_name,public_key_jwk,derivation_path,status,created_at) VALUES(?,?,?,?,?,?,'active',?)`, foreign, first.IdentityPackage.AgentDID.DID, "reasoner", "r", publicKey, "legacy", time.Now().Add(-time.Hour))
	require.NoError(t, err)
	faulty := &c5RetireFailure{StorageProvider: env.provider, failures: 1}
	reg := NewDIDRegistryWithStorage(faulty)
	require.NoError(t, reg.Initialize())
	require.NotEmpty(t, reg.QuarantineReason(foreign))
	svc = env.service(reg, "server-a")
	again := register(t, svc, "node", []string{"r"}, nil)
	require.NotEqual(t, foreign, again.IdentityPackage.ReasonerDIDs["r"].DID)
	require.Equal(t, first.IdentityPackage.ReasonerDIDs["r"].DID, again.IdentityPackage.ReasonerDIDs["r"].DID)
	rows, err := env.provider.ListAllComponentDIDs(context.Background(), first.IdentityPackage.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 2)
	for _, row := range rows {
		if row.ComponentDID != again.IdentityPackage.ReasonerDIDs["r"].DID {
			require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
			identity, err := svc.ResolveDID(row.ComponentDID)
			require.Error(t, err)
			require.Nil(t, identity)
		} else {
			require.Equal(t, storage.ComponentDIDStatusActive, row.Status)
			identity, err := svc.ResolveDID(row.ComponentDID)
			require.NoError(t, err)
			require.NotEmpty(t, identity.PrivateKeyJWK)
		}
	}
}
