package services

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func h1RetirementStore(t *testing.T, postgres bool) (storage.StorageProvider, *sql.DB) {
	t.Helper()
	var provider storage.StorageProvider
	driver := "sqlite3"
	var dsn string
	if postgres {
		provider, _, dsn = postgresTakeoverStores(t)
		driver = "pgx"
	} else {
		t.Setenv("AGENTFIELD_SQLITE_BUSY_TIMEOUT_MS", "1000")
		dir := t.TempDir()
		provider = openReproStorage(t, dir)
		t.Cleanup(func() { require.NoError(t, provider.Close(context.Background())) })
		dsn = filepath.Join(dir, "agentfield.db") + "?_txlock=immediate"
	}
	db, err := sql.Open(driver, dsn)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	return provider, db
}

func TestH1DIDRetirementDeadlineSQLite(t *testing.T) {
	testH1DIDRetirementDeadline(t, false)
}

func TestH1DIDRetirementDeadlinePostgres(t *testing.T) {
	testH1DIDRetirementDeadline(t, true)
}

// A real row/write lock makes retirement outlast the registration deadline.
// Neither the failed request nor a subsequent reload may lose the active DID.
func testH1DIDRetirementDeadline(t *testing.T, postgres bool) {
	provider, db := h1RetirementStore(t, postgres)
	keyDir := filepath.Join(t.TempDir(), "keys")
	svc, _ := openReproService(t, provider, "server-a", keyDir)
	pkg := register(t, svc, "node", []string{"r"}, nil).IdentityPackage
	oldDID := pkg.ReasonerDIDs["r"].DID
	tx, err := db.Begin()
	require.NoError(t, err)
	defer tx.Rollback()
	if postgres {
		_, err = tx.Exec("SELECT did FROM component_dids WHERE did = $1 FOR UPDATE", oldDID)
	} else {
		_, err = tx.Exec("UPDATE component_dids SET status = status WHERE did = ?", oldDID)
	}
	require.NoError(t, err)
	original := didRegistrationTimeout
	didRegistrationTimeout = 300 * time.Millisecond
	t.Cleanup(func() { didRegistrationTimeout = original })
	response, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node"})
	require.NoError(t, err)
	require.False(t, response.Success)
	require.Contains(t, response.Error, "deadline")
	registry, err := svc.GetRegistry("server-a")
	require.NoError(t, err)
	require.Equal(t, oldDID, registry.AgentNodes["node"].Reasoners["r"].DID,
		"failed retirement must leave the published component intact")
	rows, err := provider.ListComponentDIDs(context.Background(), pkg.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, oldDID, rows[0].ComponentDID)
	require.Equal(t, storage.ComponentDIDStatusActive, rows[0].Status)
	require.NoError(t, tx.Rollback())
	didRegistrationTimeout = original

	register(t, svc, "node", nil, nil)
	registry, err = svc.GetRegistry("server-a")
	require.NoError(t, err)
	require.Empty(t, registry.AgentNodes["node"].Reasoners)
	rows, err = provider.ListAllComponentDIDs(context.Background(), pkg.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.Equal(t, storage.ComponentDIDStatusRetired, rows[0].Status)
	reloaded, _ := openReproService(t, provider, "server-a", keyDir)
	agent, err := reloaded.GetExistingAgentDID("node")
	require.NoError(t, err)
	require.Empty(t, agent.Reasoners, "a successful removal must survive reload")
	readded := register(t, reloaded, "node", []string{"r"}, nil).IdentityPackage
	require.NotEqual(t, oldDID, readded.ReasonerDIDs["r"].DID)
	requireKeyMatchesDID(t, readded.ReasonerDIDs["r"].DID, readded.ReasonerDIDs["r"].PrivateKeyJWK)
}

func TestH1DIDDeregisterDurableOnlyComponents(t *testing.T) {
	for _, postgres := range []bool{false, true} {
		name := "sqlite"
		if postgres {
			name = "postgres"
		}
		t.Run(name, func(t *testing.T) {
			provider, _ := h1RetirementStore(t, postgres)
			keyDir := filepath.Join(t.TempDir(), "keys")
			svc, registry := openReproService(t, provider, "server-a", keyDir)
			pkg := register(t, svc, "node", []string{"same"}, []string{"same"}).IdentityPackage
			other := register(t, svc, "other", []string{"same"}, nil).IdentityPackage
			// Model a previously published removal whose durable retirement failed.
			require.NoError(t, registry.UpdateAgents("server-a", func(live *types.DIDRegistry) ([]string, error) {
				agent := cloneAgentInfo(live.AgentNodes["node"])
				delete(agent.Reasoners, "same")
				live.AgentNodes["node"] = agent
				live.TotalDIDs--
				return []string{"node"}, nil
			}))
			before, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			response, err := svc.DeregisterComponents(&types.ComponentDeregistrationRequest{
				AgentNodeID: "node", ReasonerIDsToRemove: []string{"same", "same", "missing"},
			})
			require.NoError(t, err)
			require.True(t, response.Success, response.Error)
			require.Equal(t, 1, response.RemovedCount, "a durable-only name counts once")
			after, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.Equal(t, before.TotalDIDs, after.TotalDIDs, "a durable-only removal must not decrement the memory count")
			rows, err := provider.ListAllComponentDIDs(context.Background(), pkg.AgentDID.DID)
			require.NoError(t, err)
			require.Len(t, rows, 2)
			for _, row := range rows {
				if row.ComponentType == "reasoner" {
					require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
				} else {
					require.Equal(t, storage.ComponentDIDStatusActive, row.Status)
				}
			}
			otherRows, err := provider.ListComponentDIDs(context.Background(), other.AgentDID.DID)
			require.NoError(t, err)
			require.Len(t, otherRows, 1, "another agent's same-name component must remain active")
			response, err = svc.DeregisterComponents(&types.ComponentDeregistrationRequest{
				AgentNodeID: "node", ReasonerIDsToRemove: []string{"same"},
			})
			require.NoError(t, err)
			require.True(t, response.Success, response.Error)
			require.Zero(t, response.RemovedCount)
			reloaded, _ := openReproService(t, provider, "server-a", keyDir)
			agent, err := reloaded.GetExistingAgentDID("node")
			require.NoError(t, err)
			require.Empty(t, agent.Reasoners)
			require.Equal(t, pkg.SkillDIDs["same"].DID, agent.Skills["same"].DID)
		})
	}
}

type h1PublicationFailureStore struct {
	storage.StorageProvider
	failPublication bool
	retireContext   context.Context
	publishContext  context.Context
}

func (s *h1PublicationFailureStore) RetireComponentDIDs(ctx context.Context, agentDID string, dids []string) error {
	s.retireContext = ctx
	return s.StorageProvider.RetireComponentDIDs(ctx, agentDID, dids)
}

func (s *h1PublicationFailureStore) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	if s.failPublication {
		s.publishContext = ctx
		return errors.New("publication failure for test")
	}
	return s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

func TestH1DIDRetirementSurvivesPublicationFailure(t *testing.T) {
	for _, postgres := range []bool{false, true} {
		name := "sqlite"
		if postgres {
			name = "postgres"
		}
		t.Run(name, func(t *testing.T) {
			provider, _ := h1RetirementStore(t, postgres)
			wrapped := &h1PublicationFailureStore{StorageProvider: provider}
			keyDir := filepath.Join(t.TempDir(), "keys")
			svc, registry := openReproService(t, wrapped, "server-a", keyDir)
			pkg := register(t, svc, "node", []string{"r"}, nil).IdentityPackage
			before, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			wrapped.failPublication = true
			response, err := svc.RegisterAgent(&types.DIDRegistrationRequest{AgentNodeID: "node"})
			require.NoError(t, err)
			require.False(t, response.Success)
			require.Contains(t, response.Error, "publication failure for test")
			require.NotNil(t, wrapped.retireContext)
			require.Same(t, wrapped.retireContext, wrapped.publishContext, "both writes must use the registration context")
			_, hasDeadline := wrapped.retireContext.Deadline()
			require.True(t, hasDeadline)
			live, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.NotContains(t, live.AgentNodes["node"].Reasoners, "r")
			require.Equal(t, before.TotalDIDs-1, live.TotalDIDs)
			rows, err := provider.ListAllComponentDIDs(context.Background(), pkg.AgentDID.DID)
			require.NoError(t, err)
			require.Len(t, rows, 1)
			require.Equal(t, storage.ComponentDIDStatusRetired, rows[0].Status)

			// A stale writer must not reactivate the retired DID (the R6 guard).
			wrapped.failPublication = false
			require.NoError(t, registry.UpdateAgents("server-a", func(live *types.DIDRegistry) ([]string, error) {
				live.AgentNodes["node"] = before.AgentNodes["node"]
				live.TotalDIDs = before.TotalDIDs
				return []string{"node"}, nil
			}))
			rows, err = provider.ListComponentDIDs(context.Background(), pkg.AgentDID.DID)
			require.NoError(t, err)
			require.Empty(t, rows)
			register(t, svc, "node", nil, nil)
			live, err = svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.Empty(t, live.AgentNodes["node"].Reasoners)
			reloaded, _ := openReproService(t, provider, "server-a", keyDir)
			agent, err := reloaded.GetExistingAgentDID("node")
			require.NoError(t, err)
			require.Empty(t, agent.Reasoners)
		})
	}
}
