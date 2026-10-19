package services

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestG2DIDDuplicateLifecycleSQLite(t *testing.T) {
	for _, quarantine := range []bool{true, false} {
		t.Run(map[bool]string{true: "quarantine", false: "deregistration"}[quarantine], func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
			require.NoError(t, err)
			defer db.Close()
			runG2DIDDuplicateLifecycle(t, env.provider, db, false, quarantine)
		})
	}
}

func TestG2DIDDuplicateLifecyclePostgres(t *testing.T) {
	for _, quarantine := range []bool{true, false} {
		t.Run(map[bool]string{true: "quarantine", false: "deregistration"}[quarantine], func(t *testing.T) {
			provider, _, dsn := postgresTakeoverStores(t)
			db, err := sql.Open("pgx", dsn)
			require.NoError(t, err)
			defer db.Close()
			runG2DIDDuplicateLifecycle(t, provider, db, true, quarantine)
		})
	}
}

func runG2DIDDuplicateLifecycle(t *testing.T, provider storage.StorageProvider, db *sql.DB, postgres, quarantine bool) {
	t.Helper()
	svc, _ := openReproService(t, provider, "server-a", filepath.Join(t.TempDir(), "keys"))
	pkg := register(t, svc, "node", []string{"r"}, []string{"r"}).IdentityPackage
	registry, err := svc.GetRegistry("server-a")
	require.NoError(t, err)
	insert := `INSERT INTO component_dids(did,agent_did,component_type,function_name,public_key_jwk,derivation_path,status,created_at) VALUES(?,?,?,?,?,?,'active',?)`
	update := `UPDATE component_dids SET created_at = ? WHERE agent_did = ?`
	if postgres {
		insert = `INSERT INTO component_dids(did,agent_did,component_type,function_name,public_key_jwk,derivation_path,status,created_at) VALUES($1,$2,$3,$4,$5,$6,'active',$7)`
		update = `UPDATE component_dids SET created_at = $1 WHERE agent_did = $2`
	}
	_, err = db.Exec(update, time.Now(), pkg.AgentDID.DID)
	require.NoError(t, err)
	duplicates := make(map[string]bool)
	for branch, kind := range []string{"reasoner", "skill"} {
		path := componentDerivationPath(pkg.AgentDID.DerivationPath, branch, 1)
		did, _, publicKey, err := svc.generateDIDWithKeys(registry.MasterSeed, path)
		require.NoError(t, err)
		if quarantine {
			did, publicKey = foreignDIDKey(t)
			path = "m/44'/0'/0'/0"
		}
		// The unverifiable duplicate must be selected before the healthy row.
		_, err = db.Exec(insert, did, pkg.AgentDID.DID, kind, "r", publicKey, path, time.Now().Add(-time.Hour))
		require.NoError(t, err)
		duplicates[did] = true
	}
	reload := func() {
		reg := NewDIDRegistryWithStorage(provider)
		require.NoError(t, reg.Initialize())
		svc = NewDIDService(svc.config, svc.keystore, reg)
		require.NoError(t, svc.Initialize("server-a"))
	}
	reload()
	if quarantine {
		// Startup itself adopts verified survivors, before another registration.
		for i := 0; i < 2; i++ {
			reg, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.Equal(t, pkg.ReasonerDIDs["r"].DID, reg.AgentNodes["node"].Reasoners["r"].DID)
			require.Equal(t, pkg.SkillDIDs["r"].DID, reg.AgentNodes["node"].Skills["r"].DID)
			again := register(t, svc, "node", []string{"r"}, []string{"r"}).IdentityPackage
			require.Equal(t, pkg.ReasonerDIDs["r"].DID, again.ReasonerDIDs["r"].DID)
			require.Equal(t, pkg.SkillDIDs["r"].DID, again.SkillDIDs["r"].DID)
			reload()
		}
	} else {
		resp, err := svc.DeregisterComponents(&types.ComponentDeregistrationRequest{AgentNodeID: "node", ReasonerIDsToRemove: []string{"r"}})
		require.NoError(t, err)
		require.True(t, resp.Success, resp.Error)
		reload()
		agent, err := svc.GetExistingAgentDID("node")
		require.NoError(t, err)
		require.Empty(t, agent.Reasoners)
		require.Contains(t, agent.Skills, "r", "removing a reasoner must preserve the same-name skill")
		resp, err = svc.DeregisterComponents(&types.ComponentDeregistrationRequest{AgentNodeID: "node", SkillIDsToRemove: []string{"r"}})
		require.NoError(t, err)
		require.True(t, resp.Success, resp.Error)
		reload()
	}
	rows, err := provider.ListAllComponentDIDs(context.Background(), pkg.AgentDID.DID)
	require.NoError(t, err)
	require.Len(t, rows, 4)
	for _, row := range rows {
		if quarantine && !duplicates[row.ComponentDID] {
			require.Equal(t, storage.ComponentDIDStatusActive, row.Status)
			identity, err := svc.ResolveDID(row.ComponentDID)
			require.NoError(t, err)
			require.NotEmpty(t, identity.PrivateKeyJWK)
		} else {
			require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
		}
	}
	if !quarantine {
		again := register(t, svc, "node", []string{"r"}, []string{"r"}).IdentityPackage
		for _, row := range rows {
			require.NotEqual(t, row.ComponentDID, again.ReasonerDIDs["r"].DID)
			require.NotEqual(t, row.ComponentDID, again.SkillDIDs["r"].DID)
		}
	}
}
