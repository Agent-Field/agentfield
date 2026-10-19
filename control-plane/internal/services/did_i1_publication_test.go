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

// SQLite write locks cover retirement too, so block only the subsequent agent
// upsert. Retirement and all reads still use real SQLite storage.
type i1PublicationDeadlineStore struct {
	storage.StorageProvider
	blockPublication bool
	blockedWrites    int
}

func (s *i1PublicationDeadlineStore) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	if s.blockPublication {
		s.blockedWrites++
		<-ctx.Done()
		return ctx.Err()
	}
	return s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
}

func TestI1DIDPublicationDeadlineSQLite(t *testing.T) {
	testI1DIDPublicationDeadline(t, false)
}

func TestI1DIDPublicationDeadlinePostgres(t *testing.T) {
	testI1DIDPublicationDeadline(t, true)
}

func testI1DIDPublicationDeadline(t *testing.T, postgres bool) {
	for _, keepComponents := range []bool{false, true} {
		name := "reasoner-only"
		if keepComponents {
			name = "reasoner-and-skill-with-survivors"
		}
		t.Run(name, func(t *testing.T) {
			provider, db := h1RetirementStore(t, postgres)
			serviceStore := provider
			wrapped := &i1PublicationDeadlineStore{StorageProvider: provider}
			if !postgres {
				serviceStore = wrapped
			}
			keyDir := filepath.Join(t.TempDir(), "keys")
			svc, _ := openReproService(t, serviceStore, "server-a", keyDir)
			reasoners := []string{"r"}
			var skills, kept []string
			if keepComponents {
				reasoners = append(reasoners, "keep")
				skills = []string{"s", "keep"}
				kept = []string{"keep"}
			}
			pkg := register(t, svc, "node", reasoners, skills).IdentityPackage
			other := register(t, svc, "other", []string{"r"}, nil).IdentityPackage
			before, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			retired := []string{pkg.ReasonerDIDs["r"].DID}
			if keepComponents {
				retired = append(retired, pkg.SkillDIDs["s"].DID)
			}
			svc.config.VCRequirements.RequireVCForExecution = true
			svc.config.VCRequirements.PersistExecutionVC = false
			vc := NewVCService(svc.config, svc, serviceStore)
			sign := func(did string) (*types.ExecutionVC, error) {
				return vc.GenerateExecutionVC(&types.ExecutionContext{
					ExecutionID: "publication-deadline", WorkflowID: "w",
					CallerDID: did, AgentNodeDID: pkg.AgentDID.DID, Timestamp: time.Now(),
				}, []byte(`{}`), []byte(`{}`), "succeeded", nil, 1)
			}
			for _, did := range retired {
				signed, err := sign(did)
				require.NoError(t, err)
				require.NotNil(t, signed, "the component must be usable before retirement")
			}

			var tx *sql.Tx
			if postgres {
				tx, err = db.Begin()
				require.NoError(t, err)
				defer tx.Rollback()
				// Lock the agent, not the component: retirement must commit first.
				_, err = tx.Exec("SELECT did FROM agent_dids WHERE did = $1 FOR UPDATE", pkg.AgentDID.DID)
				require.NoError(t, err)
			} else {
				wrapped.blockPublication = true
			}
			original := didRegistrationTimeout
			didRegistrationTimeout = 600 * time.Millisecond
			t.Cleanup(func() { didRegistrationTimeout = original })
			req := &types.DIDRegistrationRequest{AgentNodeID: "node"}
			if keepComponents {
				req.Reasoners = []types.ReasonerDefinition{{ID: "keep"}}
				req.Skills = []types.SkillDefinition{{ID: "keep"}}
			}
			response, err := svc.RegisterAgent(req)
			require.NoError(t, err)
			require.False(t, response.Success)
			require.Contains(t, response.Error, "failed to store updated registry")
			require.Contains(t, response.Error, "deadline")
			if !postgres {
				require.Equal(t, 1, wrapped.blockedWrites, "memory removal must not retry persistence")
			}

			rows, err := provider.ListAllComponentDIDs(context.Background(), pkg.AgentDID.DID)
			require.NoError(t, err)
			require.Len(t, rows, len(reasoners)+len(skills))
			for _, row := range rows {
				if row.ComponentName == "keep" {
					require.Equal(t, storage.ComponentDIDStatusActive, row.Status)
				} else {
					require.Contains(t, retired, row.ComponentDID)
					require.Equal(t, storage.ComponentDIDStatusRetired, row.Status)
				}
			}
			after, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.NotContains(t, after.AgentNodes["node"].Reasoners, "r")
			require.NotContains(t, after.AgentNodes["node"].Skills, "s")
			require.Equal(t, before.TotalDIDs-len(retired), after.TotalDIDs)
			require.Equal(t, pkg.ReasonerDIDs["r"].DID, before.AgentNodes["node"].Reasoners["r"].DID,
				"the previously published snapshot must not be mutated")
			if keepComponents {
				require.Equal(t, pkg.SkillDIDs["s"].DID, before.AgentNodes["node"].Skills["s"].DID)
			}
			for _, did := range retired {
				identity, err := svc.ResolveDID(did)
				require.Error(t, err)
				require.Nil(t, identity, "retired components must not expose a private key")
				signed, err := sign(did)
				require.ErrorContains(t, err, "failed to resolve caller DID")
				require.Nil(t, signed)
			}
			// Ordinary same-node registration must not return a retired key,
			// even while publication remains blocked.
			same := register(t, svc, "node", kept, kept).IdentityPackage
			require.NotContains(t, same.ReasonerDIDs, "r")
			require.NotContains(t, same.SkillDIDs, "s")
			require.Equal(t, pkg.AgentDID.DID, same.AgentDID.DID)
			if keepComponents {
				require.Equal(t, pkg.ReasonerDIDs["keep"].DID, same.ReasonerDIDs["keep"].DID)
				require.Equal(t, pkg.SkillDIDs["keep"].DID, same.SkillDIDs["keep"].DID)
				_, err = sign(same.ReasonerDIDs["keep"].DID)
				require.NoError(t, err)
				_, err = sign(same.SkillDIDs["keep"].DID)
				require.NoError(t, err)
			}
			identity, err := svc.ResolveDID(other.ReasonerDIDs["r"].DID)
			require.NoError(t, err)
			require.NotEmpty(t, identity.PrivateKeyJWK, "another agent's same-name component must remain usable")
			// Re-requesting the removed names must attempt a new allocation,
			// which is still blocked, instead of returning the retired keys.
			readdRequest := &types.DIDRegistrationRequest{AgentNodeID: "node"}
			for _, id := range reasoners {
				readdRequest.Reasoners = append(readdRequest.Reasoners, types.ReasonerDefinition{ID: id})
			}
			for _, id := range skills {
				readdRequest.Skills = append(readdRequest.Skills, types.SkillDefinition{ID: id})
			}
			blocked, err := svc.RegisterAgent(readdRequest)
			require.NoError(t, err)
			require.False(t, blocked.Success)
			require.Contains(t, blocked.Error, "deadline")
			require.NotContains(t, blocked.IdentityPackage.ReasonerDIDs, "r")
			require.NotContains(t, blocked.IdentityPackage.SkillDIDs, "s")

			if tx != nil {
				require.NoError(t, tx.Rollback())
			}
			wrapped.blockPublication = false
			didRegistrationTimeout = original
			register(t, svc, "node", kept, kept)
			retry, err := svc.GetRegistry("server-a")
			require.NoError(t, err)
			require.Equal(t, after.TotalDIDs, retry.TotalDIDs, "retry must not decrement the count again")
			fresh, _ := openReproService(t, provider, "server-a", keyDir)
			for _, did := range retired {
				_, err := fresh.ResolveDID(did)
				require.Error(t, err, "memory and storage must agree after reload")
			}
			readded := register(t, svc, "node", reasoners, skills).IdentityPackage
			require.NotEqual(t, pkg.ReasonerDIDs["r"].DID, readded.ReasonerDIDs["r"].DID)
			requireKeyMatchesDID(t, readded.ReasonerDIDs["r"].DID, readded.ReasonerDIDs["r"].PrivateKeyJWK)
			if keepComponents {
				require.NotEqual(t, pkg.SkillDIDs["s"].DID, readded.SkillDIDs["s"].DID)
				requireKeyMatchesDID(t, readded.SkillDIDs["s"].DID, readded.SkillDIDs["s"].PrivateKeyJWK)
			}
		})
	}
}
