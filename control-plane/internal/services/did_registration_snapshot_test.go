package services

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// Run the other replica after a completed storage read, without sleeps or
// replacing any of the SQLite persistence behavior.
type registrationSnapshotStore struct {
	storage.StorageProvider
	afterRead func()
}

func (s *registrationSnapshotStore) ListComponentDIDs(ctx context.Context, did string) ([]*types.ComponentDIDInfo, error) {
	rows, err := s.StorageProvider.ListComponentDIDs(ctx, did)
	if err == nil && s.afterRead != nil {
		callback := s.afterRead
		s.afterRead = nil
		callback()
	}
	return rows, err
}

func TestRegisterAgentConcurrentComponentSnapshot(t *testing.T) {
	for _, component := range []string{"reasoner", "skill"} {
		for _, interleave := range []bool{false, true} {
			timing := "serial"
			if interleave {
				timing = "after_first_read"
			}
			t.Run(component+"/"+timing, func(t *testing.T) {
				env := newContinuityEnv(t)
				env.restart()
				keyDir := filepath.Join(t.TempDir(), "keys")
				first, _ := openReproService(t, env.provider, "server-a", keyDir)
				original := register(t, first, "component-node", nil, nil)
				store := &registrationSnapshotStore{StorageProvider: env.provider}
				second, _ := openReproService(t, store, "server-a", keyDir)

				var reasoners, skills []string
				req := &types.DIDRegistrationRequest{AgentNodeID: "component-node"}
				if component == "reasoner" {
					reasoners = []string{"new-component"}
					req.Reasoners = []types.ReasonerDefinition{{ID: "new-component"}}
				} else {
					skills = []string{"new-component"}
					req.Skills = []types.SkillDefinition{{ID: "new-component"}}
				}

				var winner *types.DIDRegistrationResponse
				issue := func() {
					winner = register(t, first, "component-node", reasoners, skills)
				}
				if interleave {
					store.afterRead = issue
				} else {
					issue()
				}
				got, err := second.RegisterAgent(req)
				require.NoError(t, err)
				require.True(t, got.Success, got.Error)
				require.NotNil(t, winner)
				require.Equal(t, original.IdentityPackage.AgentDID.DID, got.IdentityPackage.AgentDID.DID)
				require.True(t, original.IdentityPackage.AgentDID.PrivateKeyJWK == got.IdentityPackage.AgentDID.PrivateKeyJWK, "agent signing key must survive re-registration")
				var want, actual types.DIDIdentity
				if component == "reasoner" {
					want = winner.IdentityPackage.ReasonerDIDs["new-component"]
					actual = got.IdentityPackage.ReasonerDIDs["new-component"]
				} else {
					want = winner.IdentityPackage.SkillDIDs["new-component"]
					actual = got.IdentityPackage.SkillDIDs["new-component"]
				}
				require.NotEmpty(t, want.DID)
				require.Equal(t, want.DID, actual.DID)
				require.Equal(t, want.DerivationPath, actual.DerivationPath)
				require.True(t, want.PrivateKeyJWK == actual.PrivateKeyJWK, "component signing key must be preserved")
				requireKeyMatchesDID(t, actual.DID, actual.PrivateKeyJWK)

				// A later retry also returns the issued identity, rather than
				// allocating a replacement or losing it after a refresh.
				again := register(t, second, "component-node", reasoners, skills)
				if component == "reasoner" {
					require.Equal(t, want.DID, again.IdentityPackage.ReasonerDIDs["new-component"].DID)
				} else {
					require.Equal(t, want.DID, again.IdentityPackage.SkillDIDs["new-component"].DID)
				}
			})
		}
	}
}

// A requested batch can contain both a component already issued by another
// replica and one this request must still allocate.
func TestConcurrentRegistrationPartialOverlap(t *testing.T) {
	for _, directPartial := range []bool{false, true} {
		api := "RegisterAgent"
		if directPartial {
			api = "PartialRegisterAgent"
		}
		for _, tc := range []struct {
			name              string
			reasoners, skills []string
		}{
			{"reasoners", []string{"issued", "new"}, nil},
			{"skills", nil, []string{"issued", "new"}},
			{"cross_type", []string{"issued"}, []string{"new"}},
		} {
			t.Run(api+"/"+tc.name, func(t *testing.T) {
				env := newContinuityEnv(t)
				env.restart()
				first, _ := openReproService(t, env.provider, "server-a", env.keyDir)
				original := register(t, first, "node", []string{"keep-r"}, []string{"keep-s"})
				store := &registrationSnapshotStore{StorageProvider: env.provider}
				second, _ := openReproService(t, store, "server-a", env.keyDir)
				before, readErr := second.GetRegistry("server-a")
				require.NoError(t, readErr)
				reasoners := append([]string{"keep-r"}, tc.reasoners...)
				skills := append([]string{"keep-s"}, tc.skills...)
				full := &types.DIDRegistrationRequest{AgentNodeID: "node"}
				for _, id := range reasoners {
					full.Reasoners = append(full.Reasoners, types.ReasonerDefinition{ID: id})
				}
				for _, id := range skills {
					full.Skills = append(full.Skills, types.SkillDefinition{ID: id})
				}
				var winner *types.DIDRegistrationResponse
				store.afterRead = func() {
					r, s := []string{"keep-r"}, []string{"keep-s"}
					if len(tc.reasoners) > 0 {
						r = append(r, "issued")
					} else {
						s = append(s, "issued")
					}
					winner = register(t, first, "node", r, s)
				}

				var got *types.DIDRegistrationResponse
				var err error
				if directPartial {
					got, err = second.PartialRegisterAgent(&types.PartialDIDRegistrationRequest{
						AgentNodeID: "node", NewReasonerIDs: tc.reasoners, NewSkillIDs: tc.skills,
						AllReasoners: full.Reasoners, AllSkills: full.Skills,
					})
				} else {
					got, err = second.RegisterAgent(full)
				}
				require.NoError(t, err)
				require.True(t, got.Success, got.Error)
				require.NotNil(t, winner)
				for _, id := range tc.reasoners {
					require.NotEmpty(t, got.IdentityPackage.ReasonerDIDs[id].DID, "requested reasoner %s", id)
				}
				for _, id := range tc.skills {
					require.NotEmpty(t, got.IdentityPackage.SkillDIDs[id].DID, "requested skill %s", id)
				}
				if len(tc.reasoners) > 0 {
					require.Equal(t, winner.IdentityPackage.ReasonerDIDs["issued"].DID, got.IdentityPackage.ReasonerDIDs["issued"].DID)
				} else {
					require.Equal(t, winner.IdentityPackage.SkillDIDs["issued"].DID, got.IdentityPackage.SkillDIDs["issued"].DID)
				}
				if directPartial {
					require.Empty(t, got.IdentityPackage.AgentDID.PrivateKeyJWK)
					require.Len(t, got.IdentityPackage.ReasonerDIDs, len(tc.reasoners))
					require.Len(t, got.IdentityPackage.SkillDIDs, len(tc.skills))
				} else {
					require.True(t, original.IdentityPackage.AgentDID.PrivateKeyJWK == got.IdentityPackage.AgentDID.PrivateKeyJWK, "full registration must preserve the agent signing key")
					require.Equal(t, original.IdentityPackage.ReasonerDIDs["keep-r"].DID, got.IdentityPackage.ReasonerDIDs["keep-r"].DID)
					require.Equal(t, original.IdentityPackage.SkillDIDs["keep-s"].DID, got.IdentityPackage.SkillDIDs["keep-s"].DID)
				}
				after, readErr := second.GetRegistry("server-a")
				require.NoError(t, readErr)
				require.Equal(t, before.TotalDIDs+2, after.TotalDIDs, "only two component identities were issued across both replicas")
				if tc.name == "reasoners" {
					require.Equal(t, "Partial registration successful: 1 new reasoners, 0 new skills", got.Message)
				} else {
					require.Equal(t, "Partial registration successful: 0 new reasoners, 1 new skills", got.Message)
				}
				persisted := register(t, second, "node", reasoners, skills)
				for _, id := range tc.reasoners {
					require.Equal(t, persisted.IdentityPackage.ReasonerDIDs[id].DID, got.IdentityPackage.ReasonerDIDs[id].DID)
				}
				for _, id := range tc.skills {
					require.Equal(t, persisted.IdentityPackage.SkillDIDs[id].DID, got.IdentityPackage.SkillDIDs[id].DID)
				}
			})
		}
	}
}

type registrationRefreshFailureStore struct {
	storage.StorageProvider
	failReads bool
}

func (s *registrationRefreshFailureStore) ListComponentDIDs(ctx context.Context, did string) ([]*types.ComponentDIDInfo, error) {
	if s.failReads {
		return nil, errors.New("component read unavailable")
	}
	return s.StorageProvider.ListComponentDIDs(ctx, did)
}

func (s *registrationRefreshFailureStore) UpsertAgentDIDWithComponents(ctx context.Context, record storage.AgentDIDRecord) error {
	err := s.StorageProvider.UpsertAgentDIDWithComponents(ctx, record)
	if err == nil {
		s.failReads = true
	}
	return err
}

func TestRegisterAgentRefreshFailureAfterUpdate(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	first, _ := openReproService(t, env.provider, "server-a", env.keyDir)
	register(t, first, "node", nil, nil)
	store := &registrationRefreshFailureStore{StorageProvider: env.provider}
	second, _ := openReproService(t, store, "server-a", env.keyDir)

	got, err := second.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "new"}},
	})
	require.NoError(t, err)
	require.False(t, got.Success, "a failed response refresh must not report success")
	require.Contains(t, got.Error, "failed to get updated agent info")
	require.Contains(t, got.Error, "component read unavailable")

	// Persistence succeeded; after reads recover, retry returns the issued
	// identity rather than requiring another allocation.
	store.failReads = false
	recovered := register(t, second, "node", []string{"new"}, nil)
	require.NotEmpty(t, recovered.IdentityPackage.ReasonerDIDs["new"].DID)
	require.NotEmpty(t, recovered.IdentityPackage.AgentDID.PrivateKeyJWK)
}
