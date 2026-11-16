package services

import (
	"context"
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
