package storage

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// C1.2 and C1.5: conflicts are identifiable and a failed component insert
// rolls back the preceding agent insert; validation errors remain distinct.
func TestDIDTakeover_AllocationConflictIsTypedAndAtomic(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv", "did:root", []byte("seed"), now, now))
	base := AgentDIDRecord{
		AgentNodeID: "first", DID: "did:agent:first", AgentFieldServerID: "srv",
		PublicKeyJWK: `{}`, DerivationPath: "m/44'/0'/0'",
		Components: []ComponentDIDRequest{{ComponentDID: "did:component:first", ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: `{}`, DerivationPath: "m/44'/0'/0'/0'/0'"}},
	}
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, base))
	other := base
	other.AgentNodeID = "second"
	err := ls.UpsertAgentDIDWithComponents(ctx, other)
	require.ErrorIs(t, err, ErrDIDAllocationConflict)
	other.DID = "did:agent:second"
	err = ls.UpsertAgentDIDWithComponents(ctx, other)
	require.ErrorIs(t, err, ErrDIDAllocationConflict)
	agents, err := ls.ListAgentDIDs(ctx)
	require.NoError(t, err)
	require.Len(t, agents, 1, "component conflict must roll back agent insert")
	other.Components = nil
	other.AgentNodeID = ""
	err = ls.UpsertAgentDIDWithComponents(ctx, other)
	require.Error(t, err)
	require.False(t, errors.Is(err, ErrDIDAllocationConflict))
	attempts := 0
	err = ls.retryOnConstraintFailure(context.Background(), func() error {
		attempts++
		return fmt.Errorf("wrapped: %w", ErrDIDAllocationConflict)
	}, 3)
	require.ErrorIs(t, err, ErrDIDAllocationConflict)
	require.Equal(t, 1, attempts)
}

func componentStatuses(components []*types.ComponentDIDInfo) map[string]string {
	statuses := make(map[string]string, len(components))
	for _, component := range components {
		statuses[component.ComponentDID] = component.Status
	}
	return statuses
}

func TestComponentDIDRetirement(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-retire", "did:root:retire", []byte("seed"), now, now))

	r1 := ComponentDIDRequest{ComponentDID: "did:component:r1", ComponentType: "reasoner", ComponentName: "r1", PublicKeyJWK: `{"kid":"r1"}`, DerivationPath: "m/44'/1'/0'/0'/0'"}
	s1 := ComponentDIDRequest{ComponentDID: "did:component:s1", ComponentType: "skill", ComponentName: "s1", PublicKeyJWK: `{"kid":"s1"}`, DerivationPath: "m/44'/1'/0'/1'/0'"}
	record := AgentDIDRecord{
		AgentNodeID:        "agent-retire",
		DID:                "did:agent:retire",
		AgentFieldServerID: "srv-retire",
		PublicKeyJWK:       `{"kid":"agent"}`,
		DerivationPath:     "m/44'/1'/0'",
		Components:         []ComponentDIDRequest{r1, s1},
	}
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))

	all, err := ls.ListAllComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive, s1.ComponentDID: ComponentDIDStatusActive}, componentStatuses(all))

	// Empty list is a no-op; DIDs of another agent are not touched.
	require.NoError(t, ls.RetireComponentDIDs(ctx, record.DID, nil))
	require.NoError(t, ls.RetireComponentDIDs(ctx, "did:agent:other", []string{s1.ComponentDID}))
	active, err := ls.ListComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Len(t, active, 2)

	require.NoError(t, ls.RetireComponentDIDs(ctx, record.DID, []string{s1.ComponentDID}))

	active, err = ls.ListComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive}, componentStatuses(active))
	everyAgent, err := ls.ListComponentDIDs(ctx, "")
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive}, componentStatuses(everyAgent))

	all, err = ls.ListAllComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive, s1.ComponentDID: ComponentDIDStatusRetired}, componentStatuses(all))
	for _, component := range all {
		require.NotEmpty(t, component.DerivationPath, "retired rows must keep their derivation path")
	}

	retired, err := ls.GetComponentDID(ctx, "s1")
	require.NoError(t, err)
	require.Equal(t, ComponentDIDStatusRetired, retired.Status)

	// G2.b: saving a stale copy cannot re-activate a retired DID.
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))
	active, err = ls.ListComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive}, componentStatuses(active))
}

// G2.a, G2.b, G2.c: a stale upsert preserves revoked and retired status and
// cannot lower an X25519 rotation epoch, while a newer epoch still advances.
func TestDIDTakeover_StaleUpsertPreservesMonotonicIdentityState(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-monotonic", "did:root", []byte("seed"), now, now))
	record := AgentDIDRecord{
		AgentNodeID: "node", DID: "did:agent:monotonic", AgentFieldServerID: "srv-monotonic",
		PublicKeyJWK: `{}`, DerivationPath: "m/1", X25519Epoch: 1,
		Components: []ComponentDIDRequest{{ComponentDID: "did:component:monotonic", ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: `{}`, DerivationPath: "m/2"}},
	}
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))
	require.NoError(t, ls.RetireComponentDIDs(ctx, record.DID, []string{record.Components[0].ComponentDID}))
	revoked := record
	revoked.Status = string(types.AgentDIDStatusRevoked)
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, revoked))
	record.X25519Epoch = 0
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))
	assertStored := func(status types.AgentDIDStatus, epoch int, componentStatus string) {
		t.Helper()
		rows, err := ls.ListAgentDIDs(ctx)
		require.NoError(t, err)
		require.Len(t, rows, 1)
		require.Equal(t, status, rows[0].Status)
		require.Equal(t, epoch, rows[0].X25519Epoch)
		components, err := ls.ListAllComponentDIDs(ctx, record.DID)
		require.NoError(t, err)
		require.Len(t, components, 1)
		require.Equal(t, componentStatus, components[0].Status)
	}
	assertStored(types.AgentDIDStatusRevoked, 1, ComponentDIDStatusRetired)
	record.X25519Epoch = 2
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))
	assertStored(types.AgentDIDStatusRevoked, 2, ComponentDIDStatusRetired)
}

func TestComponentDIDRowsFromBeforeStatusColumnAreActive(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-legacy", "did:root:legacy", []byte("seed"), now, now))
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, AgentDIDRecord{
		AgentNodeID:        "agent-legacy",
		DID:                "did:agent:legacy",
		AgentFieldServerID: "srv-legacy",
		PublicKeyJWK:       `{"kid":"agent"}`,
		DerivationPath:     "m/44'/2'/0'",
	}))

	// Recreate the schema as it was before the status column existed, insert a
	// row the way older versions did, then run the startup migration again.
	_, err := ls.db.ExecContext(ctx, "DROP INDEX IF EXISTS idx_component_dids_status")
	require.NoError(t, err)
	_, err = ls.db.ExecContext(ctx, "ALTER TABLE component_dids DROP COLUMN status")
	require.NoError(t, err)
	_, err = ls.db.ExecContext(ctx, `INSERT INTO component_dids (did, agent_did, component_type, function_name, public_key_jwk, derivation_path)
		VALUES (?, ?, ?, ?, ?, ?)`, "did:component:legacy", "did:agent:legacy", "reasoner", "legacy", `{"kid":"legacy"}`, "m/44'/2'/0'/0'/0'")
	require.NoError(t, err)
	require.NoError(t, ls.autoMigrateSchema(ctx))

	active, err := ls.ListComponentDIDs(ctx, "did:agent:legacy")
	require.NoError(t, err)
	require.Equal(t, map[string]string{"did:component:legacy": ComponentDIDStatusActive}, componentStatuses(active))
	all, err := ls.ListAllComponentDIDs(ctx, "did:agent:legacy")
	require.NoError(t, err)
	require.Len(t, all, 1)
}

// C3.3: a different DID cannot claim an active node or component name, while
// revocation and retirement allow a fresh identity for the same name.
func TestDIDTakeover_ActiveNamesAreUniqueButRetiredNamesCanReturn(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv", "did:root", []byte("seed"), now, now))
	agent := AgentDIDRecord{AgentNodeID: "node", DID: "did:agent:one", AgentFieldServerID: "srv", PublicKeyJWK: `{}`, DerivationPath: "m/1"}
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, agent))
	other := agent
	other.DID = "did:agent:two"
	err := ls.UpsertAgentDIDWithComponents(ctx, other)
	require.ErrorIs(t, err, ErrDIDNodeAlreadyRegistered)
	agent.Status = string(types.AgentDIDStatusRevoked)
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, agent))
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, other))

	component := ComponentDIDRequest{ComponentDID: "did:component:one", ComponentType: "reasoner", ComponentName: "r", PublicKeyJWK: `{}`, DerivationPath: "m/2"}
	other.Components = []ComponentDIDRequest{component}
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, other))
	component.ComponentDID = "did:component:two"
	component.DerivationPath = "m/3"
	other.Components = []ComponentDIDRequest{component}
	err = ls.UpsertAgentDIDWithComponents(ctx, other)
	require.ErrorIs(t, err, ErrDIDNodeAlreadyRegistered)
	require.NoError(t, ls.RetireComponentDIDs(ctx, other.DID, []string{"did:component:one"}))
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, other))
	active, err := ls.ListComponentDIDs(ctx, other.DID)
	require.NoError(t, err)
	require.Len(t, active, 1)
	require.Equal(t, "did:component:two", active[0].ComponentDID)
}
