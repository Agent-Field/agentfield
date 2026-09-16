package storage

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

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

	// Upserting the retired DID again re-activates it.
	require.NoError(t, ls.UpsertAgentDIDWithComponents(ctx, record))
	active, err = ls.ListComponentDIDs(ctx, record.DID)
	require.NoError(t, err)
	require.Equal(t, map[string]string{r1.ComponentDID: ComponentDIDStatusActive, s1.ComponentDID: ComponentDIDStatusActive}, componentStatuses(active))
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
