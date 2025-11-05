package services

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func setupTestStorage(t *testing.T) (storage.StorageProvider, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "haxen.db"),
			KVStorePath:  filepath.Join(tempDir, "haxen.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := provider.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping DID registry test")
		}
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	return provider, ctx
}

func TestDIDRegistryInitializeAndLookup(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	haxenID := "haxen-1"
	now := time.Now().UTC().Truncate(time.Second)

	require.NoError(t, provider.StoreHaxenServerDID(ctx, haxenID, "did:haxen:root", []byte("seed"), now, now))

	components := []storage.ComponentDIDRequest{
		{
			ComponentDID:    "did:reasoner:1",
			ComponentType:   "reasoner",
			ComponentName:   "reasoner.fn",
			PublicKeyJWK:    "{}",
			DerivationIndex: 1,
		},
		{
			ComponentDID:    "did:skill:1",
			ComponentType:   "skill",
			ComponentName:   "skill.fn",
			PublicKeyJWK:    "{}",
			DerivationIndex: 2,
		},
	}

	require.NoError(t, provider.StoreAgentDIDWithComponents(ctx, "agent-1", "did:agent:1", haxenID, "{}", 0, components))

	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	loaded, err := registry.GetRegistry(haxenID)
	require.NoError(t, err)
	require.NotNil(t, loaded)
	require.Contains(t, loaded.AgentNodes, "agent-1")

	// Validate reasoner lookup
	reasonerID, err := registry.FindDIDByComponent(haxenID, "reasoner", "reasoner.fn")
	require.NoError(t, err)
	require.Equal(t, "did:reasoner:1", reasonerID.DID)

	// Validate skill lookup
	skillID, err := registry.FindDIDByComponent(haxenID, "skill", "skill.fn")
	require.NoError(t, err)
	require.Equal(t, "did:skill:1", skillID.DID)

	// Update status and ensure it is persisted in-memory
	require.NoError(t, registry.UpdateAgentStatus(haxenID, "agent-1", types.AgentDIDStatusActive))

	loadedAfterUpdate, err := registry.GetRegistry(haxenID)
	require.NoError(t, err)
	require.Equal(t, types.AgentDIDStatusActive, loadedAfterUpdate.AgentNodes["agent-1"].Status)

	packageResult, err := registry.GetAgentDIDs(haxenID, "agent-1")
	require.NoError(t, err)
	require.Equal(t, "did:agent:1", packageResult.AgentDID.DID)
	require.Contains(t, packageResult.ReasonerDIDs, "reasoner.fn")
	require.Contains(t, packageResult.SkillDIDs, "skill.fn")

	registries, err := registry.ListRegistries()
	require.NoError(t, err)
	require.Len(t, registries, 1)
}
