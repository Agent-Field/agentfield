package server

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/adminpb"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestListReasonersAggregatesNodes(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "haxen.db"),
			KVStorePath:  filepath.Join(tempDir, "haxen.bolt"),
		},
	}

	localStore := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := localStore.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping reasoner aggregation test")
		}
		require.NoError(t, err)
	}
	t.Cleanup(func() { _ = localStore.Close(ctx) })

	srv := &HaxenServer{storage: localStore}

	schema := json.RawMessage("{}")
	node := &types.AgentNode{
		ID:            "node-1",
		HealthStatus:  types.HealthStatusActive,
		Version:       "1.0.0",
		LastHeartbeat: time.Now().UTC(),
		Reasoners: []types.ReasonerDefinition{
			{ID: "reason", InputSchema: schema, OutputSchema: schema},
			{ID: "another", InputSchema: schema, OutputSchema: schema},
		},
	}
	require.NoError(t, localStore.RegisterAgent(ctx, node))

	resp, err := srv.ListReasoners(ctx, &adminpb.ListReasonersRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Reasoners, 2)
	require.Equal(t, "node-1.reason", resp.Reasoners[0].ReasonerId)
	require.Equal(t, "node-1", resp.Reasoners[0].AgentNodeId)
	require.NotEmpty(t, resp.Reasoners[0].LastHeartbeat)
}
