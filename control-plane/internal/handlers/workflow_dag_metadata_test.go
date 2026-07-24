package handlers

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestExecutionGraphServiceLoadRunMetadata(t *testing.T) {
	ctx := context.Background()
	store := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := store.Initialize(ctx, storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(t.TempDir(), "agentfield.db"),
			KVStorePath:  filepath.Join(t.TempDir(), "agentfield.bolt"),
		},
	})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = store.Close(ctx)
	})

	svc := newExecutionGraphService(store)

	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID: "run-restart",
		Metadata: json.RawMessage(`{
			"lineage": {
				"kind": "fork",
				"source_run_id": "old-run",
				"source_execution_id": "old-child",
				"restarted_execution_id": "old-root",
				"reuse": "succeeded-before",
				"scope": "workflow"
			},
			"golden": {
				"name": "Known good retry",
				"tags": ["smoke", "restart"],
				"saved_by": "user",
				"saved_at": "2026-04-08T12:00:00Z"
			}
		}`),
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}))

	lineage, golden := svc.loadRunMetadata(ctx, "run-restart")
	require.NotNil(t, lineage)
	require.Equal(t, "fork", lineage.Kind)
	require.Equal(t, "old-run", lineage.SourceRunID)
	require.Equal(t, "old-child", lineage.SourceExecutionID)
	require.Equal(t, "old-root", lineage.RestartedExecutionID)
	require.Equal(t, "succeeded-before", lineage.Reuse)
	require.Equal(t, "workflow", lineage.Scope)
	require.NotNil(t, golden)
	require.Equal(t, "Known good retry", golden.Name)
	require.Equal(t, []string{"smoke", "restart"}, golden.Tags)

	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:     "run-invalid",
		Metadata:  json.RawMessage(`{"lineage":`),
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}))
	lineage, golden = svc.loadRunMetadata(ctx, "run-invalid")
	require.Nil(t, lineage)
	require.Nil(t, golden)

	lineage, golden = svc.loadRunMetadata(ctx, "run-missing")
	require.Nil(t, lineage)
	require.Nil(t, golden)
}

func TestFillReuseSourceRun(t *testing.T) {
	reused := "replayed_from_execution:src-exec"
	child := &types.Execution{ExecutionID: "child", StatusReason: &reused}
	root := WorkflowDAGNode{
		ExecutionID: "root",
		Children:    []WorkflowDAGNode{executionToDAGNode(child, 1)},
	}

	// Per-node reuse info only carries the source execution id until back-filled.
	require.NotNil(t, root.Children[0].Reuse)
	require.Equal(t, "src-exec", root.Children[0].Reuse.SourceExecutionID)
	require.Empty(t, root.Children[0].Reuse.SourceRunID)

	fillReuseSourceRunDAG(&root, "old-run")
	require.Equal(t, "old-run", root.Children[0].Reuse.SourceRunID)
	require.Nil(t, root.Reuse, "non-reused nodes must not gain a reuse marker")

	// Existing source run ids are preserved, and nil markers are a no-op.
	preset := &ExecutionReuseInfo{Hit: true, SourceExecutionID: "e", SourceRunID: "keep"}
	fillReuseSourceRunNode(preset, "other-run")
	require.Equal(t, "keep", preset.SourceRunID)
	fillReuseSourceRunNode(nil, "old-run")
}
