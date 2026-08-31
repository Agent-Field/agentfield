package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
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
			},
			"run": {"display_name":"Release","labels":["smoke"],"links":[{"label":"PR","url":"https://x.test/pr"}]}
		}`),
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}))

	lineage, golden, runMetadata := svc.loadRunMetadata(ctx, "run-restart")
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
	require.Equal(t, "Release", runMetadata.DisplayName)
	require.Equal(t, []string{"smoke"}, runMetadata.Labels)
	require.Equal(t, "https://x.test/pr", runMetadata.Links[0].URL)

	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:     "run-invalid",
		Metadata:  json.RawMessage(`{"lineage":`),
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}))
	lineage, golden, _ = svc.loadRunMetadata(ctx, "run-invalid")
	require.Nil(t, lineage)
	require.Nil(t, golden)

	lineage, golden, _ = svc.loadRunMetadata(ctx, "run-missing")
	require.Nil(t, lineage)
	require.Nil(t, golden)
}

func TestWorkflowDAGResponsesCarryRunMetadata(t *testing.T) {
	ctx := context.Background()
	store := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := store.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: storage.LocalStorageConfig{DatabasePath: filepath.Join(t.TempDir(), "agentfield.db"), KVStorePath: filepath.Join(t.TempDir(), "agentfield.bolt")}})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close(ctx) })
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: "exec-dag-metadata", RunID: "run-dag-metadata", AgentNodeID: "node", ReasonerID: "reasoner", Status: "succeeded", StartedAt: now, CreatedAt: now, UpdatedAt: now}))
	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{RunID: "run-dag-metadata", RootWorkflowID: "run-dag-metadata", Status: "succeeded", Metadata: json.RawMessage(`{"run":{"display_name":"Release","labels":["smoke"],"links":[{"label":"PR","url":"https://x.test/pr"}]}}`), CreatedAt: now, UpdatedAt: now}))

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/runs/:workflowId", GetWorkflowDAGHandler(store))
	// isLightweightRequest (workflow_dag.go) selects the lightweight branch on
	// ?mode=lightweight; both response shapes must carry run_metadata.
	for _, test := range []struct {
		path        string
		lightweight bool
	}{
		{"/runs/run-dag-metadata", false},
		{"/runs/run-dag-metadata?mode=lightweight", true},
	} {
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, test.path, nil))
		require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
		var response map[string]interface{}
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
		// Prove the two iterations really took different branches, so this is
		// not the full response asserted twice.
		if test.lightweight {
			require.Equal(t, "lightweight", response["mode"], recorder.Body.String())
		} else {
			require.Contains(t, response, "dag", recorder.Body.String())
		}
		metadata := response["run_metadata"].(map[string]interface{})
		require.Equal(t, "Release", metadata["display_name"])
		require.Equal(t, "smoke", metadata["labels"].([]interface{})[0])
		require.Equal(t, "https://x.test/pr", metadata["links"].([]interface{})[0].(map[string]interface{})["url"])
	}
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
