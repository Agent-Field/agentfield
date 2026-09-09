package ui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestWorkflowRunListAndDetailCarryRunMetadataAlongsideLineage(t *testing.T) {
	store, ctx := setupUIHandlerStorage(t)
	now := time.Now().UTC()
	for _, test := range []struct{ runID, metadata string }{
		{"run-both", `{"lineage":{"kind":"fork","source_run_id":"source"},"run":{"display_name":"Release","labels":["smoke"],"links":[{"label":"PR","url":"https://x.test/pr"}]}}`},
		{"run-only", `{"run":{"display_name":"Only run","labels":["solo"],"links":[{"label":"Docs","url":"https://x.test/docs"}]}}`},
	} {
		require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{RunID: test.runID, RootWorkflowID: test.runID, Status: "succeeded", Metadata: json.RawMessage(test.metadata), CreatedAt: now, UpdatedAt: now}))
		require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: "exec-" + test.runID, RunID: test.runID, AgentNodeID: "node", ReasonerID: "reasoner", Status: "succeeded", StartedAt: now, CreatedAt: now, UpdatedAt: now}))
	}
	gin.SetMode(gin.TestMode)
	handler := NewWorkflowRunHandler(store)
	router := gin.New()
	router.GET("/runs", handler.ListWorkflowRunsHandler)
	router.GET("/runs/:run_id", handler.GetWorkflowRunDetailHandler)

	listRecorder := httptest.NewRecorder()
	router.ServeHTTP(listRecorder, httptest.NewRequest(http.MethodGet, "/runs", nil))
	require.Equal(t, http.StatusOK, listRecorder.Code, listRecorder.Body.String())
	var list map[string]interface{}
	require.NoError(t, json.Unmarshal(listRecorder.Body.Bytes(), &list))
	rows, ok := list["runs"].([]interface{})
	require.True(t, ok, "list response should carry a runs array: %s", listRecorder.Body.String())
	found := map[string]map[string]interface{}{}
	for _, value := range rows {
		row := value.(map[string]interface{})
		found[row["run_id"].(string)] = row
	}
	require.Contains(t, found, "run-both")
	require.Contains(t, found, "run-only")
	assertRunMetadataJSON(t, found["run-both"], "Release", "smoke", "https://x.test/pr")
	require.Equal(t, "fork", found["run-both"]["lineage"].(map[string]interface{})["kind"])
	assertRunMetadataJSON(t, found["run-only"], "Only run", "solo", "https://x.test/docs")

	detailRecorder := httptest.NewRecorder()
	router.ServeHTTP(detailRecorder, httptest.NewRequest(http.MethodGet, "/runs/run-both", nil))
	require.Equal(t, http.StatusOK, detailRecorder.Code, detailRecorder.Body.String())
	var detail map[string]interface{}
	require.NoError(t, json.Unmarshal(detailRecorder.Body.Bytes(), &detail))
	// The detail response nests the run fields under "run"; run_metadata must
	// sit beside lineage there, not at the top level.
	detailRun, ok := detail["run"].(map[string]interface{})
	require.True(t, ok, "detail response should nest run fields: %s", detailRecorder.Body.String())
	assertRunMetadataJSON(t, detailRun, "Release", "smoke", "https://x.test/pr")
	require.Equal(t, "fork", detailRun["lineage"].(map[string]interface{})["kind"])
}

func assertRunMetadataJSON(t *testing.T, object map[string]interface{}, name, label, url string) {
	t.Helper()
	metadata := object["run_metadata"].(map[string]interface{})
	require.Equal(t, name, metadata["display_name"])
	require.Equal(t, label, metadata["labels"].([]interface{})[0])
	require.Equal(t, url, metadata["links"].([]interface{})[0].(map[string]interface{})["url"])
}
