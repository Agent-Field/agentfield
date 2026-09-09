package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func raw(value string) json.RawMessage { return json.RawMessage(value) }

func TestApplyRunMetadataInputPatchAndBounds(t *testing.T) {
	current := types.RunMetadata{
		DisplayName: "Old",
		Labels:      []string{"old"},
		Links:       []types.RunMetadataLink{{Label: "PR", URL: "https://x.test/pr"}},
	}
	merged, err := applyRunMetadataInput(current, RunMetadataInput{Labels: raw(`[" a ","a","","b"]`)})
	require.NoError(t, err)
	require.Equal(t, "Old", merged.DisplayName)
	require.Equal(t, []string{"a", "b"}, merged.Labels)
	require.Equal(t, current.Links, merged.Links)

	merged, err = applyRunMetadataInput(merged, RunMetadataInput{DisplayName: raw(`null`)})
	require.NoError(t, err)
	require.Empty(t, merged.DisplayName)
	require.Equal(t, []string{"a", "b"}, merged.Labels)

	tests := []RunMetadataInput{
		{DisplayName: raw(`"` + strings.Repeat("x", types.MaxRunDisplayNameRunes+1) + `"`)},
		{Labels: raw(`[` + strings.TrimSuffix(strings.Repeat(`"x",`, types.MaxRunLabels+1), ",") + `]`)},
		{Labels: raw(`["` + strings.Repeat("x", types.MaxRunLabelRunes+1) + `"]`)},
		{Links: raw(`[` + strings.TrimSuffix(strings.Repeat(`{"url":"https://x.test"},`, types.MaxRunLinks+1), ",") + `]`)},
		{Links: raw(`[{"url":"javascript:alert(1)"}]`)},
	}
	for _, input := range tests {
		_, err := applyRunMetadataInput(types.RunMetadata{}, input)
		require.Error(t, err)
	}
}

func TestApplyRunMetadataInputRejectsTypesAndLinkBoundsAndClearsFields(t *testing.T) {
	current := types.RunMetadata{DisplayName: "keep", Labels: []string{"old"}, Links: []types.RunMetadataLink{{Label: "Docs", URL: "https://x.test"}}}
	tests := []struct {
		name  string
		input RunMetadataInput
	}{
		{"display name type", RunMetadataInput{DisplayName: raw(`5`)}},
		{"labels type", RunMetadataInput{Labels: raw(`"x"`)}},
		{"links type", RunMetadataInput{Links: raw(`5`)}},
		{"link label bound", RunMetadataInput{Links: raw(`[{"label":"` + strings.Repeat("x", types.MaxRunLinkLabelRunes+1) + `","url":"https://x.test"}]`)}},
		{"link URL bound", RunMetadataInput{Links: raw(`[{"url":"https://x.test/` + strings.Repeat("x", types.MaxRunLinkURLBytes) + `"}]`)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := applyRunMetadataInput(current, test.input)
			require.Error(t, err)
		})
	}

	got, err := applyRunMetadataInput(current, RunMetadataInput{Labels: raw(`null`)})
	require.NoError(t, err)
	require.Nil(t, got.Labels)
	require.Equal(t, current.DisplayName, got.DisplayName)
	require.Equal(t, current.Links, got.Links)
	got, err = applyRunMetadataInput(current, RunMetadataInput{Links: raw(`null`)})
	require.NoError(t, err)
	require.Nil(t, got.Links)
	require.Equal(t, current.DisplayName, got.DisplayName)
	require.Equal(t, current.Labels, got.Labels)
}

// TestRunMetadataIsExcludedFromCanonicalReplayPayload pins the guarantee that
// run_metadata never reaches the replay dedupe key. It drives the production
// helper prepareExecutionForTargetWithAdmission uses to build
// executions.input_payload, so adding run_metadata to that payload fails here.
func TestRunMetadataIsExcludedFromCanonicalReplayPayload(t *testing.T) {
	replayKey := func(req ExecuteRequest) string {
		encoded, err := json.Marshal(buildClientPayload(req))
		require.NoError(t, err)
		key, ok := canonicalReplayPayload(encoded)
		require.True(t, ok)
		return key
	}

	a := ExecuteRequest{
		Input:       map[string]interface{}{"x": 1},
		Context:     map[string]interface{}{"provider": "p"},
		RunMetadata: &RunMetadataInput{DisplayName: raw(`"A"`), Labels: raw(`["release"]`)},
	}
	b := a
	b.RunMetadata = &RunMetadataInput{DisplayName: raw(`"B"`)}
	c := a
	c.RunMetadata = nil

	require.Equal(t, replayKey(a), replayKey(b))
	require.Equal(t, replayKey(a), replayKey(c))
	require.NotContains(t, replayKey(a), "run_metadata")
	require.NotContains(t, replayKey(a), "release")

	// A differing input must still change the key — proving the equality above
	// is not just an inert constant.
	d := a
	d.Input = map[string]interface{}{"x": 2}
	require.NotEqual(t, replayKey(a), replayKey(d))
}

func TestExecuteHandler_RunMetadataDifferenceStillReturnsReplayHit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var agentCalls int
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":42}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second, ""))
	execute := func(runID, body string, headers map[string]string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("X-Run-ID", runID)
		for key, value := range headers {
			request.Header.Set(key, value)
		}
		router.ServeHTTP(recorder, request)
		return recorder
	}

	first := execute("source-run", `{"input":{"foo":"bar"},"context":{"provider":"p"},"run_metadata":{"display_name":"First"}}`, nil)
	require.Equal(t, http.StatusOK, first.Code, first.Body.String())
	var firstResponse ExecuteResponse
	require.NoError(t, json.Unmarshal(first.Body.Bytes(), &firstResponse))

	second := execute("replay-run", `{"input":{"foo":"bar"},"context":{"provider":"p"},"run_metadata":{"display_name":"Second"}}`, map[string]string{
		"X-Parent-Execution-ID":             "new-parent",
		"X-AgentField-Replay-Source-Run-ID": "source-run",
		"X-AgentField-Replay-Mode":          "all-succeeded",
	})
	require.Equal(t, http.StatusOK, second.Code, second.Body.String())
	require.Equal(t, firstResponse.ExecutionID, second.Header().Get("X-AgentField-Replay-Hit"))
	require.Equal(t, 1, agentCalls)
}

func TestSetRunMetadataHandlerRoundTripAndRejectsBeforeWrite(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store, ctx := setupTestStorage(t)
	getter := store.(interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	runID := "run-metadata-handler"
	require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-metadata-handler", RunID: runID, AgentNodeID: "node",
		ReasonerID: "reasoner", Status: "succeeded",
	}))
	router := gin.New()
	router.POST("/runs/:run_id/metadata", SetRunMetadataHandler(store))

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata",
		bytes.NewBufferString(`{"display_name":" Release ","labels":["one","one"],"links":[{"label":"PR","url":"https://x.test/pr"}]}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Actor-ID", "tester")
	router.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	stored, err := getter.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	require.Equal(t, "Release", types.ParseRunMetadata(stored.Metadata).DisplayName)
	require.Equal(t, "tester", types.ParseRunMetadata(stored.Metadata).SetBy)

	before := string(stored.Metadata)
	recorder = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata",
		bytes.NewBufferString(`{"links":[{"url":"javascript:alert(1)"}]}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	stored, err = getter.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	require.Equal(t, before, string(stored.Metadata))

	recorder = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodPost, "/runs/missing/metadata", bytes.NewBufferString(`{"labels":["x"]}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusNotFound, recorder.Code)
	missing, err := getter.GetWorkflowRun(ctx, "missing")
	require.NoError(t, err)
	require.Nil(t, missing)
}

func TestSetRunMetadataHandlerBoundsBodyAndActorBeforeWrite(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store, ctx := setupTestStorage(t)
	getter := store.(interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	runID := "run-metadata-bounds"
	require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-metadata-bounds", RunID: runID, AgentNodeID: "node",
		ReasonerID: "reasoner", Status: "succeeded",
	}))
	router := gin.New()
	router.POST("/runs/:run_id/metadata", SetRunMetadataHandler(store))

	oversizedBody := `{"display_name":"ok","padding":"` + strings.Repeat("x", int(maxRunMetadataRequestBytes)) + `"}`
	for _, test := range []struct {
		name    string
		chunked bool
	}{
		{name: "content length"},
		{name: "chunked", chunked: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata", strings.NewReader(oversizedBody))
			request.Header.Set("Content-Type", "application/json")
			if test.chunked {
				request.ContentLength = -1
				request.TransferEncoding = []string{"chunked"}
			}
			router.ServeHTTP(recorder, request)
			require.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code, recorder.Body.String())
			require.JSONEq(t, `{"error":"request body too large"}`, recorder.Body.String())
		})
	}

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata", strings.NewReader(`{"display_name":"Release"}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Actor-ID", strings.Repeat("a", types.MaxRunMetadataSetByRunes+1))
	router.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusBadRequest, recorder.Code, recorder.Body.String())
	require.Contains(t, recorder.Body.String(), "X-Actor-ID exceeds")

	stored, err := getter.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	require.Nil(t, stored)
}

func TestNormalizeRunMetadataActor(t *testing.T) {
	actor, err := normalizeRunMetadataActor("   ")
	require.NoError(t, err)
	require.Equal(t, "api", actor)

	actor, err = normalizeRunMetadataActor(" release-bot ")
	require.NoError(t, err)
	require.Equal(t, "release-bot", actor)

	_, err = normalizeRunMetadataActor(strings.Repeat("漢", types.MaxRunMetadataSetByRunes+1))
	require.ErrorContains(t, err, "X-Actor-ID exceeds")
	_, err = normalizeRunMetadataActor(string([]byte{0xff}))
	require.ErrorContains(t, err, "valid UTF-8")
}

type executionStoreWithoutMetadataWriter struct{ storage.StorageProvider }

func TestSetRunMetadataHandlerBranchesAndPartialUpdates(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store, ctx := setupTestStorage(t)
	getter := store.(interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	runID := "run-metadata-branches"
	require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: "exec-metadata-branches", RunID: runID, AgentNodeID: "node", ReasonerID: "reasoner", Status: "succeeded"}))
	require.NoError(t, store.(workflowRunMetadataWriter).UpdateWorkflowRunMetadata(ctx, runID, func(namespaces map[string]json.RawMessage) error {
		namespaces["lineage"] = raw(`{"kind":"fork"}`)
		return nil
	}))

	router := gin.New()
	router.POST("/runs/:run_id/metadata", SetRunMetadataHandler(store))
	post := func(body string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata", bytes.NewBufferString(body))
		request.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(recorder, request)
		return recorder
	}

	invalid := []string{
		`{"display_name":5}`, `{"labels":"x"}`, `{"links":5}`,
		`{"links":[{"label":"` + strings.Repeat("x", types.MaxRunLinkLabelRunes+1) + `","url":"https://x.test"}]}`,
		`{"links":[{"url":"https://x.test/` + strings.Repeat("x", types.MaxRunLinkURLBytes) + `"}]}`,
		`{`,
	}
	for _, body := range invalid {
		require.Equal(t, http.StatusBadRequest, post(body).Code, body)
	}

	require.Equal(t, http.StatusOK, post(`{"display_name":"Release","labels":["one"],"links":[{"label":"PR","url":"https://x.test/pr"}]}`).Code)
	require.Equal(t, http.StatusOK, post(`{"labels":["two"]}`).Code)
	stored, err := getter.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	metadata := types.ParseRunMetadata(stored.Metadata)
	require.Equal(t, "Release", metadata.DisplayName)
	require.Equal(t, []string{"two"}, metadata.Labels)
	require.Equal(t, "https://x.test/pr", metadata.Links[0].URL)

	require.Equal(t, http.StatusOK, post(`{"display_name":null,"labels":null,"links":null}`).Code)
	stored, err = getter.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	var namespaces map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(stored.Metadata, &namespaces))
	require.NotContains(t, namespaces, types.RunMetadataNamespace)
	require.Contains(t, namespaces, "lineage")

	unsupported := gin.New()
	unsupported.POST("/runs/:run_id/metadata", SetRunMetadataHandler(executionStoreWithoutMetadataWriter{store}))
	recorder := httptest.NewRecorder()
	unsupported.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata", bytes.NewBufferString(`{}`)))
	require.Equal(t, http.StatusNotImplemented, recorder.Code)
}

func TestSetRunMetadataHandlerRejectsCapAndURLMatrixWithoutWriting(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store, ctx := setupTestStorage(t)
	getter := store.(interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	runID := "run-metadata-negative-matrix"
	require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: "exec-negative-matrix", RunID: runID, AgentNodeID: "node", ReasonerID: "reasoner", Status: "succeeded"}))
	require.NoError(t, store.(workflowRunMetadataWriter).UpdateWorkflowRunMetadata(ctx, runID, func(namespaces map[string]json.RawMessage) error {
		namespaces["lineage"] = raw(`{ "kind" : "fork" }`)
		return nil
	}))

	router := gin.New()
	router.POST("/runs/:run_id/metadata", SetRunMetadataHandler(store))
	labels := strings.TrimSuffix(strings.Repeat(`"x",`, types.MaxRunLabels+1), ",")
	links := strings.TrimSuffix(strings.Repeat(`{"url":"https://x.test"},`, types.MaxRunLinks+1), ",")
	tests := []struct {
		name string
		body string
	}{
		{"display name over cap", `{"display_name":"` + strings.Repeat("x", types.MaxRunDisplayNameRunes+1) + `"}`},
		{"labels over cap", `{"labels":[` + labels + `]}`},
		{"label over cap", `{"labels":["` + strings.Repeat("x", types.MaxRunLabelRunes+1) + `"]}`},
		{"links over cap", `{"links":[` + links + `]}`},
		{"url over cap", `{"links":[{"url":"https://x.test/` + strings.Repeat("x", types.MaxRunLinkURLBytes) + `"}]}`},
		{"javascript URL", `{"links":[{"url":"javascript:alert(1)"}]}`},
		{"data URL", `{"links":[{"url":"data:text/html,hello"}]}`},
		{"file URL", `{"links":[{"url":"file:///etc/passwd"}]}`},
		{"scheme-less URL", `{"links":[{"url":"example.com/path"}]}`},
		{"credentialed URL", `{"links":[{"url":"https://user:pass@example.com/path"}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before, err := getter.GetWorkflowRun(ctx, runID)
			require.NoError(t, err)
			beforeRaw := string(before.Metadata)
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/metadata", bytes.NewBufferString(test.body))
			request.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(recorder, request)
			require.GreaterOrEqual(t, recorder.Code, 400, recorder.Body.String())
			require.Less(t, recorder.Code, 500, recorder.Body.String())
			after, err := getter.GetWorkflowRun(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, beforeRaw, string(after.Metadata))
		})
	}
}

func TestPersistRestartLineageRacingRunMetadataPreservesNamespacesAndState(t *testing.T) {
	store, ctx := setupTestStorage(t)
	runID := "restart-lineage-race"
	now := time.Now().UTC()
	runStore := store.(interface {
		StoreWorkflowRun(context.Context, *types.WorkflowRun) error
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	require.NoError(t, runStore.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID: runID, RootWorkflowID: "root", Status: "failed", TotalSteps: 9,
		CompletedSteps: 6, FailedSteps: 3, StateVersion: 17, LastEventSequence: 23,
		Metadata: json.RawMessage(`{"golden":{ "name" : "G" }}`), CreatedAt: now, UpdatedAt: now,
	}))
	controller := newExecutionController(store, nil, nil, 0, "")
	plan := &preparedExecution{exec: &types.Execution{ExecutionID: "restart-exec", RunID: runID}}
	source := &types.Execution{ExecutionID: "source-exec", RunID: "source-run"}
	restart := &types.Execution{ExecutionID: "restart-exec", RunID: runID}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		controller.persistRestartRunMetadata(ctx, plan, source, restart, "workflow", "succeeded-before", "restart", "because")
	}()
	go func() {
		defer wg.Done()
		<-start
		controller.persistExecuteRunMetadata(ctx, runID, RunMetadataInput{DisplayName: raw(`"Release"`)}, nil)
	}()
	close(start)
	wg.Wait()

	run, err := runStore.GetWorkflowRun(ctx, runID)
	require.NoError(t, err)
	require.Equal(t, "failed", run.Status)
	require.Equal(t, 9, run.TotalSteps)
	require.Equal(t, 6, run.CompletedSteps)
	require.Equal(t, 3, run.FailedSteps)
	require.Equal(t, int64(17), run.StateVersion)
	require.Equal(t, int64(23), run.LastEventSequence)
	var namespaces map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(run.Metadata, &namespaces))
	require.Contains(t, namespaces, "lineage")
	require.Contains(t, namespaces, types.RunMetadataNamespace)
	require.Equal(t, `{ "name" : "G" }`, string(namespaces["golden"]))
}

func TestPersistExecuteRunMetadataAndRootGuard(t *testing.T) {
	store, ctx := setupTestStorage(t)
	getter := store.(interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
	})
	controller := newExecutionController(store, nil, nil, 0, "")
	seed := func(t *testing.T, runID string) {
		require.NoError(t, store.CreateExecutionRecord(ctx, &types.Execution{ExecutionID: "exec-" + runID, RunID: runID, AgentNodeID: "node", ReasonerID: "reasoner", Status: "succeeded"}))
	}

	t.Run("root persists and merges", func(t *testing.T) {
		seed(t, "root")
		actor := " user "
		controller.persistExecuteRunMetadata(ctx, "root", RunMetadataInput{DisplayName: raw(`"Release"`), Labels: raw(`["one"]`)}, &actor)
		controller.persistExecuteRunMetadata(ctx, "root", RunMetadataInput{Labels: raw(`["two"]`)}, nil)
		run, err := getter.GetWorkflowRun(ctx, "root")
		require.NoError(t, err)
		metadata := types.ParseRunMetadata(run.Metadata)
		require.Equal(t, "Release", metadata.DisplayName)
		require.Equal(t, []string{"two"}, metadata.Labels)
		require.Equal(t, "api", metadata.SetBy)
	})

	for _, actor := range []*string{nil, func() *string { value := "  "; return &value }()} {
		t.Run("api actor fallback", func(t *testing.T) {
			runID := "fallback"
			if actor != nil {
				runID += "-blank"
			}
			seed(t, runID)
			controller.persistExecuteRunMetadata(ctx, runID, RunMetadataInput{DisplayName: raw(`"X"`)}, actor)
			run, err := getter.GetWorkflowRun(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, "api", types.ParseRunMetadata(run.Metadata).SetBy)
		})
	}

	t.Run("validation error does not write", func(t *testing.T) {
		seed(t, "invalid")
		controller.persistExecuteRunMetadata(ctx, "invalid", RunMetadataInput{Labels: raw(`"bad"`)}, nil)
		run, err := getter.GetWorkflowRun(ctx, "invalid")
		require.NoError(t, err)
		if run != nil {
			require.Nil(t, types.ParseRunMetadata(run.Metadata))
		}
	})

	t.Run("oversized actor does not write", func(t *testing.T) {
		seed(t, "invalid-actor")
		actor := strings.Repeat("a", types.MaxRunMetadataSetByRunes+1)
		controller.persistExecuteRunMetadata(ctx, "invalid-actor", RunMetadataInput{DisplayName: raw(`"X"`)}, &actor)
		run, err := getter.GetWorkflowRun(ctx, "invalid-actor")
		require.NoError(t, err)
		if run != nil {
			require.Nil(t, types.ParseRunMetadata(run.Metadata))
		}
	})

	t.Run("child execute ignores metadata at root guard", func(t *testing.T) {
		bad := &RunMetadataInput{Labels: raw(`"bad"`)}
		parent := "parent"
		_, rootErr := controller.prepareExecutionForTargetWithAdmission(ctx, "missing.reasoner", ExecuteRequest{RunMetadata: bad}, executionHeaders{runID: "guard-root"}, "", "", false)
		require.ErrorContains(t, rootErr, "invalid run_metadata")
		_, childErr := controller.prepareExecutionForTargetWithAdmission(ctx, "missing.reasoner", ExecuteRequest{RunMetadata: bad}, executionHeaders{runID: "guard-child", parentExecutionID: &parent}, "", "", false)
		require.ErrorContains(t, childErr, "agent 'missing' not found")
		actor := strings.Repeat("a", types.MaxRunMetadataSetByRunes+1)
		valid := &RunMetadataInput{DisplayName: raw(`"X"`)}
		_, rootActorErr := controller.prepareExecutionForTargetWithAdmission(ctx, "missing.reasoner", ExecuteRequest{RunMetadata: valid}, executionHeaders{runID: "guard-root-actor", actorID: &actor}, "", "", false)
		require.ErrorContains(t, rootActorErr, "X-Actor-ID exceeds")
		_, childActorErr := controller.prepareExecutionForTargetWithAdmission(ctx, "missing.reasoner", ExecuteRequest{RunMetadata: valid}, executionHeaders{runID: "guard-child-actor", parentExecutionID: &parent, actorID: &actor}, "", "", false)
		require.ErrorContains(t, childActorErr, "agent 'missing' not found")
		for _, runID := range []string{"guard-root", "guard-child"} {
			run, err := getter.GetWorkflowRun(ctx, runID)
			require.NoError(t, err)
			if run != nil {
				require.Nil(t, types.ParseRunMetadata(run.Metadata))
			}
		}
	})
}
