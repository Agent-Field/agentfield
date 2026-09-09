package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type blockingGoldenMetadataStore struct {
	storage.StorageProvider
	metadataStore interface {
		GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error)
		UpdateWorkflowRunMetadata(context.Context, string, func(map[string]json.RawMessage) error) error
	}
	called     chan struct{}
	entered    chan struct{}
	release    <-chan struct{}
	executions []*types.Execution
}

func (s *blockingGoldenMetadataStore) QueryExecutionRecords(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
	return s.executions, nil
}

func (s *blockingGoldenMetadataStore) GetWorkflowRun(ctx context.Context, runID string) (*types.WorkflowRun, error) {
	return s.metadataStore.GetWorkflowRun(ctx, runID)
}

func (s *blockingGoldenMetadataStore) UpdateWorkflowRunMetadata(ctx context.Context, runID string, mutate func(map[string]json.RawMessage) error) error {
	close(s.called)
	return s.metadataStore.UpdateWorkflowRunMetadata(ctx, runID, func(namespaces map[string]json.RawMessage) error {
		close(s.entered)
		<-s.release
		return mutate(namespaces)
	})
}

func TestSaveGoldenRunConcurrentNamespaceUpdatesPreserveState(t *testing.T) {
	gin.SetMode(gin.TestMode)

	for _, goldenFirst := range []bool{true, false} {
		name := "run metadata commits first"
		if goldenFirst {
			name = "golden metadata commits first"
		}
		t.Run(name, func(t *testing.T) {
			ls, ctx := setupUIHandlerStorage(t)
			runID := "run-golden-concurrent"
			executionID := "exec-golden-concurrent"
			now := time.Date(2026, 8, 31, 20, 0, 0, 0, time.UTC)
			completed := now.Add(time.Second)
			require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
				ExecutionID: executionID, RunID: runID, AgentNodeID: "agent", NodeID: "agent",
				ReasonerID: "reasoner", Status: types.ExecutionStatusSucceeded,
				StartedAt: now, CompletedAt: &completed, CreatedAt: now, UpdatedAt: completed,
			}))
			require.NoError(t, ls.StoreWorkflowRun(ctx, &types.WorkflowRun{
				RunID: runID, RootWorkflowID: "root", RootExecutionID: &executionID,
				Status: "failed", TotalSteps: 9, CompletedSteps: 6, FailedSteps: 3,
				StateVersion: 17, LastEventSequence: 23,
				Metadata:  json.RawMessage(`{"lineage":{ "source_run_id" : "source" }}`),
				CreatedAt: now, UpdatedAt: now,
			}))

			goldenRelease := make(chan struct{})
			if !goldenFirst {
				close(goldenRelease)
			}
			wrapper := &blockingGoldenMetadataStore{
				StorageProvider: ls,
				metadataStore:   ls,
				called:          make(chan struct{}),
				entered:         make(chan struct{}),
				release:         goldenRelease,
				executions: []*types.Execution{{
					ExecutionID: executionID, RunID: runID, AgentNodeID: "agent", NodeID: "agent",
					ReasonerID: "reasoner", Status: types.ExecutionStatusSucceeded,
					StartedAt: now, CompletedAt: &completed, CreatedAt: now, UpdatedAt: completed,
				}},
			}
			router := gin.New()
			router.POST("/runs/:run_id/golden", NewWorkflowRunHandler(wrapper).SaveGoldenRunHandler)

			goldenDone := make(chan *httptest.ResponseRecorder, 1)
			startGolden := func() {
				go func() {
					recorder := httptest.NewRecorder()
					request := httptest.NewRequest(http.MethodPost, "/runs/"+runID+"/golden", strings.NewReader(`{"name":"Baseline","tags":["smoke"]}`))
					request.Header.Set("Content-Type", "application/json")
					router.ServeHTTP(recorder, request)
					goldenDone <- recorder
				}()
			}

			runEntered := make(chan struct{})
			runCalled := make(chan struct{})
			runRelease := make(chan struct{})
			if goldenFirst {
				close(runRelease)
			}
			runDone := make(chan error, 1)
			startRun := func() {
				go func() {
					close(runCalled)
					runDone <- ls.UpdateWorkflowRunMetadata(ctx, runID, func(namespaces map[string]json.RawMessage) error {
						close(runEntered)
						<-runRelease
						namespaces[types.RunMetadataNamespace] = json.RawMessage(`{"display_name":"Release","labels":["important"]}`)
						return nil
					})
				}()
			}

			if goldenFirst {
				startGolden()
				waitForTestSignal(t, wrapper.entered)
				startRun()
				waitForTestSignal(t, runCalled)
				close(goldenRelease)
			} else {
				startRun()
				waitForTestSignal(t, runEntered)
				startGolden()
				waitForTestSignal(t, wrapper.called)
				close(runRelease)
			}

			recorder := waitForTestValue(t, goldenDone)
			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			require.NoError(t, waitForTestValue(t, runDone))

			run, err := ls.GetWorkflowRun(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, "root", run.RootWorkflowID)
			require.Equal(t, "failed", run.Status)
			require.Equal(t, 9, run.TotalSteps)
			require.Equal(t, 6, run.CompletedSteps)
			require.Equal(t, 3, run.FailedSteps)
			require.Equal(t, int64(17), run.StateVersion)
			require.Equal(t, int64(23), run.LastEventSequence)
			var namespaces map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(run.Metadata, &namespaces))
			require.JSONEq(t, `{"source_run_id":"source"}`, string(namespaces["lineage"]))
			require.JSONEq(t, `{"display_name":"Release","labels":["important"]}`, string(namespaces[types.RunMetadataNamespace]))
			var golden GoldenRunMetadata
			require.NoError(t, json.Unmarshal(namespaces["golden"], &golden))
			require.Equal(t, "Baseline", golden.Name)
			require.Equal(t, []string{"smoke"}, golden.Tags)
		})
	}
}

func waitForTestSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for concurrent metadata operation")
	}
}

func waitForTestValue[T any](t *testing.T, values <-chan T) T {
	t.Helper()
	select {
	case value := <-values:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for concurrent metadata result")
		var zero T
		return zero
	}
}
