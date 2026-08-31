package storage

import (
	"context"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestUpdateWorkflowRunMetadataCreatesAndMergesWithoutChangingState(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	require.NoError(t, store.UpdateWorkflowRunMetadata(ctx, "new-run", func(m map[string]json.RawMessage) error {
		m["run"] = json.RawMessage(`{"display_name":"New"}`)
		return nil
	}))
	created, err := store.GetWorkflowRun(ctx, "new-run")
	require.NoError(t, err)
	require.Equal(t, "pending", created.Status)
	require.Equal(t, "new-run", created.RootWorkflowID)

	now := time.Now().UTC()
	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID: "existing", RootWorkflowID: "root", Status: "failed", TotalSteps: 9,
		CompletedSteps: 6, FailedSteps: 3, StateVersion: 17, LastEventSequence: 23,
		Metadata:  json.RawMessage("{\n  \"lineage\": { \"source_run_id\" : \"old\" },\n  \"golden\": {\n    \"name\" : \"G\"\n  }\n}"),
		CreatedAt: now, UpdatedAt: now,
	}))
	require.NoError(t, store.UpdateWorkflowRunMetadata(ctx, "existing", func(m map[string]json.RawMessage) error {
		m[types.RunMetadataNamespace] = json.RawMessage(`{"labels":["release"]}`)
		return nil
	}))
	got, err := store.GetWorkflowRun(ctx, "existing")
	require.NoError(t, err)
	require.Equal(t, "failed", got.Status)
	require.Equal(t, 9, got.TotalSteps)
	require.Equal(t, 6, got.CompletedSteps)
	require.Equal(t, 3, got.FailedSteps)
	require.Equal(t, int64(17), got.StateVersion)
	require.Equal(t, int64(23), got.LastEventSequence)
	var namespaces map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(got.Metadata, &namespaces))
	require.Equal(t, `{ "source_run_id" : "old" }`, string(namespaces["lineage"]))
	require.Equal(t, "{\n    \"name\" : \"G\"\n  }", string(namespaces["golden"]))
	require.JSONEq(t, `{"labels":["release"]}`, string(namespaces["run"]))
}

func TestUpdateWorkflowRunMetadataConcurrentFirstWritesPreserveBothNamespaces(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	store.db.SetMaxOpenConns(2)
	start := make(chan struct{})
	errs := make(chan error, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	write := func(namespace, value string) {
		defer ready.Done()
		<-start
		errs <- store.UpdateWorkflowRunMetadata(ctx, "concurrent-run", func(metadata map[string]json.RawMessage) error {
			metadata[namespace] = json.RawMessage(value)
			return nil
		})
	}
	go write("run", `{"display_name":"Release"}`)
	go write("lineage", `{ "kind" : "fork" }`)
	close(start)
	ready.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	run, err := store.GetWorkflowRun(ctx, "concurrent-run")
	require.NoError(t, err)
	var namespaces map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(run.Metadata, &namespaces))
	require.JSONEq(t, `{"display_name":"Release"}`, string(namespaces["run"]))
	require.Equal(t, `{ "kind" : "fork" }`, string(namespaces["lineage"]))
}

func TestUpdateWorkflowRunMetadataPostgresLocksAndMerges(t *testing.T) {
	state := &scriptedSQLState{
		execs: []scriptedExecResponse{{result: driver.RowsAffected(0)}, {result: driver.RowsAffected(1)}},
		queries: []scriptedQueryResponse{{
			columns: []string{"metadata"},
			rows:    [][]driver.Value{{`{"lineage":{"kind":"fork"}}`}},
		}},
	}
	db := openScriptedSQLDB(t, state)
	store := &LocalStorage{db: newSQLDatabase(db, "postgres"), mode: "postgres"}
	require.NoError(t, store.UpdateWorkflowRunMetadata(context.Background(), "run-postgres", func(metadata map[string]json.RawMessage) error {
		metadata[types.RunMetadataNamespace] = json.RawMessage(`{"display_name":"Release"}`)
		return nil
	}))
	state.assertConsumed(t)

	state.mu.Lock()
	executedQueries := append([]string(nil), state.executedQueries...)
	queriedQueries := append([]string(nil), state.queriedQueries...)
	executedArgs := append([][]driver.NamedValue(nil), state.executedArgs...)
	state.mu.Unlock()
	require.Len(t, executedQueries, 2)
	require.Contains(t, executedQueries[0], "ON CONFLICT(run_id) DO NOTHING")
	require.Contains(t, executedQueries[0], "$1")
	require.Contains(t, executedQueries[1], "UPDATE workflow_runs SET metadata = $1")
	require.Len(t, queriedQueries, 1)
	require.True(t, strings.HasSuffix(strings.TrimSpace(queriedQueries[0]), "FOR UPDATE"), queriedQueries[0])
	require.Contains(t, queriedQueries[0], "$1")
	require.Len(t, executedArgs, 2)
	require.JSONEq(t, `{"lineage":{"kind":"fork"},"run":{"display_name":"Release"}}`, executedArgs[1][0].Value.(string))
}

func TestUpdateWorkflowRunMetadataRejectsInvalidCallsAndRollsBack(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	require.Error(t, store.UpdateWorkflowRunMetadata(ctx, "", func(map[string]json.RawMessage) error { return nil }))
	require.Error(t, store.UpdateWorkflowRunMetadata(ctx, "run", nil))
	want := errors.New("stop")
	require.ErrorIs(t, store.UpdateWorkflowRunMetadata(ctx, "run", func(m map[string]json.RawMessage) error {
		m["run"] = json.RawMessage(`{}`)
		return want
	}), want)
	got, err := store.GetWorkflowRun(ctx, "run")
	require.NoError(t, err)
	require.Nil(t, got)
}

// RUN_METADATA.md promises restarted runs do not inherit source run metadata.
func TestRunMetadataIsNotInheritedByARestartedRun(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	require.NoError(t, store.UpdateWorkflowRunMetadata(ctx, "run-a", func(metadata map[string]json.RawMessage) error {
		metadata[types.RunMetadataNamespace] = json.RawMessage(`{"display_name":"Source"}`)
		return nil
	}))
	require.NoError(t, store.StoreWorkflowRun(ctx, &types.WorkflowRun{RunID: "run-b", RootWorkflowID: "run-b", Status: "pending", CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}))
	restarted, err := store.GetWorkflowRun(ctx, "run-b")
	require.NoError(t, err)
	require.Nil(t, types.ParseRunMetadata(restarted.Metadata))
}
