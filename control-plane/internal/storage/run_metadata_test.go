package storage

import (
	"encoding/json"
	"errors"
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

func TestUpdateWorkflowRunMetadataConcurrentFirstWritesPostgres(t *testing.T) {
	t.Skip("no live PostgreSQL integration harness is available in the storage test suite")
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
