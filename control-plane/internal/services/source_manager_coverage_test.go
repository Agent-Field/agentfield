package services

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"

	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

type sourceManagerTestLoop struct {
	ran chan struct{}
}

func (s *sourceManagerTestLoop) Name() string                   { return "source_manager_test_loop" }
func (s *sourceManagerTestLoop) Kind() sources.Kind             { return sources.KindLoop }
func (s *sourceManagerTestLoop) ConfigSchema() json.RawMessage  { return json.RawMessage(`{}`) }
func (s *sourceManagerTestLoop) SecretRequired() bool           { return false }
func (s *sourceManagerTestLoop) Validate(json.RawMessage) error { return nil }
func (s *sourceManagerTestLoop) Run(ctx context.Context, cfg json.RawMessage, secret string, emit func(sources.Event)) error {
	emit(sources.Event{
		Type:           "tick",
		IdempotencyKey: "loop-once",
	})
	close(s.ran)
	<-ctx.Done()
	return ctx.Err()
}

var registerSourceManagerTestLoop sync.Once

func setupSourceManagerStorage(t *testing.T) (*storage.LocalStorage, context.Context) {
	t.Helper()
	ctx := context.Background()
	tempDir := t.TempDir()
	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := provider.Initialize(ctx, storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	})
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5")
		}
		require.NoError(t, err)
	}
	t.Cleanup(func() { _ = provider.Close(ctx) })
	return provider, ctx
}

func TestSourceManagerLifecycleAndEmitCoverage(t *testing.T) {
	loop := &sourceManagerTestLoop{ran: make(chan struct{})}
	registerSourceManagerTestLoop.Do(func() { sources.Register(loop) })

	store, ctx := setupSourceManagerStorage(t)
	dispatcher := NewTriggerDispatcher(store, nil)
	manager := NewSourceManager(store, dispatcher)

	require.EqualError(t, manager.Start(nil), "nil trigger")
	require.Error(t, manager.Start(&types.Trigger{ID: "unknown", SourceName: "missing_source"}))
	require.Error(t, manager.Start(&types.Trigger{ID: "http", SourceName: "generic_bearer"}))
	manager.Stop("not-running")

	loopTrigger := &types.Trigger{
		ID:             "loop-trigger",
		SourceName:     loop.Name(),
		Config:         json.RawMessage(`{}`),
		TargetNodeID:   "missing-node",
		TargetReasoner: "handle",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, store.CreateTrigger(ctx, loopTrigger))
	require.NoError(t, manager.LoadAll(ctx))

	select {
	case <-loop.ran:
	case <-time.After(2 * time.Second):
		t.Fatal("loop source did not run")
	}
	manager.Stop(loopTrigger.ID)
	manager.StopAll()

	events, err := store.ListInboundEvents(ctx, loopTrigger.ID, 10)
	require.NoError(t, err)
	require.Len(t, events, 1)
	require.Equal(t, "tick", events[0].EventType)
	require.JSONEq(t, `{}`, string(events[0].RawPayload))
	require.Equal(t, types.InboundEventStatusFailed, events[0].Status)

	manager.handleEmit(ctx, loopTrigger.ID, loop.Name(), sources.Event{Type: "tick", IdempotencyKey: "loop-once"})
	events, err = store.ListInboundEvents(ctx, loopTrigger.ID, 10)
	require.NoError(t, err)
	require.Len(t, events, 1, "duplicate idempotency key should not insert a second event")

	require.NoError(t, store.SetTriggerOverride(ctx, loopTrigger.ID, false, false))
	manager.handleEmit(ctx, loopTrigger.ID, loop.Name(), sources.Event{Type: "tick", IdempotencyKey: "disabled"})
	events, err = store.ListInboundEvents(ctx, loopTrigger.ID, 10)
	require.NoError(t, err)
	require.Len(t, events, 1, "disabled trigger should not insert events")

	manager.handleEmit(ctx, "missing-trigger", loop.Name(), sources.Event{Type: "tick"})
}

func TestSourceManagerLoadAllListError(t *testing.T) {
	manager := NewSourceManager(&sourceManagerErrorStorage{}, nil)
	require.Error(t, manager.LoadAll(context.Background()))
}

type sourceManagerErrorStorage struct {
	storage.StorageProvider
}

func (s *sourceManagerErrorStorage) ListTriggers(context.Context, string, string) ([]*types.Trigger, error) {
	return nil, errors.New("list failed")
}
