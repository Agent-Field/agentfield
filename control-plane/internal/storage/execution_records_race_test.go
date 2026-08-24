package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// TestForUpdateSuffixByMode pins the dialect gate: postgres gets a row lock,
// sqlite (which serializes writers and rejects FOR UPDATE syntax) gets none.
func TestForUpdateSuffixByMode(t *testing.T) {
	require.Equal(t, " FOR UPDATE", (&sqlDatabase{mode: "postgres"}).forUpdate())
	require.Equal(t, "", (&sqlDatabase{mode: "local"}).forUpdate())
	require.Equal(t, " FOR UPDATE", (&sqlTx{mode: "postgres"}).forUpdate())
	require.Equal(t, "", (&sqlTx{mode: "local"}).forUpdate())
	var nilDB *sqlDatabase
	var nilTx *sqlTx
	require.Equal(t, "", nilDB.forUpdate())
	require.Equal(t, "", nilTx.forUpdate())
}

// TestPostgresUpdateExecutionRecord_ConcurrentRMWNoLostUpdate reproduces the
// lost-update race that stranded live builds: an execution-note write and the
// terminal status callback both run UpdateExecutionRecord (read-modify-write
// of the full row). Under READ COMMITTED, whichever transaction commits last
// used to write its stale snapshot back, silently discarding the other's
// columns — a note write reverted status "succeeded" to "running" and dropped
// the result, so the caller's poll loop never saw the reasoner finish.
//
// Contract: when two read-modify-writes of the same execution overlap, both
// effects must survive, regardless of commit order.
//
// The interleaving is made deterministic: updater A holds its transaction open
// (row read, not yet committed) while updater B runs to completion, then A
// finishes. With row locking B blocks until A commits and re-reads; without it
// B's write is clobbered when A commits (this test fails on the pre-fix code).
func TestPostgresUpdateExecutionRecord_ConcurrentRMWNoLostUpdate(t *testing.T) {
	postgresURL := os.Getenv("POSTGRES_TEST_URL")
	if postgresURL == "" {
		t.Skip("POSTGRES_TEST_URL not set, skipping postgres tests")
	}

	ctx := context.Background()
	cfg := StorageConfig{
		Mode: "postgres",
		Postgres: PostgresStorageConfig{
			DSN:          postgresURL,
			MaxOpenConns: 10,
			MaxIdleConns: 5,
		},
	}

	ls := NewPostgresStorage(PostgresStorageConfig{})
	err := ls.Initialize(ctx, cfg)
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") || strings.Contains(err.Error(), "does not exist") {
			t.Skip("PostgreSQL not available, skipping test")
		}
		require.NoError(t, err)
	}
	defer ls.Close(ctx)

	execID := fmt.Sprintf("exec-race-%d", time.Now().UnixNano())
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: execID,
		RunID:       fmt.Sprintf("run-race-%d", time.Now().UnixNano()),
		AgentNodeID: "agent-race",
		ReasonerID:  "plan",
		NodeID:      "agent-race",
		Status:      string(types.ExecutionStatusRunning),
		StartedAt:   time.Now().UTC(),
	}))

	entered := make(chan struct{})
	release := make(chan struct{})
	statusDone := make(chan error, 1)

	// Updater A: the terminal status callback (succeeded + result). It parks
	// inside the updater with the transaction open so B overlaps it.
	go func() {
		_, err := ls.UpdateExecutionRecord(ctx, execID, func(current *types.Execution) (*types.Execution, error) {
			close(entered)
			<-release
			current.Status = string(types.ExecutionStatusSucceeded)
			current.ResultPayload = json.RawMessage(`{"ok":true}`)
			now := time.Now().UTC()
			current.CompletedAt = &now
			return current, nil
		})
		statusDone <- err
	}()

	<-entered

	// Updater B: the note write, racing A. Run it with a timeout so a
	// deadlocked implementation fails the test instead of hanging it.
	noteDone := make(chan error, 1)
	go func() {
		noteCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err := ls.UpdateExecutionRecord(noteCtx, execID, func(current *types.Execution) (*types.Execution, error) {
			current.Notes = append(current.Notes, types.ExecutionNote{
				Message:   "issue_writers complete",
				Timestamp: time.Now().UTC(),
			})
			return current, nil
		})
		noteDone <- err
	}()

	// Give B time to reach the row before releasing A. With locking B is
	// parked on the SELECT; without locking B commits a stale full row.
	time.Sleep(500 * time.Millisecond)
	close(release)

	require.NoError(t, <-statusDone, "status update failed")
	require.NoError(t, <-noteDone, "note update failed")

	final, err := ls.GetExecutionRecord(ctx, execID)
	require.NoError(t, err)
	require.NotNil(t, final)
	require.Equal(t, string(types.ExecutionStatusSucceeded), final.Status,
		"terminal status was lost to the concurrent note write")
	require.NotEmpty(t, final.ResultPayload, "result payload was lost")
	require.NotNil(t, final.CompletedAt, "completed_at was lost")
	require.Len(t, final.Notes, 1, "note was lost to the concurrent status write")
}
