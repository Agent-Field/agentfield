package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Contract S1: A batch put for an execution is held and read back in seq order
// after a given seq, with a limit, so a reconnecting subscriber receives
// ordered transcript chunks without missing or reordering lines.
func TestRelayStorage_S1_BatchPutAndReadAfterWithLimit(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	// Insert entries out of seq order and include both JSON and empty payloads to
	// confirm SQLite orders by seq rather than insertion order.
	batch := []RelayEntry{
		{Seq: 3, Kind: "tool_result", Data: json.RawMessage(`{"exit":0}`)},
		{Seq: 1, Kind: "assistant", Data: json.RawMessage(`{"text":"hello"}`)},
		{Seq: 4, Kind: "done"},
		{Seq: 2, Kind: "tool_use", Data: json.RawMessage(`{"name":"bash"}`)},
	}
	stored, err := ls.RelayPut(ctx, "exec-s1", batch)
	require.NoError(t, err)
	require.Equal(t, 4, stored)

	// Passing limit <= 0 uses the default ceiling and returns every held entry in
	// ascending seq order.
	all, err := ls.RelayAfter(ctx, "exec-s1", 0, 0)
	require.NoError(t, err)
	require.Len(t, all, 4)
	require.Equal(t, []int64{1, 2, 3, 4}, []int64{all[0].Seq, all[1].Seq, all[2].Seq, all[3].Seq})
	require.Equal(t, "assistant", all[0].Kind)
	require.JSONEq(t, `{"text":"hello"}`, string(all[0].Data))
	require.False(t, all[0].At.IsZero())
	require.Nil(t, all[3].Data)

	// Reading after seq 1 with limit 2 returns only seqs 2 and 3 in order.
	window, err := ls.RelayAfter(ctx, "exec-s1", 1, 2)
	require.NoError(t, err)
	require.Len(t, window, 2)
	require.Equal(t, int64(2), window[0].Seq)
	require.Equal(t, "tool_use", window[0].Kind)
	require.Equal(t, int64(3), window[1].Seq)
	require.Equal(t, "tool_result", window[1].Kind)

	// Reading after the highest seq returns an empty slice with no error.
	tail, err := ls.RelayAfter(ctx, "exec-s1", 4, 10)
	require.NoError(t, err)
	require.Empty(t, tail)
}

// Contract S2: Putting the same (execution, seq) twice keeps one row, reports
// only newly stored rows, and preserves the first entry that arrived.
func TestRelayStorage_S2_DuplicateSeqIgnoredFirstWins(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	first := []RelayEntry{
		{Seq: 1, Kind: "assistant", Data: json.RawMessage(`{"turn":"first"}`)},
		{Seq: 2, Kind: "tool_use", Data: json.RawMessage(`{"cmd":"ls"}`)},
	}
	stored, err := ls.RelayPut(ctx, "exec-s2", first)
	require.NoError(t, err)
	require.Equal(t, 2, stored)

	// Replay seq 1 and seq 2 with different words alongside a genuinely new seq 3.
	replay := []RelayEntry{
		{Seq: 1, Kind: "mutated", Data: json.RawMessage(`{"turn":"second"}`)},
		{Seq: 2, Kind: "mutated", Data: json.RawMessage(`{"cmd":"rm"}`)},
		{Seq: 3, Kind: "tool_result", Data: json.RawMessage(`{"ok":true}`)},
	}
	stored, err = ls.RelayPut(ctx, "exec-s2", replay)
	require.NoError(t, err)
	require.Equal(t, 1, stored)

	// A pure replay batch stores zero new rows.
	stored, err = ls.RelayPut(ctx, "exec-s2", []RelayEntry{{Seq: 1, Kind: "mutated"}})
	require.NoError(t, err)
	require.Equal(t, 0, stored)

	got, err := ls.RelayAfter(ctx, "exec-s2", 0, 10)
	require.NoError(t, err)
	require.Len(t, got, 3)
	require.Equal(t, "assistant", got[0].Kind)
	require.JSONEq(t, `{"turn":"first"}`, string(got[0].Data))
	require.Equal(t, "tool_use", got[1].Kind)
	require.JSONEq(t, `{"cmd":"ls"}`, string(got[1].Data))
	require.Equal(t, int64(3), got[2].Seq)
}

// Contract S3: An ack through seq N deletes every row at or below N for that
// execution only, reports how many rows it dropped, and leaves other
// executions untouched.
func TestRelayStorage_S3_AckDeletesThroughSeqForExecutionOnly(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	_, err := ls.RelayPut(ctx, "exec-a", []RelayEntry{
		{Seq: 1, Kind: "a1"},
		{Seq: 2, Kind: "a2"},
		{Seq: 3, Kind: "a3"},
		{Seq: 4, Kind: "a4"},
	})
	require.NoError(t, err)

	_, err = ls.RelayPut(ctx, "exec-b", []RelayEntry{
		{Seq: 1, Kind: "b1"},
		{Seq: 2, Kind: "b2"},
	})
	require.NoError(t, err)

	dropped, err := ls.RelayAck(ctx, "exec-a", 2)
	require.NoError(t, err)
	require.Equal(t, int64(2), dropped)

	// Repeating the same ack drops zero additional rows.
	droppedAgain, err := ls.RelayAck(ctx, "exec-a", 2)
	require.NoError(t, err)
	require.Equal(t, int64(0), droppedAgain)

	remA, err := ls.RelayAfter(ctx, "exec-a", 0, 10)
	require.NoError(t, err)
	require.Len(t, remA, 2)
	require.Equal(t, []int64{3, 4}, []int64{remA[0].Seq, remA[1].Seq})

	remB, err := ls.RelayAfter(ctx, "exec-b", 0, 10)
	require.NoError(t, err)
	require.Len(t, remB, 2)
	require.Equal(t, []int64{1, 2}, []int64{remB[0].Seq, remB[1].Seq})
}

// Contract S4: RelayPrune drops rows recorded before the cutoff and keeps rows
// recorded at or after the cutoff so unacked entries cannot grow without bound.
func TestRelayStorage_S4_PruneDropsOlderAndKeepsNewer(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	_, err := ls.RelayPut(ctx, "exec-prune", []RelayEntry{
		{Seq: 1, Kind: "old"},
		{Seq: 2, Kind: "at_cutoff"},
		{Seq: 3, Kind: "recent"},
	})
	require.NoError(t, err)

	now := time.Now().UTC().Truncate(time.Second)
	cutoff := now.Add(-1 * time.Hour)

	// Stamp deterministic recorded_at values around the cutoff so the test does
	// not depend on wall-clock sleeps.
	_, err = ls.db.ExecContext(ctx, `UPDATE execution_relay SET recorded_at = ? WHERE execution_id = ? AND seq = ?`,
		cutoff.Add(-10*time.Minute), "exec-prune", 1)
	require.NoError(t, err)
	_, err = ls.db.ExecContext(ctx, `UPDATE execution_relay SET recorded_at = ? WHERE execution_id = ? AND seq = ?`,
		cutoff, "exec-prune", 2)
	require.NoError(t, err)
	_, err = ls.db.ExecContext(ctx, `UPDATE execution_relay SET recorded_at = ? WHERE execution_id = ? AND seq = ?`,
		cutoff.Add(10*time.Minute), "exec-prune", 3)
	require.NoError(t, err)

	require.NoError(t, ls.RelayPrune(ctx, cutoff))

	kept, err := ls.RelayAfter(ctx, "exec-prune", 0, 10)
	require.NoError(t, err)
	require.Len(t, kept, 2)
	require.Equal(t, []int64{2, 3}, []int64{kept[0].Seq, kept[1].Seq})
}

// Contract S5: RelayCount matches held rows for one execution and across all
// executions, and RelayExecutions lists only executions that currently hold rows.
func TestRelayStorage_S5_CountAndExecutionsListing(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	total, err := ls.RelayCount(ctx, "")
	require.NoError(t, err)
	require.Equal(t, int64(0), total)

	execs, err := ls.RelayExecutions(ctx)
	require.NoError(t, err)
	require.Empty(t, execs)

	_, err = ls.RelayPut(ctx, "exec-beta", []RelayEntry{
		{Seq: 1, Kind: "line"},
		{Seq: 2, Kind: "line"},
		{Seq: 3, Kind: "line"},
	})
	require.NoError(t, err)
	_, err = ls.RelayPut(ctx, "exec-alpha", []RelayEntry{
		{Seq: 1, Kind: "line"},
		{Seq: 2, Kind: "line"},
	})
	require.NoError(t, err)

	alphaCount, err := ls.RelayCount(ctx, "exec-alpha")
	require.NoError(t, err)
	require.Equal(t, int64(2), alphaCount)

	betaCount, err := ls.RelayCount(ctx, "exec-beta")
	require.NoError(t, err)
	require.Equal(t, int64(3), betaCount)

	missingCount, err := ls.RelayCount(ctx, "exec-missing")
	require.NoError(t, err)
	require.Equal(t, int64(0), missingCount)

	total, err = ls.RelayCount(ctx, "")
	require.NoError(t, err)
	require.Equal(t, int64(5), total)

	execs, err = ls.RelayExecutions(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"exec-alpha", "exec-beta"}, execs)

	// Once exec-alpha acks all of its rows, it disappears from RelayExecutions.
	_, err = ls.RelayAck(ctx, "exec-alpha", 2)
	require.NoError(t, err)

	total, err = ls.RelayCount(ctx, "")
	require.NoError(t, err)
	require.Equal(t, int64(3), total)

	execs, err = ls.RelayExecutions(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"exec-beta"}, execs)
}

// Contract S6: A run's owner is set once so nested calls cannot move ownership
// to another subscriber, and RunOwner of an unknown run is empty with no error.
func TestRelayStorage_S6_RunOwnerFirstWinsAndUnknownIsEmpty(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	owner, err := ls.RunOwner(ctx, "run-unknown")
	require.NoError(t, err)
	require.Equal(t, "", owner)

	// Blank run IDs or blank owners are ignored rather than stored.
	require.NoError(t, ls.SetRunOwner(ctx, "", "workaf"))
	require.NoError(t, ls.SetRunOwner(ctx, "run-1", ""))
	owner, err = ls.RunOwner(ctx, "run-1")
	require.NoError(t, err)
	require.Equal(t, "", owner)

	require.NoError(t, ls.SetRunOwner(ctx, "run-1", "workaf-primary"))
	owner, err = ls.RunOwner(ctx, "run-1")
	require.NoError(t, err)
	require.Equal(t, "workaf-primary", owner)

	require.NoError(t, ls.SetRunOwner(ctx, "run-1", "workaf-intruder"))
	owner, err = ls.RunOwner(ctx, "run-1")
	require.NoError(t, err)
	require.Equal(t, "workaf-primary", owner)
}

// Contract S7: RunsOwnedBy lists only that owner's runs created at or after
// since ("every run owner has owned since since, oldest first").
func TestRelayStorage_S7_RunsOwnedByScopingAndSinceFilter(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	for _, pair := range []struct {
		runID string
		owner string
	}{
		{runID: "run-expired", owner: "owner-a"},
		{runID: "run-first", owner: "owner-a"},
		{runID: "run-second", owner: "owner-a"},
		{runID: "run-other", owner: "owner-b"},
	} {
		require.NoError(t, ls.SetRunOwner(ctx, pair.runID, pair.owner))
	}

	base := time.Now().UTC().Truncate(time.Second)
	cutoff := base.Add(-30 * time.Minute)
	stamps := map[string]time.Time{
		"run-expired": cutoff.Add(-15 * time.Minute),
		"run-first":   cutoff,
		"run-other":   cutoff.Add(5 * time.Minute),
		"run-second":  cutoff.Add(20 * time.Minute),
	}
	for runID, ts := range stamps {
		_, err := ls.db.ExecContext(ctx, `UPDATE run_owners SET created_at = ? WHERE run_id = ?`, ts, runID)
		require.NoError(t, err)
	}

	// Pass since in a non-UTC zone to verify RunsOwnedBy normalizes to UTC,
	// includes created_at == since, excludes older runs and other owners, and
	// orders oldest first.
	estZone := time.FixedZone("UTC-5", -5*3600)
	runsA, err := ls.RunsOwnedBy(ctx, "owner-a", cutoff.In(estZone))
	require.NoError(t, err)
	require.Equal(t, []string{"run-first", "run-second"}, runsA)

	runsB, err := ls.RunsOwnedBy(ctx, "owner-b", cutoff)
	require.NoError(t, err)
	require.Equal(t, []string{"run-other"}, runsB)

	runsNone, err := ls.RunsOwnedBy(ctx, "owner-nobody", cutoff)
	require.NoError(t, err)
	require.Empty(t, runsNone)
}

// Contract S9: Both execution_relay and run_owners are created lazily on first
// use of a store constructed without autoMigrateSchema, and propagate database
// errors when the context or store is closed.
func TestRelayStorage_S9_LazyTableCreationWithoutMigration(t *testing.T) {
	ls, ctx := newUnmigratedLocalStorage(t)

	tableExists := func(s *LocalStorage, name string) bool {
		var count int
		err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?`, name).Scan(&count)
		require.NoError(t, err)
		return count == 1
	}

	// Neither table exists on a store that skipped schema migration.
	require.False(t, tableExists(ls, "execution_relay"))
	require.False(t, tableExists(ls, "run_owners"))

	// First relay call creates execution_relay lazily without touching run_owners.
	n, err := ls.RelayCount(ctx, "")
	require.NoError(t, err)
	require.Equal(t, int64(0), n)
	require.True(t, tableExists(ls, "execution_relay"))
	require.False(t, tableExists(ls, "run_owners"))

	// First run-owner call creates run_owners lazily.
	owner, err := ls.RunOwner(ctx, "run-lazy")
	require.NoError(t, err)
	require.Equal(t, "", owner)
	require.True(t, tableExists(ls, "run_owners"))

	// Cancelled contexts after lazy creation surface query errors cleanly.
	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.RelayPut(cancelledCtx, "exec-c", []RelayEntry{{Seq: 1, Kind: "k"}})
	require.Error(t, err)
	_, err = ls.RelayAfter(cancelledCtx, "exec-c", 0, 10)
	require.Error(t, err)
	_, err = ls.RelayAck(cancelledCtx, "exec-c", 1)
	require.Error(t, err)
	require.Error(t, ls.RelayPrune(cancelledCtx, time.Now()))
	_, err = ls.RelayCount(cancelledCtx, "exec-c")
	require.Error(t, err)
	_, err = ls.RelayExecutions(cancelledCtx)
	require.Error(t, err)
	require.Error(t, ls.SetRunOwner(cancelledCtx, "run-c", "owner-c"))
	_, err = ls.RunOwner(cancelledCtx, "run-c")
	require.Error(t, err)
	_, err = ls.RunsOwnedBy(cancelledCtx, "owner-c", time.Now())
	require.Error(t, err)

	// A closed unmigrated store surfaces the lazy DDL error on its db handle.
	closedLS, closedCtx := newUnmigratedLocalStorage(t)
	require.NoError(t, closedLS.Close(closedCtx))
	_, err = closedLS.RelayPut(closedCtx, "exec-closed", []RelayEntry{{Seq: 1, Kind: "k"}})
	require.Error(t, err)
	_, err = closedLS.RelayAfter(closedCtx, "exec-closed", 0, 10)
	require.Error(t, err)
	_, err = closedLS.RelayAck(closedCtx, "exec-closed", 1)
	require.Error(t, err)
	require.Error(t, closedLS.RelayPrune(closedCtx, time.Now()))
	_, err = closedLS.RelayCount(closedCtx, "")
	require.Error(t, err)
	_, err = closedLS.RelayExecutions(closedCtx)
	require.Error(t, err)
	require.Error(t, closedLS.SetRunOwner(closedCtx, "run-closed", "owner"))
	_, err = closedLS.RunOwner(closedCtx, "run-closed")
	require.Error(t, err)
	_, err = closedLS.RunsOwnedBy(closedCtx, "owner", time.Now())
	require.Error(t, err)
}
