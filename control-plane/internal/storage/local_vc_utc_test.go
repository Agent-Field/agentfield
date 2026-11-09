package storage

import (
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// withLocalZone runs the test with time.Local set to a zone west of UTC, the
// case where a timestamp written in the host's zone sorts before a UTC bound
// of an earlier instant.
func withLocalZone(t *testing.T) {
	t.Helper()
	previous := time.Local
	time.Local = time.FixedZone("UTC-5", -5*60*60)
	t.Cleanup(func() { time.Local = previous })
}

func storedExecutionVCCreatedAt(t *testing.T, ls *LocalStorage, vcID string) string {
	t.Helper()
	var raw string
	require.NoError(t, ls.db.QueryRow(`SELECT CAST(created_at AS TEXT) FROM execution_vcs WHERE vc_id = ?`, vcID).Scan(&raw))
	return raw
}

func TestStoreExecutionVCWritesCreatedAtInUTC(t *testing.T) {
	withLocalZone(t)
	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.StoreExecutionVC(ctx, "vc-scalar", "exec-1", "wf-1", "sess-1",
		"did:issuer", "did:target", "did:caller", "in", "out", "succeeded",
		[]byte(`{}`), "sig", "", 2))
	require.NoError(t, ls.StoreExecutionVCRecord(ctx, &types.ExecutionVC{
		VCID: "vc-record", ExecutionID: "exec-2", WorkflowID: "wf-1", SessionID: "sess-1",
		IssuerDID: "did:issuer", TargetDID: "did:target", CallerDID: "did:caller",
		VCDocument: []byte(`{}`), Signature: "sig", Status: "succeeded",
		CreatedAt: time.Now(), // in the host's zone, as callers pass it
	}))

	for _, id := range []string{"vc-scalar", "vc-record"} {
		require.True(t, strings.HasSuffix(storedExecutionVCCreatedAt(t, ls, id), "+00:00"),
			"%s created_at stored as %q", id, storedExecutionVCCreatedAt(t, ls, id))
	}

	// A created_after bound a minute in the past must include both VCs.
	after := time.Now().Add(-time.Minute)
	vcs, err := ls.ListExecutionVCs(ctx, types.VCFilters{CreatedAfter: &after})
	require.NoError(t, err)
	require.Len(t, vcs, 2)

	count, err := ls.CountExecutionVCs(ctx, types.VCFilters{CreatedAfter: &after})
	require.NoError(t, err)
	require.Equal(t, 2, count)
}

func TestNormalizeExecutionVCTimestampsRewritesHostZoneRows(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	// Rows an older build wrote in the host's zone: the later instant is
	// spelled with the earlier wall clock, so it sorts first as text.
	older := time.Date(2026, 1, 2, 14, 0, 0, 0, time.UTC)
	newer := time.Date(2026, 1, 2, 10, 30, 0, 0, time.FixedZone("UTC-5", -5*60*60)) // 15:30 UTC
	for id, at := range map[string]time.Time{"vc-older": older, "vc-newer": newer} {
		_, err := ls.db.Exec(`INSERT INTO execution_vcs (vc_id, execution_id, workflow_id, session_id,
			issuer_did, target_did, caller_did, vc_document, signature, input_hash, output_hash, status, created_at)
			VALUES (?, ?, 'wf', 'sess', 'i', 't', 'c', '{}', 'sig', 'in', 'out', 'succeeded', ?)`, id, "exec-"+id, at)
		require.NoError(t, err)
	}
	_, err := ls.db.Exec(`DELETE FROM schema_migrations WHERE version = ?`, executionVCUTCMigration)
	require.NoError(t, err)

	require.NoError(t, ls.normalizeExecutionVCTimestamps(ctx))

	require.True(t, strings.HasSuffix(storedExecutionVCCreatedAt(t, ls, "vc-newer"), "+00:00"))
	vcs, err := ls.ListExecutionVCs(ctx, types.VCFilters{})
	require.NoError(t, err)
	require.Len(t, vcs, 2)
	require.Equal(t, "vc-newer", vcs[0].VCID, "newest VC must sort first")
	require.True(t, vcs[0].CreatedAt.Equal(newer), "rewrite must keep the instant, got %s", vcs[0].CreatedAt)

	// The rewrite runs once.
	var applied int
	require.NoError(t, ls.db.QueryRow(`SELECT COUNT(*) FROM schema_migrations WHERE version = ?`, executionVCUTCMigration).Scan(&applied))
	require.Equal(t, 1, applied)
	require.NoError(t, ls.normalizeExecutionVCTimestamps(ctx))
}
