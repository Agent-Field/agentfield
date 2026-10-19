package storage

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestG1DIDSQLiteBusyTimeoutUnchanged(t *testing.T) {
	t.Setenv("AGENTFIELD_SQLITE_BUSY_TIMEOUT_MS", "1000")
	ls, _ := setupTestLocalStorage(t)
	ls.db.SetMaxOpenConns(1)
	var path string
	var seq int
	var name string
	require.NoError(t, ls.db.QueryRow("PRAGMA database_list").Scan(&seq, &name, &path))
	other, err := sql.Open("sqlite3", path+"?_txlock=immediate")
	require.NoError(t, err)
	defer other.Close()
	tx, err := other.Begin()
	require.NoError(t, err)
	defer tx.Rollback()
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()
	start := time.Now()
	err = ls.StoreAgentFieldServerDID(ctx, "server", "did:key:root", []byte("seed"), time.Now(), time.Now())
	require.ErrorContains(t, err, "database is locked")
	require.GreaterOrEqual(t, time.Since(start), 800*time.Millisecond, "DID writes must keep SQLite's configured 1s busy wait, even with a shorter context deadline")
	require.ErrorIs(t, ctx.Err(), context.DeadlineExceeded)
	var timeout int
	require.NoError(t, ls.db.QueryRow("PRAGMA busy_timeout").Scan(&timeout))
	require.Equal(t, 1000, timeout)
	require.NoError(t, tx.Rollback())

	ctx, cancel = context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "server", "did:key:root", []byte("seed"), time.Now(), time.Now()))
	require.NoError(t, ls.db.QueryRow("PRAGMA busy_timeout").Scan(&timeout))
	require.Equal(t, 1000, timeout, "successful DID writes must leave the pooled connection's timeout unchanged")
}
