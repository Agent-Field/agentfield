package storage

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestLocalLockWrappersAndSQLHelpers(t *testing.T) {
	t.Run("local lock wrappers handle success and cancellation", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		lock, err := ls.AcquireLock(ctx, "lock-1", time.Second)
		require.NoError(t, err)
		require.Nil(t, lock)

		require.NoError(t, ls.ReleaseLock(ctx, "lock-1"))

		lock, err = ls.RenewLock(ctx, "lock-1")
		require.NoError(t, err)
		require.Nil(t, lock)

		lock, err = ls.GetLockStatus(ctx, "lock-1")
		require.NoError(t, err)
		require.Nil(t, lock)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()

		_, err = ls.AcquireLock(cancelled, "lock-2", time.Second)
		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, ls.ReleaseLock(cancelled, "lock-2"), context.Canceled)
		_, err = ls.RenewLock(cancelled, "lock-2")
		require.ErrorIs(t, err, context.Canceled)
		_, err = ls.GetLockStatus(cancelled, "lock-2")
		require.ErrorIs(t, err, context.Canceled)
	})

	t.Run("query helpers rebind and stream rows", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		db := newSQLDatabase(rawDB, "local")
		_, err = db.Exec(`CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)`)
		require.NoError(t, err)
		_, err = db.Exec(`INSERT INTO items (name) VALUES (?), (?)`, "alpha", "beta")
		require.NoError(t, err)

		rows, err := db.QueryContext(context.Background(), `SELECT name FROM items ORDER BY id`)
		require.NoError(t, err)
		defer rows.Close()

		var names []string
		for rows.Next() {
			var name string
			require.NoError(t, rows.Scan(&name))
			names = append(names, name)
		}
		require.Equal(t, []string{"alpha", "beta"}, names)

		tx, err := db.BeginTx(context.Background(), nil)
		require.NoError(t, err)
		defer func() { _ = tx.Rollback() }()

		rows, err = tx.QueryContext(context.Background(), `SELECT name FROM items WHERE name = ?`, "beta")
		require.NoError(t, err)
		defer rows.Close()
		require.True(t, rows.Next())
		var name string
		require.NoError(t, rows.Scan(&name))
		require.Equal(t, "beta", name)
	})
}

func TestSQLiteVectorStoreBranches(t *testing.T) {
	newStore := func(t *testing.T) *sqliteVectorStore {
		t.Helper()

		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE memory_vectors (
				scope TEXT NOT NULL,
				scope_id TEXT NOT NULL,
				key TEXT NOT NULL,
				dimension INTEGER NOT NULL,
				embedding BLOB NOT NULL,
				metadata TEXT,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP NOT NULL,
				PRIMARY KEY (scope, scope_id, key)
			)
		`)
		require.NoError(t, err)

		return newSQLiteVectorStore(newSQLDatabase(rawDB, "local"), VectorDistanceCosine)
	}

	t.Run("delete and delete by prefix affect stored rows", func(t *testing.T) {
		store := newStore(t)
		ctx := context.Background()

		for _, record := range []*types.VectorRecord{
			{Scope: "session", ScopeID: "scope-1", Key: "doc-1", Embedding: []float32{1, 0}},
			{Scope: "session", ScopeID: "scope-1", Key: "doc-2", Embedding: []float32{0, 1}},
			{Scope: "session", ScopeID: "scope-1", Key: "note-1", Embedding: []float32{1, 1}},
		} {
			require.NoError(t, store.Set(ctx, record))
		}

		require.NoError(t, store.Delete(ctx, "session", "scope-1", "note-1"))
		record, err := store.Get(ctx, "session", "scope-1", "note-1")
		require.NoError(t, err)
		require.Nil(t, record)

		deleted, err := store.DeleteByPrefix(ctx, "session", "scope-1", "doc-")
		require.NoError(t, err)
		require.Equal(t, 2, deleted)
	})

	t.Run("context and query validation fail fast", func(t *testing.T) {
		store := newStore(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		require.ErrorIs(t, store.Delete(cancelled, "session", "scope-1", "doc-1"), context.Canceled)
		_, err := store.DeleteByPrefix(cancelled, "session", "scope-1", "doc-")
		require.ErrorIs(t, err, context.Canceled)

		_, err = store.Search(context.Background(), "session", "scope-1", nil, 1, nil)
		require.EqualError(t, err, "query embedding cannot be empty")
	})

	t.Run("corrupt rows surface decode and metadata errors", func(t *testing.T) {
		store := newStore(t)
		ctx := context.Background()

		now := time.Now().UTC()
		_, err := store.db.ExecContext(ctx, `
			INSERT INTO memory_vectors (scope, scope_id, key, dimension, embedding, metadata, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		`, "session", "scope-1", "bad-embedding", 1, []byte{1, 2, 3}, `{"kind":"doc"}`, now, now)
		require.NoError(t, err)

		_, err = store.Get(ctx, "session", "scope-1", "bad-embedding")
		require.EqualError(t, err, "decode embedding: invalid embedding length: 3")

		_, err = store.Search(ctx, "session", "scope-1", []float32{1}, 10, nil)
		require.EqualError(t, err, "decode embedding: invalid embedding length: 3")

		_, err = store.db.Exec(`DELETE FROM memory_vectors`)
		require.NoError(t, err)

		_, err = store.db.ExecContext(ctx, `
			INSERT INTO memory_vectors (scope, scope_id, key, dimension, embedding, metadata, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		`, "session", "scope-1", "bad-metadata", 1, encodeEmbedding([]float32{1}), `{"kind":`, now, now)
		require.NoError(t, err)

		_, err = store.Get(ctx, "session", "scope-1", "bad-metadata")
		require.ErrorContains(t, err, "unmarshal metadata")

		_, err = store.Search(ctx, "session", "scope-1", []float32{1}, 10, nil)
		require.ErrorContains(t, err, "unmarshal metadata")
	})
}
