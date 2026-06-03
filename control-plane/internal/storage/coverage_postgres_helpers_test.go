package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	sqlite3 "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/require"
)

var registerSQLiteNowDriver sync.Once

func openSQLiteNowDB(t *testing.T) *sql.DB {
	t.Helper()

	registerSQLiteNowDriver.Do(func() {
		sql.Register("sqlite3_with_now", &sqlite3.SQLiteDriver{
			ConnectHook: func(conn *sqlite3.SQLiteConn) error {
				return conn.RegisterFunc("NOW", func() string {
					return time.Now().UTC().Format(time.RFC3339Nano)
				}, true)
			},
		})
	})

	db, err := sql.Open("sqlite3_with_now", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestPostgresVectorStoreCoverage(t *testing.T) {
	t.Run("helper constructors and expression builders", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		store := newPostgresVectorStore(newSQLDatabase(rawDB, "postgres"), VectorDistanceCosine)
		require.NotNil(t, store)
		require.Equal(t, VectorDistanceCosine, store.metric)

		require.Equal(t, "[1.5,-2,3]", vectorLiteral([]float32{1.5, -2, 3}))

		tests := []struct {
			name         string
			metric       VectorDistanceMetric
			wantScore    string
			wantDistance string
		}{
			{
				name:         "dot",
				metric:       VectorDistanceDot,
				wantScore:    "-(mv.embedding <#> query_vec.qv)",
				wantDistance: "mv.embedding <#> query_vec.qv",
			},
			{
				name:         "l2",
				metric:       VectorDistanceL2,
				wantScore:    "-(mv.embedding <-> query_vec.qv)",
				wantDistance: "mv.embedding <-> query_vec.qv",
			},
			{
				name:         "cosine",
				metric:       VectorDistanceCosine,
				wantScore:    "1 - (mv.embedding <=> query_vec.qv)",
				wantDistance: "mv.embedding <=> query_vec.qv",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				scoreExpr, distanceExpr := buildPostgresVectorExpressions(tc.metric, "query_vec.qv")
				require.Equal(t, tc.wantScore, scoreExpr)
				require.Equal(t, tc.wantDistance, distanceExpr)
			})
		}
	})

	t.Run("set validates context and metadata marshal errors", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		store := newPostgresVectorStore(newSQLDatabase(rawDB, "postgres"), VectorDistanceCosine)

		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		err = store.Set(cancelled, &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "scope-1",
			Key:       "key-1",
			Embedding: []float32{1},
		})
		require.ErrorIs(t, err, context.Canceled)

		err = store.Set(context.Background(), &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "scope-1",
			Key:       "key-1",
			Embedding: []float32{1},
			Metadata:  map[string]interface{}{"bad": make(chan int)},
		})
		require.ErrorContains(t, err, "marshal metadata")
	})

	t.Run("db-backed operations surface postgres-specific query errors", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		store := newPostgresVectorStore(newSQLDatabase(rawDB, "postgres"), VectorDistanceDot)
		ctx := context.Background()

		_, err = store.Get(ctx, "session", "scope-1", "missing")
		require.ErrorContains(t, err, "get postgres vector")

		err = store.Delete(ctx, "session", "scope-1", "missing")
		require.Error(t, err)

		_, err = store.DeleteByPrefix(ctx, "session", "scope-1", "doc-")
		require.Error(t, err)

		_, err = store.Search(ctx, "session", "scope-1", []float32{1}, 2, map[string]interface{}{"bad": make(chan int)})
		require.ErrorContains(t, err, "marshal filter")

		_, err = store.Search(ctx, "session", "scope-1", []float32{1}, 2, map[string]interface{}{"kind": "doc"})
		require.ErrorContains(t, err, "query postgres vectors")
	})

	t.Run("context validation short-circuits all read paths", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		store := newPostgresVectorStore(newSQLDatabase(rawDB, "postgres"), VectorDistanceL2)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		_, err = store.Get(cancelled, "session", "scope-1", "key-1")
		require.ErrorIs(t, err, context.Canceled)

		err = store.Delete(cancelled, "session", "scope-1", "key-1")
		require.ErrorIs(t, err, context.Canceled)

		_, err = store.DeleteByPrefix(cancelled, "session", "scope-1", "prefix-")
		require.ErrorIs(t, err, context.Canceled)

		_, err = store.Search(cancelled, "session", "scope-1", []float32{1}, 1, nil)
		require.ErrorIs(t, err, context.Canceled)

		_, err = store.Search(context.Background(), "session", "scope-1", nil, 1, nil)
		require.EqualError(t, err, "query embedding cannot be empty")
	})
}

func TestPostgresLockHelpersCoverage(t *testing.T) {
	newPostgresLocksStorage := func(t *testing.T) *LocalStorage {
		t.Helper()

		rawDB := openSQLiteNowDB(t)

		_, err := rawDB.Exec(`
			CREATE TABLE distributed_locks (
				lock_id TEXT PRIMARY KEY,
				key TEXT NOT NULL UNIQUE,
				owner TEXT NOT NULL,
				expires_at TIMESTAMP NOT NULL,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP NOT NULL
			)
		`)
		require.NoError(t, err)

		return &LocalStorage{
			db:   newSQLDatabase(rawDB, "postgres"),
			mode: "postgres",
		}
	}

	t.Run("wrapper methods dispatch into postgres helpers", func(t *testing.T) {
		ls := newPostgresLocksStorage(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := ls.AcquireLock(cancelled, "lock-1", time.Second)
		require.ErrorIs(t, err, context.Canceled)

		err = ls.ReleaseLock(cancelled, "lock-1")
		require.ErrorIs(t, err, context.Canceled)

		_, err = ls.RenewLock(cancelled, "lock-1")
		require.ErrorIs(t, err, context.Canceled)

		_, err = ls.GetLockStatus(cancelled, "lock-1")
		require.ErrorIs(t, err, context.Canceled)
	})

	t.Run("postgres lock helpers cover not found and generic error branches", func(t *testing.T) {
		ls := newPostgresLocksStorage(t)
		ctx := context.Background()

		err := ls.releaseLockPostgres(ctx, "missing-lock")
		require.EqualError(t, err, "lock 'missing-lock' not found")

		lock, err := ls.getLockStatusPostgres(ctx, "missing-key")
		require.NoError(t, err)
		require.Nil(t, lock)

		_, err = ls.renewLockPostgres(ctx, "missing-lock")
		require.EqualError(t, err, "lock 'missing-lock' not found")

		require.NoError(t, ls.db.DB.Close())

		_, err = ls.acquireLockPostgres(ctx, "lock-closed", 0)
		require.ErrorContains(t, err, "failed to acquire postgres lock")

		err = ls.releaseLockPostgres(ctx, "lock-closed")
		require.ErrorContains(t, err, "failed to release postgres lock")

		_, err = ls.renewLockPostgres(ctx, "lock-closed")
		require.ErrorContains(t, err, "failed to renew postgres lock")

		_, err = ls.getLockStatusPostgres(ctx, "lock-closed")
		require.ErrorContains(t, err, "failed to get postgres lock status")
	})

	t.Run("postgres lock helpers cover success and already-held paths", func(t *testing.T) {
		ls := newPostgresLocksStorage(t)
		ctx := context.Background()

		lock, err := ls.acquireLockPostgres(ctx, "lock-live", 0)
		require.NoError(t, err)
		require.NotNil(t, lock)
		require.Equal(t, "lock-live", lock.Key)

		status, err := ls.getLockStatusPostgres(ctx, "lock-live")
		require.NoError(t, err)
		require.NotNil(t, status)
		require.Equal(t, lock.LockID, status.LockID)

		renewed, err := ls.renewLockPostgres(ctx, lock.LockID)
		require.NoError(t, err)
		require.Equal(t, lock.LockID, renewed.LockID)

		_, err = ls.db.ExecContext(ctx, `UPDATE distributed_locks SET expires_at = ? WHERE lock_id = ?`, "9999-12-31T23:59:59Z", lock.LockID)
		require.NoError(t, err)
		_, err = ls.acquireLockPostgres(ctx, "lock-live", time.Second)
		require.EqualError(t, err, "lock 'lock-live' is already held")

		require.NoError(t, ls.releaseLockPostgres(ctx, lock.LockID))
		status, err = ls.getLockStatusPostgres(ctx, "lock-live")
		require.NoError(t, err)
		require.Nil(t, status)
	})
}

func TestStorageFactoryCoverageBranches(t *testing.T) {
	factory := &StorageFactory{}

	t.Run("default mode uses local storage and normalizes vector config", func(t *testing.T) {
		tempDir := t.TempDir()
		store, cache, err := factory.CreateStorage(StorageConfig{
			Local: LocalStorageConfig{
				DatabasePath: tempDir + "/agentfield.db",
				KVStorePath:  tempDir + "/agentfield.bolt",
			},
		})
		require.NoError(t, err)
		require.NotNil(t, store)
		require.NotNil(t, cache)
		t.Cleanup(func() { _ = store.Close(context.Background()) })
	})

	t.Run("local initialization failures are wrapped", func(t *testing.T) {
		store, cache, err := factory.CreateStorage(StorageConfig{
			Mode: "local",
			Local: LocalStorageConfig{
				DatabasePath: "/",
				KVStorePath:  "/",
			},
		})
		require.Nil(t, store)
		require.Nil(t, cache)
		require.ErrorContains(t, err, "failed to initialize local storage")
	})

	t.Run("postgres initialization failures are wrapped", func(t *testing.T) {
		store, cache, err := factory.CreateStorage(StorageConfig{
			Mode: "postgres",
			Postgres: PostgresStorageConfig{
				Host:     "127.0.0.1",
				Port:     1,
				Database: "missing",
				User:     "missing",
				Password: "missing",
				SSLMode:  "disable",
			},
		})
		require.Nil(t, store)
		require.Nil(t, cache)
		require.ErrorContains(t, err, "failed to initialize postgres storage")
	})
}

func TestSQLiteVectorStoreAdditionalBranches(t *testing.T) {
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

	store := newSQLiteVectorStore(newSQLDatabase(rawDB, "local"), VectorDistanceCosine)
	ctx := context.Background()

	require.NoError(t, store.Set(ctx, &types.VectorRecord{
		Scope:     "session",
		ScopeID:   "scope-1",
		Key:       "valid",
		Embedding: []float32{1, 0},
		Metadata:  map[string]interface{}{"kind": "doc"},
	}))

	_, err = rawDB.Exec(`
		INSERT INTO memory_vectors (scope, scope_id, key, dimension, embedding, metadata, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, "session", "scope-1", "dim-mismatch", 3, encodeEmbedding([]float32{1, 2, 3}), string(mustJSON(t, map[string]string{"kind": "doc"})), time.Now().UTC(), time.Now().UTC())
	require.NoError(t, err)

	results, err := store.Search(ctx, "session", "scope-1", []float32{1, 0}, 10, map[string]interface{}{"kind": "doc"})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "valid", results[0].Key)
}

func TestVectorWrappersAndInitializationCoverage(t *testing.T) {
	t.Run("initialize vector store honors mode and enabled flag", func(t *testing.T) {
		disabled := false
		ls := &LocalStorage{vectorConfig: VectorStoreConfig{Enabled: &disabled}}
		require.NoError(t, ls.initializeVectorStore())
		require.Nil(t, ls.vectorStore)

		ls = &LocalStorage{
			db:           &sqlDatabase{},
			mode:         "postgres",
			vectorConfig: VectorStoreConfig{},
			vectorMetric: VectorDistanceDot,
		}
		require.NoError(t, ls.initializeVectorStore())
		_, ok := ls.vectorStore.(*postgresVectorStore)
		require.True(t, ok)

		ls = &LocalStorage{
			db:           &sqlDatabase{},
			mode:         "local",
			vectorConfig: VectorStoreConfig{},
			vectorMetric: VectorDistanceL2,
		}
		require.NoError(t, ls.initializeVectorStore())
		_, ok = ls.vectorStore.(*sqliteVectorStore)
		require.True(t, ok)
	})

	t.Run("vector wrapper methods enforce store availability", func(t *testing.T) {
		ls := &LocalStorage{}
		require.EqualError(t, ls.requireVectorStore(), "vector store is not initialized")

		ls.vectorConfig = VectorStoreConfig{}
		require.EqualError(t, ls.requireVectorStore(), "vector store is not initialized")

		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		require.ErrorIs(t, ls.SetVector(cancelled, &types.VectorRecord{}), context.Canceled)
		_, err := ls.GetVector(cancelled, "scope", "scope-1", "key")
		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, ls.DeleteVector(cancelled, "scope", "scope-1", "key"), context.Canceled)
		_, err = ls.DeleteVectorsByPrefix(cancelled, "scope", "scope-1", "pre")
		require.ErrorIs(t, err, context.Canceled)
		_, err = ls.SimilaritySearch(cancelled, "scope", "scope-1", []float32{1}, 1, nil)
		require.ErrorIs(t, err, context.Canceled)

		ctx := context.Background()
		require.EqualError(t, ls.SetVector(ctx, &types.VectorRecord{}), "vector store is not initialized")
		_, err = ls.GetVector(ctx, "scope", "scope-1", "key")
		require.EqualError(t, err, "vector store is not initialized")
		require.EqualError(t, ls.DeleteVector(ctx, "scope", "scope-1", "key"), "vector store is not initialized")
		_, err = ls.DeleteVectorsByPrefix(ctx, "scope", "scope-1", "pre")
		require.EqualError(t, err, "vector store is not initialized")
		_, err = ls.SimilaritySearch(ctx, "scope", "scope-1", []float32{1}, 1, nil)
		require.EqualError(t, err, "vector store is not initialized")
	})
}

func TestPostgresMemoryHelpersCoverage(t *testing.T) {
	newMemoryStorage := func(t *testing.T) *LocalStorage {
		t.Helper()

		rawDB := openSQLiteNowDB(t)

		_, err := rawDB.Exec(`
			CREATE TABLE kv_store (
				scope TEXT NOT NULL,
				scope_id TEXT NOT NULL,
				key TEXT NOT NULL,
				value BLOB NOT NULL,
				updated_at TIMESTAMP,
				PRIMARY KEY (scope, scope_id, key)
			)
		`)
		require.NoError(t, err)

		return &LocalStorage{
			db:    newSQLDatabase(rawDB, "postgres"),
			mode:  "postgres",
			cache: &sync.Map{},
		}
	}

	t.Run("set memory validates context and marshal errors", func(t *testing.T) {
		ls := newMemoryStorage(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		err := ls.setMemoryPostgres(cancelled, &types.Memory{})
		require.EqualError(t, err, "context cancelled before postgres SetMemory operation: context canceled")

		err = ls.setMemoryPostgres(context.Background(), &types.Memory{
			Scope:   "session",
			ScopeID: "scope-1",
			Key:     "key-1",
			Data:    json.RawMessage(`{"broken"`),
		})
		require.ErrorContains(t, err, "failed to marshal memory payload")

		memory := &types.Memory{
			Scope:   "session",
			ScopeID: "scope-1",
			Key:     "good-set",
			Data:    json.RawMessage(`{"ok":true}`),
		}
		require.NoError(t, ls.setMemoryPostgres(context.Background(), memory))
		cached, ok := ls.cache.Load("session:scope-1:good-set")
		require.True(t, ok)
		require.Equal(t, memory, cached)
	})

	t.Run("get memory covers cache hit db lookup and payload errors", func(t *testing.T) {
		ls := newMemoryStorage(t)
		ctx := context.Background()

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.getMemoryPostgres(cancelled, "session", "scope-1", "key-1")
		require.EqualError(t, err, "context cancelled before postgres GetMemory operation: context canceled")

		cached := &types.Memory{Scope: "session", ScopeID: "scope-1", Key: "cached"}
		ls.cache.Store("session:scope-1:cached", cached)
		got, err := ls.getMemoryPostgres(ctx, "session", "scope-1", "cached")
		require.NoError(t, err)
		require.Same(t, cached, got)

		_, err = ls.getMemoryPostgres(ctx, "session", "scope-1", "missing")
		require.EqualError(t, err, "memory with key 'missing' not found in scope 'session' for ID 'scope-1'")

		_, err = ls.db.ExecContext(ctx, `INSERT INTO kv_store(scope, scope_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)`,
			"session", "scope-1", "bad", []byte(`{"scope":"session","scope_id":"scope-1","key":"bad","data":{"broken"}`), time.Now().UTC())
		require.NoError(t, err)
		_, err = ls.getMemoryPostgres(ctx, "session", "scope-1", "bad")
		require.ErrorContains(t, err, "failed to unmarshal postgres memory payload")

		payload, err := json.Marshal(&types.Memory{
			Scope:   "session",
			ScopeID: "scope-1",
			Key:     "good",
			Data:    json.RawMessage(`{"ok":true}`),
		})
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `INSERT INTO kv_store(scope, scope_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)`,
			"session", "scope-1", "good", payload, time.Now().UTC())
		require.NoError(t, err)

		got, err = ls.getMemoryPostgres(ctx, "session", "scope-1", "good")
		require.NoError(t, err)
		require.Equal(t, "good", got.Key)
	})

	t.Run("delete and list memory cover success and failure paths", func(t *testing.T) {
		ls := newMemoryStorage(t)
		ctx := context.Background()

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		err := ls.deleteMemoryPostgres(cancelled, "session", "scope-1", "key-1")
		require.EqualError(t, err, "context cancelled before postgres DeleteMemory operation: context canceled")
		_, err = ls.listMemoryPostgres(cancelled, "session", "scope-1")
		require.EqualError(t, err, "context cancelled before postgres ListMemory operation: context canceled")

		err = ls.deleteMemoryPostgres(ctx, "session", "scope-1", "missing")
		require.EqualError(t, err, "memory with key 'missing' not found in scope 'session' for ID 'scope-1'")

		payload, err := json.Marshal(&types.Memory{
			Scope:   "session",
			ScopeID: "scope-1",
			Key:     "delete-me",
			Data:    json.RawMessage(`{"ok":true}`),
		})
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `INSERT INTO kv_store(scope, scope_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)`,
			"session", "scope-1", "delete-me", payload, time.Now().UTC())
		require.NoError(t, err)
		ls.cache.Store("session:scope-1:delete-me", &types.Memory{Key: "delete-me"})
		require.NoError(t, ls.deleteMemoryPostgres(ctx, "session", "scope-1", "delete-me"))
		_, ok := ls.cache.Load("session:scope-1:delete-me")
		require.False(t, ok)

		_, err = ls.db.ExecContext(ctx, `INSERT INTO kv_store(scope, scope_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)`,
			"session", "scope-1", "bad", []byte(`{`), time.Now().UTC())
		require.NoError(t, err)
		_, err = ls.listMemoryPostgres(ctx, "session", "scope-1")
		require.ErrorContains(t, err, "failed to unmarshal postgres memory payload")

		_, err = ls.db.ExecContext(ctx, `DELETE FROM kv_store WHERE key = ?`, "bad")
		require.NoError(t, err)
		goodPayload, err := json.Marshal(&types.Memory{
			Scope:   "session",
			ScopeID: "scope-1",
			Key:     "list-good",
			Data:    json.RawMessage(`{"ok":true}`),
		})
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `INSERT INTO kv_store(scope, scope_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)`,
			"session", "scope-1", "list-good", goodPayload, time.Now().UTC())
		require.NoError(t, err)

		memories, err := ls.listMemoryPostgres(ctx, "session", "scope-1")
		require.NoError(t, err)
		require.NotEmpty(t, memories)
	})
}

func TestPostgresEventHelpersCoverage(t *testing.T) {
	newEventStorage := func(t *testing.T) *LocalStorage {
		t.Helper()

		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE memory_events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				scope TEXT NOT NULL,
				scope_id TEXT NOT NULL,
				key TEXT NOT NULL,
				event_type TEXT,
				action TEXT,
				data BLOB,
				previous_data BLOB,
				metadata BLOB,
				timestamp TIMESTAMP NOT NULL
			)
		`)
		require.NoError(t, err)

		return &LocalStorage{
			db:    newSQLDatabase(rawDB, "postgres"),
			mode:  "postgres",
			cache: &sync.Map{},
		}
	}

	t.Run("store event handles cancellation and successful insert", func(t *testing.T) {
		ls := newEventStorage(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		err := ls.storeEventPostgres(cancelled, &types.MemoryChangeEvent{})
		require.EqualError(t, err, "context cancelled during store event: context canceled")

		event := &types.MemoryChangeEvent{
			Scope:     "session",
			ScopeID:   "scope-1",
			Key:       "doc-1",
			Type:      "memory.updated",
			Action:    "set",
			Data:      json.RawMessage(`{"value":1}`),
			Timestamp: time.Now().UTC(),
		}
		require.NoError(t, ls.storeEventPostgres(context.Background(), event))
		require.NotEmpty(t, event.ID)
		require.False(t, event.Timestamp.IsZero())
	})

	t.Run("history query covers errors filtering limit and bad metadata", func(t *testing.T) {
		ls := newEventStorage(t)
		ctx := context.Background()

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.getEventHistoryPostgres(cancelled, types.EventFilter{})
		require.EqualError(t, err, "context cancelled during get event history: context canceled")

		since := time.Now().UTC().Add(-time.Hour)
		filter := types.EventFilter{
			Scope:    strPtr("session"),
			ScopeID:  strPtr("scope-1"),
			Since:    &since,
			Patterns: []string{"doc-*"},
			Limit:    1,
		}

		_, err = ls.db.ExecContext(ctx, `
			INSERT INTO memory_events(scope, scope_id, key, event_type, action, data, previous_data, metadata, timestamp)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, "session", "scope-1", "bad", "memory.updated", "set", []byte(`{}`), []byte(`{}`), []byte(`{`), time.Now().UTC())
		require.NoError(t, err)
		_, err = ls.getEventHistoryPostgres(ctx, filter)
		require.ErrorContains(t, err, "failed to unmarshal memory event metadata")

		_, err = ls.db.ExecContext(ctx, `DELETE FROM memory_events`)
		require.NoError(t, err)

		meta, err := json.Marshal(types.EventMetadata{ActorID: "actor-1"})
		require.NoError(t, err)
		for _, key := range []string{"doc-1", "skip-1", "doc-2"} {
			_, err = ls.db.ExecContext(ctx, `
				INSERT INTO memory_events(scope, scope_id, key, event_type, action, data, previous_data, metadata, timestamp)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			`, "session", "scope-1", key, "memory.updated", "set", []byte(`{"v":1}`), []byte(`{"v":0}`), meta, time.Now().UTC())
			require.NoError(t, err)
		}

		events, err := ls.getEventHistoryPostgres(ctx, filter)
		require.NoError(t, err)
		require.Len(t, events, 1)
		require.Equal(t, "doc-2", events[0].Key)
		require.Equal(t, "actor-1", events[0].Metadata.ActorID)
	})

	t.Run("cleanup expired events executes against postgres path", func(t *testing.T) {
		ls := newEventStorage(t)
		_, err := ls.db.ExecContext(context.Background(), `
			INSERT INTO memory_events(scope, scope_id, key, event_type, action, data, previous_data, metadata, timestamp)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, "session", "scope-1", "doc-1", "memory.updated", "set", []byte(`{}`), []byte(`{}`), []byte(`{}`), time.Now().UTC().Add(-2*defaultEventTTL))
		require.NoError(t, err)

		ls.cleanupExpiredEventsPostgres()

		var count int
		require.NoError(t, ls.db.QueryRow(`SELECT COUNT(*) FROM memory_events`).Scan(&count))
		require.Equal(t, 0, count)
	})
}

func TestAgentExecutionCoverageBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	t.Run("agent execution converters handle marshal and unmarshal failures", func(t *testing.T) {
		_, err := agentExecutionToModel(&types.AgentExecution{
			WorkflowID:  "wf-1",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-1",
			Metadata: types.ExecutionMetadata{
				Custom: map[string]interface{}{"bad": make(chan int)},
			},
		})
		require.ErrorContains(t, err, "failed to marshal execution metadata")

		_, err = agentExecutionFromModel(&AgentExecutionModel{
			ID:          1,
			WorkflowID:  "wf-1",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-1",
			Metadata:    []byte(`{`),
		})
		require.ErrorContains(t, err, "failed to unmarshal execution metadata")
	})

	t.Run("store and get execution cover success not found cancelled and corrupted metadata", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.GetExecution(cancelled, 1)
		require.EqualError(t, err, "context cancelled during get execution: context canceled")

		_, err = ls.GetExecution(ctx, 999999)
		require.EqualError(t, err, "execution with ID 999999 not found")

		exec := &types.AgentExecution{
			WorkflowID:  "wf-exec",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-1",
			InputData:   json.RawMessage(`{"input":1}`),
			OutputData:  json.RawMessage(`{"output":1}`),
			Status:      "completed",
			CreatedAt:   time.Now().UTC(),
		}
		require.NoError(t, ls.StoreExecution(ctx, exec))
		require.NotZero(t, exec.ID)

		got, err := ls.GetExecution(ctx, exec.ID)
		require.NoError(t, err)
		require.Equal(t, exec.WorkflowID, got.WorkflowID)

		_, err = ls.db.ExecContext(ctx, `UPDATE agent_executions SET metadata = ? WHERE id = ?`, []byte(`{`), exec.ID)
		require.NoError(t, err)
		_, err = ls.GetExecution(ctx, exec.ID)
		require.ErrorContains(t, err, "failed to unmarshal execution metadata")
	})
}

func TestCreateAccessPolicyPostgresCoverage(t *testing.T) {
	newPolicy := func() *types.AccessPolicy {
		description := "demo policy"
		return &types.AccessPolicy{
			Name:           "policy-demo",
			CallerTags:     []string{"caller"},
			TargetTags:     []string{"target"},
			AllowFunctions: []string{"summarize"},
			DenyFunctions:  []string{"delete"},
			Constraints: map[string]types.AccessConstraint{
				"max_cost": {Operator: "<=", Value: 5},
			},
			Action:      "allow",
			Priority:    1,
			Enabled:     true,
			Description: &description,
			CreatedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}
	}

	t.Run("create access policy postgres covers cancel marshal success and query failure", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		ls.mode = "postgres"

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		err := ls.createAccessPolicyPostgres(cancelled, newPolicy())
		require.EqualError(t, err, "context cancelled during create access policy: context canceled")

		badPolicy := newPolicy()
		badPolicy.Constraints["bad"] = types.AccessConstraint{Operator: "==", Value: make(chan int)}
		err = ls.createAccessPolicyPostgres(ctx, badPolicy)
		require.ErrorContains(t, err, "failed to marshal access policy fields")

		policy := newPolicy()
		require.NoError(t, ls.createAccessPolicyPostgres(ctx, policy))
		require.NotZero(t, policy.ID)

		require.NoError(t, ls.db.DB.Close())
		err = ls.createAccessPolicyPostgres(ctx, newPolicy())
		require.ErrorContains(t, err, "failed to create access policy")
	})
}

func TestLocalStorageAdditionalCoverage(t *testing.T) {
	t.Run("health check covers cancel nil local and postgres success", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		require.EqualError(t, (&LocalStorage{}).HealthCheck(cancelled), "context cancelled during health check: context canceled")
		require.EqualError(t, (&LocalStorage{}).HealthCheck(context.Background()), "database connection is not initialized")

		ls, ctx := setupLocalStorage(t)
		require.NoError(t, ls.HealthCheck(ctx))

		rawDB := openSQLiteNowDB(t)
		pgLS := &LocalStorage{db: newSQLDatabase(rawDB, "postgres"), mode: "postgres"}
		require.NoError(t, pgLS.HealthCheck(context.Background()))
	})

	t.Run("get agent covers cancel not found success and json decode failures", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.GetAgent(cancelled, "agent-1")
		require.EqualError(t, err, "context cancelled during get agent: context canceled")

		_, err = ls.GetAgent(ctx, "missing-agent")
		require.EqualError(t, err, "agent node with ID 'missing-agent' not found")

		invocationURL := " https://example.com/invoke "
		agent := &types.AgentNode{
			ID:              "agent-1",
			Version:         "",
			GroupID:         "group-1",
			TeamID:          "team-1",
			BaseURL:         "https://example.com",
			TrafficWeight:   100,
			DeploymentType:  "serverless",
			InvocationURL:   &invocationURL,
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastHeartbeat:   time.Now().UTC(),
			RegisteredAt:    time.Now().UTC(),
		}
		require.NoError(t, ls.RegisterAgent(ctx, agent))

		got, err := ls.GetAgent(ctx, "agent-1")
		require.NoError(t, err)
		require.Equal(t, "group-1", got.GroupID)
		require.NotNil(t, got.InvocationURL)
		require.Equal(t, "https://example.com/invoke", *got.InvocationURL)

		_, err = ls.db.ExecContext(ctx, `UPDATE agent_nodes SET reasoners = ? WHERE id = ?`, []byte(`{`), "agent-1")
		require.NoError(t, err)
		_, err = ls.GetAgent(ctx, "agent-1")
		require.ErrorContains(t, err, "failed to unmarshal agent reasoners")
	})

	t.Run("get af server did covers cancel not found and success", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.GetAgentFieldServerDID(cancelled, "srv-1")
		require.EqualError(t, err, "context cancelled during get af server DID: context canceled")

		info, err := ls.GetAgentFieldServerDID(ctx, "missing")
		require.NoError(t, err)
		require.Nil(t, info)

		now := time.Now().UTC().Truncate(time.Second)
		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-1", "did:root:1", []byte("seed"), now, now))
		info, err = ls.GetAgentFieldServerDID(ctx, "srv-1")
		require.NoError(t, err)
		require.Equal(t, "did:root:1", info.RootDID)
	})

	t.Run("get execution vc covers cancel not found and success", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.GetExecutionVC(cancelled, "vc-1")
		require.EqualError(t, err, "context cancelled during get execution VC: context canceled")

		_, err = ls.GetExecutionVC(ctx, "missing-vc")
		require.EqualError(t, err, "execution VC missing-vc not found")

		require.NoError(t, ls.StoreExecutionVC(ctx, "vc-1", "exec-1", "wf-1", "session-1", "did:issuer:1", "did:target:1", "did:caller:1", "in", "out", "active", []byte(`{"vc":1}`), "sig", "s3://vc", 42))
		info, err := ls.GetExecutionVC(ctx, "vc-1")
		require.NoError(t, err)
		require.Equal(t, "vc-1", info.VCID)
		require.Equal(t, int64(42), info.DocumentSize)
	})
}

func mustJSON(t *testing.T, value interface{}) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	require.NoError(t, err)
	return data
}
