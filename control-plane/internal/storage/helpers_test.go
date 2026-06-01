package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"math"
	"os"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func ptr[T any](v T) *T {
	return &v
}

func ptrString(s string) *string {
	return &s
}

func ptrTime(t time.Time) *time.Time {
	return &t
}

func ptrInt64(i int64) *int64 {
	return &i
}

func TestSQLDatabaseHelpers(t *testing.T) {
	t.Run("nil database guards and rebinding", func(t *testing.T) {
		var nilDB *sqlDatabase
		require.Equal(t, "", nilDB.Mode())
		_, err := nilDB.Begin()
		require.EqualError(t, err, "sql database is not initialized")
		_, err = nilDB.BeginTx(context.Background(), nil)
		require.EqualError(t, err, "sql database is not initialized")
		require.Equal(t, "select ?", nilDB.rebind("select ?"))

		db := newSQLDatabase(nil, "postgres")
		require.Equal(t, "postgres", db.Mode())
		require.Equal(t, "select $1, $2", db.rebind("select ?, ?"))

		var nilTx *sqlTx
		require.Equal(t, "select ?", nilTx.rebind("select ?"))
		require.Equal(t, "select $1", newSQLTx(nil, "postgres").rebind("select ?"))
	})

	t.Run("executes and queries through database and transaction wrappers", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite", "file::memory:?cache=shared")
		require.NoError(t, err)
		defer rawDB.Close()
		rawDB.SetMaxOpenConns(1)

		db := newSQLDatabase(rawDB, "sqlite")
		_, err = db.Exec(`create table items (id integer primary key, name text)`)
		require.NoError(t, err)

		_, err = db.ExecContext(context.Background(), `insert into items (name) values (?)`, "alpha")
		require.NoError(t, err)

		rows, err := db.Query(`select name from items order by id`)
		require.NoError(t, err)
		require.True(t, rows.Next())
		var name string
		require.NoError(t, rows.Scan(&name))
		require.Equal(t, "alpha", name)
		require.NoError(t, rows.Close())

		var queryRowName string
		require.NoError(t, db.QueryRowContext(context.Background(), `select name from items where id = ?`, 1).Scan(&queryRowName))
		require.Equal(t, "alpha", queryRowName)

		stmt, err := db.PrepareContext(context.Background(), `insert into items (name) values (?)`)
		require.NoError(t, err)
		defer stmt.Close()
		_, err = stmt.ExecContext(context.Background(), "beta")
		require.NoError(t, err)

		tx, err := db.BeginTx(context.Background(), nil)
		require.NoError(t, err)
		_, err = tx.Exec(`insert into items (name) values (?)`, "gamma")
		require.NoError(t, err)
		var txName string
		require.NoError(t, tx.QueryRow(`select name from items where name = ?`, "gamma").Scan(&txName))
		require.Equal(t, "gamma", txName)
		require.NoError(t, tx.Commit())
	})
}

func TestSafeJSONRawMessageAndMin(t *testing.T) {
	require.JSONEq(t, `{"fallback":true}`, string(safeJSONRawMessage("", `{"fallback":true}`, "blank")))
	require.JSONEq(t, `{"ok":true}`, string(safeJSONRawMessage(`{"ok":true}`, `{"fallback":true}`, "valid")))

	logFile, err := os.CreateTemp(t.TempDir(), "storage-log-*.txt")
	require.NoError(t, err)
	defer logFile.Close()
	previousWriter := log.Writer()
	log.SetOutput(logFile)
	defer log.SetOutput(previousWriter)

	raw := safeJSONRawMessage(`{"bad":`, `{"fallback":true}`, `config-sync`)
	var decoded map[string]string
	require.NoError(t, json.Unmarshal(raw, &decoded))
	require.Equal(t, "corrupted_json_data", decoded["error"])
	require.Equal(t, "config-sync", decoded["context"])
	require.Contains(t, decoded["preview"], `{"bad":`)

	contents, err := os.ReadFile(logFile.Name())
	require.NoError(t, err)
	require.Contains(t, string(contents), "Corrupted JSON data detected in config-sync")

	require.Equal(t, 2, min(2, 9))
	require.Equal(t, -3, min(4, -3))
}

func TestVectorStoreHelpers(t *testing.T) {
	require.Equal(t, VectorDistanceCosine, parseDistanceMetric(""))
	require.Equal(t, VectorDistanceDot, parseDistanceMetric(" inner "))
	require.Equal(t, VectorDistanceDot, parseDistanceMetric("ip"))
	require.Equal(t, VectorDistanceL2, parseDistanceMetric("euclidean"))

	encoded := encodeEmbedding([]float32{1.5, -2.25, 3})
	decoded, err := decodeEmbedding(encoded)
	require.NoError(t, err)
	require.Equal(t, []float32{1.5, -2.25, 3}, decoded)
	_, err = decodeEmbedding([]byte{1, 2, 3})
	require.EqualError(t, err, "invalid embedding length: 3")

	require.EqualError(t, ensureVectorPayload(nil), "vector record cannot be nil")
	require.EqualError(t, ensureVectorPayload(&types.VectorRecord{Scope: "scope"}), "scope, scope_id, and key are required")
	require.EqualError(t, ensureVectorPayload(&types.VectorRecord{Scope: "scope", ScopeID: "id", Key: "key"}), "embedding cannot be empty")
	require.NoError(t, ensureVectorPayload(&types.VectorRecord{Scope: "scope", ScopeID: "id", Key: "key", Embedding: []float32{1, 2}}))

	meta := normalizeMetadata(nil)
	require.NotNil(t, meta)
	require.Empty(t, meta)
	providedMeta := map[string]interface{}{"team": "ops"}
	require.Equal(t, providedMeta, normalizeMetadata(providedMeta))

	cos := cosineSimilarity([]float32{1, 0}, []float32{1, 0})
	require.InDelta(t, 1.0, cos, 0.0001)
	require.Equal(t, 0.0, cosineSimilarity([]float32{0, 0}, []float32{1, 1}))
	require.InDelta(t, 5.0, dotProduct([]float32{1, 2}, []float32{1, 2}), 0.0001)
	require.InDelta(t, 5.0, l2Distance([]float32{1, 2}, []float32{4, 6}), 0.0001)

	score, distance := computeSimilarity(VectorDistanceCosine, []float32{1, 0}, []float32{0, 1})
	require.InDelta(t, 0.0, score, 0.0001)
	require.InDelta(t, 1.0, distance, 0.0001)
	score, distance = computeSimilarity(VectorDistanceDot, []float32{1, 2}, []float32{1, 2})
	require.InDelta(t, 5.0, score, 0.0001)
	require.InDelta(t, -5.0, distance, 0.0001)
	score, distance = computeSimilarity(VectorDistanceL2, []float32{1, 2}, []float32{4, 6})
	require.InDelta(t, -5.0, score, 0.0001)
	require.InDelta(t, 5.0, distance, 0.0001)

	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	results := sortAndLimit([]*types.VectorSearchResult{
		{Key: "low", Score: 0.5, Distance: 0.4, CreatedAt: now},
		{Key: "best-near", Score: 0.9, Distance: 0.1, CreatedAt: now},
		{Key: "best-far", Score: 0.9, Distance: 0.3, CreatedAt: now},
	}, 2)
	require.Len(t, results, 2)
	require.Equal(t, []string{"best-near", "best-far"}, []string{results[0].Key, results[1].Key})
	require.True(t, metadataMatchesFilters(map[string]interface{}{"team": "ops", "rank": 3}, map[string]interface{}{"team": "ops", "rank": "3"}))
	require.False(t, metadataMatchesFilters(map[string]interface{}{"team": "ops"}, map[string]interface{}{"team": "dev"}))
	require.False(t, metadataMatchesFilters(map[string]interface{}{"team": "ops"}, map[string]interface{}{"missing": true}))
	require.True(t, metadataMatchesFilters(map[string]interface{}{"team": "ops"}, nil))

	delta := nowUTC().Sub(time.Now().UTC())
	require.Less(t, math.Abs(delta.Seconds()), 2.0)
}
