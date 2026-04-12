//go:build integration
// +build integration

package storage

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func setupPostgresTest(t *testing.T) (context.Context, *LocalStorage, func()) {
	if testing.Short() {
		t.Skip("skipping postgres test in short mode")
	}

	ctx := context.Background()
	// a unique name for the database for each test
	dbName := fmt.Sprintf("test_db_%s", t.Name())
	// postgres db names can't contain uppercase letters
	dbName = strings.ToLower(dbName)

	baseDSN := "host=localhost user=postgres password=postgres port=5432 sslmode=disable"
	// Connect to postgres db to be able to create/drop the test one
	gormDB, err := gorm.Open(postgres.Open(baseDSN+" dbname=postgres"), &gorm.Config{})
	require.NoError(t, err, "failed to connect to postgres db")

	// Drop test database if it exists
	err = gormDB.Exec(fmt.Sprintf("DROP DATABASE IF EXISTS %s WITH (FORCE);", dbName)).Error
	require.NoError(t, err, "failed to drop test database")

	cfg := StorageConfig{
		Mode: "postgres",
		Postgres: PostgresStorageConfig{
			DSN:      baseDSN,
			Database: dbName,
		},
		Vector: VectorStoreConfig{
			Enabled: ptr(true),
		},
	}

	ls := NewPostgresStorage(cfg.Postgres)
	err = ls.Initialize(ctx, cfg)
	require.NoError(t, err)

	return ctx, ls, func() {
		ls.Close(ctx)
		gormDB, err := gorm.Open(postgres.Open(baseDSN+" dbname=postgres"), &gorm.Config{})
		require.NoError(t, err)
		err = gormDB.Exec(fmt.Sprintf("DROP DATABASE IF EXISTS %s WITH (FORCE);", dbName)).Error
		require.NoError(t, err)
	}
}

func TestInitializePostgres(t *testing.T) {
	_, ls, cleanup := setupPostgresTest(t)
	defer cleanup()

	require.NotNil(t, ls.db)
	require.NotNil(t, ls.gormDB)
	require.NotNil(t, ls.vectorStore)

	// Verify that migrations ran
	var exists bool
	err := ls.gormDB.Raw("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'schema_migrations')").Scan(&exists).Error
	require.NoError(t, err)
	require.True(t, exists)
}

func TestVectorStorePostgres(t *testing.T) {
	ctx, ls, cleanup := setupPostgresTest(t)
	defer cleanup()

	t.Run("GetNonExistentKey", func(t *testing.T) {
		_, err := ls.vectorStore.Get(ctx, "scope", "scopeID", "non-existent-key")
		require.Nil(t, err)
	})

	t.Run("DeleteByPrefix", func(t *testing.T) {
		record1 := &types.VectorRecord{
			Scope:     "scope",
			ScopeID:   "scopeID",
			Key:       "prefix_1",
			Embedding: []float32{1, 2, 3},
		}
		err := ls.vectorStore.Set(ctx, record1)
		require.NoError(t, err)

		record2 := &types.VectorRecord{
			Scope:     "scope",
			ScopeID:   "scopeID",
			Key:       "prefix_2",
			Embedding: []float32{4, 5, 6},
		}
		err = ls.vectorStore.Set(ctx, record2)
		require.NoError(t, err)

		deleted, err := ls.vectorStore.DeleteByPrefix(ctx, "scope", "scopeID", "prefix")
		require.NoError(t, err)
		require.Equal(t, 2, deleted)

		r, err := ls.vectorStore.Get(ctx, "scope", "scopeID", "prefix_1")
		require.NoError(t, err)
		require.Nil(t, r)

		r, err = ls.vectorStore.Get(ctx, "scope", "scopeID", "prefix_2")
		require.NoError(t, err)
		require.Nil(t, r)
	})
}
