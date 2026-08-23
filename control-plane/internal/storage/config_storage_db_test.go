package storage

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLocalStorageConfigCRUD verifies the database-backed configuration lifecycle and its context guards.
func TestLocalStorageConfigCRUD(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, ls *LocalStorage, ctx context.Context)
	}{
		{
			name: "insert starts at version one with API audit fields",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				require.NoError(t, ls.SetConfig(ctx, "feature/alpha", `{"enabled":true}`, "api"))

				entry, err := ls.GetConfig(ctx, "feature/alpha")
				require.NoError(t, err)
				require.NotNil(t, entry)
				require.Equal(t, "feature/alpha", entry.Key)
				require.Equal(t, `{"enabled":true}`, entry.Value)
				require.Equal(t, 1, entry.Version)
				require.Equal(t, "api", entry.CreatedBy)
				require.Equal(t, "api", entry.UpdatedBy)
				require.False(t, entry.CreatedAt.IsZero())
				require.False(t, entry.UpdatedAt.IsZero())
			},
		},
		{
			name: "upsert increments version and preserves creation metadata",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				require.NoError(t, ls.SetConfig(ctx, "feature/alpha", "old", "api"))
				created, err := ls.GetConfig(ctx, "feature/alpha")
				require.NoError(t, err)
				require.NotNil(t, created)

				require.NoError(t, ls.SetConfig(ctx, "feature/alpha", "new", "operator"))
				updated, err := ls.GetConfig(ctx, "feature/alpha")
				require.NoError(t, err)
				require.NotNil(t, updated)
				require.Equal(t, "new", updated.Value)
				require.Equal(t, 2, updated.Version)
				require.Equal(t, "api", updated.CreatedBy)
				require.Equal(t, "operator", updated.UpdatedBy)
				require.Equal(t, created.CreatedAt, updated.CreatedAt)
				require.False(t, updated.UpdatedAt.Before(created.UpdatedAt))
			},
		},
		{
			name: "list returns keys in lexical order",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				for _, key := range []string{"c", "a", "b"} {
					require.NoError(t, ls.SetConfig(ctx, key, key+"-value", "api"))
				}

				entries, err := ls.ListConfigs(ctx)
				require.NoError(t, err)
				require.Len(t, entries, 3)
				require.Equal(t, []string{"a", "b", "c"}, []string{entries[0].Key, entries[1].Key, entries[2].Key})
			},
		},
		{
			name: "get missing key returns nil entry",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				entry, err := ls.GetConfig(ctx, "missing")
				require.NoError(t, err)
				require.Nil(t, entry)
			},
		},
		{
			name: "delete removes an existing key",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				require.NoError(t, ls.SetConfig(ctx, "obsolete", "value", "api"))
				require.NoError(t, ls.DeleteConfig(ctx, "obsolete"))

				entry, err := ls.GetConfig(ctx, "obsolete")
				require.NoError(t, err)
				require.Nil(t, entry)
			},
		},
		{
			name: "delete missing key reports its quoted name",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				err := ls.DeleteConfig(ctx, "missing/key")
				require.EqualError(t, err, `config "missing/key" not found`)
			},
		},
		{
			name: "cancelled context rejects set list and delete",
			run: func(t *testing.T, ls *LocalStorage, ctx context.Context) {
				// Seed a key so the cancelled delete can prove it never reaches the database.
				require.NoError(t, ls.SetConfig(ctx, "retained", "value", "api"))
				cancelledCtx, cancel := context.WithCancel(ctx)
				cancel()

				require.ErrorIs(t, ls.SetConfig(cancelledCtx, "not-created", "value", "api"), context.Canceled)
				_, err := ls.ListConfigs(cancelledCtx)
				require.ErrorIs(t, err, context.Canceled)
				require.ErrorIs(t, ls.DeleteConfig(cancelledCtx, "retained"), context.Canceled)

				notCreated, err := ls.GetConfig(ctx, "not-created")
				require.NoError(t, err)
				require.Nil(t, notCreated)
				retained, err := ls.GetConfig(ctx, "retained")
				require.NoError(t, err)
				require.NotNil(t, retained)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Isolate every scenario in its own SQLite and BoltDB pair.
			ls, ctx := setupLocalStorage(t)
			test.run(t, ls, ctx)
		})
	}
}
