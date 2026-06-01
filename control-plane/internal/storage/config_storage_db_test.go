package storage

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestLocalStorageConfigCRUD(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	t.Run("first insert creates version 1", func(t *testing.T) {
		require.NoError(t, ls.SetConfig(ctx, "config/agentfield.yaml", "workers: 2\n", "api"))

		entry, err := ls.GetConfig(ctx, "config/agentfield.yaml")
		require.NoError(t, err)
		require.NotNil(t, entry)
		require.Equal(t, "config/agentfield.yaml", entry.Key)
		require.Equal(t, "workers: 2\n", entry.Value)
		require.Equal(t, 1, entry.Version)
		require.Equal(t, "api", entry.CreatedBy)
		require.Equal(t, "api", entry.UpdatedBy)
		require.False(t, entry.CreatedAt.IsZero())
		require.False(t, entry.UpdatedAt.IsZero())
	})

	t.Run("upsert increments version and preserves created fields", func(t *testing.T) {
		const key = "config/upsert.yaml"
		require.NoError(t, ls.SetConfig(ctx, key, "workers: 1\n", "bootstrap"))

		first, err := ls.GetConfig(ctx, key)
		require.NoError(t, err)
		require.NotNil(t, first)

		require.NoError(t, ls.SetConfig(ctx, key, "workers: 3\n", "operator"))

		second, err := ls.GetConfig(ctx, key)
		require.NoError(t, err)
		require.NotNil(t, second)
		require.Equal(t, "workers: 3\n", second.Value)
		require.Equal(t, 2, second.Version)
		require.Equal(t, first.CreatedBy, second.CreatedBy)
		require.True(t, first.CreatedAt.Equal(second.CreatedAt))
		require.Equal(t, "operator", second.UpdatedBy)
		require.False(t, second.UpdatedAt.Before(first.UpdatedAt))
	})

	t.Run("list returns lexical order", func(t *testing.T) {
		prefix := "config/list/"
		require.NoError(t, ls.SetConfig(ctx, prefix+"c", "c", "api"))
		require.NoError(t, ls.SetConfig(ctx, prefix+"a", "a", "api"))
		require.NoError(t, ls.SetConfig(ctx, prefix+"b", "b", "api"))

		entries, err := ls.ListConfigs(ctx)
		require.NoError(t, err)

		var keys []string
		for _, entry := range entries {
			if len(entry.Key) >= len(prefix) && entry.Key[:len(prefix)] == prefix {
				keys = append(keys, entry.Key)
			}
		}
		require.Equal(t, []string{prefix + "a", prefix + "b", prefix + "c"}, keys)
	})

	t.Run("get missing key returns nil entry and nil error", func(t *testing.T) {
		entry, err := ls.GetConfig(ctx, "config/missing.yaml")
		require.NoError(t, err)
		require.Nil(t, entry)
	})

	t.Run("delete missing key returns not found error", func(t *testing.T) {
		err := ls.DeleteConfig(ctx, "config/missing-delete.yaml")
		require.EqualError(t, err, `config "config/missing-delete.yaml" not found`)
	})

	t.Run("cancelled context fails fast", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()

		require.ErrorIs(t, ls.SetConfig(cancelled, "config/cancelled.yaml", "value", "api"), context.Canceled)

		entries, err := ls.ListConfigs(cancelled)
		require.Nil(t, entries)
		require.ErrorIs(t, err, context.Canceled)

		require.ErrorIs(t, ls.DeleteConfig(cancelled, "config/agentfield.yaml"), context.Canceled)

		entry, err := ls.GetConfig(cancelled, "config/agentfield.yaml")
		require.Nil(t, entry)
		require.ErrorIs(t, err, context.Canceled)
	})
}

func TestLocalStorageConfigCRUDDoesNotMaskContextErrors(t *testing.T) {
	ls, _ := setupLocalStorage(t)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	err := ls.SetConfig(ctx, "config/deadline.yaml", "value", "api")
	require.True(t, errors.Is(err, context.DeadlineExceeded))
}
