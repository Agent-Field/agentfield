package storage

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConfigStorageSetConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		key       string
		value     string
		updatedBy string
	}{
		{
			name:      "first insert creates version 1",
			key:       "config/agentfield.yaml",
			value:     "key: value\nlogging:\n  level: debug\n",
			updatedBy: "api",
		},
		{
			name:      "second config entry",
			key:       "config/another.yaml",
			value:     "setting: true",
			updatedBy: "cli",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ls, ctx := setupLocalStorage(t)
			err := ls.SetConfig(ctx, tt.key, tt.value, tt.updatedBy)
			require.NoError(t, err)

			entry, err := ls.GetConfig(ctx, tt.key)
			require.NoError(t, err)
			require.NotNil(t, entry)
			require.Equal(t, tt.key, entry.Key)
			require.Equal(t, tt.value, entry.Value)
			require.Equal(t, 1, entry.Version)
			require.Equal(t, tt.updatedBy, entry.CreatedBy)
			require.Equal(t, tt.updatedBy, entry.UpdatedBy)
			require.False(t, entry.CreatedAt.IsZero())
			require.False(t, entry.UpdatedAt.IsZero())
		})
	}
}

func TestConfigStorageUpsertIncrementsVersion(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)
	key := "config/agentfield.yaml"

	err := ls.SetConfig(ctx, key, "version1", "alice")
	require.NoError(t, err)

	entry, err := ls.GetConfig(ctx, key)
	require.NoError(t, err)
	require.Equal(t, "version1", entry.Value)
	require.Equal(t, 1, entry.Version)
	originalCreatedAt := entry.CreatedAt
	originalCreatedBy := entry.CreatedBy

	err = ls.SetConfig(ctx, key, "version2", "bob")
	require.NoError(t, err)

	entry, err = ls.GetConfig(ctx, key)
	require.NoError(t, err)
	require.Equal(t, "version2", entry.Value)
	require.Equal(t, 2, entry.Version)
	require.Equal(t, originalCreatedBy, entry.CreatedBy)
	require.Equal(t, originalCreatedAt, entry.CreatedAt)
	require.Equal(t, "bob", entry.UpdatedBy)
	require.False(t, entry.UpdatedAt.IsZero())
}

func TestConfigStorageListConfigsLexicalOrder(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.SetConfig(ctx, "c", "3", "user"))
	require.NoError(t, ls.SetConfig(ctx, "a", "1", "user"))
	require.NoError(t, ls.SetConfig(ctx, "b", "2", "user"))

	entries, err := ls.ListConfigs(ctx)
	require.NoError(t, err)
	require.Len(t, entries, 3)
	require.Equal(t, "a", entries[0].Key)
	require.Equal(t, "b", entries[1].Key)
	require.Equal(t, "c", entries[2].Key)
}

func TestConfigStorageGetMissingKey(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)

	entry, err := ls.GetConfig(ctx, "nonexistent")
	require.NoError(t, err)
	require.Nil(t, entry)
}

func TestConfigStorageDeleteConfig(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.SetConfig(ctx, "test-key", "value", "admin"))

	err := ls.DeleteConfig(ctx, "test-key")
	require.NoError(t, err)

	entry, err := ls.GetConfig(ctx, "test-key")
	require.NoError(t, err)
	require.Nil(t, entry)
}

func TestConfigStorageDeleteMissingKeyReturnsError(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)

	err := ls.DeleteConfig(ctx, "absent-key")
	require.ErrorContains(t, err, `"absent-key"`)
}

func TestConfigStorageCancelledContext(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	err := ls.SetConfig(cancelled, "key", "val", "user")
	require.ErrorIs(t, err, context.Canceled)

	_, err = ls.GetConfig(cancelled, "key")
	require.ErrorIs(t, err, context.Canceled)

	_, err = ls.ListConfigs(cancelled)
	require.ErrorIs(t, err, context.Canceled)

	err = ls.DeleteConfig(cancelled, "key")
	require.ErrorIs(t, err, context.Canceled)
}

func TestConfigStorageDeleteConfigTwice(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)
	require.NoError(t, ls.SetConfig(ctx, "once", "val", "admin"))
	require.NoError(t, ls.DeleteConfig(ctx, "once"))
	err := ls.DeleteConfig(ctx, "once")
	require.ErrorContains(t, err, `"once"`)
}

func TestConfigStorageListConfigsEmpty(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)

	entries, err := ls.ListConfigs(ctx)
	require.NoError(t, err)
	require.Empty(t, entries)
}

func TestConfigStorageMultipleUpsertRoundTrip(t *testing.T) {
	t.Parallel()

	ls, ctx := setupLocalStorage(t)
	key := "roundtrip"

	versions := []struct {
		value     string
		updatedBy string
	}{
		{"v1", "alice"},
		{"v2", "bob"},
		{"v3", "charlie"},
	}

	for _, v := range versions {
		require.NoError(t, ls.SetConfig(ctx, key, v.value, v.updatedBy))
	}

	entry, err := ls.GetConfig(ctx, key)
	require.NoError(t, err)
	require.Equal(t, "v3", entry.Value)
	require.Equal(t, 3, entry.Version)
	require.Equal(t, "alice", entry.CreatedBy)
	require.Equal(t, "charlie", entry.UpdatedBy)
}
