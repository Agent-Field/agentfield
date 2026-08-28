package services

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestFilePayloadStoreLifecycle(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewFilePayloadStore(t.TempDir())

	original := []byte("hello world")
	record, err := store.SaveFromReader(ctx, bytes.NewReader(original))
	require.NoError(t, err)
	require.NotNil(t, record)
	require.Greater(t, record.Size, int64(0))
	require.True(t, strings.HasPrefix(record.URI, payloadURIPrefix))

	sum := sha256.Sum256(original)
	require.Equal(t, hex.EncodeToString(sum[:]), record.SHA256)

	rc, err := store.Open(ctx, record.URI)
	require.NoError(t, err)
	data, err := io.ReadAll(rc)
	require.NoError(t, err)
	require.Equal(t, original, data)
	require.NoError(t, rc.Close())

	require.NoError(t, store.Remove(ctx, record.URI))
	require.NoError(t, store.Remove(ctx, record.URI))

	_, err = store.Open(ctx, record.URI)
	require.Error(t, err)
}

func TestFilePayloadStoreSweepPreservesReferencesAndGrace(t *testing.T) {
	ctx := context.Background()
	store := NewFilePayloadStore(t.TempDir())
	referenced, err := store.SaveBytes(ctx, []byte("referenced"))
	require.NoError(t, err)
	orphan, err := store.SaveBytes(ctx, []byte("orphan"))
	require.NoError(t, err)
	recent, err := store.SaveBytes(ctx, []byte("recent"))
	require.NoError(t, err)
	old := time.Now().Add(-2 * time.Hour)
	for _, uri := range []string{referenced.URI, orphan.URI} {
		path, err := store.resolvePath(uri)
		require.NoError(t, err)
		require.NoError(t, os.Chtimes(path, old, old))
	}
	inspected, removed, err := store.Sweep(ctx, map[string]struct{}{referenced.URI: {}}, time.Hour, 10000)
	require.NoError(t, err)
	require.Equal(t, 3, inspected)
	require.Equal(t, 1, removed)
	reader, err := store.Open(ctx, referenced.URI)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	reader, err = store.Open(ctx, recent.URI)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	_, err = store.Open(ctx, orphan.URI)
	require.Error(t, err)
}

func TestFilePayloadStoreSweepDeletionCapDoesNotLimitInspection(t *testing.T) {
	dir := t.TempDir()
	store := NewFilePayloadStore(dir)
	old := time.Now().Add(-2 * time.Hour)
	references := make(map[string]struct{})
	for _, name := range []string{"aaa", "aab", "aac"} {
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte("referenced"), 0o600))
		require.NoError(t, os.Chtimes(filepath.Join(dir, name), old, old))
		references[payloadURIPrefix+name] = struct{}{}
	}
	require.NoError(t, os.WriteFile(filepath.Join(dir, "zzz"), []byte("orphan"), 0o600))
	require.NoError(t, os.Chtimes(filepath.Join(dir, "zzz"), old, old))

	inspected, removed, err := store.Sweep(context.Background(), references, time.Hour, 2)
	require.NoError(t, err)
	require.Equal(t, 4, inspected)
	require.Equal(t, 1, removed)
	_, err = os.Stat(filepath.Join(dir, "zzz"))
	require.ErrorIs(t, err, os.ErrNotExist)
}

func TestFilePayloadStoreSweepGuardAndEntryBranches(t *testing.T) {
	ctx := context.Background()
	var nilStore *FilePayloadStore
	inspected, removed, err := nilStore.Sweep(ctx, nil, 0, 10)
	require.NoError(t, err)
	require.Zero(t, inspected)
	require.Zero(t, removed)

	store := NewFilePayloadStore(filepath.Join(t.TempDir(), "missing"))
	inspected, removed, err = store.Sweep(ctx, nil, 0, 0)
	require.NoError(t, err)
	require.Zero(t, inspected)
	require.Zero(t, removed)
	inspected, removed, err = store.Sweep(ctx, nil, 0, 10)
	require.NoError(t, err)
	require.Zero(t, inspected)
	require.Zero(t, removed)

	dir := t.TempDir()
	store = NewFilePayloadStore(dir)
	require.NoError(t, os.Mkdir(filepath.Join(dir, "subdir"), 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "payload-temporary"), []byte("temp"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "orphan"), []byte("old"), 0o600))
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	inspected, removed, err = store.Sweep(cancelled, nil, 0, 10)
	require.ErrorIs(t, err, context.Canceled)
	require.Zero(t, inspected)
	require.Zero(t, removed)

	inspected, removed, err = store.Sweep(ctx, nil, -time.Hour, 10)
	require.NoError(t, err)
	require.Equal(t, 1, inspected)
	require.Equal(t, 1, removed)
}

func TestFilePayloadStoreSaveBytes(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewFilePayloadStore(t.TempDir())

	record, err := store.SaveBytes(ctx, []byte("abc"))
	require.NoError(t, err)
	require.Equal(t, int64(3), record.Size)
}

func TestFilePayloadStoreErrors(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewFilePayloadStore(t.TempDir())

	_, err := store.SaveFromReader(ctx, nil)
	require.Error(t, err)

	_, err = store.Open(ctx, "invalid://uri")
	require.Error(t, err)
}

func TestCopyWithContextCancels(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()

	go func() {
		defer pw.Close()
		buf := bytes.Repeat([]byte("a"), 1024)
		for i := 0; i < 4; i++ {
			if _, err := pw.Write(buf); err != nil {
				return
			}
			time.Sleep(5 * time.Millisecond)
			if i == 1 {
				cancel()
			}
		}
	}()

	err := copyWithContext(ctx, io.Discard, pr)
	require.ErrorIs(t, err, context.Canceled)
}

func TestNopPayloadStoreReturnsNilOnSave(t *testing.T) {
	store := NopPayloadStore{}
	ctx := context.Background()

	rec, err := store.SaveBytes(ctx, []byte(`{"hello":"world"}`))
	require.NoError(t, err)
	require.Nil(t, rec)

	rec, err = store.SaveFromReader(ctx, strings.NewReader("data"))
	require.NoError(t, err)
	require.Nil(t, rec)
}

func TestNopPayloadStoreOpenReturnsError(t *testing.T) {
	store := NopPayloadStore{}
	_, err := store.Open(context.Background(), "payload://abc")
	require.Error(t, err)
}

func TestNopPayloadStoreRemoveIsNoop(t *testing.T) {
	store := NopPayloadStore{}
	err := store.Remove(context.Background(), "payload://abc")
	require.NoError(t, err)
}
