package services

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
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
