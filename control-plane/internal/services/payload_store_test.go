package services

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
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
