package services

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const payloadURIPrefix = "payload://"

// PayloadRecord captures metadata about a stored payload blob.
type PayloadRecord struct {
	URI    string
	Size   int64
	SHA256 string
}

// PayloadStore provides streaming persistence for request/response payloads.
type PayloadStore interface {
	SaveFromReader(ctx context.Context, r io.Reader) (*PayloadRecord, error)
	SaveBytes(ctx context.Context, data []byte) (*PayloadRecord, error)
	Open(ctx context.Context, uri string) (io.ReadCloser, error)
	Remove(ctx context.Context, uri string) error
}

// NopPayloadStore is a no-op implementation of PayloadStore used when payloads
// are persisted inline (e.g. in Postgres) and disk storage is unnecessary.
type NopPayloadStore struct{}

func (NopPayloadStore) SaveFromReader(context.Context, io.Reader) (*PayloadRecord, error) {
	return nil, nil
}
func (NopPayloadStore) SaveBytes(context.Context, []byte) (*PayloadRecord, error) { return nil, nil }
func (NopPayloadStore) Open(context.Context, string) (io.ReadCloser, error) {
	return nil, errors.New("nop payload store does not support Open")
}
func (NopPayloadStore) Remove(context.Context, string) error { return nil }

// FilePayloadStore persists payloads on the local filesystem under a base directory.
type FilePayloadStore struct {
	baseDir string
}

// Sweep removes unreferenced payload files older than grace, up to limit.
// It returns the number inspected and removed.
func (s *FilePayloadStore) Sweep(ctx context.Context, referenced map[string]struct{}, grace time.Duration, limit int) (int, int, error) {
	if s == nil || limit <= 0 {
		return 0, 0, nil
	}
	dir, err := os.Open(s.baseDir)
	if os.IsNotExist(err) {
		return 0, 0, nil
	}
	if err != nil {
		return 0, 0, fmt.Errorf("open payload directory: %w", err)
	}
	defer dir.Close()
	cutoff := time.Now().Add(-grace)
	inspected, removed := 0, 0
	for removed < limit {
		entries, readErr := dir.ReadDir(256)
		for _, entry := range entries {
			if err := ctx.Err(); err != nil {
				return inspected, removed, err
			}
			if entry.IsDir() {
				continue
			}
			inspected++
			uri := payloadURIPrefix + entry.Name()
			if _, ok := referenced[uri]; ok {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				return inspected, removed, fmt.Errorf("stat payload orphan: %w", err)
			}
			if info.ModTime().After(cutoff) {
				continue
			}
			if err := s.Remove(ctx, uri); err != nil {
				return inspected, removed, err
			}
			removed++
			if removed >= limit {
				break
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return inspected, removed, fmt.Errorf("read payload directory: %w", readErr)
		}
	}
	return inspected, removed, nil
}

// NewFilePayloadStore creates a payload store rooted at baseDir. The directory must exist.
func NewFilePayloadStore(baseDir string) *FilePayloadStore {
	return &FilePayloadStore{baseDir: baseDir}
}

// SaveFromReader streams the reader to disk while hashing it.
func (s *FilePayloadStore) SaveFromReader(ctx context.Context, r io.Reader) (*PayloadRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("payload reader cannot be nil")
	}

	id, err := randomPayloadID()
	if err != nil {
		return nil, err
	}

	tmpFile, err := os.CreateTemp(s.baseDir, "payload-*")
	if err != nil {
		return nil, fmt.Errorf("create payload temp file: %w", err)
	}
	defer func() {
		_ = tmpFile.Close()
		if err != nil {
			_ = os.Remove(tmpFile.Name())
		}
	}()

	hasher := sha256.New()
	writer := io.MultiWriter(tmpFile, hasher)

	if err = copyWithContext(ctx, writer, r); err != nil {
		return nil, err
	}

	if err = tmpFile.Close(); err != nil {
		return nil, fmt.Errorf("close payload temp file: %w", err)
	}

	finalPath := filepath.Join(s.baseDir, id)
	if err = os.Rename(tmpFile.Name(), finalPath); err != nil {
		return nil, fmt.Errorf("finalize payload file: %w", err)
	}

	info, err := os.Stat(finalPath)
	if err != nil {
		return nil, fmt.Errorf("stat payload file: %w", err)
	}

	record := &PayloadRecord{
		URI:    payloadURIPrefix + id,
		Size:   info.Size(),
		SHA256: hex.EncodeToString(hasher.Sum(nil)),
	}
	return record, nil
}

// SaveBytes writes an in-memory payload.
func (s *FilePayloadStore) SaveBytes(ctx context.Context, data []byte) (*PayloadRecord, error) {
	return s.SaveFromReader(ctx, bytes.NewReader(data))
}

// Open returns a reader for the payload at the given URI.
func (s *FilePayloadStore) Open(ctx context.Context, uri string) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path, err := s.resolvePath(uri)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open payload: %w", err)
	}
	return file, nil
}

// Remove deletes a payload from disk. It is safe to call on missing URIs.
func (s *FilePayloadStore) Remove(ctx context.Context, uri string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.resolvePath(uri)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove payload: %w", err)
	}
	return nil
}

func (s *FilePayloadStore) resolvePath(uri string) (string, error) {
	if !strings.HasPrefix(uri, payloadURIPrefix) {
		return "", fmt.Errorf("unsupported payload URI: %s", uri)
	}
	name := strings.TrimPrefix(uri, payloadURIPrefix)
	if name == "" {
		return "", fmt.Errorf("invalid payload URI: %s", uri)
	}
	return filepath.Join(s.baseDir, name), nil
}

func randomPayloadID() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generate payload id: %w", err)
	}
	return hex.EncodeToString(buf), nil
}

func copyWithContext(ctx context.Context, dst io.Writer, src io.Reader) error {
	buf := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := src.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return werr
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}
