package workspace

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
)

// The blob-transport sender lives here so both hops that push blobs — the CLI to
// the control plane and the control plane to a node — share one implementation.
// Cold transfer used to be a sequential PUT per missing blob, so on a workspace
// with thousands of small files nearly all the wall-clock time was per-request
// round-trip latency rather than bandwidth. UploadBlobs instead packs the
// missing blobs into a handful of gzip-compressed tar batches (bounding memory
// and allowing retry) and falls back to bounded-parallel PUTs against servers
// that predate the batch route.

const (
	// defaultMaxBatchBytes is the target compressed size of a single batch
	// request. Splitting at this bound keeps sender/receiver memory flat and
	// lets a failed batch be retried in isolation.
	defaultMaxBatchBytes = 16 << 20 // 16 MiB compressed
	// defaultUploadParallelism bounds the fallback worker pool so an older
	// server still gets concurrent PUTs (never sequential) without unbounded
	// fan-out.
	defaultUploadParallelism = 16
	// batchGzipLevel trades a little CPU for a solid ratio on source trees.
	batchGzipLevel = 6

	batchPath = "/api/v1/workspace/blobs/batch"
	blobPath  = "/api/v1/workspace/blobs/"
)

// BlobSource yields the raw bytes for a blob keyed by its sha256 hex. *CAS
// satisfies it.
type BlobSource interface {
	Get(sha string) ([]byte, error)
}

// UploadOptions configures a blob upload. BaseURL is the server root (no
// trailing slash needed beyond what the caller supplies); Decorate stamps auth
// headers onto every outgoing request (the CLI adds X-API-Key, the control
// plane adds the internal bearer token).
type UploadOptions struct {
	BaseURL       string
	Client        *http.Client
	Decorate      func(*http.Request)
	MaxBatchBytes int
	Parallelism   int
}

// UploadStats reports the outcome of an upload for public-safe instrumentation.
type UploadStats struct {
	Blobs             int
	UncompressedBytes int64
	CompressedBytes   int64
	Requests          int
	// Mode is "batch" when the batch endpoint served the upload, or
	// "parallel-fallback" when the server lacked the route and per-blob PUTs
	// were used instead.
	Mode string
}

// batchRejection mirrors one entry of the batch endpoint's rejected list.
type batchRejection struct {
	SHA256 string `json:"sha256"`
	Error  string `json:"error"`
}

type batchResponse struct {
	Stored   int              `json:"stored"`
	Rejected []batchRejection `json:"rejected"`
}

// UploadBlobs pushes every blob named in shas from src to the server at
// opts.BaseURL. It packs them into gzip-compressed tar batches of roughly
// MaxBatchBytes compressed each and POSTs them to the batch endpoint. If the
// server answers 404/405 (no batch route) it transparently falls back to
// bounded-parallel per-blob PUTs. It returns instrumentation describing what
// was sent.
func UploadBlobs(ctx context.Context, src BlobSource, shas []string, opts UploadOptions) (UploadStats, error) {
	stats := UploadStats{Mode: "batch"}
	if len(shas) == 0 {
		return stats, nil
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	maxBatch := opts.MaxBatchBytes
	if maxBatch <= 0 {
		maxBatch = defaultMaxBatchBytes
	}

	i := 0
	for i < len(shas) {
		buf, uncompressed, next, err := buildBatch(src, shas, i, maxBatch)
		if err != nil {
			return stats, err
		}

		fellBack, ferr := sendBatch(ctx, opts, buf.Bytes())
		if ferr != nil {
			return stats, ferr
		}
		if fellBack {
			// The server has no batch route. Upload everything from this batch's
			// start onward via parallel PUTs; nothing in this batch was stored.
			reqs, bytesUp, perr := parallelUpload(ctx, src, shas[i:], opts)
			stats.Mode = "parallel-fallback"
			stats.Requests += reqs
			stats.Blobs += len(shas[i:])
			stats.UncompressedBytes += bytesUp
			return stats, perr
		}

		stats.Requests++
		stats.Blobs += next - i
		stats.UncompressedBytes += uncompressed
		stats.CompressedBytes += int64(buf.Len())
		i = next
	}
	return stats, nil
}

// buildBatch packs blobs from shas[start:] into one gzip-compressed tar, stopping
// once the compressed size reaches maxBatch (always emitting at least one entry).
// It returns the compressed buffer, the uncompressed byte total, and the index
// of the next unpacked blob.
func buildBatch(src BlobSource, shas []string, start, maxBatch int) (*bytes.Buffer, int64, int, error) {
	var buf bytes.Buffer
	gz, err := gzip.NewWriterLevel(&buf, batchGzipLevel)
	if err != nil {
		return nil, 0, start, fmt.Errorf("init gzip writer: %w", err)
	}
	tw := tar.NewWriter(gz)

	var uncompressed int64
	j := start
	for j < len(shas) {
		sha := shas[j]
		data, err := src.Get(sha)
		if err != nil {
			return nil, 0, start, fmt.Errorf("read blob %s from content store: %w", sha, err)
		}
		hdr := &tar.Header{Name: sha, Mode: 0o644, Size: int64(len(data)), Typeflag: tar.TypeReg}
		if err := tw.WriteHeader(hdr); err != nil {
			return nil, 0, start, fmt.Errorf("write tar header for %s: %w", sha, err)
		}
		if _, err := tw.Write(data); err != nil {
			return nil, 0, start, fmt.Errorf("write blob %s to batch: %w", sha, err)
		}
		uncompressed += int64(len(data))
		j++
		// Flush so buf reflects the compressed size accumulated so far, letting
		// the batch be bounded by actual on-wire bytes.
		if err := gz.Flush(); err != nil {
			return nil, 0, start, fmt.Errorf("flush batch: %w", err)
		}
		if buf.Len() >= maxBatch {
			break
		}
	}
	if err := tw.Close(); err != nil {
		return nil, 0, start, fmt.Errorf("close tar: %w", err)
	}
	if err := gz.Close(); err != nil {
		return nil, 0, start, fmt.Errorf("close gzip: %w", err)
	}
	return &buf, uncompressed, j, nil
}

// sendBatch POSTs one compressed batch. fellBack is true when the server has no
// batch route (404/405) so the caller can switch to per-blob PUTs.
func sendBatch(ctx context.Context, opts UploadOptions, body []byte) (fellBack bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, opts.BaseURL+batchPath, bytes.NewReader(body))
	if err != nil {
		return false, fmt.Errorf("build batch request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-tar")
	req.Header.Set("Content-Encoding", "gzip")
	if opts.Decorate != nil {
		opts.Decorate(req)
	}

	resp, err := opts.Client.Do(req)
	if err != nil {
		return false, fmt.Errorf("send blob batch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
		return true, nil
	}
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= http.StatusBadRequest {
		return false, fmt.Errorf("blob batch rejected (%d): %s", resp.StatusCode, string(respBody))
	}

	var decoded batchResponse
	if len(bytes.TrimSpace(respBody)) > 0 {
		if err := json.Unmarshal(respBody, &decoded); err != nil {
			return false, fmt.Errorf("decode batch response: %w", err)
		}
	}
	if len(decoded.Rejected) > 0 {
		// A rejected entry means the server could not verify a blob's hash. That
		// is a hard integrity failure for a cold transfer, so surface it.
		return false, fmt.Errorf("server rejected %d blob(s), first: %s (%s)",
			len(decoded.Rejected), decoded.Rejected[0].SHA256, decoded.Rejected[0].Error)
	}
	return false, nil
}

// parallelUpload uploads each blob with an individual PUT, using a bounded
// worker pool. It is the compatibility fallback for servers without the batch
// route; it is never sequential. It returns the number of requests issued and
// the total uncompressed bytes sent.
func parallelUpload(ctx context.Context, src BlobSource, shas []string, opts UploadOptions) (int, int64, error) {
	workers := opts.Parallelism
	if workers <= 0 {
		workers = defaultUploadParallelism
	}
	if workers > len(shas) {
		workers = len(shas)
	}

	jobs := make(chan string)
	var (
		mu        sync.Mutex
		firstErr  error
		bytesUp   int64
		requests  int
		wg        sync.WaitGroup
		cancelCtx context.Context
		cancel    context.CancelFunc
	)
	cancelCtx, cancel = context.WithCancel(ctx)
	defer cancel()

	record := func(n int64, err error) {
		mu.Lock()
		defer mu.Unlock()
		requests++
		bytesUp += n
		if err != nil && firstErr == nil {
			firstErr = err
			cancel()
		}
	}

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sha := range jobs {
				data, err := src.Get(sha)
				if err != nil {
					record(0, fmt.Errorf("read blob %s from content store: %w", sha, err))
					continue
				}
				record(int64(len(data)), putBlob(cancelCtx, opts, sha, data))
			}
		}()
	}

feed:
	for _, sha := range shas {
		select {
		case <-cancelCtx.Done():
			break feed
		case jobs <- sha:
		}
	}
	close(jobs)
	wg.Wait()
	return requests, bytesUp, firstErr
}

// putBlob PUTs a single raw blob to the per-blob endpoint.
func putBlob(ctx context.Context, opts UploadOptions, sha string, data []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, opts.BaseURL+blobPath+sha, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("build blob upload for %s: %w", sha, err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if opts.Decorate != nil {
		opts.Decorate(req)
	}
	resp, err := opts.Client.Do(req)
	if err != nil {
		return fmt.Errorf("upload blob %s: %w", sha, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("server rejected blob %s (%d): %s", sha, resp.StatusCode, string(body))
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}
