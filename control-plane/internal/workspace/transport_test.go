package workspace

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// batchTestServer emulates the control-plane/node batch + per-blob endpoints
// backed by a real CAS, and counts the HTTP requests it serves so tests can
// assert the sender collapses many blobs into few requests. When
// disableBatch is set it answers the batch route with 404 to exercise fallback.
type batchTestServer struct {
	cas          *CAS
	server       *httptest.Server
	batchReqs    int64
	putReqs      int64
	disableBatch bool
}

func newBatchTestServer(t *testing.T, disableBatch bool) *batchTestServer {
	t.Helper()
	bs := &batchTestServer{
		cas:          NewCAS(t.TempDir()),
		disableBatch: disableBatch,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/workspace/blobs/batch", func(w http.ResponseWriter, r *http.Request) {
		if bs.disableBatch {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt64(&bs.batchReqs, 1)
		var reader io.Reader = r.Body
		if strings.Contains(r.Header.Get("Content-Encoding"), "gzip") {
			gz, err := gzip.NewReader(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			defer gz.Close()
			reader = gz
		}
		tr := tar.NewReader(reader)
		stored := 0
		rejected := make([]map[string]string, 0)
		for {
			hdr, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			data, _ := io.ReadAll(tr)
			if err := bs.cas.PutVerified(hdr.Name, data); err != nil {
				rejected = append(rejected, map[string]string{"sha256": hdr.Name, "error": err.Error()})
				continue
			}
			stored++
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"stored": stored, "rejected": rejected})
	})
	mux.HandleFunc("/api/v1/workspace/blobs/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		atomic.AddInt64(&bs.putReqs, 1)
		sha := strings.TrimPrefix(r.URL.Path, "/api/v1/workspace/blobs/")
		data, _ := io.ReadAll(r.Body)
		if err := bs.cas.PutVerified(sha, data); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
	bs.server = httptest.NewServer(mux)
	t.Cleanup(bs.server.Close)
	return bs
}

// seedBlobs stores n blobs of the given size in src and returns their shas. The
// contents are pseudo-random (deterministic per index) so they are effectively
// incompressible — this keeps size-bounded batch-splitting tests honest rather
// than letting gzip collapse a repeating pattern into one tiny batch.
func seedBlobs(t *testing.T, src *CAS, n, size int) []string {
	t.Helper()
	shas := make([]string, n)
	for i := 0; i < n; i++ {
		rng := rand.New(rand.NewSource(int64(i)*2654435761 + 1))
		data := make([]byte, size)
		rng.Read(data)
		sha, err := src.PutBytes(data)
		if err != nil {
			t.Fatalf("seed blob %d: %v", i, err)
		}
		shas[i] = sha
	}
	return shas
}

func TestUploadBlobsBatchRoundTrip(t *testing.T) {
	src := NewCAS(t.TempDir())
	shas := seedBlobs(t, src, 5, 64)
	server := newBatchTestServer(t, false)

	stats, err := UploadBlobs(context.Background(), src, shas, UploadOptions{
		BaseURL: server.server.URL,
		Client:  server.server.Client(),
	})
	if err != nil {
		t.Fatalf("UploadBlobs: %v", err)
	}
	if stats.Mode != "batch" {
		t.Fatalf("mode = %q, want batch", stats.Mode)
	}
	if stats.Blobs != len(shas) {
		t.Fatalf("uploaded %d blobs, want %d", stats.Blobs, len(shas))
	}
	// Every blob landed byte-exact in the server CAS.
	for _, sha := range shas {
		want, _ := src.Get(sha)
		got, err := server.cas.Get(sha)
		if err != nil {
			t.Fatalf("server missing blob %s: %v", sha, err)
		}
		if string(got) != string(want) {
			t.Fatalf("blob %s content mismatch", sha)
		}
	}
	if atomic.LoadInt64(&server.putReqs) != 0 {
		t.Fatalf("expected no per-blob PUTs in batch mode, got %d", server.putReqs)
	}
}

func TestUploadBlobsManySmallFewRequests(t *testing.T) {
	// The core latency win: 500 small blobs must go out in a handful of
	// requests, not one request per blob.
	src := NewCAS(t.TempDir())
	shas := seedBlobs(t, src, 500, 128)
	server := newBatchTestServer(t, false)

	stats, err := UploadBlobs(context.Background(), src, shas, UploadOptions{
		BaseURL: server.server.URL,
		Client:  server.server.Client(),
	})
	if err != nil {
		t.Fatalf("UploadBlobs: %v", err)
	}
	if stats.Blobs != 500 {
		t.Fatalf("uploaded %d, want 500", stats.Blobs)
	}
	batchReqs := atomic.LoadInt64(&server.batchReqs)
	if batchReqs > 3 {
		t.Fatalf("500 small blobs issued %d batch requests; expected a handful (<=3)", batchReqs)
	}
	if int64(stats.Requests) != batchReqs {
		t.Fatalf("stats.Requests=%d disagrees with server count=%d", stats.Requests, batchReqs)
	}
	t.Logf("500 blobs uploaded in %d batch request(s); uncompressed=%d compressed=%d",
		batchReqs, stats.UncompressedBytes, stats.CompressedBytes)
}

func TestUploadBlobsOversizedSplitsIntoMultipleRequests(t *testing.T) {
	src := NewCAS(t.TempDir())
	// Random-ish 200 KiB blobs resist gzip, so a small compressed bound forces
	// several batches.
	shas := seedBlobs(t, src, 40, 200<<10)
	server := newBatchTestServer(t, false)

	stats, err := UploadBlobs(context.Background(), src, shas, UploadOptions{
		BaseURL:       server.server.URL,
		Client:        server.server.Client(),
		MaxBatchBytes: 512 << 10, // 512 KiB compressed per batch
	})
	if err != nil {
		t.Fatalf("UploadBlobs: %v", err)
	}
	batchReqs := atomic.LoadInt64(&server.batchReqs)
	if batchReqs < 2 {
		t.Fatalf("expected the oversized upload to split into multiple batches, got %d", batchReqs)
	}
	if stats.Blobs != len(shas) {
		t.Fatalf("uploaded %d, want %d", stats.Blobs, len(shas))
	}
	// All blobs still arrived despite the split.
	for _, sha := range shas {
		if !server.cas.Has(sha) {
			t.Fatalf("server missing blob %s after split upload", sha)
		}
	}
	t.Logf("%d blobs split across %d batches", len(shas), batchReqs)
}

func TestUploadBlobsFallsBackToParallelPuts(t *testing.T) {
	src := NewCAS(t.TempDir())
	shas := seedBlobs(t, src, 20, 64)
	server := newBatchTestServer(t, true) // batch route returns 404

	stats, err := UploadBlobs(context.Background(), src, shas, UploadOptions{
		BaseURL: server.server.URL,
		Client:  server.server.Client(),
	})
	if err != nil {
		t.Fatalf("UploadBlobs fallback: %v", err)
	}
	if stats.Mode != "parallel-fallback" {
		t.Fatalf("mode = %q, want parallel-fallback", stats.Mode)
	}
	if stats.Blobs != len(shas) {
		t.Fatalf("uploaded %d, want %d", stats.Blobs, len(shas))
	}
	if got := atomic.LoadInt64(&server.putReqs); got != int64(len(shas)) {
		t.Fatalf("expected %d parallel PUTs, got %d", len(shas), got)
	}
	for _, sha := range shas {
		if !server.cas.Has(sha) {
			t.Fatalf("server missing blob %s after fallback", sha)
		}
	}
}

// probeSource records the peak number of concurrent Get calls so a test can
// assert the fallback pool runs blobs in parallel rather than sequentially.
type probeSource struct {
	inner     *CAS
	cur, peak int64
}

func (p *probeSource) Get(sha string) ([]byte, error) {
	c := atomic.AddInt64(&p.cur, 1)
	for {
		pk := atomic.LoadInt64(&p.peak)
		if c <= pk || atomic.CompareAndSwapInt64(&p.peak, pk, c) {
			break
		}
	}
	// Hold briefly so overlapping workers are observable.
	time.Sleep(20 * time.Millisecond)
	atomic.AddInt64(&p.cur, -1)
	return p.inner.Get(sha)
}

func TestUploadBlobsFallbackIsConcurrent(t *testing.T) {
	inner := NewCAS(t.TempDir())
	shas := seedBlobs(t, inner, 32, 32)
	server := newBatchTestServer(t, true) // no batch route -> fallback
	ps := &probeSource{inner: inner}

	stats, err := UploadBlobs(context.Background(), ps, shas, UploadOptions{
		BaseURL:     server.server.URL,
		Client:      server.server.Client(),
		Parallelism: 8,
	})
	if err != nil {
		t.Fatalf("concurrent fallback upload: %v", err)
	}
	if stats.Mode != "parallel-fallback" {
		t.Fatalf("mode = %q, want parallel-fallback", stats.Mode)
	}
	peak := atomic.LoadInt64(&ps.peak)
	if peak < 2 {
		t.Fatalf("fallback ran sequentially (peak concurrent Get=%d), expected parallelism", peak)
	}
	if peak > 8 {
		t.Fatalf("fallback exceeded the bounded pool (peak=%d > 8)", peak)
	}
}
