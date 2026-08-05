package launchdsvc

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// probeServer starts an httptest server and returns its port. The probes build
// their own URLs as http://localhost:<port>/…, which resolves to the same
// loopback address httptest listens on — so no URL seam is needed to point them
// at a stub.
func probeServer(t *testing.T, h http.Handler) int {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	addr, ok := srv.Listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("unexpected listener address %T", srv.Listener.Addr())
	}
	return addr.Port
}

// freePort returns a port with nothing listening on it, so a probe against it
// fails to connect the way it would against a stopped control plane.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func TestURLBuilders(t *testing.T) {
	if got := HealthURL(9111); got != "http://localhost:9111/health" {
		t.Errorf("HealthURL = %q", got)
	}
	if got := ActiveExecutionsURL(9111); got != "http://localhost:9111/api/v1/executions/active" {
		t.Errorf("ActiveExecutionsURL = %q", got)
	}
}

func TestServerHealthy(t *testing.T) {
	t.Run("responding", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/health" {
				t.Errorf("unexpected path %q", r.URL.Path)
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		}))
		if !ServerHealthy(port) {
			t.Error("a 200 on /health must report healthy")
		}
	})

	t.Run("error status is not healthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		if ServerHealthy(port) {
			t.Error("503 must not report healthy")
		}
	})

	t.Run("nothing listening", func(t *testing.T) {
		if ServerHealthy(freePort(t)) {
			t.Error("a refused connection must not report healthy")
		}
	})
}

func TestActiveExecutions(t *testing.T) {
	t.Run("counts in-flight runs", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/executions/active" {
				t.Errorf("unexpected path %q", r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"count":3,"runs":[{"run_id":"a"},{"run_id":"b"},{"run_id":"c"}]}`))
		}))
		n, ok := ActiveExecutions(port, "")
		if !ok || n != 3 {
			t.Fatalf("ActiveExecutions = (%d, %v), want (3, true)", n, ok)
		}
	})

	t.Run("idle server reports zero", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"count":0,"runs":[]}`))
		}))
		n, ok := ActiveExecutions(port, "")
		if !ok || n != 0 {
			t.Fatalf("ActiveExecutions = (%d, %v), want (0, true)", n, ok)
		}
	})

	t.Run("falls back to the run list when count is absent", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"runs":[{"run_id":"a"},{"run_id":"b"}]}`))
		}))
		n, ok := ActiveExecutions(port, "")
		if !ok || n != 2 {
			t.Fatalf("ActiveExecutions = (%d, %v), want (2, true)", n, ok)
		}
	})

	t.Run("forwards the api key", func(t *testing.T) {
		var seen string
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			seen = r.Header.Get("X-API-Key")
			_, _ = w.Write([]byte(`{"count":1}`))
		}))
		if _, ok := ActiveExecutions(port, "sekret"); !ok {
			t.Fatal("expected a usable answer")
		}
		if seen != "sekret" {
			t.Errorf("X-API-Key = %q, want it forwarded", seen)
		}
	})

	// Every un-interpretable answer must report ok=false, because the install
	// path treats "unknown" as not-busy rather than blocking forever.
	t.Run("unauthorized is not trustworthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"nope"}`))
		}))
		if n, ok := ActiveExecutions(port, ""); ok {
			t.Fatalf("401 reported as trustworthy (n=%d)", n)
		}
	})

	t.Run("malformed json is not trustworthy", func(t *testing.T) {
		port := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"count": `))
		}))
		if _, ok := ActiveExecutions(port, ""); ok {
			t.Fatal("truncated JSON reported as trustworthy")
		}
	})

	t.Run("nothing listening is not trustworthy", func(t *testing.T) {
		if _, ok := ActiveExecutions(freePort(t), ""); ok {
			t.Fatal("a refused connection reported as trustworthy")
		}
	})
}

func TestFileSHA256AndBytesSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f")
	body := []byte("agentfield")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}

	sum, ok := FileSHA256(path)
	if !ok {
		t.Fatal("FileSHA256 failed on a readable file")
	}
	if sum != BytesSHA256(body) {
		t.Errorf("FileSHA256 = %q, BytesSHA256 = %q — must agree", sum, BytesSHA256(body))
	}
	if len(sum) != 64 || strings.ContainsAny(sum, "ghijklmnopqrstuvwxyz") {
		t.Errorf("not a hex sha256: %q", sum)
	}

	if _, ok := FileSHA256(filepath.Join(dir, "absent")); ok {
		t.Error("a missing file must report ok=false")
	}
	// A directory is openable but not readable as a stream.
	if _, ok := FileSHA256(dir); ok {
		t.Error("a directory must report ok=false")
	}
}

func TestFileHasContentsMatrix(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "plist")
	body := []byte(fmt.Sprintf("<plist>%s</plist>", ServerLabel))
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if !FileHasContents(path, body) {
		t.Error("identical contents must match")
	}
	if FileHasContents(path, append(body, '\n')) {
		t.Error("a trailing byte must not match")
	}
	if FileHasContents(filepath.Join(dir, "absent"), body) {
		t.Error("a missing file must not match")
	}
}
