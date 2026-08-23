package skillkit

import (
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
)

func TestInstallAllEnsuresAforgeOnce(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("executable mode checks are not meaningful on Windows")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("CODEX_HOME", filepath.Join(home, "codex"))
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")

	payload := []byte("aforge from skill install")
	sum := sha256.Sum256(payload)
	asset := fmt.Sprintf("aforge-%s-%s", runtime.GOOS, runtime.GOARCH)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		switch r.URL.Path {
		case "/checksums.txt":
			_, _ = fmt.Fprintf(w, "%x  %s\n", sum, asset)
		case "/" + asset + ".gz":
			gz := gzip.NewWriter(w)
			_, _ = gz.Write(payload)
			_ = gz.Close()
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("AGENTFIELD_AFORGE_BASE_URL", server.URL)

	if _, err := InstallAll(InstallOptions{Targets: []string{"codex"}}); err != nil {
		t.Fatalf("InstallAll: %v", err)
	}
	info, err := os.Stat(filepath.Join(home, "bin", "aforge"))
	if err != nil {
		t.Fatalf("stat provisioned aforge: %v", err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("provisioned aforge mode = %o, want executable", info.Mode().Perm())
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("HTTP requests = %d, want 2 (checksums + one asset)", got)
	}
}

func TestInstallAllDryRunDoesNotEnsureAforge(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("CODEX_HOME", filepath.Join(home, "codex"))
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "unexpected request", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	t.Setenv("AGENTFIELD_AFORGE_BASE_URL", server.URL)

	if _, err := InstallAll(InstallOptions{Targets: []string{"codex"}, DryRun: true}); err != nil {
		t.Fatalf("InstallAll(dry-run): %v", err)
	}
	if _, err := os.Stat(filepath.Join(home, "bin", "aforge")); !os.IsNotExist(err) {
		t.Fatalf("stat aforge after dry-run: %v, want not exist", err)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("HTTP requests = %d, want 0", got)
	}
}
