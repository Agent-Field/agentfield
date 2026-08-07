package furrow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestAssetName(t *testing.T) {
	tests := []struct {
		goos, goarch, want string
		ok                 bool
	}{
		{"linux", "amd64", "furrow-linux-amd64", true},
		{"darwin", "arm64", "furrow-darwin-arm64", true},
		{"darwin", "amd64", "furrow-darwin-amd64", true},
		{"windows", "amd64", "", false},
		{"linux", "arm64", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.goos+"_"+tt.goarch, func(t *testing.T) {
			got, ok := AssetName(tt.goos, tt.goarch)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("AssetName() = %q, %v; want %q, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestEnsureWindowsIsNoOp(t *testing.T) {
	home := t.TempDir()
	if err := Ensure(Options{GOOS: "windows", GOARCH: "amd64", Home: home, BaseURL: "http://must-not-be-used.invalid"}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, "bin", "furrow")); !os.IsNotExist(err) {
		t.Fatalf("furrow unexpectedly exists: %v", err)
	}
}

func TestEnsureDefaultsRuntimePlatform(t *testing.T) {
	asset, ok := AssetName(runtime.GOOS, runtime.GOARCH)
	if !ok {
		t.Skipf("furrow is not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	home := t.TempDir()
	payload := []byte("runtime furrow")
	sum := sha256.Sum256(payload)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/SHA256SUMS":
			_, _ = fmt.Fprintf(w, "%x  %s\n", sum, asset)
		case "/" + asset:
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	if err := Ensure(Options{Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(home, "bin", "furrow"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content = %q, want %q", got, payload)
	}
}

func TestEnsureFailsWithoutResolvableHome(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("empty HOME behavior is specific to Unix")
	}
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64"})
	if err == nil || !strings.Contains(err.Error(), "home") {
		t.Fatalf("error = %v, want home resolution failure", err)
	}
}

func TestEnsureFailsToCreateBinDirectory(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "bin"), []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home})
	if err == nil || !strings.Contains(err.Error(), "create furrow bin directory") {
		t.Fatalf("error = %v", err)
	}
}

func TestEnsureSkipsExecutableWithoutDownload(t *testing.T) {
	home := t.TempDir()
	path := filepath.Join(home, "bin", "furrow")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("existing"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "bin", versionMarker), []byte(Version+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	defer server.Close()
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 0 {
		t.Fatalf("got %d requests, want 0", requests.Load())
	}
}

// Bumping Version has to actually reach machines that already have furrow;
// skipping on the binary's existence alone would pin them forever.
func TestEnsureUpgradesWhenInstalledVersionDiffers(t *testing.T) {
	home := t.TempDir()
	path := filepath.Join(home, "bin", "furrow")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("stale binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "bin", versionMarker), []byte("v0.0.1\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	payload := []byte("fresh binary")
	sum := sha256.Sum256(payload)
	server := releaseServer(t, payload, hex.EncodeToString(sum[:]))
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("binary = %q, want the freshly downloaded one", got)
	}
	marker, err := os.ReadFile(filepath.Join(home, "bin", versionMarker))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(marker)) != Version {
		t.Fatalf("marker = %q, want %q", strings.TrimSpace(string(marker)), Version)
	}
}

// A successful install must leave the marker behind, or every later Ensure
// re-downloads a binary it already has.
func TestEnsureRecordsVersionSoTheNextRunSkips(t *testing.T) {
	home := t.TempDir()
	payload := []byte("furrow")
	sum := sha256.Sum256(payload)
	var requests atomic.Int32
	server := countingReleaseServer(t, payload, hex.EncodeToString(sum[:]), &requests)

	for i := 0; i < 2; i++ {
		if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
			t.Fatalf("Ensure() run %d: %v", i+1, err)
		}
	}
	if requests.Load() == 0 {
		t.Fatal("first Ensure() made no requests; the test server was not exercised")
	}
	before := requests.Load()
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != before {
		t.Fatalf("a third Ensure() made %d extra requests, want 0", requests.Load()-before)
	}
}

func TestEnsureConcurrentDownloadsAssetOnce(t *testing.T) {
	home := t.TempDir()
	payload := []byte("concurrently installed furrow")
	sum := sha256.Sum256(payload)
	var assetRequests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/SHA256SUMS":
			_, _ = fmt.Fprintf(w, "%x  furrow-linux-amd64\n", sum)
		case "/furrow-linux-amd64":
			assetRequests.Add(1)
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	const ensures = 12
	errors := make(chan error, ensures)
	var group sync.WaitGroup
	for range ensures {
		group.Add(1)
		go func() {
			defer group.Done()
			errors <- Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
		}()
	}
	group.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatalf("Ensure() = %v", err)
		}
	}

	binary, err := os.ReadFile(filepath.Join(home, "bin", "furrow"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(binary, payload) {
		t.Fatalf("binary = %q, want %q", binary, payload)
	}
	marker, err := os.ReadFile(filepath.Join(home, "bin", versionMarker))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(marker)) != Version {
		t.Fatalf("marker = %q, want %q", strings.TrimSpace(string(marker)), Version)
	}
	if assetRequests.Load() != 1 {
		t.Fatalf("asset downloads = %d, want 1", assetRequests.Load())
	}
}

func TestEnsureRejectsChecksumMismatchAndWritesNothing(t *testing.T) {
	home := t.TempDir()
	server := releaseServer(t, []byte("tampered"), strings.Repeat("0", 64))
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("error = %v", err)
	}
	assertNoFurrowArtifacts(t, home)
}

func TestEnsureDownloadFailureIsNonFatal(t *testing.T) {
	home := t.TempDir()
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	var warnings bytes.Buffer
	if err := EnsureBestEffort(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}, &warnings); err != nil {
		t.Fatalf("EnsureBestEffort returned %v", err)
	}
	if strings.Count(strings.TrimSpace(warnings.String()), "\n") != 0 {
		t.Fatalf("warning was not one line: %q", warnings.String())
	}
	assertNoFurrowArtifacts(t, home)
}

func TestEnsureBestEffortAcceptsNilWarnings(t *testing.T) {
	home := t.TempDir()
	server := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(server.Close)
	if err := EnsureBestEffort(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}, nil); err != nil {
		t.Fatalf("EnsureBestEffort returned %v", err)
	}
}

func TestEnsureRejectsInvalidChecksum(t *testing.T) {
	home := t.TempDir()
	server := releaseServer(t, []byte("unused"), "not-a-valid-checksum")
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "invalid checksum for") {
		t.Fatalf("error = %v", err)
	}
}

func TestEnsureRejectsChecksumsMissingTheAsset(t *testing.T) {
	home := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/SHA256SUMS" {
			_, _ = fmt.Fprintf(w, "%x  some-other-asset\n", sha256.Sum256([]byte("x")))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(server.Close)
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "checksum missing for") {
		t.Fatalf("error = %v", err)
	}
	assertNoFurrowArtifacts(t, home)
}

func TestEnsureBinaryDownloadFailureLeavesNoArtifacts(t *testing.T) {
	home := t.TempDir()
	payload := []byte("furrow binary")
	sum := sha256.Sum256(payload)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/SHA256SUMS" {
			_, _ = fmt.Fprintf(w, "%x  furrow-linux-amd64\n", sum)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(server.Close)
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "download furrow-linux-amd64") {
		t.Fatalf("error = %v", err)
	}
	assertNoFurrowArtifacts(t, home)
}

func TestEnsureAtomicRenameFailureLeavesNoPartialFile(t *testing.T) {
	home := t.TempDir()
	destination := filepath.Join(home, "bin", "furrow")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := []byte("valid furrow")
	sum := sha256.Sum256(payload)
	server := releaseServer(t, payload, fmt.Sprintf("%x", sum))
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "install furrow") {
		t.Fatalf("error = %v", err)
	}
	entries, readErr := os.ReadDir(filepath.Join(home, "bin"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	if len(entries) != 2 || entries[0].Name() != ".furrow.lock" || entries[1].Name() != "furrow" || !entries[1].IsDir() {
		t.Fatalf("partial artifacts left behind: %+v", entries)
	}
}

func TestEnsureInstallsExecutable(t *testing.T) {
	home := t.TempDir()
	payload := []byte("valid furrow")
	sum := sha256.Sum256(payload)
	server := releaseServer(t, payload, fmt.Sprintf("%x", sum))
	t.Setenv("AGENTFIELD_FURROW_BASE_URL", server.URL)
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(home, "bin", "furrow")
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("content = %q", got)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o755 {
		t.Fatalf("mode = %o", info.Mode().Perm())
	}
}

func TestEnsureHonorsOptOut(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: "http://must-not-be-used.invalid"}); err != nil {
		t.Fatal(err)
	}
	assertNoFurrowArtifacts(t, home)
}

func releaseServer(t *testing.T, payload []byte, checksum string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/SHA256SUMS":
			_, _ = fmt.Fprintf(w, "%s  furrow-linux-amd64\n", checksum)
		case "/furrow-linux-amd64":
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

// countingReleaseServer is releaseServer with a request tally, for asserting
// that a second Ensure does no network work.
func countingReleaseServer(t *testing.T, payload []byte, checksum string, requests *atomic.Int32) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		switch r.URL.Path {
		case "/SHA256SUMS":
			_, _ = fmt.Fprintf(w, "%s  furrow-linux-amd64\n", checksum)
		case "/furrow-linux-amd64":
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func assertNoFurrowArtifacts(t *testing.T, home string) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(home, "bin"))
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != ".furrow.lock" {
		t.Fatalf("unexpected artifacts: %+v", entries)
	}
}
