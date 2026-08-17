package aforge

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestAssetName(t *testing.T) {
	tests := []struct {
		goos, goarch, want string
		ok                 bool
	}{
		{"linux", "amd64", "aforge-linux-amd64", true},
		{"linux", "arm64", "aforge-linux-arm64", true},
		{"darwin", "amd64", "aforge-darwin-amd64", true},
		{"darwin", "arm64", "aforge-darwin-arm64", true},
		{"windows", "amd64", "aforge-windows-amd64.exe", true},
		{"windows", "arm64", "aforge-windows-arm64.exe", true},
		{"freebsd", "riscv64", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.goos+"_"+tt.goarch, func(t *testing.T) {
			got, ok := AssetName(tt.goos, tt.goarch)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("AssetName() = %q, %v; want %q, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
	if got := BinaryName("windows"); got != "aforge.exe" {
		t.Fatalf("BinaryName(windows) = %q", got)
	}
}

func TestEnsureSkipEnvironmentIsNoOp(t *testing.T) {
	home := filepath.Join(t.TempDir(), "uncreated")
	t.Setenv(skipEnv, "1")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	t.Cleanup(server.Close)
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	assertNoRequestsOrFiles(t, requests.Load(), home)
}

func TestEnsureUnsupportedPlatformIsNoOp(t *testing.T) {
	home := filepath.Join(t.TempDir(), "uncreated")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	t.Cleanup(server.Close)
	if err := Ensure(Options{GOOS: "freebsd", GOARCH: "riscv64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	assertNoRequestsOrFiles(t, requests.Load(), home)
}

func TestEnsureInstallsGzippedAsset(t *testing.T) {
	home := t.TempDir()
	payload := []byte("aforge fixture")
	server, _ := releaseServer(t, payload, "")
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(home, "bin", "aforge")
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, payload) {
		t.Fatalf("installed binary = %q, %v", got, err)
	}
	info, err := os.Stat(path)
	if err != nil || info.Mode().Perm() != 0o755 {
		t.Fatalf("installed mode = %v, %v", info.Mode(), err)
	}
	marker, err := os.ReadFile(filepath.Join(home, "bin", versionMarker))
	if err != nil || string(marker) != Version+"\n" {
		t.Fatalf("marker = %q, %v", marker, err)
	}
}

func TestEnsureRejectsChecksumMismatch(t *testing.T) {
	home := t.TempDir()
	server, _ := releaseServer(t, []byte("tampered"), strings.Repeat("0", 64))
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), "aforge-linux-amd64") {
		t.Fatalf("error = %v", err)
	}
	assertNoInstall(t, home)
}

func TestEnsureSkipsWhenMarkerMatchesVersion(t *testing.T) {
	home := installedHome(t, Version)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	t.Cleanup(server.Close)
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 0 {
		t.Fatalf("requests = %d, want 0", requests.Load())
	}
}

func TestEnsureReinstallsWhenMarkerDiffers(t *testing.T) {
	home := installedHome(t, "old-version")
	server, requests := releaseServer(t, []byte("fresh"), "")
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatalf("requests = %d, want 2", requests.Load())
	}
	got, _ := os.ReadFile(filepath.Join(home, "bin", "aforge"))
	if string(got) != "fresh" {
		t.Fatalf("binary = %q", got)
	}
}

func TestEnsureForceReinstallsMatchingVersion(t *testing.T) {
	home := installedHome(t, Version)
	server, requests := releaseServer(t, []byte("forced"), "")
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL, Force: true}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatalf("requests = %d, want 2", requests.Load())
	}
}

func TestEnsureBaseURLPrecedence(t *testing.T) {
	envServer, envRequests := releaseServer(t, []byte("from env"), "")
	optionServer, optionRequests := releaseServer(t, []byte("from option"), "")
	t.Setenv(baseURLEnv, envServer.URL+"/")
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: t.TempDir(), BaseURL: optionServer.URL + "/"}); err != nil {
		t.Fatal(err)
	}
	if optionRequests.Load() != 2 || envRequests.Load() != 0 {
		t.Fatalf("option requests = %d, env requests = %d", optionRequests.Load(), envRequests.Load())
	}
	home := t.TempDir()
	if err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home}); err != nil {
		t.Fatal(err)
	}
	if envRequests.Load() != 2 {
		t.Fatalf("env requests = %d, want 2", envRequests.Load())
	}
}

func TestEnsureRejectsOversizedDecompressedPayload(t *testing.T) {
	home := t.TempDir()
	payload := bytes.Repeat([]byte{'x'}, maxDecompressedBytes+1)
	server, _ := releaseServer(t, payload, "")
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
	if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("exceeds %d bytes", maxDecompressedBytes)) {
		t.Fatalf("error = %v", err)
	}
	assertNoInstall(t, home)
}

func TestEnsureBestEffortWarnsOnce(t *testing.T) {
	home := t.TempDir()
	server := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(server.Close)
	var warnings bytes.Buffer
	if err := EnsureBestEffort(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL}, &warnings); err != nil {
		t.Fatal(err)
	}
	want := "warning: aforge was not installed: download checksums: 404 Not Found\n"
	if warnings.String() != want {
		t.Fatalf("warning = %q, want %q", warnings.String(), want)
	}
	if err := EnsureBestEffort(Options{GOOS: "linux", GOARCH: "amd64", Home: t.TempDir(), BaseURL: server.URL}, nil); err != nil {
		t.Fatal(err)
	}
}

func releaseServer(t *testing.T, payload []byte, checksum string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	if checksum == "" {
		sum := sha256.Sum256(payload)
		checksum = fmt.Sprintf("%x", sum)
	}
	var compressed bytes.Buffer
	gz := gzip.NewWriter(&compressed)
	if _, err := gz.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		switch r.URL.Path {
		case "/checksums.txt":
			for _, platform := range [][2]string{{"linux", "amd64"}, {"linux", "arm64"}, {"darwin", "amd64"}, {"darwin", "arm64"}, {"windows", "amd64"}, {"windows", "arm64"}} {
				asset, _ := AssetName(platform[0], platform[1])
				_, _ = fmt.Fprintf(w, "%s  %s\n", checksum, asset)
			}
		case "/aforge-linux-amd64.gz":
			_, _ = w.Write(compressed.Bytes())
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, &requests
}

func installedHome(t *testing.T, marker string) string {
	t.Helper()
	home := t.TempDir()
	binDir := filepath.Join(home, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "aforge"), []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, versionMarker), []byte(marker+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return home
}

func assertNoRequestsOrFiles(t *testing.T, requests int32, home string) {
	t.Helper()
	if requests != 0 {
		t.Fatalf("requests = %d, want 0", requests)
	}
	if _, err := os.Stat(home); !os.IsNotExist(err) {
		t.Fatalf("home unexpectedly exists: %v", err)
	}
}

func assertNoInstall(t *testing.T, home string) {
	t.Helper()
	for _, name := range []string{"aforge", versionMarker} {
		if _, err := os.Stat(filepath.Join(home, "bin", name)); !os.IsNotExist(err) {
			t.Fatalf("%s unexpectedly exists: %v", name, err)
		}
	}
}
