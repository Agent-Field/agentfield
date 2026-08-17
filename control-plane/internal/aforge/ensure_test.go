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
	"runtime"
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

func TestEnsureDefaultsToRuntimePlatformAndAgentfieldHome(t *testing.T) {
	asset, supported := AssetName(runtime.GOOS, runtime.GOARCH)
	if !supported {
		t.Skipf("unsupported test platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	payload := []byte("native aforge fixture")
	server := releaseServerForAsset(t, asset, payload)
	if err := Ensure(Options{BaseURL: server.URL}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(home, "bin", BinaryName(runtime.GOOS)))
	if err != nil || !bytes.Equal(got, payload) {
		t.Fatalf("installed binary = %q, %v", got, err)
	}
}

func TestAgentfieldHomePrecedenceAndFallback(t *testing.T) {
	override := t.TempDir()
	environment := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", environment)
	got, err := agentfieldHome(override)
	if err != nil || got != override {
		t.Fatalf("agentfieldHome(override) = %q, %v", got, err)
	}
	got, err = agentfieldHome("")
	if err != nil || got != environment {
		t.Fatalf("agentfieldHome(environment) = %q, %v", got, err)
	}

	t.Setenv("AGENTFIELD_HOME", "")
	userHome := t.TempDir()
	t.Setenv("HOME", userHome)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", userHome)
	}
	got, err = agentfieldHome("")
	want := filepath.Join(userHome, ".agentfield")
	if err != nil || got != want {
		t.Fatalf("agentfieldHome(user home) = %q, %v; want %q", got, err, want)
	}
}

func TestAgentfieldHomeReportsMissingUserHome(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("UserHomeDir uses additional Windows environment variables")
	}
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")
	_, err := agentfieldHome("")
	if err == nil || !strings.Contains(err.Error(), "resolve home directory") {
		t.Fatalf("error = %v", err)
	}
	err = Ensure(Options{GOOS: "linux", GOARCH: "amd64"})
	if err == nil || !strings.Contains(err.Error(), "resolve home directory") {
		t.Fatalf("Ensure error = %v", err)
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

func TestEnsureUsesDefaultBaseURL(t *testing.T) {
	t.Setenv(baseURLEnv, "")
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if !strings.HasPrefix(request.URL.String(), defaultBaseURL+"/") {
			t.Fatalf("request URL = %q, want prefix %q", request.URL, defaultBaseURL+"/")
		}
		return nil, fmt.Errorf("fixture transport failure")
	})}
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: t.TempDir(), Client: client})
	if err == nil || !strings.Contains(err.Error(), "fixture transport failure") {
		t.Fatalf("error = %v", err)
	}
}

func TestDownloadErrors(t *testing.T) {
	closed := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closed.Close()
	if _, err := download(http.DefaultClient, closed.URL); err == nil {
		t.Fatal("download transport error = nil")
	}
}

func TestDownloadGzipErrors(t *testing.T) {
	var valid bytes.Buffer
	gz := gzip.NewWriter(&valid)
	_, _ = gz.Write(bytes.Repeat([]byte("truncated payload"), 100))
	_ = gz.Close()
	tests := []struct {
		name   string
		status int
		body   []byte
	}{
		{name: "non-200", status: http.StatusNotFound},
		{name: "invalid header", status: http.StatusOK, body: []byte("not gzip")},
		{name: "truncated stream", status: http.StatusOK, body: valid.Bytes()[:valid.Len()-4]},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
				_, _ = w.Write(test.body)
			}))
			t.Cleanup(server.Close)
			if _, err := downloadGzip(http.DefaultClient, server.URL); err == nil {
				t.Fatal("downloadGzip error = nil")
			}
		})
	}
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	if _, err := downloadGzip(http.DefaultClient, closed.URL); err == nil {
		t.Fatal("downloadGzip transport error = nil")
	}
}

func TestChecksumForRejectsInvalidAndMissingChecksums(t *testing.T) {
	asset := "aforge-linux-amd64"
	tests := []struct {
		name string
		data string
	}{
		{name: "malformed hex", data: strings.Repeat("z", 64) + "  " + asset},
		{name: "wrong length", data: "0123456789  " + asset},
		{name: "missing asset", data: strings.Repeat("0", 64) + "  another-asset"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := checksumFor([]byte(test.data), asset); err == nil {
				t.Fatal("checksumFor error = nil")
			}
		})
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

func releaseServerForAsset(t *testing.T, asset string, payload []byte) *httptest.Server {
	t.Helper()
	sum := sha256.Sum256(payload)
	var compressed bytes.Buffer
	gz := gzip.NewWriter(&compressed)
	_, _ = gz.Write(payload)
	_ = gz.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/checksums.txt":
			_, _ = fmt.Fprintf(w, "%x  %s\n", sum, asset)
		case "/" + asset + ".gz":
			_, _ = w.Write(compressed.Bytes())
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
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
