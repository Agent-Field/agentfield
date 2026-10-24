package codeaf

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func noSkip(t *testing.T) {
	t.Helper()
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "")
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "")
}

func fixture(t *testing.T) (Options, map[string]string, *atomic.Int32) {
	t.Helper()
	noSkip(t)
	payload := []byte("fixture executable")
	sum := sha256.Sum256(payload)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.URL.Path != "/release/codeaf-linux-amd64" {
			t.Errorf("unexpected request: %s", r.URL.Path)
		}
		_, _ = w.Write(payload)
	}))
	t.Cleanup(server.Close)
	return Options{Home: t.TempDir(), GOOS: "linux", GOARCH: "amd64", BaseURL: server.URL + "/release/"}, map[string]string{"codeaf-linux-amd64": fmt.Sprintf("%x", sum)}, &requests
}

// V6: every raw asset and embedded SHA-256 must match the release contract.
func TestAssetsAndPins(t *testing.T) {
	cases := []struct{ os, arch, name, pin string }{
		{"darwin", "amd64", "codeaf-darwin-amd64", "a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7"},
		{"darwin", "arm64", "codeaf-darwin-arm64", "e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1"},
		{"linux", "amd64", "codeaf-linux-amd64", "681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c"},
		{"linux", "arm64", "codeaf-linux-arm64", "c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78"},
		{"windows", "amd64", "codeaf-windows-amd64.exe", "ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d"},
		{"windows", "arm64", "codeaf-windows-arm64.exe", "bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80"},
	}
	for _, tc := range cases {
		name, ok := AssetName(tc.os, tc.arch)
		if !ok || name != tc.name || checksums[name] != tc.pin {
			t.Fatalf("asset/pin for %s/%s: %s, %s", tc.os, tc.arch, name, checksums[name])
		}
		want := "codeaf"
		if tc.os == "windows" {
			want += ".exe"
		}
		if BinaryName(tc.os) != want {
			t.Fatal("binary name")
		}
	}
	for _, platform := range [][2]string{{"freebsd", "amd64"}, {"linux", "386"}, {"windows", "386"}} {
		if name, ok := AssetName(platform[0], platform[1]); ok || name != "" {
			t.Fatal("unsupported platform accepted")
		}
	}
}

func TestEnsureRawMarkerForceRepair(t *testing.T) {
	opts, pins, requests := fixture(t)
	for i := 0; i < 2; i++ {
		if err := ensure(opts, pins); err != nil {
			t.Fatal(err)
		}
	}
	binary := filepath.Join(opts.Home, "bin", "codeaf")
	marker := filepath.Join(opts.Home, "bin", versionMarker)
	if requests.Load() != 1 || !alreadyInstalled(binary, marker) {
		t.Fatal("not idempotently installed")
	}
	data, _ := os.ReadFile(binary)
	if string(data) != "fixture executable" {
		t.Fatal("raw payload changed")
	}
	data, _ = os.ReadFile(marker)
	if string(data) != Version+"\n" {
		t.Fatalf("marker: %q", data)
	}
	opts.Force = true
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	opts.Force = false
	if requests.Load() != 2 {
		t.Fatal("force did not fetch")
	}
	if err := os.WriteFile(marker, []byte("old"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(binary); err != nil {
		t.Fatal(err)
	}
	if alreadyInstalled(binary, marker) {
		t.Fatal("marker alone is not installed")
	}
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		if err := os.Chmod(binary, 0644); err != nil {
			t.Fatal(err)
		}
		if err := ensure(opts, pins); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(marker, []byte(" \n"+Version+" \n"), 0644); err != nil {
		t.Fatal(err)
	}
	if !alreadyInstalled(binary, marker) {
		t.Fatal("marker whitespace not trimmed")
	}
}

func TestEnsureUsesEachPlatformAsset(t *testing.T) {
	for _, goos := range []string{"linux", "darwin", "windows"} {
		for _, arch := range []string{"amd64", "arm64"} {
			t.Run(goos+"/"+arch, func(t *testing.T) {
				noSkip(t)
				asset, _ := AssetName(goos, arch)
				sum := sha256.Sum256([]byte("raw"))
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/"+asset {
						t.Errorf("asset path %s", r.URL.Path)
					}
					_, _ = io.WriteString(w, "raw")
				}))
				defer server.Close()
				opts := Options{Home: t.TempDir(), GOOS: goos, GOARCH: arch, BaseURL: server.URL}
				if err := ensure(opts, map[string]string{asset: fmt.Sprintf("%x", sum)}); err != nil {
					t.Fatal(err)
				}
				if _, err := os.Stat(filepath.Join(opts.Home, "bin", BinaryName(goos))); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func TestEnsureRejectsMismatchAndPreservesOldInstall(t *testing.T) {
	opts, pins, _ := fixture(t)
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	opts.Force = true
	pins["codeaf-linux-amd64"] = strings.Repeat("0", 64)
	if err := ensure(opts, pins); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("error: %v", err)
	}
	binary := filepath.Join(opts.Home, "bin", "codeaf")
	if !alreadyInstalled(binary, filepath.Join(opts.Home, "bin", versionMarker)) {
		t.Fatal("old install damaged")
	}
	opts.Home = t.TempDir()
	if err := ensure(opts, pins); err == nil {
		t.Fatal("bad hash accepted")
	}
	for _, name := range []string{"codeaf", versionMarker} {
		if _, err := os.Stat(filepath.Join(opts.Home, "bin", name)); !os.IsNotExist(err) {
			t.Fatalf("left %s after mismatch", name)
		}
	}
	entries, _ := os.ReadDir(filepath.Join(opts.Home, "bin"))
	if len(entries) != 1 || entries[0].Name() != ".codeaf.lock" {
		t.Fatalf("temporary files left: %v", entries)
	}
}

func TestEnsureSkipAndUnsupportedNeverTouchDiskOrNetwork(t *testing.T) {
	for _, key := range []string{"AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE"} {
		for _, value := range []string{"1", "true", "TRUE", "Yes", "on", " ON "} {
			t.Run(key+"/"+value, func(t *testing.T) {
				opts, _, requests := fixture(t)
				t.Setenv(key, value)
				opts.Force = true
				if err := Ensure(opts); err != nil {
					t.Fatal(err)
				}
				if requests.Load() != 0 {
					t.Fatal("skip downloaded")
				}
				if _, err := os.Stat(filepath.Join(opts.Home, "bin")); !os.IsNotExist(err) {
					t.Fatal("skip wrote disk")
				}
			})
		}
	}
	noSkip(t)
	for _, value := range []string{"0", "false", "no", "off", "invalid", ""} {
		t.Setenv("AGENTFIELD_SKIP_CODEAF", value)
		if SkipRequested() {
			t.Fatalf("false skip %q", value)
		}
	}
	opts, _, requests := fixture(t)
	opts.GOARCH = "386"
	if err := Ensure(opts); err != nil || requests.Load() != 0 {
		t.Fatal("unsupported downloaded")
	}
}

func TestEnsureBaseURLPrecedence(t *testing.T) {
	opts, pins, requests := fixture(t)
	t.Setenv(baseURLEnv, opts.BaseURL)
	opts.BaseURL = ""
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	opts.Home = t.TempDir()
	opts.BaseURL = os.Getenv(baseURLEnv)
	t.Setenv(baseURLEnv, "http://127.0.0.1:1/unused")
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatal("mirror override unused")
	}
}

func TestAgentfieldHome(t *testing.T) {
	noSkip(t)
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	if got, err := agentfieldHome("explicit"); err != nil || got != "explicit" {
		t.Fatal(got, err)
	}
	if got, err := agentfieldHome(""); err != nil || got != os.Getenv("AGENTFIELD_HOME") {
		t.Fatal(got, err)
	}
	t.Setenv("AGENTFIELD_HOME", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if got, err := agentfieldHome(""); err != nil || got != filepath.Join(home, ".agentfield") {
		t.Fatal(got, err)
	}
	if runtime.GOOS == "linux" {
		t.Setenv("HOME", "")
		if _, err := agentfieldHome(""); err == nil {
			t.Fatal("missing home accepted")
		}
		if err := Ensure(Options{}); err == nil {
			t.Fatal("missing home accepted by ensure")
		}
	}
}

func TestEnsureConcurrent(t *testing.T) {
	opts, pins, requests := fixture(t)
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := ensure(opts, pins); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if runtime.GOOS != "windows" && requests.Load() != 1 {
		t.Fatalf("concurrent downloads: %d", requests.Load())
	}
}

func TestEnsureFilesystemErrors(t *testing.T) {
	for _, tc := range []struct{ name, block, want string }{
		{"bin directory", "bin", "create codeaf bin directory"},
		{"lock", "bin/.codeaf.lock", "lock codeaf installation"},
		{"destination", "bin/codeaf", "install codeaf"},
		{"marker", "bin/.codeaf.version", "record codeaf version"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts, pins, _ := fixture(t)
			path := filepath.Join(opts.Home, tc.block)
			if tc.name == "bin directory" {
				if err := os.WriteFile(path, []byte("file"), 0644); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.MkdirAll(path, 0755); err != nil {
					t.Fatal(err)
				}
			}
			if err := ensure(opts, pins); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want %q, got %v", tc.want, err)
			}
			if tc.name == "destination" {
				if _, err := os.Stat(filepath.Join(opts.Home, "bin", versionMarker)); !os.IsNotExist(err) {
					t.Fatal("version marker published before binary rename succeeded")
				}
			}
		})
	}
}

func TestEnsureBestEffortSharesAttemptAndQuiet(t *testing.T) {
	noSkip(t)
	resetBestEffort()
	t.Cleanup(func() { resetBestEffort() })
	var count atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { count.Add(1); http.Error(w, "bad", 503) }))
	defer server.Close()
	opts := Options{Home: t.TempDir(), BaseURL: server.URL}
	var warnings bytes.Buffer
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := EnsureBestEffort(opts, &warnings); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if count.Load() != 1 || strings.Count(warnings.String(), "warning:") != 1 {
		t.Fatal(count.Load(), warnings.String())
	}
	resetBestEffort()
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "yes")
	warnings.Reset()
	_ = EnsureBestEffort(opts, &warnings)
	if warnings.Len() != 0 || count.Load() != 1 {
		t.Fatal("skip was not quiet")
	}
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "")
	_ = EnsureBestEffort(opts, nil)
	if count.Load() != 2 {
		t.Fatal("skip consumed guard")
	}
	resetBestEffort()
	opts, pins, _ := fixture(t)
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	warnings.Reset()
	_ = EnsureBestEffort(opts, &warnings)
	if warnings.Len() != 0 {
		t.Fatal("installed not quiet")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type zeros struct{}

func (zeros) Read(b []byte) (int, error) { clear(b); return len(b), nil }

func TestDownloadFailuresAndCap(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		length int64
		body   io.Reader
		want   string
	}{
		{"http error", 404, 0, strings.NewReader("missing"), "404"},
		{"advertised size", 200, maxDownloadBytes + 1, strings.NewReader(""), "exceeds"},
		{"stream size", 200, -1, io.LimitReader(zeros{}, maxDownloadBytes+1), "exceeds"},
		{"short body", 200, 100, strings.NewReader("short"), "checksum mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Status: fmt.Sprint(tc.status), ContentLength: tc.length, Body: io.NopCloser(tc.body), Header: make(http.Header)}, nil
			})}
			if err := download(client, "https://fixture.invalid/codeaf", io.Discard, ""); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatal(err)
			}
		})
	}
	opts, _, _ := fixture(t)
	opts.BaseURL = ":invalid"
	if err := Ensure(opts); err == nil {
		t.Fatal("bad URL accepted")
	}
}

func TestEnsureDefaultsUsePinnedReleaseAndRuntimePlatform(t *testing.T) {
	noSkip(t)
	t.Setenv(baseURLEnv, "")
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	asset, supported := AssetName(runtime.GOOS, runtime.GOARCH)
	if !supported {
		t.Skip("host platform not installable")
	}
	sum := sha256.Sum256([]byte("fixture"))
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != defaultBaseURL+"/"+asset {
			t.Errorf("default URL: %s", r.URL)
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("fixture")), Header: make(http.Header)}, nil
	})}
	if err := ensure(Options{Client: client}, map[string]string{asset: fmt.Sprintf("%x", sum)}); err != nil {
		t.Fatal(err)
	}
	if !alreadyInstalled(filepath.Join(home, "bin", BinaryName(runtime.GOOS)), filepath.Join(home, "bin", versionMarker)) {
		t.Fatal("default home not used")
	}
}

func TestEnsurePublicRejectsUnpinnedMirrorPayload(t *testing.T) {
	opts, _, requests := fixture(t)
	if err := Ensure(opts); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("untrusted mirror accepted: %v", err)
	}
	if requests.Load() != 1 {
		t.Fatal("must fetch only the raw asset, never trust a downloaded checksums file")
	}
}

func resetBestEffort() {
	bestEffort.Lock()
	bestEffort.lastDownload = time.Time{}
	bestEffort.Unlock()
}

// C1: CLI installation and read-only lookup never create a system-temp fallback.
func TestUnwritableHomeDoesNotUseTemp(t *testing.T) {
	home, temporary, searchPath := t.TempDir(), t.TempDir(), t.TempDir()
	blocked := filepath.Join(home, "bin")
	if err := os.WriteFile(blocked, []byte("not a directory"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("TMPDIR", temporary)
	t.Setenv("PATH", searchPath)
	t.Setenv("CODEAF_BIN", "")
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "0")
	if err := Ensure(Options{}); err == nil || !strings.Contains(err.Error(), blocked) {
		t.Fatalf("install error=%v", err)
	}
	if _, err := Lookup(""); err == nil {
		t.Fatal("found an unavailable binary")
	}
	entries, err := os.ReadDir(temporary)
	if err != nil || len(entries) != 0 {
		t.Fatalf("temp fallback entries=%v err=%v", entries, err)
	}
}
