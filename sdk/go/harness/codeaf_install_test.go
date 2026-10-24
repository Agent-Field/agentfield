package harness

import (
	"context"
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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeCodeAFCLI(t *testing.T, path, stdout string, code int) string {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' '%s'\nexit %d\n", stdout, code)
	require.NoError(t, os.WriteFile(path, []byte(script), 0o755))
	return path
}

func writeManagedCodeAF(t *testing.T, path, stdout string) string {
	t.Helper()
	writeCodeAFCLI(t, path, stdout, 0)
	require.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(path), codeAFVersionFilename), []byte(CodeAFVersion+"\n"), 0o644))
	return path
}

func useCodeAFTestHash(t *testing.T, data string) string {
	t.Helper()
	asset, old := codeAFAsset(runtime.GOOS, runtime.GOARCH)
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
	// Test-only injection; production has no env/config hash override.
	codeAFHashes[asset] = hash
	t.Cleanup(func() { codeAFHashes[asset] = old })
	return hash
}

// V1, V3, V4: default and alias use CodeAF; both legacy pins preserve the old adapter.
func TestCodeAFDefaultAliasAndLegacy(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv(ProviderEnvVar, "")
	logs := captureCodeAFLogs(t)
	home := codeAFManagedPaths()
	writeManagedCodeAF(t, home, `{"ok":true,"stop":"done","answer":"canonical","spend_usd":0,"tokens":{"in":10,"out":5}}`)
	runner := NewRunner(Options{})
	result, err := runner.Run(context.Background(), "task", nil, nil, Options{})
	require.NoError(t, err)
	assert.Equal(t, ProviderCodeAF, result.Provider)
	assert.Equal(t, "canonical", result.Result)
	assert.Equal(t, 15, result.TotalTokens)
	require.NotNil(t, result.CostUSD)
	assert.Zero(t, *result.CostUSD)
	for i := 0; i < 2; i++ {
		result, err = runner.Run(context.Background(), "task", nil, nil, Options{Provider: ProviderAforge})
		require.NoError(t, err)
		assert.Equal(t, ProviderCodeAF, result.Provider)
	}
	assert.Equal(t, 1, strings.Count(logs.String(), "provider 'aforge' now runs CodeAF"))
	legacy := writeCodeAFCLI(t, filepath.Join(t.TempDir(), "aforge"), `{"text":"legacy","stop":"done","turns":2,"usage":{"cost":0.1}}`, 0)
	for _, pin := range []string{"option", "env"} {
		t.Run(pin, func(t *testing.T) {
			opts := Options{Provider: ProviderAforge}
			if pin == "option" {
				opts.BinPath = legacy
			} else {
				t.Setenv("AFORGE_BIN", legacy)
			}
			result, err := runner.Run(context.Background(), "task", nil, nil, opts)
			require.NoError(t, err)
			assert.Equal(t, ProviderAforge, result.Provider)
			assert.Equal(t, "legacy", result.Result)
			assert.Equal(t, 2, result.NumTurns)
		})
	}
	t.Setenv("AFORGE_BIN", legacy)
	result, err = runner.Run(context.Background(), "task", nil, nil, Options{Provider: ProviderCodeAF})
	require.NoError(t, err)
	assert.Equal(t, "canonical", result.Result)
}

// V5: the resolution matrix includes all managed and unmanaged fallbacks.
func TestCodeAFResolutionOrder(t *testing.T) {
	for _, tc := range []struct {
		name, want                                                  string
		override, env, home, download, path, legacyHome, legacyPath bool
		requests                                                    int32
	}{
		{name: "override", want: "override", override: true, env: true, home: true, path: true},
		{name: "env", want: "env", env: true, home: true, path: true},
		{name: "home", want: "home", home: true, download: true, path: true},
		{name: "download", want: "home", download: true, path: true, requests: 1},
		{name: "path", want: "path", path: true, legacyHome: true, legacyPath: true},
		{name: "legacy home", want: "legacyHome", legacyHome: true, legacyPath: true},
		{name: "legacy path", want: "legacyPath", legacyPath: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolateCodeAF(t)
			paths := map[string]string{}
			paths["home"] = codeAFManagedPaths()
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			paths["path"] = filepath.Join(dir, "codeaf")
			paths["legacyHome"] = filepath.Join(filepath.Dir(paths["home"]), "aforge")
			paths["legacyPath"] = filepath.Join(dir, "aforge")
			paths["override"] = filepath.Join(t.TempDir(), "override")
			paths["env"] = filepath.Join(t.TempDir(), "env")
			for name, enabled := range map[string]bool{"override": tc.override, "env": tc.env, "home": tc.home, "path": tc.path, "legacyHome": tc.legacyHome, "legacyPath": tc.legacyPath} {
				if !enabled {
					continue
				}
				if name == "home" {
					writeManagedCodeAF(t, paths[name], "{}")
				} else {
					writeCodeAFCLI(t, paths[name], "{}", 0)
				}
			}
			if tc.env {
				t.Setenv("CODEAF_BIN", paths["env"])
			}
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests.Add(1); _, _ = io.WriteString(w, "executable") }))
			defer server.Close()
			t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL+"/")
			if tc.download {
				t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
				useCodeAFTestHash(t, "executable")
			}
			override := ""
			if tc.override {
				override = paths["override"]
			}
			binary, err := resolveCodeAF(context.Background(), override)
			require.NoError(t, err)
			assert.Equal(t, paths[tc.want], binary.path)
			assert.Equal(t, strings.HasPrefix(tc.want, "legacy"), binary.legacy)
			assert.Equal(t, tc.requests, requests.Load())
		})
	}
}

func TestCodeAFResolutionFailureAndFallback(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests.Add(1); http.Error(w, "failure", 503) }))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	_, err := resolveCodeAF(context.Background(), "")
	require.ErrorIs(t, err, ErrProviderUnavailable)
	var unavailable *ProviderUnavailableError
	require.ErrorAs(t, err, &unavailable)
	assert.Equal(t, ProviderCodeAF, unavailable.Provider)
	for _, part := range []string{"af codeaf ensure", "CODEAF_BIN", "503"} {
		assert.Contains(t, err.Error(), part)
	}
	_, err = resolveCodeAF(context.Background(), "")
	require.Error(t, err)
	assert.Equal(t, int32(1), requests.Load())
	for _, name := range []string{"codeaf", "aforge"} {
		t.Run(name, func(t *testing.T) {
			resetCodeAFResolution()
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			stdout := `{"ok":true,"stop":"done","answer":"new"}`
			if name == "aforge" {
				stdout = `{"text":"legacy","stop":"done"}`
			}
			writeCodeAFCLI(t, filepath.Join(dir, name), stdout, 0)
			logs := captureCodeAFLogs(t)
			for i := 0; i < 2; i++ {
				raw, err := NewCodeAFProvider("").Execute(context.Background(), "task", Options{})
				require.NoError(t, err)
				assert.False(t, raw.IsError)
				assert.Equal(t, name, raw.Provider)
			}
			if name == "aforge" {
				assert.Equal(t, 1, strings.Count(logs.String(), "falling back to the legacy aforge binary"))
			}
		})
	}
	_, err = resolveCodeAF(context.Background(), "/missing/explicit-codeaf")
	require.ErrorIs(t, err, ErrProviderUnavailable)
	assert.Contains(t, err.Error(), "/missing/explicit-codeaf")
}

// V6: raw bytes, platform URL, embedded verification, atomic install, marker last.
func TestCodeAFManagedDownload(t *testing.T) {
	isolateCodeAF(t)
	data := "plain executable bytes"
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
	home := codeAFManagedPaths()
	asset, _ := codeAFAsset(runtime.GOOS, runtime.GOARCH)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/release/"+asset, r.URL.Path)
		assert.NoFileExists(t, home)
		assert.NoFileExists(t, filepath.Join(filepath.Dir(home), codeAFVersionFilename))
		_, _ = io.WriteString(w, data)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL+"/release///")
	path, err := installCodeAF(context.Background(), home, asset, hash)
	require.NoError(t, err)
	assert.Equal(t, home, path)
	assert.True(t, codeAFInstalled(path))
	content, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, data, string(content))
	marker, err := os.ReadFile(filepath.Join(filepath.Dir(path), codeAFVersionFilename))
	require.NoError(t, err)
	assert.Equal(t, CodeAFVersion+"\n", string(marker))
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
	leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".codeaf-download-*"))
	assert.Empty(t, leftovers)
}

type zeroCodeAFReader struct{}

func (zeroCodeAFReader) Read(p []byte) (int, error) { clear(p); return len(p), nil }

func TestCodeAFDownloadRejections(t *testing.T) {
	for _, tc := range []struct{ name, want string }{
		{"checksum", "sha256 mismatch"}, {"status", "HTTP 404"}, {"large header", "128 MiB"}, {"large stream", "128 MiB"}, {"truncated", "unexpected EOF"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolateCodeAF(t)
			home := codeAFManagedPaths()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch tc.name {
				case "status":
					w.WriteHeader(404)
				case "large header":
					w.Header().Set("Content-Length", fmt.Sprint(maxCodeAFDownloadSize+1))
				case "large stream":
					w.(http.Flusher).Flush()
					_, _ = io.CopyN(w, zeroCodeAFReader{}, maxCodeAFDownloadSize+1)
				case "truncated":
					w.Header().Set("Content-Length", "100")
					_, _ = io.WriteString(w, "short")
				default:
					_, _ = io.WriteString(w, "untrusted bytes")
				}
			}))
			defer server.Close()
			t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
			_, err := installCodeAF(context.Background(), home, "codeaf-linux-amd64", strings.Repeat("0", 64))
			require.ErrorContains(t, err, tc.want)
			assert.NoFileExists(t, home)
			assert.NoFileExists(t, filepath.Join(filepath.Dir(home), codeAFVersionFilename))
			leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(home), ".codeaf-download-*"))
			assert.Empty(t, leftovers)
		})
	}
}

func TestCodeAFDownloadFilesystemErrors(t *testing.T) {
	isolateCodeAF(t)
	data := "fixture"
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, data) }))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	blocked := filepath.Join(t.TempDir(), "file")
	require.NoError(t, os.WriteFile(blocked, []byte("not a directory"), 0o644))
	t.Setenv("AGENTFIELD_HOME", blocked)
	home := codeAFManagedPaths()
	_, err := installCodeAF(context.Background(), home, "asset", hash)
	require.ErrorContains(t, err, "cannot install CodeAF into")
	assert.Empty(t, findManagedCodeAF())
	target := filepath.Join(t.TempDir(), "directory")
	require.NoError(t, os.Mkdir(target, 0o755))
	_, err = installCodeAF(context.Background(), target, "asset", hash)
	require.Error(t, err)
	assert.NoFileExists(t, filepath.Join(filepath.Dir(target), codeAFVersionFilename))
	target = filepath.Join(t.TempDir(), "codeaf")
	require.NoError(t, os.Mkdir(filepath.Join(filepath.Dir(target), codeAFVersionFilename), 0o755))
	_, err = installCodeAF(context.Background(), target, "asset", hash)
	require.Error(t, err)
	assert.False(t, codeAFInstalled(target))
}

func TestCodeAFConcurrentLazyDownload(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	data := "#!/bin/sh\nprintf '%s\\n' '{\"ok\":true,\"stop\":\"done\",\"answer\":\"downloaded\"}'\n"
	useCodeAFTestHash(t, data)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		_, _ = io.WriteString(w, data)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	provider := NewCodeAFProvider("")
	assert.Zero(t, requests.Load(), "construction never downloads")
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			raw, err := provider.Execute(context.Background(), "task", Options{})
			if assert.NoError(t, err) {
				assert.Equal(t, "downloaded", raw.Result)
				assert.Equal(t, ProviderCodeAF, raw.Provider)
			}
		}()
	}
	wg.Wait()
	assert.Equal(t, int32(1), requests.Load())
}

func TestCodeAFSkipSwitches(t *testing.T) {
	for _, key := range []string{"AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE"} {
		for _, value := range []string{"1", "TrUe", "YES", "on"} {
			t.Run(key+value, func(t *testing.T) {
				isolateCodeAF(t)
				t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
				t.Setenv(key, value)
				var requests atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests.Add(1) }))
				defer server.Close()
				t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
				_, err := resolveCodeAF(context.Background(), "")
				require.NoError(t, err)
				assert.Zero(t, requests.Load())
			})
		}
	}
}

func TestCodeAFInstalledRequiresExecutableAndMarker(t *testing.T) {
	isolateCodeAF(t)
	home := codeAFManagedPaths()
	assert.False(t, codeAFInstalled(home))
	require.NoError(t, os.MkdirAll(filepath.Dir(home), 0o755))
	require.NoError(t, os.WriteFile(home, []byte("binary"), 0o644))
	marker := filepath.Join(filepath.Dir(home), codeAFVersionFilename)
	require.NoError(t, os.WriteFile(marker, []byte(CodeAFVersion+"\n"), 0o644))
	assert.False(t, codeAFInstalled(home))
	require.NoError(t, os.Chmod(home, 0o755))
	assert.True(t, codeAFInstalled(home))
	require.NoError(t, os.WriteFile(marker, []byte("old-version"), 0o644))
	assert.False(t, codeAFInstalled(home))
	assert.Empty(t, findManagedCodeAF())
	require.NoError(t, os.Remove(home))
	require.NoError(t, os.Mkdir(home, 0o755))
	assert.False(t, codeAFInstalled(home))
}

func TestCodeAFPlatformPinsAndHome(t *testing.T) {
	assert.Equal(t, "v0.8.2", CodeAFVersion)
	assert.Equal(t, "https://github.com/Agent-Field/CodeAF/releases/download/v0.8.2", CodeAFBaseURL)
	for _, tc := range []struct{ goos, arch, hash string }{
		{"darwin", "amd64", "a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7"},
		{"darwin", "arm64", "e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1"},
		{"linux", "amd64", "681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c"},
		{"linux", "arm64", "c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78"},
		{"windows", "amd64", "ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d"},
		{"windows", "arm64", "bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80"},
	} {
		asset, hash := codeAFAsset(tc.goos, tc.arch)
		want := "codeaf-" + tc.goos + "-" + tc.arch
		if tc.goos == "windows" {
			want += ".exe"
		}
		assert.Equal(t, want, asset)
		assert.Equal(t, tc.hash, hash)
	}
	_, hash := codeAFAsset("freebsd", "amd64")
	assert.Empty(t, hash)
	_, hash = codeAFAsset("linux", "386")
	assert.Empty(t, hash)
	assert.Equal(t, "codeaf.exe", codeAFExecutable("windows"))
	assert.Equal(t, "codeaf", codeAFExecutable("linux"))
	t.Setenv("AGENTFIELD_HOME", "")
	home, err := os.UserHomeDir()
	require.NoError(t, err)
	managed, err := agentFieldHome()
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(home, ".agentfield"), managed)
	t.Setenv("HOME", "")
	_, err = agentFieldHome()
	require.Error(t, err)
	assert.Empty(t, codeAFManagedPaths())
}

func TestCodeAFDownloadTransportErrors(t *testing.T) {
	isolateCodeAF(t)
	home := codeAFManagedPaths()
	for _, base := range []string{"http://[::1", "http://127.0.0.1:1"} {
		t.Setenv("AGENTFIELD_CODEAF_BASE_URL", base)
		_, err := installCodeAF(context.Background(), home, "asset", "hash")
		require.Error(t, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _ = resolveCodeAF(ctx, "")
	codeAFResolver.Lock()
	assert.Empty(t, codeAFResolver.cache)
	codeAFResolver.Unlock()
}

func TestCodeAFExplicitBareNameOverridesManaged(t *testing.T) {
	isolateCodeAF(t)
	home := codeAFManagedPaths()
	writeManagedCodeAF(t, home, `{"ok":true,"stop":"done","answer":"managed"}`)
	dir := t.TempDir()
	path := writeCodeAFCLI(t, filepath.Join(dir, "codeaf"), "{}", 0)
	t.Setenv("PATH", dir)
	t.Setenv("CODEAF_BIN", home)
	binary, err := resolveCodeAF(context.Background(), "codeaf")
	require.NoError(t, err)
	assert.Equal(t, path, binary.path)
}
