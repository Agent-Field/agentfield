package harness

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func expectedCodeAFTempBase() string {
	name := "agentfield-codeaf"
	if runtime.GOOS != "windows" {
		name += fmt.Sprintf("-%d", os.Getuid())
	}
	return filepath.Join(os.TempDir(), name)
}

func blockCodeAFHome(t *testing.T) {
	t.Helper()
	home := filepath.Join(t.TempDir(), "home-file")
	require.NoError(t, os.WriteFile(home, []byte("not a directory"), 0o600))
	t.Setenv("AGENTFIELD_HOME", home)
}

// N2 / A5 / V5-V8: existing fallback directories must become private.
func TestCodeAFTempExistingPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix directory permissions")
	}
	for _, surface := range []string{"resolution", "state", "doctor", "download"} {
		t.Run(surface, func(t *testing.T) {
			isolateCodeAF(t)
			blockCodeAFHome(t)
			t.Setenv("PATH", t.TempDir())
			_, binary := codeAFManagedPaths()
			base := filepath.Dir(filepath.Dir(binary))
			leaf := filepath.Dir(binary)
			if surface == "state" {
				leaf = filepath.Join(base, "state")
			}
			require.NoError(t, os.MkdirAll(leaf, 0o777))
			for _, dir := range []string{base, leaf} {
				require.NoError(t, os.Chmod(dir, 0o777))
			}
			switch surface {
			case "resolution", "doctor":
				writeManagedCodeAF(t, binary, "codeaf v0.8.2 built fixture")
				if surface == "resolution" {
					resolved, err := resolveCodeAF(context.Background(), "")
					require.NoError(t, err)
					assert.Equal(t, binary, resolved.path)
				} else {
					reports, err := Doctor(context.Background(), DoctorOptions{Providers: []string{ProviderCodeAF}, Probe: true})
					require.NoError(t, err)
					assert.Equal(t, binary, reports[0].Binary)
					assert.Equal(t, "v0.8.2", reports[0].Version)
				}
			case "state":
				env, err := codeAFEnvironment(Options{}, "")
				require.NoError(t, err)
				assert.Equal(t, leaf, env["CODEAF_HOME"])
			case "download":
				data := "fixture executable"
				useCodeAFTestHash(t, data)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, data) }))
				defer server.Close()
				t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
				path, err := downloadManagedCodeAF(context.Background())
				require.NoError(t, err)
				assert.Equal(t, binary, path)
				assert.True(t, codeAFInstalled(path))
			}
			for _, dir := range []string{base, leaf} {
				info, err := os.Lstat(dir)
				require.NoError(t, err)
				assert.Equal(t, os.FileMode(0o700), info.Mode().Perm(), dir)
			}
		})
	}
}

// A5 / V5-V6 / V11: the old shared temp directory is never trusted.
func TestCodeAFTempPerUserPathsAndOldBinary(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows already uses a per-user temp directory")
	}
	isolateCodeAF(t)
	blockCodeAFHome(t)
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	safe := writeCodeAFCLI(t, filepath.Join(dir, "codeaf"), "{}", 0)
	old := filepath.Join(os.TempDir(), "agentfield-codeaf", CodeAFVersion, codeAFExecutable(runtime.GOOS))
	writeManagedCodeAF(t, old, "codeaf v0.8.2 built planted")
	_, managedTemp := codeAFManagedPaths()
	assert.Equal(t, filepath.Join(expectedCodeAFTempBase(), CodeAFVersion, codeAFExecutable(runtime.GOOS)), managedTemp)
	resolved, err := resolveCodeAF(context.Background(), "")
	require.NoError(t, err)
	assert.Equal(t, safe, resolved.path)
	reports, err := Doctor(context.Background(), DoctorOptions{Providers: []string{ProviderCodeAF}})
	require.NoError(t, err)
	assert.Equal(t, safe, reports[0].Binary)
	env, err := codeAFEnvironment(Options{}, "")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(expectedCodeAFTempBase(), "state"), env["CODEAF_HOME"])
}

func TestCodeAFTempCachedResolutionRechecksTrust(t *testing.T) {
	isolateCodeAF(t)
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	safe := writeCodeAFCLI(t, filepath.Join(dir, "codeaf"), "{}", 0)
	_, binary := codeAFManagedPaths()
	writeManagedCodeAF(t, binary, "{}")
	resolved, err := resolveCodeAF(context.Background(), "")
	require.NoError(t, err)
	require.Equal(t, binary, resolved.path)
	leaf := filepath.Dir(binary)
	require.NoError(t, os.RemoveAll(leaf))
	outside := t.TempDir()
	writeManagedCodeAF(t, filepath.Join(outside, codeAFExecutable(runtime.GOOS)), "{}")
	require.NoError(t, os.Symlink(outside, leaf))
	resolved, err = resolveCodeAF(context.Background(), "")
	require.NoError(t, err)
	assert.Equal(t, safe, resolved.path)
}

// A5: reject directories through symlinks and non-directories before access.
func TestCodeAFTempRejectsUntrustedDirectories(t *testing.T) {
	for _, surface := range []string{"resolution", "doctor", "download", "state"} {
		for _, level := range []string{"base", "leaf"} {
			for _, kind := range []string{"symlink", "file"} {
				t.Run(surface+"/"+level+"/"+kind, func(t *testing.T) {
					isolateCodeAF(t)
					blockCodeAFHome(t)
					logs := captureCodeAFLogs(t)
					dir := t.TempDir()
					t.Setenv("PATH", dir)
					safe := writeCodeAFCLI(t, filepath.Join(dir, "codeaf"), "{}", 0)
					_, binary := codeAFManagedPaths()
					base := filepath.Dir(filepath.Dir(binary))
					leaf := filepath.Dir(binary)
					if surface == "state" {
						leaf = filepath.Join(base, "state")
					}
					untrusted := base
					if level == "leaf" {
						require.NoError(t, os.MkdirAll(base, 0o700))
						untrusted = leaf
					}
					outside := t.TempDir()
					if kind == "symlink" {
						require.NoError(t, os.Symlink(outside, untrusted))
						if surface == "resolution" || surface == "doctor" {
							writeManagedCodeAF(t, binary, "codeaf v0.8.2 built planted")
						}
					} else {
						require.NoError(t, os.WriteFile(untrusted, []byte("blocked"), 0o600))
					}
					before, err := os.ReadDir(outside)
					require.NoError(t, err)
					var requests atomic.Int32
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						requests.Add(1)
						_, _ = io.WriteString(w, "fixture")
					}))
					defer server.Close()
					t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
					useCodeAFTestHash(t, "fixture")
					for i := 0; i < 2; i++ {
						switch surface {
						case "resolution":
							resolved, err := resolveCodeAF(context.Background(), "")
							require.NoError(t, err)
							assert.Equal(t, safe, resolved.path)
						case "doctor":
							reports, err := Doctor(context.Background(), DoctorOptions{Providers: []string{ProviderCodeAF}})
							require.NoError(t, err)
							assert.Equal(t, safe, reports[0].Binary)
						case "download":
							_, err := downloadManagedCodeAF(context.Background())
							require.Error(t, err)
						case "state":
							env, err := codeAFEnvironment(Options{}, "")
							require.Error(t, err)
							assert.Empty(t, env)
						}
					}
					assert.Zero(t, requests.Load())
					assert.Equal(t, 1, strings.Count(logs.String(), "WARNING"), logs.String())
					after, err := os.ReadDir(outside)
					require.NoError(t, err)
					assert.Equal(t, before, after, "rejected symlink targets must not be modified")
				})
			}
		}
	}
}
