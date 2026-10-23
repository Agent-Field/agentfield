package harness

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Finding 1 / A1 / V5-V6: repair is visible without resetting process state.
func TestCodeAFResolutionRecoversAfterFailure(t *testing.T) {
	for _, location := range []string{"home", "temp", "path", "legacy home", "legacy path", "option", "env"} {
		t.Run(location, func(t *testing.T) {
			isolateCodeAF(t)
			t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
			pathDir := t.TempDir()
			t.Setenv("PATH", pathDir)
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				requests.Add(1)
				w.WriteHeader(http.StatusServiceUnavailable)
			}))
			defer server.Close()
			t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
			home, temp := codeAFManagedPaths()
			paths := map[string]string{
				"home": home, "temp": temp, "path": filepath.Join(pathDir, "codeaf"),
				"legacy home": filepath.Join(filepath.Dir(home), "aforge"), "legacy path": filepath.Join(pathDir, "aforge"),
				"option": filepath.Join(t.TempDir(), "explicit"), "env": filepath.Join(t.TempDir(), "explicit"),
			}
			override := ""
			if location == "option" {
				override = paths[location]
			} else if location == "env" {
				t.Setenv("CODEAF_BIN", paths[location])
			}
			_, err := resolveCodeAF(context.Background(), override)
			require.ErrorIs(t, err, ErrProviderUnavailable)
			if location == "home" || location == "temp" {
				writeManagedCodeAF(t, paths[location], "{}")
			} else {
				writeCodeAFCLI(t, paths[location], "{}", 0)
			}
			binary, err := resolveCodeAF(context.Background(), override)
			require.NoError(t, err)
			assert.Equal(t, paths[location], binary.path)
			assert.Equal(t, location == "legacy home" || location == "legacy path", binary.legacy)
			wantRequests := int32(1)
			if location == "option" || location == "env" {
				wantRequests = 0
			}
			assert.Equal(t, wantRequests, requests.Load(), "repair must not trigger another download during cooldown")
		})
	}
}

func TestCodeAFFailedConcurrentDownloadIsThrottled(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("PATH", t.TempDir())
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := resolveCodeAF(context.Background(), "")
			assert.ErrorIs(t, err, ErrProviderUnavailable)
			assert.ErrorContains(t, err, "503")
		}()
	}
	wg.Wait()
	assert.Equal(t, int32(1), requests.Load())
	home, _ := codeAFManagedPaths()
	writeManagedCodeAF(t, home, `{"ok":true,"stop":"done","answer":"repaired"}`)
	raw, err := NewCodeAFProvider("").Execute(context.Background(), "fixture only", Options{})
	require.NoError(t, err)
	assert.Equal(t, "repaired", raw.Result)
}

func TestCodeAFDownloadRetriesAfterCooldown(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("PATH", t.TempDir())
	data := "fixture executable"
	useCodeAFTestHash(t, data)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if requests.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, data)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	_, err := resolveCodeAF(context.Background(), "")
	require.ErrorContains(t, err, "503")
	codeAFResolver.Lock()
	assert.Empty(t, codeAFResolver.cache, "negative resolutions are never cached")
	codeAFResolver.lastDownloadAttempt = time.Now().Add(-59 * time.Second)
	codeAFResolver.Unlock()
	_, err = resolveCodeAF(context.Background(), "")
	require.ErrorContains(t, err, "503")
	assert.Equal(t, int32(1), requests.Load())
	codeAFResolver.Lock()
	codeAFResolver.lastDownloadAttempt = time.Now().Add(-61 * time.Second)
	codeAFResolver.Unlock()
	binary, err := resolveCodeAF(context.Background(), "")
	require.NoError(t, err)
	assert.True(t, codeAFInstalled(binary.path))
	assert.Equal(t, int32(2), requests.Load())
	_, err = resolveCodeAF(context.Background(), "")
	require.NoError(t, err)
	assert.Equal(t, int32(2), requests.Load(), "successful resolution is cached")
}

// Finding 2 / A2 / V6-V8: the temp install must be usable through execution.
func TestCodeAFDownloadExecutesWithUnwritableHome(t *testing.T) {
	for _, blocked := range []string{"home is a file", "home is read-only", "existing state is read-only", "overlay home is a file"} {
		t.Run(blocked, func(t *testing.T) {
			isolateCodeAF(t)
			t.Setenv("PATH", t.TempDir())
			t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
			home := os.Getenv("AGENTFIELD_HOME")
			opts := Options{}
			switch blocked {
			case "home is a file", "overlay home is a file":
				home = filepath.Join(t.TempDir(), "file")
				require.NoError(t, os.WriteFile(home, []byte("blocked"), 0o600))
				if blocked == "overlay home is a file" {
					opts.Env = map[string]string{"AGENTFIELD_HOME": home}
				} else {
					t.Setenv("AGENTFIELD_HOME", home)
				}
			case "home is read-only":
				require.NoError(t, os.Chmod(home, 0o555))
				t.Cleanup(func() { _ = os.Chmod(home, 0o755) })
				requireCodeAFReadOnlyDirectory(t, home)
			case "existing state is read-only":
				state := filepath.Join(home, "codeaf")
				require.NoError(t, os.Mkdir(state, 0o555))
				t.Cleanup(func() { _ = os.Chmod(state, 0o755) })
				requireCodeAFReadOnlyDirectory(t, state)
			}
			data := `#!/bin/sh
if [ "$CODEAF_HOME" != "$EXPECTED_STATE" ]; then exit 10; fi
if ! ( : > "$CODEAF_HOME/child-write" ); then exit 11; fi
printf '%s\n' '{"ok":true,"stop":"done","answer":"temp state works"}'
`
			useCodeAFTestHash(t, data)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, data) }))
			defer server.Close()
			t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
			state := filepath.Join(os.TempDir(), "agentfield-codeaf", "state")
			t.Setenv("EXPECTED_STATE", state)
			raw, err := NewCodeAFProvider("").Execute(context.Background(), "fixture only", opts)
			require.NoError(t, err)
			require.False(t, raw.IsError, raw.ErrorMessage)
			assert.Equal(t, "temp state works", raw.Result)
			info, err := os.Stat(state)
			require.NoError(t, err)
			assert.Equal(t, os.FileMode(0o700), info.Mode().Perm())
			assert.FileExists(t, filepath.Join(state, "child-write"))
			if blocked == "home is a file" || blocked == "home is read-only" {
				_, temp := codeAFManagedPaths()
				assert.True(t, codeAFInstalled(temp))
			}
		})
	}
}

func requireCodeAFReadOnlyDirectory(t *testing.T, path string) {
	t.Helper()
	probe, err := os.CreateTemp(path, ".permission-probe-*")
	if err == nil {
		_ = probe.Close()
		_ = os.Remove(probe.Name())
		t.Skip("current user can write to mode-0555 directories")
	}
	require.ErrorIs(t, err, os.ErrPermission)
}

func TestCodeAFExplicitStateHomeIsPreserved(t *testing.T) {
	for _, source := range []string{"inherited", "overlay"} {
		t.Run(source, func(t *testing.T) {
			isolateCodeAF(t)
			blocked := filepath.Join(t.TempDir(), "home-file")
			require.NoError(t, os.WriteFile(blocked, []byte("blocked"), 0o600))
			t.Setenv("AGENTFIELD_HOME", blocked)
			explicit := filepath.Join(t.TempDir(), "explicit-state")
			opts := Options{}
			if source == "inherited" {
				t.Setenv("CODEAF_HOME", explicit)
			} else {
				opts.Env = map[string]string{"CODEAF_HOME": explicit}
			}
			path := filepath.Join(t.TempDir(), "codeaf")
			script := "#!/bin/sh\nprintf '{\"ok\":true,\"stop\":\"done\",\"answer\":\"%s\"}\\n' \"$CODEAF_HOME\"\n"
			require.NoError(t, os.WriteFile(path, []byte(script), 0o755))
			raw, err := NewCodeAFProvider(path).Execute(context.Background(), "fixture only", opts)
			require.NoError(t, err)
			assert.Equal(t, explicit, raw.Result)
			assert.DirExists(t, explicit)
			assert.NoDirExists(t, filepath.Join(os.TempDir(), "agentfield-codeaf", "state"))
		})
	}
}

// Finding 4 / V8: presence and an empty value are distinct in the real child.
func TestCodeAFChildPreservesEmptyOverlay(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("CODEAF_MODELS", "inherited")
	t.Setenv("AFORGE_MODELS", "inherited")
	t.Setenv("CODEAF_HOME", filepath.Join(t.TempDir(), "inherited"))
	t.Setenv("CI", "true")
	path := filepath.Join(t.TempDir(), "codeaf")
	data := `#!/bin/sh
for key in CODEAF_TELEMETRY_ENDPOINT CODEAF_HOME CODEAF_TELEMETRY CODEAF_NO_UPDATE_CHECK CODEAF_MODELS AFORGE_MODELS OR_APP_NAME; do
  eval 'present=${'"$key"'+x}'
  eval 'value=${'"$key"'-}'
  if [ "$present" != x ] || [ -n "$value" ]; then exit 10; fi
done
printf '%s\n' '{"ok":true,"stop":"done","answer":"empty values preserved"}'
`
	require.NoError(t, os.WriteFile(path, []byte(data), 0o755))
	overlay := map[string]string{}
	for _, key := range []string{"CODEAF_TELEMETRY_ENDPOINT", "CODEAF_HOME", "CODEAF_TELEMETRY", "CODEAF_NO_UPDATE_CHECK", "CODEAF_MODELS", "AFORGE_MODELS", "OR_APP_NAME"} {
		overlay[key] = ""
	}
	raw, err := NewCodeAFProvider(path).Execute(context.Background(), "fixture only", Options{Env: overlay})
	require.NoError(t, err)
	assert.False(t, raw.IsError, raw.ErrorMessage)
	assert.Equal(t, "empty values preserved", raw.Result)
}

func TestCodeAFLegacyFallbackKeepsEmptyOverlaySemantics(t *testing.T) {
	isolateCodeAF(t)
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	t.Setenv("LEGACY_EMPTY_TEST", "inherited")
	path := filepath.Join(dir, "aforge")
	data := `#!/bin/sh
if [ "${LEGACY_EMPTY_TEST+x}" = x ]; then exit 10; fi
printf '%s\n' '{"text":"legacy unset preserved","stop":"done"}'
`
	require.NoError(t, os.WriteFile(path, []byte(data), 0o755))
	raw, err := NewCodeAFProvider("").Execute(context.Background(), "fixture only", Options{Env: map[string]string{"LEGACY_EMPTY_TEST": ""}})
	require.NoError(t, err)
	assert.Equal(t, ProviderAforge, raw.Provider)
	assert.Equal(t, "legacy unset preserved", raw.Result)
}
