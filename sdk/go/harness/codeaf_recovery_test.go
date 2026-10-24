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
	for _, location := range []string{"home", "path", "legacy home", "legacy path", "option", "env"} {
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
			home := codeAFManagedPaths()
			paths := map[string]string{
				"home": home, "path": filepath.Join(pathDir, "codeaf"),
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
			if location == "home" {
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
	home := codeAFManagedPaths()
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
			entries, err := os.ReadDir(os.TempDir())
			require.NoError(t, err)
			assert.Empty(t, entries)
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

// C1: failed installation never creates a temp fallback, and the throttle retains its error.
func TestCodeAFUnwritableHomeResolution(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("PATH", t.TempDir())
	blocked := filepath.Join(t.TempDir(), "home-file")
	require.NoError(t, os.WriteFile(blocked, []byte("blocked"), 0600))
	t.Setenv("AGENTFIELD_HOME", blocked)
	for i := 0; i < 2; i++ {
		_, err := resolveCodeAF(context.Background(), "")
		require.ErrorIs(t, err, ErrProviderUnavailable)
		require.ErrorContains(t, err, "cannot install CodeAF into")
		require.ErrorContains(t, err, "af codeaf ensure")
	}
	first := codeAFResolver.lastDownloadAttempt
	require.False(t, first.IsZero())
	codeAFResolver.lastDownloadAttempt = time.Now().Add(-61 * time.Second)
	_, err := resolveCodeAF(context.Background(), "")
	require.ErrorContains(t, err, "cannot install CodeAF into")
	require.True(t, codeAFResolver.lastDownloadAttempt.After(first))
	entries, err := os.ReadDir(os.TempDir())
	require.NoError(t, err)
	assert.Empty(t, entries)
}

// C2: an explicit binary still runs when the default state cannot be created.
func TestCodeAFExplicitBinaryWithUnwritableHome(t *testing.T) {
	isolateCodeAF(t)
	blocked := filepath.Join(t.TempDir(), "home-file")
	require.NoError(t, os.WriteFile(blocked, []byte("blocked"), 0600))
	t.Setenv("AGENTFIELD_HOME", blocked)
	binary := filepath.Join(t.TempDir(), "codeaf")
	script := "#!/bin/sh\nprintf '{\"ok\":true,\"stop\":\"done\",\"answer\":\"%s\"}\\n' \"$CODEAF_HOME\"\n"
	require.NoError(t, os.WriteFile(binary, []byte(script), 0755))
	t.Setenv("CODEAF_BIN", binary)
	raw, err := NewCodeAFProvider("").Execute(context.Background(), "fixture", Options{})
	require.NoError(t, err)
	require.False(t, raw.IsError, raw.ErrorMessage)
	assert.Equal(t, filepath.Join(blocked, "codeaf"), raw.Result)
	entries, err := os.ReadDir(os.TempDir())
	require.NoError(t, err)
	assert.Empty(t, entries)
}
