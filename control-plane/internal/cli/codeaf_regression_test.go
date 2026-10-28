package cli

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
	"github.com/stretchr/testify/require"
)

// Finding 6 / A3: both doctors inspect the same binary as execution, read-only.
func TestCodeafDoctorsLookupOrder(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixtures")
	}
	t.Setenv("CODEAF_BIN", "")
	home, tmp, path := t.TempDir(), t.TempDir(), t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("TMPDIR", tmp)
	t.Setenv("PATH", path)
	spec := harnessProviderSpecs[0]
	var downloads int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/health" {
			downloads++
		}
		w.WriteHeader(503)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL+"/releases")
	require.False(t, probeHarnessBinary(spec).Available)
	write := func(dir, version string, managed bool) string {
		require.NoError(t, os.MkdirAll(dir, 0755))
		writeHarnessTestScript(t, dir, "codeaf", "[ \"$1\" = version ] || exit 99\nprintf 'codeaf "+version+" built fixture\\n'")
		if managed {
			require.NoError(t, os.WriteFile(filepath.Join(dir, ".codeaf.version"), []byte(codeaf.Version+"\n"), 0644))
		}
		return filepath.Join(dir, "codeaf")
	}
	check := func(want, version string) {
		t.Helper()
		tool := probeHarnessBinary(spec)
		require.True(t, tool.Available)
		require.Equal(t, want, tool.Path)
		require.Equal(t, version, tool.Version)
		reports, err := buildHarnessDoctorReports([]string{"codeaf"})
		require.NoError(t, err)
		require.Equal(t, want, reports[0].Binary)
		report := buildDoctorReport(server.URL)
		require.Equal(t, want, report.HarnessProviders["codeaf"].Path)
	}
	check(write(path, "vPATH", false), "vPATH")
	homeBin := write(filepath.Join(home, "bin"), "vHOME", true)
	check(homeBin, "vHOME")
	override := write(t.TempDir(), "vOVERRIDE", false)
	t.Setenv("CODEAF_BIN", override)
	check(override, "vOVERRIDE")
	require.Equal(t, 0, downloads)
}
