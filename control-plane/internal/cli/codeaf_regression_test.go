package cli

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

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
	tempBin := write(filepath.Join(tmp, fmt.Sprintf("agentfield-codeaf-%d", os.Getuid()), codeaf.Version), "vTEMP", true)
	check(tempBin, "vTEMP")
	homeBin := write(filepath.Join(home, "bin"), "vHOME", true)
	check(homeBin, "vHOME")
	override := write(t.TempDir(), "vOVERRIDE", false)
	t.Setenv("CODEAF_BIN", override)
	check(override, "vOVERRIDE")
	require.Equal(t, 0, downloads)
}

func TestCodeafEnsureUsesManagedHomeDespiteBackgroundGuard(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "0")
	home := t.TempDir()
	dir := filepath.Join(home, "bin")
	t.Setenv("AGENTFIELD_HOME", home)
	require.NoError(t, os.MkdirAll(dir, 0755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, codeaf.BinaryName(runtime.GOOS)), []byte("already installed"), 0755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".codeaf.version"), []byte(codeaf.Version), 0644))
	cmd := NewCodeafCommand()
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".codeaf.background"), []byte("active lease"), 0600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".codeaf.lastfail"), []byte(time.Now().Format(time.RFC3339Nano)), 0600))
	cmd.SetArgs([]string{"ensure"})
	require.NoError(t, cmd.Execute())
	require.FileExists(t, filepath.Join(dir, codeaf.BinaryName(runtime.GOOS)))
	// A missing managed binary must still be attempted during both guards.
	require.NoError(t, os.Remove(filepath.Join(dir, codeaf.BinaryName(runtime.GOOS))))
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	cmd = NewCodeafCommand()
	cmd.SetArgs([]string{"ensure"})
	require.ErrorContains(t, cmd.Execute(), "503")
	require.Equal(t, int32(1), requests.Load())
	lease, err := os.ReadFile(filepath.Join(dir, ".codeaf.background"))
	require.NoError(t, err)
	require.Equal(t, "active lease", string(lease))
}

func TestCodeafBackgroundChildRecordsResult(t *testing.T) {
	if os.Getenv("CODEAF_ENSURE_TEST_CHILD") == "1" {
		// TestMain opts out by default; this child must exercise provisioning.
		_ = os.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
		cmd := NewCodeafCommand()
		for i, arg := range os.Args {
			if arg == "--" {
				cmd.SetArgs(os.Args[i+1:])
				break
			}
		}
		if err := cmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "0")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	for _, installed := range []bool{false, true} {
		t.Run(fmt.Sprintf("installed=%v", installed), func(t *testing.T) {
			home := t.TempDir()
			t.Setenv("AGENTFIELD_HOME", home)
			bin := filepath.Join(home, "bin")
			require.NoError(t, os.MkdirAll(bin, 0755))
			lease := fmt.Sprintf("%d %s\n", os.Getpid(), time.Now().Format(time.RFC3339Nano))
			leasePath := filepath.Join(bin, ".codeaf.background")
			failPath := filepath.Join(bin, ".codeaf.lastfail")
			require.NoError(t, os.WriteFile(leasePath, []byte(lease), 0600))
			// The child must ignore cooldowns and clean old failures after success.
			require.NoError(t, os.WriteFile(failPath, []byte(time.Now().Format(time.RFC3339Nano)), 0600))
			if installed {
				require.NoError(t, os.WriteFile(filepath.Join(bin, codeaf.BinaryName(runtime.GOOS)), []byte("fixture"), 0755))
				require.NoError(t, os.WriteFile(filepath.Join(bin, ".codeaf.version"), []byte(codeaf.Version), 0644))
			}
			child := exec.Command(os.Args[0], "-test.run=^TestCodeafBackgroundChildRecordsResult$", "--", "ensure", "--background-lease", lease)
			child.Env = append(os.Environ(), "CODEAF_ENSURE_TEST_CHILD=1")
			out, err := child.CombinedOutput()
			if installed {
				require.NoError(t, err, string(out))
				require.NoFileExists(t, failPath)
			} else {
				require.Error(t, err)
				require.Contains(t, string(out), "503")
				stamp, err := os.ReadFile(failPath)
				require.NoError(t, err)
				failed, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(stamp)))
				require.NoError(t, err)
				require.WithinDuration(t, time.Now(), failed, 5*time.Second)
			}
			require.NoFileExists(t, leasePath)
		})
	}
}
