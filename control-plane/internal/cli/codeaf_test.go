package cli

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCodeafCommand(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {"ensure", "--help"}} {
		cmd := NewCodeafCommand()
		var out bytes.Buffer
		cmd.SetOut(&out)
		cmd.SetArgs(args)
		require.NoError(t, cmd.Execute())
		require.Contains(t, out.String(), "ensure")
	}
	for _, key := range []string{"AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv("AGENTFIELD_SKIP_CODEAF", "")
			t.Setenv("AGENTFIELD_SKIP_AFORGE", "")
			t.Setenv(key, "YES")
			for _, args := range [][]string{{"ensure"}, {"ensure", "--force"}} {
				cmd := NewCodeafCommand()
				var out bytes.Buffer
				cmd.SetOut(&out)
				cmd.SetArgs(args)
				require.NoError(t, cmd.Execute())
				require.Contains(t, out.String(), "skipped")
			}
		})
	}
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "")
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "")
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	cmd := NewCodeafCommand()
	cmd.SetArgs([]string{"ensure"})
	require.ErrorContains(t, cmd.Execute(), "404")
}

// V11: both doctors survey the managed default using only `codeaf version`.
func TestCodeafDoctorManagedVersionAndNoTaskProbe(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	require.Equal(t, "codeaf", harnessProviderSpecs[0].Name)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("PATH", t.TempDir())
	t.Setenv("OPENROUTER_API_KEY", "configured")
	bin := filepath.Join(home, "bin")
	require.NoError(t, os.MkdirAll(bin, 0755))
	require.NoError(t, os.WriteFile(filepath.Join(bin, ".codeaf.version"), []byte("v0.8.2\n"), 0644))
	writeHarnessTestScript(t, bin, "codeaf", "[ \"$#\" = 1 ] && [ \"$1\" = version ] || exit 99\nprintf 'codeaf v0.8.2 built 2026-10-23 11:39 · go1.26.5 linux/amd64\\n'")
	reports, err := buildHarnessDoctorReports([]string{"codeaf"})
	require.NoError(t, err)
	require.Len(t, reports, 1)
	require.True(t, reports[0].Usable)
	require.Equal(t, "v0.8.2", reports[0].Version)
	require.Equal(t, "af codeaf ensure", reports[0].InstallCommand)
	require.Equal(t, "configured", reports[0].Auth)
	require.Equal(t, []string{"OPENROUTER_API_KEY"}, reports[0].AuthEnvVars)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	defer server.Close()
	report := buildDoctorReport(server.URL)
	require.Equal(t, "v0.8.2", report.HarnessProviders["codeaf"].Version)
	require.Equal(t, filepath.Join(bin, "codeaf"), report.HarnessProviders["codeaf"].Path)
	require.Equal(t, "codeaf", report.Recommendation.HarnessProviders[0])
	require.NotContains(t, runHarnessProbes(report), "codeaf")
	require.NoError(t, os.WriteFile(filepath.Join(bin, ".codeaf.version"), []byte("v0.8.2\n"), 0644))
	writeHarnessTestScript(t, bin, "codeaf", "printf 'usage: codeaf version\\n'")
	reports, err = buildHarnessDoctorReports([]string{"codeaf"})
	require.NoError(t, err)
	require.False(t, reports[0].Usable)
	require.Equal(t, []string{"version_probe_failed"}, reports[0].Issues)
	report = buildDoctorReport(server.URL)
	require.Contains(t, strings.Join(report.Recommendation.Notes, "\n"), "af codeaf ensure")
}

// Automatic skill hooks must not make ambient network requests in CLI tests.
// Provisioning tests explicitly use a local HTTP server or injected hook.
func TestMain(m *testing.M) {
	_ = os.Setenv("AGENTFIELD_SKIP_CODEAF", "1")
	os.Exit(m.Run())
}
