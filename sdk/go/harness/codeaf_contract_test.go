package harness

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCodeAFF1LegacyPinPrecedence(t *testing.T) {
	for _, pin := range []string{"env", "bin"} {
		for _, explicit := range []string{"", "codeaf", "aforge"} {
			t.Run(pin+explicit, func(t *testing.T) {
				isolateCodeAF(t)
				t.Setenv(ProviderEnvVar, " ")
				bin := ""
				if pin == "env" {
					t.Setenv("AFORGE_BIN", "/legacy")
				} else {
					bin = "/legacy"
				}
				p, err := BuildProvider(explicit, bin)
				require.NoError(t, err)
				if explicit == "codeaf" {
					assert.IsType(t, &CodeAFProvider{}, p)
				} else {
					assert.IsType(t, &AforgeProvider{}, p)
				}
				r := NewRunner(Options{BinPath: bin})
				p, err = r.buildProvider(r.DefaultOptions)
				require.NoError(t, err)
				assert.IsType(t, &AforgeProvider{}, p)
			})
		}
	}
	isolateCodeAF(t)
	t.Setenv("AFORGE_BIN", "/legacy")
	t.Setenv(ProviderEnvVar, "codeaf")
	p, err := BuildProvider("", "")
	require.NoError(t, err)
	assert.IsType(t, &CodeAFProvider{}, p) // F1c
}

func TestCodeAFF2CommandAndModelFallbacks(t *testing.T) {
	for _, tc := range []struct {
		name, command, legacyCommand, model, newModel, oldModel string
		env                                                     map[string]string
		wantCommand, wantModel                                  string
	}{
		{name: "F2a", legacyCommand: "do", wantCommand: "do", wantModel: DefaultCodeAFModel},
		{name: "F2b", command: "exec", legacyCommand: "do", wantCommand: "exec", wantModel: DefaultCodeAFModel},
		{name: "R3a", legacyCommand: " DO ", wantCommand: "do", wantModel: DefaultCodeAFModel},
		{name: "R3b", command: "Exec", legacyCommand: "do", wantCommand: "exec", wantModel: DefaultCodeAFModel},
		{name: "F2c ambient", oldModel: "openrouter/old", wantCommand: "exec", wantModel: "old"},
		{name: "F2c overlay", oldModel: "old", env: map[string]string{"AFORGE_MODEL": "overlay"}, wantCommand: "exec", wantModel: "overlay"},
		{name: "F2d", newModel: "openrouter/new", oldModel: "old", wantCommand: "exec", wantModel: "new"},
		{name: "F2d blank overlay", newModel: "ambient", oldModel: "old", env: map[string]string{"CODEAF_MODEL": " ", "AFORGE_MODEL": "overlay"}, wantCommand: "exec", wantModel: "overlay"},
		{name: "F2e", model: "openrouter/explicit#high", newModel: "new", oldModel: "old", wantCommand: "exec", wantModel: "explicit"},
		{name: "F2f", wantCommand: "exec", wantModel: DefaultCodeAFModel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolateCodeAF(t)
			t.Setenv("AGENTFIELD_CODEAF_COMMAND", tc.command)
			t.Setenv("AGENTFIELD_AFORGE_COMMAND", tc.legacyCommand)
			t.Setenv("CODEAF_MODEL", tc.newModel)
			t.Setenv("AFORGE_MODEL", tc.oldModel)
			p := stubCodeAF()
			p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _, _ int, _ []byte) (*CLIResult, error) {
				assert.Equal(t, tc.wantCommand, cmd[1])
				for i, arg := range cmd {
					if arg == "--model" || arg == "--plan-model" || arg == "--check-model" {
						assert.Equal(t, tc.wantModel, cmd[i+1])
					}
				}
				return &CLIResult{Stdout: `{"ok":true,"stop":"done"}`}, nil
			}
			raw, err := p.Execute(context.Background(), "p", Options{Model: tc.model, Env: tc.env})
			require.NoError(t, err)
			assert.Equal(t, tc.wantModel, raw.Metrics.Model)
		})
	}
}

func TestCodeAFR2LegacyFallbackQueue(t *testing.T) {
	for _, contract := range []string{"R2a", "R2b", "R2b caller deadline"} {
		t.Run(contract, func(t *testing.T) {
			isolateCodeAF(t)
			t.Setenv("AFORGE_MAX_CONCURRENT", "1")
			aforgeSemaphore, aforgeSemOnce = nil, sync.Once{}
			sem := getAforgeSemaphore()
			sem <- struct{}{}
			t.Cleanup(func() {
				<-sem
				aforgeSemaphore, aforgeSemOnce = nil, sync.Once{}
			})
			p := NewCodeAFProvider("")
			p.resolveBinary = func(context.Context, string) (codeAFBinary, error) {
				return codeAFBinary{path: "/legacy/aforge", legacy: true}, nil
			}
			p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
				t.Fatal("queued fallback must not launch a process")
				return nil, nil
			}
			ctx := context.Background()
			if contract == "R2b" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				defer cancel()
				timer := time.AfterFunc(20*time.Millisecond, cancel)
				defer timer.Stop()
			} else if contract == "R2b caller deadline" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, 20*time.Millisecond)
				defer cancel()
			}
			started := time.Now()
			raw, err := p.Execute(ctx, "p", Options{Timeout: 1})
			if contract != "R2a" {
				require.ErrorIs(t, err, ctx.Err())
				assert.Nil(t, raw)
				assert.Less(t, time.Since(started), time.Second)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, raw)
			assert.NoError(t, ctx.Err())
			assert.Equal(t, ProviderAforge, raw.Provider)
			assert.True(t, raw.IsError)
			assert.Equal(t, FailureTimeout, raw.FailureType)
			assert.Contains(t, raw.ErrorMessage, "timed out")
			assert.GreaterOrEqual(t, raw.Metrics.DurationMS, 900)
			assert.Less(t, time.Since(started), 2*time.Second)
		})
	}
}

func TestCodeAFF3F4SharedProvisioningDeadlineAndCancellation(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("CODEAF_MODEL", "")
	t.Setenv("AFORGE_MODEL", "")
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("PATH", t.TempDir())
	body := "#!/bin/sh\necho '{\"ok\":true,\"stop\":\"done\",\"answer\":\"ready\"}'\n"
	useCodeAFTestHash(t, body)
	entered, release := make(chan struct{}), make(chan struct{})
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if requests.Add(1) == 1 {
			close(entered)
		}
		<-release
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", srv.URL)
	first := make(chan *RawResult, 1)
	go func() {
		raw, err := NewCodeAFProvider("").Execute(context.Background(), "p", Options{Timeout: 1})
		assert.NoError(t, err)
		first <- raw
	}()
	<-entered
	// F4a: an explicit binary ignores the stalled automatic resolver.
	ready := filepath.Join(t.TempDir(), "ready")
	writeCodeAFCLI(t, ready, `{"ok":true,"stop":"done"}`, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	begin := time.Now()
	binary, err := resolveCodeAF(ctx, ready)
	cancel()
	require.NoError(t, err)
	assert.Equal(t, ready, binary.path)
	assert.Less(t, time.Since(begin), 50*time.Millisecond)
	// F4b: a canceled waiter returns without canceling the installation.
	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	_, err = resolveCodeAF(ctx, "")
	require.ErrorIs(t, err, context.Canceled)
	// F4c: a second automatic call shares the in-flight download.
	second := make(chan error, 1)
	go func() { _, err := resolveCodeAF(context.Background(), ""); second <- err }()
	select {
	case raw := <-first: // F3a: first-use provisioning is in the 1s budget.
		require.NotNil(t, raw)
		assert.Equal(t, FailureTimeout, raw.FailureType)
		assert.Contains(t, raw.ErrorMessage, "provisioning CodeAF")
	case <-time.After(1500 * time.Millisecond):
		close(release)
		t.Fatal("provisioning exceeded the execution budget")
	}
	close(release)
	require.NoError(t, <-second)
	// F3b: timed-out callers leave a completed binary for later calls.
	raw, err := NewCodeAFProvider("").Execute(context.Background(), "p", Options{})
	require.NoError(t, err)
	assert.Equal(t, "ready", raw.Result)
	assert.Equal(t, int32(1), requests.Load())
}

func TestCodeAFF3cRemainingSubprocessBudget(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	p.resolveBinary = func(ctx context.Context, _ string) (codeAFBinary, error) {
		time.Sleep(150 * time.Millisecond)
		return codeAFBinary{path: "/fixture"}, nil
	}
	p.runCLI = func(ctx context.Context, cmd []string, _ map[string]string, _ string, timeout, _ int, _ []byte) (*CLIResult, error) {
		deadline, ok := ctx.Deadline()
		require.True(t, ok)
		remaining := time.Until(deadline).Seconds()
		// Provisioning took >= 150ms of the 10s budget; only the rest is passed on.
		assert.Greater(t, remaining, 5.0)
		assert.LessOrEqual(t, remaining, 9.855)
		// The integer cap rounds up; ctx enforces the precise remainder.
		assert.GreaterOrEqual(t, float64(timeout), remaining)
		assert.LessOrEqual(t, timeout, 10)
		require.Contains(t, cmd, "--timeout")
		for i, arg := range cmd {
			if arg == "--timeout" {
				inner, err := time.ParseDuration(cmd[i+1])
				require.NoError(t, err)
				// Taken before remaining, so it can only exceed remaining-5.
				assert.GreaterOrEqual(t, inner.Seconds()+0.001, remaining-5)
				assert.LessOrEqual(t, inner.Seconds(), 4.856)
			}
		}
		return &CLIResult{Stdout: `{"ok":true,"stop":"done"}`}, nil
	}
	_, err := p.Execute(context.Background(), "p", Options{Timeout: 10})
	require.NoError(t, err)
}

func TestCodeAFF5AliasDoctor(t *testing.T) {
	for _, location := range []string{"managed", "path", "legacy", "missing", "pin"} {
		t.Run(location, func(t *testing.T) {
			isolateCodeAF(t)
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			target := filepath.Join(dir, "codeaf")
			if location == "managed" {
				target = codeAFManagedPaths()
			}
			if location == "legacy" {
				target = filepath.Join(dir, "aforge")
			}
			if location != "missing" {
				writeCodeAFCLI(t, target, "{}", 0)
			}
			if location == "managed" {
				require.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(target), codeAFVersionFilename), []byte(CodeAFVersion), 0o644))
			}
			if location == "pin" {
				t.Setenv("AFORGE_BIN", filepath.Join(dir, "missing"))
			}
			reports, err := Doctor(context.Background(), DoctorOptions{Providers: []string{"aforge"}, Probe: true, VersionProbe: func(_ context.Context, cmd []string) (string, error) {
				assert.Equal(t, target, cmd[0])
				if location != "legacy" {
					assert.Equal(t, "version", cmd[1])
					return "codeaf v0.8.2", nil
				}
				return "aforge v0.1.0", nil
			}})
			require.NoError(t, err)
			assert.Equal(t, location != "pin" && location != "missing", reports[0].Usable)
			if location == "managed" || location == "path" {
				assert.True(t, strings.Contains(reports[0].Version, "codeaf v0.8.2"))
			}
			if !reports[0].Usable {
				assert.Contains(t, reports[0].InstallCommand, "af aforge ensure")
			}
		})
	}
}

func TestCodeAFR4AliasDoctorOverride(t *testing.T) {
	for _, contract := range []string{"R4a", "R4b", "R4c"} {
		t.Run(contract, func(t *testing.T) {
			isolateCodeAF(t)
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			legacy := filepath.Join(dir, "aforge")
			writeCodeAFCLI(t, legacy, "{}", 0)
			target := filepath.Join(t.TempDir(), "override")
			if contract != "R4b" {
				t.Setenv("CODEAF_BIN", target)
			}
			if contract == "R4c" {
				writeCodeAFCLI(t, target, "{}", 0)
			}
			calls := 0
			reports, err := Doctor(context.Background(), DoctorOptions{
				Providers: []string{"aforge"}, Probe: true,
				VersionProbe: func(_ context.Context, cmd []string) (string, error) {
					calls++
					if contract == "R4b" {
						assert.Equal(t, []string{legacy, "version"}, cmd)
						return "aforge v0.1.0", nil
					}
					assert.Equal(t, []string{target, "version"}, cmd)
					return "codeaf v0.8.2", nil
				},
			})
			require.NoError(t, err)
			require.Len(t, reports, 1)
			report := reports[0]
			assert.Equal(t, contract != "R4a", report.Usable)
			assert.Equal(t, report.Usable, report.Installed)
			if contract == "R4a" {
				assert.Empty(t, report.Binary)
				assert.Equal(t, []string{IssueBinaryNotFound}, report.Issues)
				assert.Zero(t, calls)
			} else {
				assert.Equal(t, 1, calls)
				if contract == "R4c" {
					assert.Equal(t, "codeaf v0.8.2", report.Version)
				}
			}
		})
	}
}

// Compare protocol arguments while allowing time spent resolving the binary.
func normalizeCodeAFTimeout(t *testing.T, cmd, want []string) []string {
	t.Helper()
	actual := append([]string(nil), cmd...)
	for i, arg := range want {
		if arg == "--timeout" {
			expected, err := time.ParseDuration(want[i+1])
			require.NoError(t, err)
			got, err := time.ParseDuration(actual[i+1])
			require.NoError(t, err)
			assert.InDelta(t, expected.Seconds(), got.Seconds(), 0.1)
			actual[i+1] = want[i+1]
		}
	}
	return actual
}
