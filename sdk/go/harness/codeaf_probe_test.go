package harness

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Validation contract (a watched CodeAF run and an older CodeAF build):
//
//	P1  A binary whose `chat --help` lacks the watched run's flags (v0.8.3,
//	    which also refuses `chat --json`) is never sent `chat ... --json`: the run
//	    goes through `exec`, succeeds, and logs one WARNING naming the mismatch.
//	P2  A binary whose `chat --help` lists --json, --events, --steer and --session
//	    is run as `chat --once ... --json --events --steer --session`, never `exec`.
//	P3  A binary that lists the flags and still refuses one up front ("has no
//	    --json flag") is re-run through `exec` once, and that run succeeds.
//	P4  `chat --help` runs once per binary per process, however many runs follow
//	    or start together; a binary replaced in place is asked again.
//	P5  A failed run is re-run only when it was refused for a live flag, exited
//	    non-zero and printed no envelope.
//	P6  The listing is read the way codeaf prints it, not by substring: a usage
//	    line that mentions "[--session path]" does not list --session.

const fakeCodeAFEnvelope = `{"ok":true,"stop":"done","answer":"done"}`

// fakeCodeAFScript writes a stand-in codeaf. Every `chat --help` appends to
// probes, every other call appends its arguments to calls. help is the flag
// listing it prints; rejectChat makes `chat` refuse the way v0.8.3 does.
func fakeCodeAFScript(t *testing.T, dir string, help []string, rejectChat string) (bin, calls, probes string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in for codeaf")
	}
	bin, calls, probes = filepath.Join(dir, "fake-codeaf"), filepath.Join(dir, "calls"), filepath.Join(dir, "probes")
	writeFakeCodeAF(t, bin, calls, probes, help, rejectChat)
	return bin, calls, probes
}

func writeFakeCodeAF(t *testing.T, bin, calls, probes string, help []string, rejectChat string) {
	t.Helper()
	var listing strings.Builder
	listing.WriteString("  codeaf chat [--model slug] [--session path] [--once \"text\"]\n\nflags:\n")
	for _, flag := range help {
		fmt.Fprintf(&listing, "  --%s\n      what %s does\n", flag, flag)
	}
	reject := `printf '%s\n' '` + fakeCodeAFEnvelope + `'`
	if rejectChat != "" {
		reject = `printf '%s\n' '` + rejectChat + `' >&2; exit 1`
	}
	script := fmt.Sprintf(`#!/bin/sh
if [ "$1" = chat ] && [ "$2" = --help ]; then
  echo probe >> '%s'
  cat <<'HELP'
%sHELP
  exit 0
fi
echo "$*" >> '%s'
if [ "$1" = chat ]; then
  %s
  exit 0
fi
cat >/dev/null
printf '%%s\n' '%s'
`, probes, listing.String(), calls, reject, fakeCodeAFEnvelope)
	tmp := bin + ".tmp"
	require.NoError(t, os.WriteFile(tmp, []byte(script), 0o755))
	require.NoError(t, os.Rename(tmp, bin))
}

func readLines(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	require.NoError(t, err)
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

// fakeCodeAFProvider is a provider that runs the real runners against a script.
func fakeCodeAFProvider(bin string) *CodeAFProvider {
	p := stubCodeAF()
	p.BinPath = bin
	p.probeCLI, p.runCLI = nil, nil
	return p
}

func liveOptions(t *testing.T) (Options, *int) {
	begun := 0
	return Options{ProjectDir: t.TempDir(), Live: &Live{Begin: func(LiveInfo) { begun++ }}}, &begun
}

// P1
func TestCodeAFLiveRunUsesExecOnABuildWithoutTheLiveFlags(t *testing.T) {
	isolateCodeAF(t)
	logs := captureCodeAFLogs(t)
	bin, calls, probes := fakeCodeAFScript(t, t.TempDir(), []string{"session"}, "Error: codeaf chat has no --json flag")
	options, begun := liveOptions(t)

	raw, err := fakeCodeAFProvider(bin).Execute(context.Background(), "write it", options)

	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	assert.False(t, raw.IsError)
	got := readLines(t, calls)
	require.Len(t, got, 1, "the run must go straight to exec: %v", got)
	assert.True(t, strings.HasPrefix(got[0], "exec --json "), got[0])
	assert.NotContains(t, strings.Join(got, "\n"), "chat")
	assert.Len(t, readLines(t, probes), 1)
	assert.Equal(t, 0, *begun, "an unwatched run must not announce itself as watchable")
	assert.Equal(t, 1, strings.Count(logs.String(), "WARNING CodeAF at "+bin+" is older than this SDK expects"), logs.String())
	assert.Contains(t, logs.String(), "--json, --events, --steer")
}

// P2
func TestCodeAFLiveRunUsesChatOnABuildWithTheLiveFlags(t *testing.T) {
	isolateCodeAF(t)
	logs := captureCodeAFLogs(t)
	bin, calls, _ := fakeCodeAFScript(t, t.TempDir(), []string{"events string", "json", "session string", "steer string"}, "")
	options, begun := liveOptions(t)

	raw, err := fakeCodeAFProvider(bin).Execute(context.Background(), "write it", options)

	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	got := readLines(t, calls)
	require.Len(t, got, 1, "%v", got)
	assert.True(t, strings.HasPrefix(got[0], "chat --once write it "), got[0])
	for _, flag := range []string{" --json", " --events ", " --steer ", " --session "} {
		assert.Contains(t, got[0], flag)
	}
	assert.Equal(t, 1, *begun)
	assert.NotContains(t, logs.String(), "WARNING")
}

// P3
func TestCodeAFLiveRunFallsBackOnceWhenAListedFlagIsStillRefused(t *testing.T) {
	isolateCodeAF(t)
	bin, calls, _ := fakeCodeAFScript(t, t.TempDir(), []string{"events string", "json", "session string", "steer string"}, "Error: codeaf chat has no --json flag")
	options, _ := liveOptions(t)

	raw, err := fakeCodeAFProvider(bin).Execute(context.Background(), "write it", options)

	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	got := readLines(t, calls)
	require.Len(t, got, 2, "%v", got)
	assert.True(t, strings.HasPrefix(got[0], "chat --once "), got[0])
	assert.True(t, strings.HasPrefix(got[1], "exec --json "), got[1])
}

// P4
func TestCodeAFLiveProbeRunsOncePerBinary(t *testing.T) {
	isolateCodeAF(t)
	dir := t.TempDir()
	full := []string{"events string", "json", "session string", "steer string"}
	bin, calls, probes := fakeCodeAFScript(t, dir, full, "")
	p := fakeCodeAFProvider(bin)

	// Sequential runs, then a burst that starts together.
	for i := 0; i < 2; i++ {
		options, _ := liveOptions(t)
		_, err := p.Execute(context.Background(), "write it", options)
		require.NoError(t, err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			options, _ := liveOptions(t)
			_, err := p.Execute(context.Background(), "write it", options)
			assert.NoError(t, err)
		}()
	}
	wg.Wait()
	assert.Len(t, readLines(t, probes), 1, "six runs, one probe")
	assert.Len(t, readLines(t, calls), 6)

	// The same path with a different build behind it is asked again.
	writeFakeCodeAF(t, bin, calls, probes, []string{"session"}, "Error: codeaf chat has no --json flag")
	options, _ := liveOptions(t)
	_, err := p.Execute(context.Background(), "write it", options)
	require.NoError(t, err)
	assert.Len(t, readLines(t, probes), 2)
	got := readLines(t, calls)
	assert.True(t, strings.HasPrefix(got[len(got)-1], "exec --json "), got[len(got)-1])
}

// A probe that cannot run is a "no" for that run only: the next run asks again.
func TestCodeAFLiveProbeFailureIsNotRemembered(t *testing.T) {
	isolateCodeAF(t)
	logs := captureCodeAFLogs(t)
	bin, calls, _ := fakeCodeAFScript(t, t.TempDir(), []string{"events string", "json", "session string", "steer string"}, "")
	p := fakeCodeAFProvider(bin)
	probeCalls := 0
	p.probeCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
		probeCalls++
		if probeCalls == 1 {
			return nil, fmt.Errorf("CLI command timed out after 10s")
		}
		return &CLIResult{Stdout: "  --events string\n  --json\n  --session string\n  --steer string\n"}, nil
	}

	options, _ := liveOptions(t)
	_, err := p.Execute(context.Background(), "write it", options)
	require.NoError(t, err)
	assert.Contains(t, logs.String(), "could not be probed")
	options, _ = liveOptions(t)
	_, err = p.Execute(context.Background(), "write it", options)
	require.NoError(t, err)

	assert.Equal(t, 2, probeCalls)
	got := readLines(t, calls)
	require.Len(t, got, 2)
	assert.True(t, strings.HasPrefix(got[0], "exec --json "), got[0])
	assert.True(t, strings.HasPrefix(got[1], "chat --once "), got[1])
}

// P5
func TestCodeAFRejectedLiveFlags(t *testing.T) {
	envelope := fakeCodeAFEnvelope
	for _, tc := range []struct {
		name   string
		cli    *CLIResult
		expect bool
	}{
		{"v0.8.3 has no --json", &CLIResult{ReturnCode: 1, Stderr: "Error: codeaf chat has no --json flag\n"}, true},
		{"has no --events", &CLIResult{ReturnCode: 1, Stderr: "Error: codeaf chat has no --events flag"}, true},
		{"has no --steer", &CLIResult{ReturnCode: 1, Stderr: "Error: codeaf chat has no --steer flag"}, true},
		{"has no --session", &CLIResult{ReturnCode: 1, Stderr: "Error: codeaf chat has no --session flag"}, true},
		{"flag package, -events", &CLIResult{ReturnCode: 2, Stderr: "flag provided but not defined: -events"}, true},
		{"flag package, --json", &CLIResult{ReturnCode: 2, Stderr: "flag provided but not defined: --json"}, true},
		{"flag package, -once", &CLIResult{ReturnCode: 2, Stderr: "flag provided but not defined: -once"}, true},
		{"older -events spelling", &CLIResult{ReturnCode: 2, Stderr: "option -events not defined"}, true},
		{"exited zero", &CLIResult{ReturnCode: 0, Stderr: "Error: codeaf chat has no --json flag"}, false},
		{"an envelope was printed", &CLIResult{ReturnCode: 1, Stdout: envelope, Stderr: "Error: codeaf chat has no --json flag"}, false},
		{"a flag the live run does not own", &CLIResult{ReturnCode: 1, Stderr: "Error: codeaf chat has no --model flag"}, false},
		{"an ordinary failure", &CLIResult{ReturnCode: 1, Stderr: "provider returned 500"}, false},
		{"no result", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, codeAFRejectedLiveFlags(tc.cli))
		})
	}
}

// A run refused for a live flag AND exiting with an envelope is a real
// failure of the run, not a refused command line: it is reported, not re-run.
func TestCodeAFLiveRunWithAnEnvelopeIsNotRerun(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	calls := 0
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, stdin []byte) (*CLIResult, error) {
		calls++
		return &CLIResult{ReturnCode: 1, Stdout: `{"ok":false,"stop":"error","error":"boom"}`, Stderr: "Error: codeaf chat has no --json flag"}, nil
	}
	raw, err := p.Execute(context.Background(), "write it", Options{Live: &Live{}})
	require.NoError(t, err)
	assert.Equal(t, 1, calls)
	assert.True(t, raw.IsError)
}

// P6, against the real `codeaf chat --help` of a build that refuses the watched
// run's flags (v0.8.3) and of one that takes them.
func TestCodeAFLiveFlagsReadFromRealHelpOutput(t *testing.T) {
	old, err := os.ReadFile("testdata/codeaf-chat-help-v0.8.3.txt")
	require.NoError(t, err)
	current, err := os.ReadFile("testdata/codeaf-chat-help-live-flags.txt")
	require.NoError(t, err)

	assert.Equal(t, []string{"--json", "--events", "--steer"}, codeAFMissingLiveFlags(string(old)))
	assert.Empty(t, codeAFMissingLiveFlags(string(current)))
	assert.Equal(t, []string{"--json", "--events", "--steer", "--session"}, codeAFMissingLiveFlags(""))
	// The stock flag package's spelling counts too.
	assert.Empty(t, codeAFMissingLiveFlags("Usage of chat:\n  -events string\n  \tx\n  -json\n  \tx\n  -session string\n  -steer string\n"))
	// A mention is not a listing.
	assert.Equal(t, []string{"--json", "--events", "--steer", "--session"},
		codeAFMissingLiveFlags("  codeaf chat [--json] [--events f] [--steer f] [--session p]\n"))
}
