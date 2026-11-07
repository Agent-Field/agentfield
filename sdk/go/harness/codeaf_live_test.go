package harness

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func flagValue(cmd []string, name string) string {
	for i, c := range cmd {
		if c == name && i+1 < len(cmd) {
			return cmd[i+1]
		}
	}
	return ""
}

func appendLine(t *testing.T, path, line string) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	require.NoError(t, err)
	_, err = f.WriteString(line)
	require.NoError(t, err)
	require.NoError(t, f.Close())
}

// A live codeaf run streams every event line while it is still running,
// VERBATIM, and a note sent on Steer reaches the run's steer file mid-run.
func TestCodeAFLiveStreamsWhileRunningAndDeliversASteer(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	steer := make(chan string, 1)
	var mu sync.Mutex
	var lines []string
	sawFirstWhileRunning := make(chan struct{})
	var begun, ended int
	live := &Live{
		Steer: steer,
		Begin: func() { begun++ },
		End:   func() { ended++ },
		Event: func(raw json.RawMessage) {
			mu.Lock()
			lines = append(lines, string(raw))
			n := len(lines)
			mu.Unlock()
			if n == 1 {
				close(sawFirstWhileRunning)
			}
		},
	}
	// Exactly the bytes codeaf writes, odd spacing and field order included.
	first := `{"kind":"assistant", "turn":1,"text":"looking at main.go"}`
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, stdin []byte) (*CLIResult, error) {
		events, steerPath := flagValue(cmd, "--events"), flagValue(cmd, "--steer")
		// A watched run is the native step engine: one headless chat turn.
		require.Equal(t, []string{"chat", "--once", "write it"}, cmd[1:4])
		require.Contains(t, cmd, "--json")
		require.NotEmpty(t, cwd)
		require.NotEmpty(t, events)
		require.NotEmpty(t, steerPath)
		appendLine(t, events, first+"\n")
		// The run is still going: the watcher must already have the line.
		select {
		case <-sawFirstWhileRunning:
		case <-time.After(3 * time.Second):
			t.Fatal("the first event was not handed over while the run was working")
		}
		steer <- "also add a doc comment"
		var note string
		for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
			raw, _ := os.ReadFile(steerPath)
			if len(raw) > 0 {
				note = string(raw)
				break
			}
		}
		require.Equal(t, `{"text":"also add a doc comment"}`+"\n", note)
		// A last line with no newline still reaches the watcher at the end.
		appendLine(t, events, `{"kind":"note","text":"steered: also add a doc comment"}`)
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"done"}`}, nil
	}
	raw, err := p.Execute(context.Background(), "write it", Options{Live: live})
	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	assert.Equal(t, []string{first, `{"kind":"note","text":"steered: also add a doc comment"}`}, lines)
	assert.Equal(t, 1, begun)
	assert.Equal(t, 1, ended)
}

// Without Live nothing is armed: no flags, no files.
func TestCodeAFWithoutLivePassesNoLiveFlags(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, stdin []byte) (*CLIResult, error) {
		assert.NotContains(t, strings.Join(cmd, " "), "--events")
		assert.NotContains(t, strings.Join(cmd, " "), "--steer")
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"done"}`}, nil
	}
	_, err := p.Execute(context.Background(), "write it", Options{})
	require.NoError(t, err)
}

// An older codeaf that refuses --events is run again without them.
func TestCodeAFLiveFallsBackOnAnOlderBinary(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	calls := 0
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, stdin []byte) (*CLIResult, error) {
		calls++
		if flagValue(cmd, "--events") != "" {
			return &CLIResult{ReturnCode: 2, Stderr: "flag provided but not defined: -events"}, nil
		}
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"done"}`}, nil
	}
	raw, err := p.Execute(context.Background(), "write it", Options{Live: &Live{}})
	require.NoError(t, err)
	assert.Equal(t, 2, calls)
	assert.Equal(t, "done", raw.Result)
}

// Live is a codeaf-only thing: another provider never begins it, so the
// execution it runs in is never steerable.
func TestOnlyCodeAFBeginsALiveRun(t *testing.T) {
	begun := false
	live := &Live{Begin: func() { begun = true }}
	p := NewOpenCodeProvider("/nonexistent/opencode", "")
	_, _ = p.Execute(context.Background(), "x", Options{Live: live, Timeout: 1})
	assert.False(t, begun)
}
