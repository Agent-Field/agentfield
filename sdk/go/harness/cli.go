package harness

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// defaultIdleSeconds is the no-progress watchdog window used when the env var
// AGENTFIELD_HARNESS_IDLE_SECONDS is unset or invalid. A value <= 0 disables it.
//
// 300 rather than 120: provider CLIs in JSON mode (e.g. `opencode run
// --format json`) emit events only at completion boundaries — never
// token-by-token — so one long reasoning completion over a large context is
// minutes of legitimate stdout silence. At 120s the watchdog routinely killed
// healthy runs on slower models; 300s tolerates a long single completion while
// still catching genuine hangs.
const defaultIdleSeconds = 300

// resolveIdleSeconds reads the idle watchdog window from the environment,
// falling back to defaultIdleSeconds. A value <= 0 disables the watchdog.
func resolveIdleSeconds() int {
	raw := os.Getenv("AGENTFIELD_HARNESS_IDLE_SECONDS")
	if raw == "" {
		return defaultIdleSeconds
	}
	v, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return defaultIdleSeconds
	}
	return v
}

var ansiRe = regexp.MustCompile(`\x1B\[[0-?]*[ -/]*[@-~]`)

// StripANSI removes ANSI escape sequences from text.
func StripANSI(text string) string {
	return ansiRe.ReplaceAllString(text, "")
}

// CLIResult holds the output from a CLI subprocess.
type CLIResult struct {
	Stdout     string
	Stderr     string
	ReturnCode int
}

// RunCLI runs a CLI command with no standard input: the child sees an immediate
// EOF on stdin. It is a thin wrapper over RunCLIWithStdin; see that function for
// the full contract.
func RunCLI(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error) {
	return RunCLIWithStdin(ctx, cmd, env, cwd, timeout, nil)
}

// RunCLIWithStdin runs a CLI command and returns its output. The context
// controls cancellation; timeout is in seconds (0 means no timeout beyond ctx).
//
// stdin is fed to the child process's standard input. A nil (or empty) slice
// yields an immediate EOF — identical to RunCLI. Providers whose CLI reads the
// prompt from stdin (e.g. `claude --print`) pass the prompt bytes here instead
// of appending them as a trailing positional argument, which a preceding
// variadic flag (e.g. claude's `--allowedTools`) would otherwise greedily
// consume — leaving the CLI with no prompt and exiting non-zero.
//
// Environment merging: entries in env are merged with os.Environ(). An empty
// string value ("") causes that variable to be removed from the environment
// rather than set to empty — use this to unset inherited variables.
func RunCLIWithStdin(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, stdin []byte) (*CLIResult, error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
		defer cancel()
	}

	if len(cmd) == 0 {
		return nil, fmt.Errorf("empty command")
	}

	c := exec.CommandContext(ctx, cmd[0], cmd[1:]...)

	// Merge environment: empty values unset the variable.
	merged := make(map[string]string)
	for _, entry := range os.Environ() {
		key, value, found := strings.Cut(entry, "=")
		if !found {
			continue
		}
		merged[key] = value
	}
	for k, v := range env {
		if v == "" {
			delete(merged, k)
		} else {
			merged[k] = v
		}
	}

	applyOpenRouterAttributionEnv(merged)

	var mergedEnv []string
	for k, v := range merged {
		mergedEnv = append(mergedEnv, k+"="+v)
	}
	c.Env = mergedEnv

	if cwd != "" {
		c.Dir = cwd
	}

	// Feed the provided stdin to the child. A nil/empty slice yields an
	// immediate EOF, so the child never inherits the parent's stdin (a hang
	// risk if the child blocks reading stdin). When the child exits without
	// draining stdin, the copy fails with EPIPE, which os/exec ignores.
	c.Stdin = bytes.NewReader(stdin)

	// Run the child in its own process group so the idle watchdog can kill
	// the whole tree, not just the leader. setProcessGroup is platform-guarded.
	setProcessGroup(c)

	// On ctx cancellation/timeout, kill the whole process group too.
	// exec.CommandContext's default Cancel only kills the group LEADER, which
	// orphans grandchildren (e.g. MCP servers spawned by the claude CLI) when
	// a caller times out or cancels. On Windows killProcessGroup degrades to
	// killing just the leader, matching the previous behavior.
	c.Cancel = func() error {
		killProcessGroup(c)
		return nil
	}

	stdoutPipe, err := c.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderrPipe, err := c.StderrPipe()
	if err != nil {
		return nil, err
	}

	if err := c.Start(); err != nil {
		if isExecNotFound(err) {
			return nil, err
		}
		return nil, err
	}

	var (
		mu           sync.Mutex
		stdout       bytes.Buffer
		stderr       bytes.Buffer
		lastActivity = time.Now()
		wg           sync.WaitGroup
	)

	// Drain both pipes concurrently to avoid a pipe-buffer deadlock: if we
	// read stdout fully before stderr, a child that fills the stderr pipe
	// blocks forever.
	drain := func(r io.Reader, buf *bytes.Buffer) {
		defer wg.Done()
		chunk := make([]byte, 65536)
		for {
			n, readErr := r.Read(chunk)
			if n > 0 {
				mu.Lock()
				buf.Write(chunk[:n])
				lastActivity = time.Now()
				mu.Unlock()
			}
			if readErr != nil {
				return
			}
		}
	}
	wg.Add(2)
	go drain(stdoutPipe, &stdout)
	go drain(stderrPipe, &stderr)

	// Wait for the child to exit on a separate goroutine so the watchdog
	// loop below can observe idle stalls and abort early.
	waitDone := make(chan error, 1)
	go func() {
		wg.Wait() // ensure all output is flushed before reaping
		waitDone <- c.Wait()
	}()

	idleSeconds := resolveIdleSeconds()
	idleTimedOut := false
	var waitErr error

	if idleSeconds > 0 {
		idle := time.Duration(idleSeconds) * time.Second
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
	loop:
		for {
			select {
			case waitErr = <-waitDone:
				break loop
			case <-ticker.C:
				mu.Lock()
				stalledFor := time.Since(lastActivity)
				mu.Unlock()
				if stalledFor >= idle {
					idleTimedOut = true
					killProcessGroup(c)
					waitErr = <-waitDone
					break loop
				}
			}
		}
	} else {
		waitErr = <-waitDone
	}

	result := &CLIResult{
		Stdout:     stdout.String(),
		Stderr:     stderr.String(),
		ReturnCode: 0,
	}

	if idleTimedOut {
		return result, fmt.Errorf("CLI command made no progress for %ds: %s", idleSeconds, strings.Join(cmd, " "))
	}

	if waitErr != nil {
		if exitErr, ok := waitErr.(*exec.ExitError); ok {
			result.ReturnCode = exitErr.ExitCode()
		} else if ctx.Err() != nil {
			return result, fmt.Errorf("CLI command timed out after %ds: %s", timeout, strings.Join(cmd, " "))
		} else {
			return nil, waitErr
		}
	}

	return result, nil
}

// isExecNotFound checks if an error indicates the binary was not found.
func isExecNotFound(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "executable file not found") ||
		strings.Contains(msg, "no such file or directory")
}

// truncate returns the first maxLen characters of s, or s if shorter.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}
