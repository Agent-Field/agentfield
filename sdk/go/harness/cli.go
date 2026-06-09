package harness

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

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

// RunCLI runs a CLI command and returns its output. The context controls
// cancellation; timeout is in seconds (0 means no timeout beyond ctx).
//
// Environment merging: entries in env are merged with os.Environ(). An empty
// string value ("") causes that variable to be removed from the environment
// rather than set to empty — use this to unset inherited variables.
func RunCLI(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error) {
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
	unset := make(map[string]bool)
	for k, v := range env {
		if v == "" {
			unset[k] = true
		}
	}
	var mergedEnv []string
	for _, entry := range os.Environ() {
		key, _, found := strings.Cut(entry, "=")
		if !found {
			mergedEnv = append(mergedEnv, entry)
			continue
		}
		if unset[key] {
			continue
		}
		mergedEnv = append(mergedEnv, entry)
	}
	for k, v := range env {
		if v != "" {
			mergedEnv = append(mergedEnv, k+"="+v)
		}
	}
	c.Env = mergedEnv

	if cwd != "" {
		c.Dir = cwd
	}

	var stdout, stderr bytes.Buffer
	c.Stdout = &stdout
	c.Stderr = &stderr

	err := c.Run()

	result := &CLIResult{
		Stdout:     stdout.String(),
		Stderr:     stderr.String(),
		ReturnCode: 0,
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ReturnCode = exitErr.ExitCode()
		} else if ctx.Err() != nil {
			return result, fmt.Errorf("CLI command timed out after %ds: %s", timeout, strings.Join(cmd, " "))
		} else {
			return nil, err
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
