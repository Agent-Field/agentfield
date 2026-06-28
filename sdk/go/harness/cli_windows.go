//go:build windows

package harness

import "os/exec"

// setProcessGroup is a no-op on Windows (process-group semantics differ).
func setProcessGroup(_ *exec.Cmd) {}

// killProcessGroup kills the child process. Windows lacks POSIX process groups,
// so this kills the leader directly.
func killProcessGroup(c *exec.Cmd) {
	if c.Process != nil {
		_ = c.Process.Kill()
	}
}
