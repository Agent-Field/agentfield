//go:build !windows

package harness

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the child the leader of a new process group so the
// idle watchdog can signal the whole tree at once.
func setProcessGroup(c *exec.Cmd) {
	if c.SysProcAttr == nil {
		c.SysProcAttr = &syscall.SysProcAttr{}
	}
	c.SysProcAttr.Setpgid = true
}

// killProcessGroup hard-kills the child's process group. It falls back to
// killing just the process if the group signal fails.
func killProcessGroup(c *exec.Cmd) {
	if c.Process == nil {
		return
	}
	pid := c.Process.Pid
	if pid > 0 {
		// Negative pid signals the entire process group.
		if err := syscall.Kill(-pid, syscall.SIGKILL); err == nil {
			return
		}
	}
	_ = c.Process.Kill()
}
