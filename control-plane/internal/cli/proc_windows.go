//go:build windows

package cli

// NOTE: compile-verified only (GOOS=windows cross-build); not yet exercised on
// a real Windows machine.

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// signalGracefulStop asks a process to shut down gracefully. Windows cannot
// deliver SIGINT/SIGTERM to an unrelated process (os.Process.Signal only
// supports os.Kill there), so use `taskkill` without /F, which requests the
// target to close. Callers fall back to process.Kill() when it returns an
// error.
func signalGracefulStop(process *os.Process) error {
	return exec.Command("taskkill", "/PID", strconv.Itoa(process.Pid)).Run()
}

// isProcessAlive reports whether the process is still running. On Windows
// os.FindProcess always succeeds and signal-0 probing is unsupported, so ask
// tasklist whether the PID is present. When the probe itself fails, report
// not-alive so callers skip the force-kill rather than killing blindly.
func isProcessAlive(process *os.Process) bool {
	out, err := exec.Command(
		"tasklist", "/FI", "PID eq "+strconv.Itoa(process.Pid), "/NH", "/FO", "CSV",
	).Output()
	if err != nil {
		return false
	}
	// CSV rows quote every field; a live PID appears as "...","<pid>",...
	// A no-match run prints an INFO message instead and still exits 0.
	return strings.Contains(string(out), `"`+strconv.Itoa(process.Pid)+`"`)
}
