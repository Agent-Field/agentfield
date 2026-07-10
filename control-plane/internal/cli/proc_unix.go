//go:build !windows

package cli

import (
	"os"
	"syscall"
)

// signalGracefulStop asks a process to shut down gracefully. On Unix this
// sends SIGINT (os.Interrupt), matching the historical `af stop` behaviour.
// Callers fall back to process.Kill() when it returns an error.
func signalGracefulStop(process *os.Process) error {
	return process.Signal(os.Interrupt)
}

// isProcessAlive reports whether the process is still running. On Unix,
// signal 0 probes for liveness without delivering an actual signal.
func isProcessAlive(process *os.Process) bool {
	return process.Signal(syscall.Signal(0)) == nil
}
