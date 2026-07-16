//go:build !windows

package cli

import (
	"os"
	"os/exec"
	"testing"
)

// Contract: isProcessAlive must report true for a live process and false once
// the process has exited. `af stop` uses this to decide whether the graceful
// shutdown worked or a force-kill is needed.
func TestIsProcessAlive(t *testing.T) {
	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("FindProcess(self): %v", err)
	}
	if !isProcessAlive(self) {
		t.Fatal("isProcessAlive(current process) = false; want true")
	}

	cmd := exec.Command("true")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if isProcessAlive(cmd.Process) {
		t.Fatal("isProcessAlive(exited process) = true; want false")
	}
}

// Contract: signalGracefulStop must deliver an interrupt that terminates a
// well-behaved (default-signal-disposition) child process.
func TestSignalGracefulStop(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := signalGracefulStop(cmd.Process); err != nil {
		t.Fatalf("signalGracefulStop: %v", err)
	}
	// The child dies from the signal, so Wait returns a non-nil ExitError.
	if err := cmd.Wait(); err == nil {
		t.Fatal("child exited cleanly; want interrupt-terminated")
	}
	if isProcessAlive(cmd.Process) {
		t.Fatal("process still alive after graceful stop")
	}
}
