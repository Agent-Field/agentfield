package packages

import "testing"

// Contract: under test (stdout is not a terminal), clearLine emits nothing so
// piped/captured output stays clean.
func TestClearLineNonTTY(t *testing.T) {
	if got := clearLine(); got != "" {
		t.Fatalf("clearLine() = %q; want empty when stdout is not a TTY", got)
	}
	if stdoutIsTTY() {
		t.Skip("stdout is a terminal in this environment; non-TTY assertions are moot")
	}
}

// Contract: the spinner is safe to Start/Success/Error when stdout is not a
// terminal — it must not spawn an animator or block on Stop's done channel.
func TestSpinnerNonTTYLifecycle(t *testing.T) {
	if stdoutIsTTY() {
		t.Skip("stdout is a terminal in this environment")
	}
	pi := &PackageInstaller{}

	s := pi.newSpinner("cloning")
	s.Start()          // non-TTY: no goroutine spawned
	s.Success("cloned") // must not block on the done channel

	s2 := pi.newSpinner("building")
	s2.Start()
	s2.Error("failed") // Error path also stops cleanly

	// Start followed by a bare Stop must also not block.
	s3 := pi.newSpinner("noop")
	s3.Start()
	s3.Stop()
}
