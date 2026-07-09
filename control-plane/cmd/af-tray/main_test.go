package main

import "testing"

// Contract: the CLI dispatch returns 0 for version/help variants and 2 for an
// unknown subcommand. These arms are platform-independent (they don't touch the
// tray or launchd), so they run everywhere.
func TestRunDispatchExitCodes(t *testing.T) {
	zero := []string{"version", "--version", "-v", "help", "--help", "-h"}
	for _, cmd := range zero {
		if code := run([]string{cmd}); code != 0 {
			t.Errorf("run([%q]) = %d, want 0", cmd, code)
		}
	}
	if code := run([]string{"totally-unknown"}); code != 2 {
		t.Errorf("run([unknown]) = %d, want 2", code)
	}
}
