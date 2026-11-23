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

// Contract: desktop-tray takes exactly "show|hide" and "auto|login|session".
// Anything else exits 2 before the tray is touched, on every platform, so a typo
// in Desktop's call can never register or remove a login item.
func TestDesktopTrayRejectsMalformedArguments(t *testing.T) {
	for _, args := range [][]string{
		{"desktop-tray"},
		{"desktop-tray", "show"},
		{"desktop-tray", "toggle", "auto"},
		{"desktop-tray", "show", "forever"},
		{"desktop-tray", "show", "auto", "extra"},
	} {
		if code := run(args); code != 2 {
			t.Errorf("run(%q) = %d, want 2", args, code)
		}
	}
}
