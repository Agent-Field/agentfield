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

// TestParseInstallOptions covers the install-mode flags install.sh threads
// through. Unknown flags must be ignored, not rejected: a newer install.sh may
// pass a flag an older tray binary predates, and `curl … | bash` must not fail.
func TestParseInstallOptions(t *testing.T) {
	for _, tc := range []struct {
		name         string
		args         []string
		deferRestart bool
		takeOver     bool
	}{
		{name: "no flags", args: nil},
		{name: "defer", args: []string{"--defer-restart"}, deferRestart: true},
		{name: "take over", args: []string{"--take-over"}, takeOver: true},
		{name: "both", args: []string{"--take-over", "--defer-restart"}, deferRestart: true, takeOver: true},
		{name: "unknown ignored", args: []string{"--from-the-future", "--defer-restart"}, deferRestart: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := parseInstallOptions(tc.args)
			if got.deferRestart != tc.deferRestart || got.takeOver != tc.takeOver {
				t.Fatalf("parseInstallOptions(%v) = %#v", tc.args, got)
			}
		})
	}
}
