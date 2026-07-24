//go:build !darwin

package main

import "testing"

// Contract: on non-macOS, run/install/uninstall dispatch to graceful no-op
// stubs that succeed (exit 0) rather than crashing — this is what keeps
// `curl | install.sh` and any stray invocation safe on Linux/headless/container
// hosts. This test is itself non-darwin-tagged so it never executes the real
// (blocking, side-effecting) macOS implementations.
func TestPlatformSubcommandsNoOpOffMacOS(t *testing.T) {
	for _, cmd := range []string{"run", "install", "uninstall"} {
		if code := run([]string{cmd}); code != 0 {
			t.Errorf("run([%q]) = %d, want 0 (stub must no-op cleanly)", cmd, code)
		}
	}

	if err := runTray(); err != nil {
		t.Errorf("runTray() stub returned %v, want nil", err)
	}
	if err := installDesktop(); err != nil {
		t.Errorf("installDesktop() stub returned %v, want nil", err)
	}
	if err := uninstallDesktop(); err != nil {
		t.Errorf("uninstallDesktop() stub returned %v, want nil", err)
	}
}
