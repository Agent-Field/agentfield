//go:build !darwin

package launchdsvc

import (
	"errors"
	"testing"
)

// The stubs in launchctl_other.go ARE the compiled implementation everywhere
// except macOS — including on the Linux runner that measures coverage. They
// exist so `af service` reports a clear "macOS-only" message instead of
// pretending to manage a launchd job that cannot exist, so each one is pinned
// here rather than left as unexercised platform filler.
func TestUnsupportedPlatformStubs(t *testing.T) {
	if Supported() {
		t.Fatal("Supported() must be false off macOS")
	}

	t.Run("mutating calls all report unsupported", func(t *testing.T) {
		for name, err := range map[string]error{
			"Bootstrap":     Bootstrap("/tmp/whatever.plist"),
			"Bootout":       Bootout(ServerLabel),
			"Kickstart":     Kickstart(ServerLabel, false),
			"KickstartKill": Kickstart(ServerLabel, true),
			"SignalAgent":   SignalAgent(ServerLabel, "SIGTERM"),
		} {
			if !errors.Is(err, ErrUnsupported) {
				t.Errorf("%s returned %v, want ErrUnsupported", name, err)
			}
		}
	})

	t.Run("queries answer without touching launchd", func(t *testing.T) {
		if AgentLoaded(ServerLabel) {
			t.Error("AgentLoaded must be false off macOS")
		}
		if got := GUIDomain(); got != "" {
			t.Errorf("GUIDomain = %q, want empty", got)
		}
		if got := SvcTarget(ServerLabel); got != ServerLabel {
			t.Errorf("SvcTarget = %q, want the bare label", got)
		}
		if got := KickstartArgs(ServerLabel, true); got != nil {
			t.Errorf("KickstartArgs = %v, want nil", got)
		}
	})

	t.Run("Reload is an inert no-op", func(t *testing.T) {
		// Must not panic and must not attempt any process execution.
		Reload("/tmp/whatever.plist", ServerLabel)
	})

	if ErrUnsupported.Error() != "service management is macOS-only" {
		t.Errorf("ErrUnsupported message drifted: %q", ErrUnsupported.Error())
	}
}
