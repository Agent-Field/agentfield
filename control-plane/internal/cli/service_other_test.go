//go:build !darwin

package cli

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// TestCollectServiceStatusUnsupported covers the status assembly on a platform
// with no launchd: it must report unsupported/not-loaded without probing
// launchctl, while still filling in the rest of the struct.
func TestCollectServiceStatusUnsupported(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("AGENTFIELD_PORT", "1") // nothing listening on port 1
	st := collectServiceStatus()
	if st.Supported {
		t.Error("Supported must be false off macOS")
	}
	if st.Loaded {
		t.Error("Loaded must be false when launchd is absent")
	}
	if st.Healthy {
		t.Error("no server is listening, so Healthy must be false")
	}
	if !strings.HasSuffix(st.PlistPath, "ai.agentfield.server.plist") {
		t.Errorf("PlistPath = %q", st.PlistPath)
	}
}

// TestServiceDelegatesUnsupported pins the !darwin delegates directly. They sit
// behind requireLaunchd in normal use, so testing them here is what keeps the
// non-macOS build from carrying untested lines — and asserts they fail loudly
// rather than silently succeeding if a future caller forgets the guard.
func TestServiceDelegatesUnsupported(t *testing.T) {
	for name, fn := range map[string]func() error{
		"serviceStop":      serviceStop,
		"serviceRestart":   serviceRestart,
		"serviceUninstall": serviceUninstall,
	} {
		err := fn()
		if err == nil {
			t.Errorf("%s() must fail off macOS", name)
			continue
		}
		if !strings.Contains(err.Error(), "macOS-only") {
			t.Errorf("%s() = %v, want a macOS-only message", name, err)
		}
	}
}

// TestServiceMutatingRunEOffMacOS drives the cobra RunE bodies. On this
// platform they delegate to the stubs above and return the macOS-only message,
// so the Linux coverage runner exercises them while it remains impossible for
// any test to reach launchctl.
func TestServiceMutatingRunEOffMacOS(t *testing.T) {
	for name, build := range map[string]func() *cobra.Command{
		"stop":      newServiceStopCmd,
		"restart":   newServiceRestartCmd,
		"uninstall": newServiceUninstallCmd,
	} {
		t.Run(name, func(t *testing.T) {
			err := build().RunE(nil, nil)
			if err == nil {
				t.Fatalf("%s must fail off macOS", name)
			}
			if !strings.Contains(err.Error(), "macOS-only") {
				t.Errorf("%s error = %q, want it to say macOS-only", name, err)
			}
		})
	}
}
