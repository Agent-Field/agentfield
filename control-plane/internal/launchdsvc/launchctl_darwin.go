//go:build darwin

package launchdsvc

import (
	"fmt"
	"os"
	"os/exec"
	"time"
)

// Supported reports whether launchd service management works on this platform.
func Supported() bool { return true }

// GUIDomain / SvcTarget address a per-login-session agent.
func GUIDomain() string         { return fmt.Sprintf("gui/%d", os.Getuid()) }
func SvcTarget(l string) string { return GUIDomain() + "/" + l }

// KickstartArgs builds the argv for `launchctl kickstart`. The -k flag forces a
// restart of an already-running service (kill then relaunch); without it,
// kickstart only starts a loaded-but-idle service.
func KickstartArgs(label string, kill bool) []string {
	args := []string{"kickstart"}
	if kill {
		args = append(args, "-k")
	}
	return append(args, SvcTarget(label))
}

func Bootstrap(plistPath string) error {
	return exec.Command("launchctl", "bootstrap", GUIDomain(), plistPath).Run()
}

func Bootout(label string) error {
	return exec.Command("launchctl", "bootout", SvcTarget(label)).Run()
}

func Kickstart(label string, kill bool) error {
	return exec.Command("launchctl", KickstartArgs(label, kill)...).Run()
}

// AgentLoaded reports whether launchd currently has the label registered.
func AgentLoaded(label string) bool {
	return exec.Command("launchctl", "print", SvcTarget(label)).Run() == nil
}

// SignalAgent sends a signal to a loaded agent (used for a graceful stop).
func SignalAgent(label, signal string) error {
	return exec.Command("launchctl", "kill", signal, SvcTarget(label)).Run()
}

// Reload converges a launchd agent onto the freshly written plist and binary.
// It fully unloads (bootout) then reloads (bootstrap) rather than using
// `kickstart -k`, because kickstart cannot re-exec across a binary whose code
// signature changed — and every rebuild/upgrade carries a new ad-hoc cdhash, so
// launchd rejects the relaunch with EX_CONFIG ("spawn failed") and the agent
// dies on upgrade. bootout+bootstrap always lands on the new bytes.
//
// bootout is not fully synchronous, so bootstrap is retried briefly until the
// prior job has finished tearing down. A final kickstart makes sure the agent
// is running now (not only at next login).
func Reload(plistPath, label string) {
	_ = Bootout(label) // ignored if the agent isn't currently loaded
	for i := 0; i < 20; i++ {
		if err := Bootstrap(plistPath); err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	_ = Kickstart(label, false)
}
