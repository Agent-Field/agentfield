//go:build darwin

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// ---- Install / uninstall ---------------------------------------------------

// installDesktop is idempotent and convergent: every run rewrites the .app
// bundle and both launchd plists, then bootstraps-or-force-restarts each agent.
// This is what makes `curl … | install.sh` hands-off on both a fresh install
// and an update — a stale, already-running tray is killed and relaunched onto
// the freshly installed binary, and a freshly written agent is started now
// (not just at next login).
func installDesktop() error {
	for _, d := range []string{logsDir(), launchAgentsDir(),
		filepath.Join(appBundleDir(), "Contents", "MacOS"),
		filepath.Join(appBundleDir(), "Contents", "Resources")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", d, err)
		}
	}

	// Build the .app bundle around a copy of ourselves. Using rename-over means
	// we can safely replace the binary even while an old tray is executing it.
	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate self: %w", err)
	}
	selfData, err := os.ReadFile(self)
	if err != nil {
		return fmt.Errorf("read self: %w", err)
	}
	if err := writeFileAtomic(trayBundleBinaryPath(), selfData, 0o755); err != nil {
		return fmt.Errorf("install tray binary: %w", err)
	}
	if err := writeFileAtomic(filepath.Join(appBundleDir(), "Contents", "Resources", "appicon.icns"), appIconICNS, 0o644); err != nil {
		return fmt.Errorf("write app icon: %w", err)
	}
	if err := writeFileAtomic(filepath.Join(appBundleDir(), "Contents", "Info.plist"), []byte(infoPlist()), 0o644); err != nil {
		return fmt.Errorf("write Info.plist: %w", err)
	}

	// launchd agents.
	if err := writeFileAtomic(serverPlistPath(), []byte(serverPlist()), 0o644); err != nil {
		return fmt.Errorf("write server plist: %w", err)
	}
	if err := writeFileAtomic(trayPlistPath(), []byte(trayPlist()), 0o644); err != nil {
		return fmt.Errorf("write tray plist: %w", err)
	}

	// Converge launchd state by fully reloading each agent (see reloadAgent).
	reloadAgent(serverPlistPath(), serverLabel)
	reloadAgent(trayPlistPath(), trayLabel)

	fmt.Println("AgentField desktop tray installed. Look for the icon in your menu bar.")
	return nil
}

func uninstallDesktop() error {
	_ = bootoutAgent(trayLabel)
	_ = bootoutAgent(serverLabel)
	_ = os.Remove(trayPlistPath())
	_ = os.Remove(serverPlistPath())
	_ = os.RemoveAll(appBundleDir())
	fmt.Println("AgentField desktop tray removed.")
	return nil
}

// ---- Server lifecycle (driven from the tray menu) --------------------------

func startServer() error {
	if !agentLoaded(serverLabel) {
		_ = bootstrapAgent(serverPlistPath())
	}
	return kickstartAgent(serverLabel, false)
}

// stopServer sends SIGTERM for a graceful shutdown. Because the server plist
// uses KeepAlive={SuccessfulExit: false}, a clean exit is not relaunched — so
// "Stop" actually stops it, while a genuine crash still auto-restarts.
func stopServer() error {
	return exec.Command("launchctl", "kill", "SIGTERM", svcTarget(serverLabel)).Run()
}

func restartServer() error {
	if !agentLoaded(serverLabel) {
		_ = bootstrapAgent(serverPlistPath())
	}
	return kickstartAgent(serverLabel, true)
}

// serverAutostartEnabled reflects whether the server agent is loaded (and will
// therefore start at login).
func serverAutostartEnabled() bool { return agentLoaded(serverLabel) }

func setServerAutostart(enable bool) error {
	if enable {
		if err := bootstrapAgent(serverPlistPath()); err != nil && !agentLoaded(serverLabel) {
			return err
		}
		return kickstartAgent(serverLabel, false)
	}
	return bootoutAgent(serverLabel)
}

// ---- launchctl exec wrappers -----------------------------------------------

// reloadAgent converges a launchd agent onto the freshly written plist and
// binary. It fully unloads (bootout) then reloads (bootstrap) rather than using
// `kickstart -k`, because kickstart cannot re-exec across a binary whose code
// signature changed — and every rebuild/upgrade carries a new ad-hoc cdhash, so
// launchd rejects the relaunch with EX_CONFIG ("spawn failed") and the agent
// dies on upgrade. bootout+bootstrap always lands on the new bytes.
//
// bootout is not fully synchronous, so bootstrap is retried briefly until the
// prior job has finished tearing down. A final kickstart makes sure the agent
// is running now (not only at next login).
func reloadAgent(plistPath, label string) {
	_ = bootoutAgent(label) // ignored if the agent isn't currently loaded
	for i := 0; i < 20; i++ {
		if err := bootstrapAgent(plistPath); err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	_ = kickstartAgent(label, false)
}

func bootstrapAgent(plistPath string) error {
	return exec.Command("launchctl", "bootstrap", guiDomain(), plistPath).Run()
}

func bootoutAgent(label string) error {
	return exec.Command("launchctl", "bootout", svcTarget(label)).Run()
}

func kickstartAgent(label string, kill bool) error {
	return exec.Command("launchctl", kickstartArgs(label, kill)...).Run()
}

func agentLoaded(label string) bool {
	return exec.Command("launchctl", "print", svcTarget(label)).Run() == nil
}
