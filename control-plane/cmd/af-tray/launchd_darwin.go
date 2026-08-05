//go:build darwin

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
)

// ---- Install / uninstall ---------------------------------------------------

// installOptions carry the switches that modify how far an install may go in
// taking over the shared launchd labels. Zero value is the default install.
type installOptions struct {
	// deferRestart never restarts a running server (install.sh --defer-restart).
	deferRestart bool
	// takeOver permits seizing a server agent owned by a different install.
	takeOver bool
}

// installDesktop is idempotent and convergent: every run rewrites the .app
// bundle and both launchd plists, then bootstraps-or-force-restarts each agent.
// This is what makes `curl … | install.sh` hands-off on both a fresh install
// and an update — a stale, already-running tray is killed and relaunched onto
// the freshly installed binary, and a freshly written agent is started now
// (not just at next login).
//
// That convergence is preserved for the TRAY agent, where a restart costs the
// user nothing. The SERVER agent is now gated by launchdsvc.DecideTakeover:
// the labels are global per login session, so a second install used to seize a
// running control plane — swapping the binary under a server that was mid-run,
// and respawning it via KeepAlive when the user killed it. See serverAgentStep.
func installDesktop() error { return installDesktopWith(installOptions{}) }

func installDesktopWith(opts installOptions) error {
	// Decide about the server agent BEFORE writing anything: a refusal must
	// leave the other install's files exactly as they were.
	decision, plistData, staleRuns := serverAgentDecision(opts)
	if decision.Action == launchdsvc.ActionRefuse {
		existing, _ := launchdsvc.ReadPlistOwner(serverPlistPath())
		return fmt.Errorf(
			"refusing to take over the AgentField server agent: %s\n"+
				"  currently registered: %s\n"+
				"  this install would use: %s\n"+
				"Re-run with --take-over to replace it, or AGENTFIELD_INSTALL_FORCE_RESTART=1 to force the old behaviour",
			decision.Reason, existing.Program, serverBinaryPath())
	}

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
	if err := writeFileAtomic(serverPlistPath(), plistData, 0o644); err != nil {
		return fmt.Errorf("write server plist: %w", err)
	}
	if err := writeFileAtomic(trayPlistPath(), []byte(trayPlist()), 0o644); err != nil {
		return fmt.Errorf("write tray plist: %w", err)
	}

	// The tray agent keeps converging unconditionally: restarting a menu-bar
	// app interrupts no work, and a stale tray running yesterday's binary is
	// exactly what this is for.
	reloadAgent(trayPlistPath(), trayLabel)

	// The server agent follows the policy decided above.
	switch decision.Action {
	case launchdsvc.ActionSkip:
		fmt.Println("AgentField server: already up to date; leaving it running.")
	case launchdsvc.ActionWriteOnly:
		fmt.Printf("AgentField server: %s — not restarting.%s\n",
			decision.Reason, staleSuffix(staleRuns))
		fmt.Println("  The new version takes effect on the next restart " +
			"(menu-bar Restart, or `af service restart`).")
	default:
		if decision.Reason == "server running and idle" {
			fmt.Println("AgentField server: running and idle — restarting onto the new version.")
		}
		reloadAgent(serverPlistPath(), serverLabel)
	}

	fmt.Println("AgentField desktop tray installed. Look for the icon in your menu bar.")
	return nil
}

// serverAgentDecision probes the current server agent and applies the takeover
// policy. It returns the decision and the plist bytes the install would write,
// so the caller can both act on the decision and avoid regenerating the plist.
func serverAgentDecision(opts installOptions) (launchdsvc.TakeoverDecision, []byte, int) {
	want := []byte(serverPlist())

	existing, plistExists := launchdsvc.ReadPlistOwner(serverPlistPath())
	sameOwner := !plistExists || launchdsvc.SameOwner(existing, launchdsvc.PlistOwner{
		Program:          serverBinaryPath(),
		WorkingDirectory: agentfieldDir(),
	})

	stale := 0
	in := launchdsvc.TakeoverInputs{
		PlistExists:  plistExists,
		SameOwner:    sameOwner,
		LabelLoaded:  agentLoaded(serverLabel),
		ForceEnv:     os.Getenv("AGENTFIELD_INSTALL_FORCE_RESTART") == "1",
		DeferFlag:    opts.deferRestart,
		TakeOverFlag: opts.takeOver,
	}
	if in.LabelLoaded {
		in.ServerHealthy = launchdsvc.ServerHealthy(serverPort())
		if in.ServerHealthy {
			// An unreadable endpoint (auth, older server) reports ok=false and
			// is treated as not-busy: an install must not be blocked forever by
			// a probe it cannot interpret.
			// Only runs that have done something recently block a restart;
			// a wedged run left in the active list forever must not pin an
			// install to an old binary. See launchdsvc.ActiveWindow.
			if n, s, ok := launchdsvc.ActiveExecutions(serverPort(), os.Getenv("AGENTFIELD_API_KEY")); ok {
				in.ActiveExecutions = n
				stale = s
			}
		}
	}
	// Up to date means: the plist we would write already on disk, and the
	// target binary already carrying the bytes we would install.
	in.Identical = plistExists &&
		launchdsvc.FileHasContents(serverPlistPath(), want) &&
		trayBundleUpToDate()

	return launchdsvc.DecideTakeover(in), want, stale
}

// staleSuffix names runs the probe deliberately ignored, so a user who reads
// "1 workflow in flight" against a server they believe is idle can see that the
// harness already discounted the wedged ones.
func staleSuffix(stale int) string {
	if stale <= 0 {
		return ""
	}
	return fmt.Sprintf(" (ignored %d stale run(s) with no activity for over %s)",
		stale, launchdsvc.ActiveWindow())
}

// trayBundleUpToDate reports whether the installed tray binary is already the
// one we are about to write — the other half of the "nothing changed" test.
func trayBundleUpToDate() bool {
	self, err := os.Executable()
	if err != nil {
		return false
	}
	selfSum, ok := launchdsvc.FileSHA256(self)
	if !ok {
		return false
	}
	installedSum, ok := launchdsvc.FileSHA256(trayBundleBinaryPath())
	if !ok {
		return false
	}
	return selfSum == installedSum
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
	return launchdsvc.SignalAgent(serverLabel, "SIGTERM")
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

// ---- launchctl wrappers ----------------------------------------------------
//
// The implementations live in internal/launchdsvc so `af service` and the tray
// drive launchd through exactly one code path. These thin aliases keep the
// tray's call sites unchanged.

func reloadAgent(plistPath, label string)   { launchdsvc.Reload(plistPath, label) }
func bootstrapAgent(plistPath string) error { return launchdsvc.Bootstrap(plistPath) }
func bootoutAgent(label string) error       { return launchdsvc.Bootout(label) }
func kickstartAgent(label string, kill bool) error {
	return launchdsvc.Kickstart(label, kill)
}
func agentLoaded(label string) bool { return launchdsvc.AgentLoaded(label) }
