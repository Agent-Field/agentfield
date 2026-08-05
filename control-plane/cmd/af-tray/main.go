// Command af-tray is the AgentField menu-bar companion.
//
// It is a small, separate binary from the main `af`/agentfield control-plane
// binary on purpose: it carries the GUI/systray dependency so the server binary
// never has to. In a headless deployment (Railway, ECS, EC2, a container) the
// tray simply is never installed or run — and if it is run there anyway, it
// detects the absence of a GUI session and exits cleanly instead of crashing.
//
// Subcommands:
//
//	af-tray run         Run the menu-bar tray (default).
//	af-tray install     Install the desktop tray + control-plane autostart (macOS).
//	af-tray uninstall   Remove the desktop tray and autostart.
//	af-tray version     Print version.
//
// The platform-specific behaviour lives in tray_darwin.go / launchd_darwin.go
// (real implementation) and tray_other.go (no-op stubs for every other OS).
package main

import (
	"fmt"
	"os"
)

// Build-time version information (set via ldflags during build).
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

// run dispatches a subcommand and returns the process exit code. It is split
// out of main so it can be unit-tested without spawning a process.
func run(args []string) int {
	cmd := "run"
	if len(args) > 0 {
		cmd = args[0]
	}

	switch cmd {
	case "run":
		if err := runTray(); err != nil {
			fmt.Fprintln(os.Stderr, "af-tray:", err)
			return 1
		}
	case "install":
		if err := installDesktopWith(parseInstallOptions(args[1:])); err != nil {
			fmt.Fprintln(os.Stderr, "af-tray install:", err)
			return 1
		}
	case "uninstall":
		if err := uninstallDesktop(); err != nil {
			fmt.Fprintln(os.Stderr, "af-tray uninstall:", err)
			return 1
		}
	case "version", "--version", "-v":
		fmt.Printf("af-tray %s (%s) %s\n", version, commit, date)
	case "help", "--help", "-h":
		printUsage()
	default:
		printUsage()
		return 2
	}
	return 0
}

// parseInstallOptions reads the install-mode flags. Unknown arguments are
// ignored rather than rejected: install.sh may pass flags a newer installer
// script knows about but an older tray binary does not, and a hands-off
// `curl … | bash` must not fail on that.
func parseInstallOptions(args []string) installOptions {
	var opts installOptions
	for _, a := range args {
		switch a {
		case "--defer-restart":
			opts.deferRestart = true
		case "--take-over":
			opts.takeOver = true
		}
	}
	return opts
}

func printUsage() {
	fmt.Print(`af-tray — AgentField menu-bar companion

Usage:
  af-tray run         Run the menu-bar tray (default)
  af-tray install     Install the desktop tray + control-plane autostart (macOS)
  af-tray uninstall   Remove the desktop tray and control-plane autostart
  af-tray version     Print version information

Install flags:
  --defer-restart   Update files but never restart a running control plane
  --take-over       Replace a launchd agent registered by a different install
`)
}
