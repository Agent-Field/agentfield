//go:build !darwin

package main

import "fmt"

// The AgentField desktop tray is a macOS-first feature. On every other platform
// these are graceful no-ops that exit 0, so:
//   - `curl … | install.sh` flows on Linux/headless boxes never error out, and
//   - if a launchd/systemd unit somehow invokes `af-tray run` in a container,
//     it prints one line and exits cleanly instead of crashing.
//
// Crucially, this file imports nothing GUI-related (no systray/CGO), so the
// tray package builds cleanly on Linux without pulling in any desktop deps.

func runTray() error {
	fmt.Println("af-tray: the AgentField desktop tray is only supported on macOS in this release.")
	return nil
}

func installDesktop() error {
	fmt.Println("af-tray: desktop tray install is only supported on macOS in this release.")
	return nil
}

func uninstallDesktop() error {
	fmt.Println("af-tray: nothing to uninstall on this platform.")
	return nil
}
