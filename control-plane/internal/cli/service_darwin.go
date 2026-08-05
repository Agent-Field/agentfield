//go:build darwin

package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
)

// The launchd mutations live in their own platform file so the cross-platform
// command wiring in service.go stays fully exercisable on the Linux runner that
// measures coverage — and so no test on any platform can reach a launchctl call
// by accident.

// serviceStop asks launchd to SIGTERM the control plane. The agent is
// registered with KeepAlive={SuccessfulExit:false}, so a clean shutdown is not
// relaunched — which is exactly why a plain `kill` is not equivalent.
func serviceStop() error {
	if err := launchdsvc.SignalAgent(launchdsvc.ServerLabel, "SIGTERM"); err != nil {
		return fmt.Errorf("stop control plane: %w", err)
	}
	fmt.Println("Sent SIGTERM to the control plane; it will stay stopped until started again.")
	return nil
}

// serviceRestart reloads the agent onto whatever binary the plist now names.
func serviceRestart() error {
	plist := launchdsvc.ServerPlistPath(serviceHome())
	if _, err := os.Stat(plist); err != nil {
		return fmt.Errorf("no control-plane agent installed at %s", plist)
	}
	// Full reload rather than `kickstart -k`: an upgraded binary carries a new
	// ad-hoc code signature, which launchd refuses to re-exec.
	launchdsvc.Reload(plist, launchdsvc.ServerLabel)
	fmt.Println("Control plane restarted.")
	return nil
}

// serviceUninstall deregisters both agents and removes the menu-bar app.
func serviceUninstall() error {
	home := serviceHome()
	_ = launchdsvc.Bootout(launchdsvc.TrayLabel)
	_ = launchdsvc.Bootout(launchdsvc.ServerLabel)
	_ = os.Remove(launchdsvc.TrayPlistPath(home))
	_ = os.Remove(launchdsvc.ServerPlistPath(home))
	_ = os.RemoveAll(filepath.Join(home, "Applications", "AgentField.app"))
	fmt.Println("Control plane autostart and menu-bar app removed.")
	fmt.Println("The `af` binary itself is untouched.")
	return nil
}
