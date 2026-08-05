//go:build !darwin

package cli

import (
	"fmt"
	"runtime"
)

// Off macOS there is no launchd to drive, so each command reports that plainly
// instead of pretending to manage a service. Keeping the refusal here — rather
// than in a guard inside the shared command wiring — means service.go carries
// no platform-conditional branches, and every line of it is exercised on both
// the macOS and Linux test runners.

func serviceStop() error      { return errMacOSOnly("stop") }
func serviceRestart() error   { return errMacOSOnly("restart") }
func serviceUninstall() error { return errMacOSOnly("uninstall") }

func errMacOSOnly(action string) error {
	return fmt.Errorf("af service %s: service management is macOS-only (this is %s)",
		action, runtime.GOOS)
}
