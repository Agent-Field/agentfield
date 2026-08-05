//go:build darwin

package main

import (
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
)

// TestServerPlistRoundTripsThroughOwnerParser is the contract between the plist
// TEMPLATE (serverPlist, here) and the PARSER that the ownership guard relies
// on (launchdsvc.ParsePlistOwner). If the template ever stops emitting
// ProgramArguments or WorkingDirectory in a shape the parser reads, the guard
// would silently see an empty owner, conclude "different install", and refuse
// every upgrade — so this pins the round trip against the real generator rather
// than a copied fixture.
//
// It only reads generated strings; nothing here touches launchd.
func TestServerPlistRoundTripsThroughOwnerParser(t *testing.T) {
	owner := launchdsvc.ParsePlistOwner([]byte(serverPlist()))

	if owner.Program == "" {
		t.Fatal("parser found no program path in the generated server plist")
	}
	if owner.Program != serverBinaryPath() {
		t.Errorf("Program = %q, want %q", owner.Program, serverBinaryPath())
	}
	if owner.WorkingDirectory != agentfieldDir() {
		t.Errorf("WorkingDirectory = %q, want %q", owner.WorkingDirectory, agentfieldDir())
	}

	// The generated plist must therefore be recognised as its own owner: an
	// upgrade in place is never a takeover.
	want := launchdsvc.PlistOwner{
		Program:          serverBinaryPath(),
		WorkingDirectory: agentfieldDir(),
	}
	if !launchdsvc.SameOwner(owner, want) {
		t.Errorf("generated plist not recognised as same-owner: got %#v want %#v", owner, want)
	}
}

// TestTrayPlistRoundTrips guards the same property for the tray agent, whose
// program path is the .app bundle binary.
func TestTrayPlistRoundTrips(t *testing.T) {
	owner := launchdsvc.ParsePlistOwner([]byte(trayPlist()))
	if owner.Program != trayBundleBinaryPath() {
		t.Errorf("tray Program = %q, want %q", owner.Program, trayBundleBinaryPath())
	}
}

// TestStaleSuffix: when the probe discounts wedged runs, the install says so —
// otherwise "1 workflow in flight" on a server the user believes is idle looks
// like the harness is simply wrong.
func TestStaleSuffix(t *testing.T) {
	if got := staleSuffix(0); got != "" {
		t.Errorf("staleSuffix(0) = %q, want empty", got)
	}
	if got := staleSuffix(-1); got != "" {
		t.Errorf("staleSuffix(-1) = %q, want empty", got)
	}
	got := staleSuffix(2)
	for _, want := range []string{"ignored 2 stale run(s)", "no activity for over"} {
		if !strings.Contains(got, want) {
			t.Errorf("staleSuffix(2) = %q, want it to mention %q", got, want)
		}
	}
}
