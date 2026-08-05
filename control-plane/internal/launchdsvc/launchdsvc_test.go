package launchdsvc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDecideTakeover walks every branch of the policy. The launchd labels are
// global per login session, so these decisions are the only thing standing
// between two installs on one machine and a silent server swap.
func TestDecideTakeover(t *testing.T) {
	// running is a healthy, loaded, idle server owned by this install.
	running := TakeoverInputs{PlistExists: true, SameOwner: true, LabelLoaded: true, ServerHealthy: true}

	cases := []struct {
		name   string
		in     TakeoverInputs
		want   Action
		reason string // substring
	}{
		{
			name: "fresh machine: nothing installed",
			in:   TakeoverInputs{},
			want: ActionReload, reason: "no running server",
		},
		{
			name: "installed but not loaded",
			in:   TakeoverInputs{PlistExists: true, SameOwner: true},
			want: ActionReload, reason: "no running server",
		},
		{
			name: "loaded but not answering (crashed/stopped)",
			in:   TakeoverInputs{PlistExists: true, SameOwner: true, LabelLoaded: true},
			want: ActionReload, reason: "no running server",
		},
		{
			name: "upgrade over a running idle server",
			in:   running,
			want: ActionReload, reason: "idle",
		},
		{
			name: "upgrade over a busy server does not interrupt it",
			in:   withActive(running, 3),
			want: ActionWriteOnly, reason: "3 workflow(s) in flight",
		},
		{
			// The launch-rehearsal scenario: a second install, from a different
			// home, while the first one is serving.
			name: "different install owns the label",
			in: TakeoverInputs{
				PlistExists: true, SameOwner: false,
				LabelLoaded: true, ServerHealthy: true,
			},
			want: ActionRefuse, reason: "different install",
		},
		{
			name: "different owner, but --take-over given",
			in: TakeoverInputs{
				PlistExists: true, SameOwner: false,
				LabelLoaded: true, ServerHealthy: true, TakeOverFlag: true,
			},
			want: ActionReload, reason: "idle",
		},
		{
			name: "different owner, but force env given",
			in: TakeoverInputs{
				PlistExists: true, SameOwner: false,
				LabelLoaded: true, ServerHealthy: true, ForceEnv: true,
			},
			want: ActionReload, reason: "FORCE_RESTART",
		},
		{
			name: "defer never restarts, even when idle",
			in:   withDefer(running),
			want: ActionWriteOnly, reason: "--defer-restart",
		},
		{
			name: "defer never restarts, even when busy",
			in:   withDefer(withActive(running, 9)),
			want: ActionWriteOnly, reason: "--defer-restart",
		},
		{
			name: "force restarts even when busy",
			in:   withForce(withActive(running, 5)),
			want: ActionReload, reason: "FORCE_RESTART",
		},
		{
			name: "force beats already-up-to-date",
			in:   withForce(withIdentical(running)),
			want: ActionReload, reason: "FORCE_RESTART",
		},
		{
			name: "same version reinstall does nothing",
			in:   withIdentical(running),
			want: ActionSkip, reason: "up to date",
		},
		{
			name: "up to date but busy still skips (nothing to converge)",
			in:   withIdentical(withActive(running, 2)),
			want: ActionSkip, reason: "up to date",
		},
		{
			// Ownership outranks defer: writing our plist over another
			// install's is itself the takeover, restart or not.
			name: "different owner with defer is still refused",
			in: TakeoverInputs{
				PlistExists: true, SameOwner: false, LabelLoaded: true,
				ServerHealthy: true, DeferFlag: true,
			},
			want: ActionRefuse, reason: "different install",
		},
		{
			name: "busy count is ignored when the server is not answering",
			in: TakeoverInputs{
				PlistExists: true, SameOwner: true, LabelLoaded: true,
				ServerHealthy: false, ActiveExecutions: 4,
			},
			want: ActionReload, reason: "no running server",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecideTakeover(tc.in)
			if got.Action != tc.want {
				t.Fatalf("action = %q, want %q (reason %q)", got.Action, tc.want, got.Reason)
			}
			if !strings.Contains(got.Reason, tc.reason) {
				t.Errorf("reason = %q, want it to mention %q", got.Reason, tc.reason)
			}
		})
	}
}

func withActive(in TakeoverInputs, n int) TakeoverInputs { in.ActiveExecutions = n; return in }
func withDefer(in TakeoverInputs) TakeoverInputs         { in.DeferFlag = true; return in }
func withForce(in TakeoverInputs) TakeoverInputs         { in.ForceEnv = true; return in }
func withIdentical(in TakeoverInputs) TakeoverInputs     { in.Identical = true; return in }

// TestDecideTakeoverNeverPanics guards the policy's totality across the whole
// input space: every combination must yield one of the four actions.
func TestDecideTakeoverNeverPanics(t *testing.T) {
	valid := map[Action]bool{
		ActionReload: true, ActionWriteOnly: true, ActionRefuse: true, ActionSkip: true,
	}
	for mask := 0; mask < 1<<7; mask++ {
		in := TakeoverInputs{
			PlistExists:   mask&1 != 0,
			SameOwner:     mask&2 != 0,
			LabelLoaded:   mask&4 != 0,
			ServerHealthy: mask&8 != 0,
			Identical:     mask&16 != 0,
			ForceEnv:      mask&32 != 0,
			DeferFlag:     mask&64 != 0,
		}
		for _, active := range []int{0, 1} {
			in.ActiveExecutions = active
			got := DecideTakeover(in)
			if !valid[got.Action] {
				t.Fatalf("mask=%d active=%d produced %q", mask, active, got.Action)
			}
			if got.Reason == "" {
				t.Fatalf("mask=%d active=%d produced an empty reason", mask, active)
			}
		}
	}
}

// ---- plist parsing ---------------------------------------------------------

// serverPlistFixture mirrors the template in cmd/af-tray/shared.go. The
// round-trip test below parses it, so a change to that template that this
// parser cannot read shows up here.
func serverPlistFixture(program, workdir string) string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>server</string>
    <string>--open=false</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>WorkingDirectory</key><string>%s</string>
  <key>StandardOutPath</key><string>/tmp/out.log</string>
  <key>StandardErrorPath</key><string>/tmp/out.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/bin:/bin</string>
  </dict>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
`, ServerLabel, program, workdir)
}

func TestParsePlistOwnerRoundTrip(t *testing.T) {
	const program = "/Users/demo/.agentfield/bin/agentfield"
	const workdir = "/Users/demo/.agentfield"

	owner := ParsePlistOwner([]byte(serverPlistFixture(program, workdir)))
	if owner.Program != program {
		t.Errorf("Program = %q, want %q", owner.Program, program)
	}
	if owner.WorkingDirectory != workdir {
		t.Errorf("WorkingDirectory = %q, want %q", owner.WorkingDirectory, workdir)
	}
	// The first ProgramArguments entry is the binary — not "server".
	if owner.Program == "server" {
		t.Error("parser picked an argument instead of the program")
	}
}

func TestParsePlistOwnerReadsEnvHome(t *testing.T) {
	body := strings.Replace(
		serverPlistFixture("/opt/af/agentfield", "/opt/af"),
		`<key>PATH</key><string>/usr/bin:/bin</string>`,
		`<key>PATH</key><string>/usr/bin:/bin</string>
    <key>AGENTFIELD_HOME</key><string>/opt/af-home</string>`, 1)
	owner := ParsePlistOwner([]byte(body))
	if owner.Home != "/opt/af-home" {
		t.Errorf("Home = %q, want /opt/af-home", owner.Home)
	}
}

func TestParsePlistOwnerTolerantOfGarbage(t *testing.T) {
	for _, body := range []string{"", "not a plist", "<plist></plist>"} {
		owner := ParsePlistOwner([]byte(body))
		if owner.Program != "" || owner.WorkingDirectory != "" {
			t.Errorf("garbage %q parsed as %#v", body, owner)
		}
	}
}

// TestSameOwner is the ownership rule: an in-place upgrade is not a takeover,
// a different binary or a different home is.
func TestSameOwner(t *testing.T) {
	base := PlistOwner{Program: "/Users/a/.agentfield/bin/agentfield", WorkingDirectory: "/Users/a/.agentfield"}

	if !SameOwner(base, base) {
		t.Error("identical owners must match (reinstall/upgrade in place)")
	}
	other := PlistOwner{Program: "/tmp/sandbox/.agentfield/bin/agentfield", WorkingDirectory: "/tmp/sandbox/.agentfield"}
	if SameOwner(base, other) {
		t.Error("a different binary path is a different owner")
	}
	sameBinDifferentHome := PlistOwner{Program: base.Program, WorkingDirectory: "/Users/a/.agentfield-staging"}
	if SameOwner(base, sameBinDifferentHome) {
		t.Error("a different home is a different owner")
	}
	// An explicit AGENTFIELD_HOME wins over the working directory.
	withHome := PlistOwner{Program: base.Program, WorkingDirectory: base.WorkingDirectory, Home: "/h1"}
	otherHome := PlistOwner{Program: base.Program, WorkingDirectory: base.WorkingDirectory, Home: "/h2"}
	if SameOwner(withHome, otherHome) {
		t.Error("differing AGENTFIELD_HOME must be a different owner")
	}
	if !SameOwner(withHome, withHome) {
		t.Error("matching AGENTFIELD_HOME must match")
	}
}

func TestReadPlistOwnerMissingFile(t *testing.T) {
	if _, ok := ReadPlistOwner(filepath.Join(t.TempDir(), "absent.plist")); ok {
		t.Error("a missing plist must report ok=false")
	}
}

func TestReadPlistOwnerReadsFromDisk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "server.plist")
	if err := os.WriteFile(path, []byte(serverPlistFixture("/bin/x", "/home/x")), 0o644); err != nil {
		t.Fatal(err)
	}
	owner, ok := ReadPlistOwner(path)
	if !ok || owner.Program != "/bin/x" {
		t.Fatalf("owner = %#v ok=%v", owner, ok)
	}
}

// ---- paths / hashing -------------------------------------------------------

func TestPlistPaths(t *testing.T) {
	home := "/Users/demo"
	if got := ServerPlistPath(home); got != "/Users/demo/Library/LaunchAgents/ai.agentfield.server.plist" {
		t.Errorf("ServerPlistPath = %q", got)
	}
	if got := TrayPlistPath(home); got != "/Users/demo/Library/LaunchAgents/ai.agentfield.tray.plist" {
		t.Errorf("TrayPlistPath = %q", got)
	}
}

func TestFileHasContents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f")
	body := []byte("hello")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if !FileHasContents(path, body) {
		t.Error("identical contents must match")
	}
	if FileHasContents(path, []byte("other")) {
		t.Error("different contents must not match")
	}
	if FileHasContents(filepath.Join(dir, "absent"), body) {
		t.Error("a missing file must not match")
	}
}

func TestActiveExecutionsURLMatchesServerRoute(t *testing.T) {
	// Pinned to internal/server/routes_core.go: the /api/v1 group registers
	// GET /executions/active.
	if got := ActiveExecutionsURL(8080); got != "http://localhost:8080/api/v1/executions/active" {
		t.Errorf("ActiveExecutionsURL = %q", got)
	}
}
