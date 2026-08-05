// Package launchdsvc holds the launchd integration shared by the menu-bar tray
// (cmd/af-tray) and the `af service` CLI command: the agent labels, the plist
// paths, thin launchctl wrappers, and the policy that decides whether an
// install may restart a control plane that is already running.
//
// The policy lives here as a pure function (TakeoverDecision) so it can be
// exhaustively unit-tested without invoking launchctl, which is a global,
// per-login-session mutation that tests must never perform.
package launchdsvc

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

// Agent labels. These are GLOBAL per login session: two installs on one machine
// share them, which is exactly why an install has to check who owns the label
// before reloading it.
const (
	TrayLabel   = "ai.agentfield.tray"
	ServerLabel = "ai.agentfield.server"
)

// LaunchAgentsDir is where per-user launchd agents live.
func LaunchAgentsDir(home string) string {
	return filepath.Join(home, "Library", "LaunchAgents")
}

// ServerPlistPath / TrayPlistPath name the two agent definitions.
func ServerPlistPath(home string) string {
	return filepath.Join(LaunchAgentsDir(home), ServerLabel+".plist")
}

func TrayPlistPath(home string) string {
	return filepath.Join(LaunchAgentsDir(home), TrayLabel+".plist")
}

// ---- Ownership -------------------------------------------------------------

// PlistOwner is the identifying part of an installed server agent: which binary
// it runs and which AgentField home that binary is pointed at. Two installs
// that differ in either are different owners of the shared label.
type PlistOwner struct {
	// Program is the first entry of ProgramArguments — the binary launchd runs.
	Program string
	// WorkingDirectory is the agent's cwd, which for the server agent is the
	// AgentField home directory.
	WorkingDirectory string
	// Home is the AGENTFIELD_HOME environment entry when the plist sets one.
	// Empty when absent; WorkingDirectory is the fallback signal.
	Home string
}

var (
	// The first <string> inside <array> under ProgramArguments.
	programArgsRe = regexp.MustCompile(
		`(?s)<key>ProgramArguments</key>\s*<array>\s*<string>([^<]*)</string>`)
	workingDirRe = regexp.MustCompile(
		`(?s)<key>WorkingDirectory</key>\s*<string>([^<]*)</string>`)
	envHomeRe = regexp.MustCompile(
		`(?s)<key>AGENTFIELD_HOME</key>\s*<string>([^<]*)</string>`)
)

// ParsePlistOwner pulls the identifying fields out of a launchd plist.
//
// Deliberately a targeted extraction rather than a full plist decoder: the only
// consumer is the ownership check, the two fields it needs are written by
// serverPlist() in a fixed shape, and adding an XML/plist dependency to read
// back our own template would be a poor trade. The round-trip is pinned by a
// test that parses the plist this repo generates.
func ParsePlistOwner(data []byte) PlistOwner {
	var owner PlistOwner
	if m := programArgsRe.FindSubmatch(data); m != nil {
		owner.Program = string(m[1])
	}
	if m := workingDirRe.FindSubmatch(data); m != nil {
		owner.WorkingDirectory = string(m[1])
	}
	if m := envHomeRe.FindSubmatch(data); m != nil {
		owner.Home = string(m[1])
	}
	return owner
}

// SameOwner reports whether an existing agent belongs to the install described
// by want. A reinstall or upgrade in place is the same owner — only a different
// binary path or a different AgentField home is a takeover.
func SameOwner(existing, want PlistOwner) bool {
	if existing.Program != want.Program {
		return false
	}
	// Prefer an explicit AGENTFIELD_HOME when either side declares one;
	// otherwise the working directory identifies the home.
	if existing.Home != "" || want.Home != "" {
		return existing.Home == want.Home
	}
	return existing.WorkingDirectory == want.WorkingDirectory
}

// ReadPlistOwner reads and parses a plist, reporting whether it exists.
func ReadPlistOwner(path string) (PlistOwner, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PlistOwner{}, false
	}
	return ParsePlistOwner(data), true
}

// ---- Takeover policy -------------------------------------------------------

// Action is what an install should do about the server launchd agent.
type Action string

const (
	// ActionReload is the historical behaviour: bootout, bootstrap, kickstart,
	// so the new binary is serving immediately.
	ActionReload Action = "reload"
	// ActionWriteOnly writes the binary and plists but leaves the running
	// server alone. The new version takes effect at the next restart.
	ActionWriteOnly Action = "write-only"
	// ActionRefuse aborts before writing anything: the label belongs to a
	// different install and taking it over was not requested.
	ActionRefuse Action = "refuse"
	// ActionSkip means the installed state already matches; nothing to do.
	ActionSkip Action = "skip"
)

// TakeoverInputs are the observations and switches the decision is made from.
// Every field is supplied by the caller so the policy stays pure.
type TakeoverInputs struct {
	// PlistExists reports a server plist already on disk.
	PlistExists bool
	// SameOwner reports that the existing plist points at this install.
	SameOwner bool
	// LabelLoaded reports that launchd currently has the label registered.
	LabelLoaded bool
	// ServerHealthy reports that GET /health answered.
	ServerHealthy bool
	// ActiveExecutions is the number of RECENTLY-ACTIVE in-flight runs from
	// GET /api/v1/executions/active. Only meaningful when ServerHealthy.
	// Runs that are listed but have gone quiet are excluded by the probe (see
	// launchdsvc.ActiveWindow), so a wedged run cannot make this permanently
	// non-zero and defer every future restart.
	ActiveExecutions int
	// Identical reports that the plists we would write are byte-identical to
	// what is on disk AND the target binary already has the new contents.
	Identical bool
	// ForceEnv is AGENTFIELD_INSTALL_FORCE_RESTART=1 — restore the old
	// unconditional behaviour.
	ForceEnv bool
	// DeferFlag is --defer-restart — never restart a running server.
	DeferFlag bool
	// TakeOverFlag is --take-over — permission to seize a label owned by a
	// different install.
	TakeOverFlag bool
}

// TakeoverDecision is the outcome plus a human-readable reason, which callers
// print so the user always knows why an install did or did not restart.
type TakeoverDecision struct {
	Action Action
	Reason string
}

// DecideTakeover is the whole policy, as a pure function.
//
// Precedence, most protective first:
//
//  1. Ownership. A plist that points at a different binary or home belongs to
//     another install; seizing the shared label would silently swap the server
//     out from under it. Refuse unless explicitly allowed.
//  2. --defer-restart. The caller has said not to interrupt anything.
//  3. AGENTFIELD_INSTALL_FORCE_RESTART. The caller has said to restart anyway.
//  4. Already up to date. Nothing to converge, so do not churn launchd.
//  5. Busy. A healthy server with work in flight is not interrupted; the files
//     are still written, so the next restart picks the new version up.
//  6. Otherwise converge, which is the historical hands-off behaviour.
func DecideTakeover(in TakeoverInputs) TakeoverDecision {
	if in.PlistExists && !in.SameOwner && !in.TakeOverFlag && !in.ForceEnv {
		return TakeoverDecision{ActionRefuse, "an existing AgentField server agent belongs to a different install"}
	}
	if in.DeferFlag {
		return TakeoverDecision{ActionWriteOnly, "--defer-restart: files updated, running server left alone"}
	}
	if in.ForceEnv {
		return TakeoverDecision{ActionReload, "AGENTFIELD_INSTALL_FORCE_RESTART=1: restarting unconditionally"}
	}
	if in.Identical {
		return TakeoverDecision{ActionSkip, "already up to date"}
	}
	if in.LabelLoaded && in.ServerHealthy && in.ActiveExecutions > 0 {
		return TakeoverDecision{
			ActionWriteOnly,
			fmt.Sprintf("%d workflow(s) in flight", in.ActiveExecutions),
		}
	}
	if in.LabelLoaded && in.ServerHealthy {
		return TakeoverDecision{ActionReload, "server running and idle"}
	}
	return TakeoverDecision{ActionReload, "no running server"}
}
