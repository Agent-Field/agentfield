package skillkit

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// Target is a coding-agent integration the skill can be installed into.
// Each target knows how to detect itself, install, uninstall, and report
// its current installed version (if any).
type Target interface {
	Name() string                                                              // canonical short name, e.g. "claude-code"
	DisplayName() string                                                       // pretty name for UI, e.g. "Claude Code"
	Detected() bool                                                            // is this target installed on the user's machine?
	Method() string                                                            // "symlink", "marker-block", "manual"
	TargetPath() (string, error)                                               // canonical path the target writes to
	Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error)  // performs the install (idempotent)
	Uninstall() error                                                          // removes the integration
	Status() (installed bool, version string, err error)                       // currently installed?
}

// AllTargets returns the registered list of targets in stable order. New
// targets register themselves in init() and append to this slice.
var allTargets []Target

// RegisterTarget adds a target to the global registry. Called from init() in
// each per-target file.
func RegisterTarget(t Target) {
	allTargets = append(allTargets, t)
}

// AllTargets returns the registered targets.
func AllTargets() []Target {
	return allTargets
}

// TargetByName looks up a target by its short name.
func TargetByName(name string) (Target, error) {
	for _, t := range allTargets {
		if t.Name() == name {
			return t, nil
		}
	}
	available := make([]string, len(allTargets))
	for i, t := range allTargets {
		available[i] = t.Name()
	}
	return nil, fmt.Errorf("target %q not registered (available: %v)", name, available)
}

// DetectedTargets returns the subset of registered targets that are
// currently installed on the user's machine.
func DetectedTargets() []Target {
	var out []Target
	for _, t := range allTargets {
		if t.Detected() {
			out = append(out, t)
		}
	}
	return out
}

// ── Shared utilities used by per-target install logic ────────────────────

// homeDir returns the user's home directory.
func homeDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return ""
}

// commandAvailable returns true if the named binary is on PATH.
func commandAvailable(bin string) bool {
	_, err := exec.LookPath(bin)
	return err == nil
}

// dirExists returns true if a directory exists at path.
func dirExists(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// fileExists returns true if a regular file exists at path.
func fileExists(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// markerStart returns the opening marker line for a skill's marker block.
// Used by file-append targets (Codex, Gemini, OpenCode, Aider, Windsurf).
func markerStart(skill Skill) string {
	return fmt.Sprintf("<!-- agentfield-skill:%s v%s -->", skill.Name, skill.Version)
}

// markerStartPattern returns a substring (without version) used to find an
// existing block of THIS skill regardless of installed version, so re-installs
// can replace older versions cleanly.
func markerStartPattern(skill Skill) string {
	return fmt.Sprintf("<!-- agentfield-skill:%s ", skill.Name)
}

// markerEnd returns the closing marker line for a skill's marker block.
func markerEnd(skill Skill) string {
	return fmt.Sprintf("<!-- /agentfield-skill:%s -->", skill.Name)
}

// renderPointerBlock returns the marker-bracketed text that file-append
// targets write into the agent's global rules file. The block points the
// agent at the canonical SKILL.md path so updates to the canonical store
// flow through automatically — no need to re-edit every agent rules file.
func renderPointerBlock(skill Skill, canonicalCurrentDir string) string {
	skillPath := filepath.Join(canonicalCurrentDir, skill.EntryFile)
	return fmt.Sprintf(`%s
## %s

When the user asks you to architect or build a multi-agent system on
AgentField (composite-intelligence backends, multi-reasoner pipelines,
financial reviewer / clinical triage / research agent / etc.), you MUST
read this skill first:

  %s

The skill is self-contained and every reference file is one level deep
from SKILL.md. It teaches the philosophy, the SDK primitives, the
canonical scaffold layout, the verification workflow, and the curl
smoke test.

Skill version: %s
%s`,
		markerStart(skill),
		skill.Description,
		skillPath,
		skill.Version,
		markerEnd(skill),
	)
}

// platformInfo is a small helper for diagnostics — used by Cursor's manual
// install path to give the user an OS-appropriate hint about where the
// "Settings → Rules for AI" UI lives.
func platformInfo() string {
	return runtime.GOOS
}
