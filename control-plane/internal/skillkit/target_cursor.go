package skillkit

import (
	"errors"
	"fmt"
	"path/filepath"
	"time"
)

// cursorTarget is a "manual" target — Cursor's global rules live in the
// Settings UI rather than a file we can write. Install() prints instructions
// for the user to copy/paste the SKILL.md content into Cursor → Settings →
// Rules for AI, and records the install in state so `af skill list` shows it
// as "manual / pending user action".
type cursorTarget struct{}

func init() { RegisterTarget(cursorTarget{}) }

func (cursorTarget) Name() string        { return "cursor" }
func (cursorTarget) DisplayName() string { return "Cursor" }
func (cursorTarget) Method() string      { return "manual" }

func (cursorTarget) Detected() bool {
	return commandAvailable("cursor") ||
		dirExists(filepath.Join(homeDir(), ".cursor")) ||
		dirExists(filepath.Join(homeDir(), "Library", "Application Support", "Cursor"))
}

func (cursorTarget) TargetPath() (string, error) {
	return "", errors.New("Cursor stores global rules in the Settings UI; no file path")
}

func (t cursorTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
	skillPath := filepath.Join(canonicalCurrentDir, skill.EntryFile)
	fmt.Printf(`
  ⚠ Cursor manual install required

  Cursor's global rules live in the Settings UI, not a file the af binary
  can write to. To enable the skill in Cursor:

    1. Open Cursor
    2. Cmd+, → Settings → General → Rules for AI
    3. Add a rule like:

       When the user asks you to architect or build a multi-agent system on
       AgentField, read this skill first:
         %s

       The skill is self-contained — every reference is one level deep
       from SKILL.md.

  (You can also add a per-project rule at .cursor/rules/agentfield.mdc.)
`, skillPath)

	return InstalledTarget{
		TargetName:  t.Name(),
		Method:      t.Method(),
		Path:        "Cursor Settings → Rules for AI (manual)",
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}

func (cursorTarget) Uninstall() error {
	fmt.Println("  ⚠ Cursor manual uninstall: remove the AgentField rule from Settings → Rules for AI")
	return nil
}

func (cursorTarget) Status() (bool, string, error) {
	// Cursor's UI state isn't readable from disk. Always report unknown.
	return false, "", nil
}
