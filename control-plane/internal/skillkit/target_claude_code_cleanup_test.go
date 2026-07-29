package skillkit

import (
	"os"
	"path/filepath"
	"testing"
)

// Contract: updating a skill to a version that no longer ships a slash
// command must remove the stale ~/.claude/commands link the old version
// installed (the retired /agentfield shim), while leaving live command
// links, user files, and other skills' commands untouched.
func TestClaudeCodeInstallRemovesRetiredCommandShim(t *testing.T) {
	home := withTempHome(t)
	if err := os.MkdirAll(filepath.Join(home, ".claude"), 0o755); err != nil {
		t.Fatalf("mkdir .claude: %v", err)
	}
	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	skill := Skill{Name: "agentfield", Version: "9.9.9"}

	// Old version ships a command; install links it.
	oldVer := filepath.Join(root, skill.Name, "9.9.8")
	if err := os.MkdirAll(filepath.Join(oldVer, "commands"), 0o755); err != nil {
		t.Fatalf("mkdir old commands: %v", err)
	}
	if err := os.WriteFile(filepath.Join(oldVer, "commands", "agentfield.md"), []byte("shim"), 0o644); err != nil {
		t.Fatalf("write shim: %v", err)
	}
	current := filepath.Join(root, skill.Name, "current")
	if err := os.Symlink(oldVer, current); err != nil {
		t.Fatalf("symlink current: %v", err)
	}

	target := claudeCodeTarget{}
	if _, err := target.Install(skill, current); err != nil {
		t.Fatalf("install old version: %v", err)
	}
	cmdLink := filepath.Join(home, ".claude", "commands", "agentfield.md")
	if _, err := os.Lstat(cmdLink); err != nil {
		t.Fatalf("command link missing after old install: %v", err)
	}

	// Bystanders that must survive: a user's own file and another skill's
	// live command link.
	userFile := filepath.Join(home, ".claude", "commands", "mine.md")
	if err := os.WriteFile(userFile, []byte("user"), 0o644); err != nil {
		t.Fatalf("write user file: %v", err)
	}
	otherCmd := filepath.Join(root, "other-skill", "current", "commands")
	if err := os.MkdirAll(otherCmd, 0o755); err != nil {
		t.Fatalf("mkdir other skill: %v", err)
	}
	if err := os.WriteFile(filepath.Join(otherCmd, "other.md"), []byte("other"), 0o644); err != nil {
		t.Fatalf("write other command: %v", err)
	}
	otherLink := filepath.Join(home, ".claude", "commands", "other.md")
	if err := os.Symlink(filepath.Join(otherCmd, "other.md"), otherLink); err != nil {
		t.Fatalf("symlink other command: %v", err)
	}

	// New version drops the commands dir; retarget current and reinstall.
	newVer := filepath.Join(root, skill.Name, "9.9.9")
	if err := os.MkdirAll(newVer, 0o755); err != nil {
		t.Fatalf("mkdir new version: %v", err)
	}
	if err := os.Remove(current); err != nil {
		t.Fatalf("remove current symlink: %v", err)
	}
	if err := os.Symlink(newVer, current); err != nil {
		t.Fatalf("retarget current: %v", err)
	}

	if _, err := target.Install(skill, current); err != nil {
		t.Fatalf("install new version: %v", err)
	}

	if _, err := os.Lstat(cmdLink); !os.IsNotExist(err) {
		t.Fatalf("stale command shim should be removed, lstat err=%v", err)
	}
	if _, err := os.Stat(userFile); err != nil {
		t.Fatalf("user command file must survive cleanup: %v", err)
	}
	if _, err := os.Lstat(otherLink); err != nil {
		t.Fatalf("other skill's live command link must survive cleanup: %v", err)
	}
}
