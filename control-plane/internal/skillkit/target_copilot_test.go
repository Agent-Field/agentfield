package skillkit

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCopilotTarget(t *testing.T) {
	home := withTempHome(t)

	cp := copilotTarget{}
	if cp.Name() != "copilot" || cp.DisplayName() != "GitHub Copilot CLI" || cp.Method() != "symlink" {
		t.Fatalf("unexpected copilot metadata: %q %q %q", cp.Name(), cp.DisplayName(), cp.Method())
	}

	// Not detected before ~/.copilot exists and the copilot binary is not on
	// PATH (withTempHome sets PATH to an empty bin dir).
	if cp.Detected() {
		t.Fatal("copilot target should not be detected in empty temp home")
	}

	if err := os.MkdirAll(filepath.Join(home, ".copilot"), 0o755); err != nil {
		t.Fatalf("mkdir copilot dir: %v", err)
	}
	if !cp.Detected() {
		t.Fatal("copilot target should be detected once ~/.copilot exists")
	}

	skill := Catalog[0]
	canonicalCurrentDir := filepath.Join(home, "canonical", "current")
	if err := os.MkdirAll(canonicalCurrentDir, 0o755); err != nil {
		t.Fatalf("mkdir canonical current: %v", err)
	}

	inst, err := cp.Install(skill, canonicalCurrentDir)
	if err != nil {
		t.Fatalf("copilot install: %v", err)
	}
	if inst.Method != "symlink" || inst.TargetName != "copilot" {
		t.Fatalf("unexpected install record: %+v", inst)
	}

	link := filepath.Join(home, ".copilot", "skills", skill.Name)
	info, err := os.Lstat(link)
	if err != nil {
		t.Fatalf("lstat symlink: %v", err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("expected symlink at %s", link)
	}
	dest, err := os.Readlink(link)
	if err != nil || dest != canonicalCurrentDir {
		t.Fatalf("readlink = %q err=%v", dest, err)
	}

	installed, version, err := cp.Status()
	if err != nil || !installed || version != "current" {
		t.Fatalf("copilot status = %v %q %v", installed, version, err)
	}

	// Re-install should be idempotent (replaces existing symlink cleanly).
	if _, err := cp.Install(skill, canonicalCurrentDir); err != nil {
		t.Fatalf("copilot reinstall: %v", err)
	}

	if err := cp.Uninstall(); err != nil {
		t.Fatalf("copilot uninstall: %v", err)
	}
	if installed, _, _ := cp.Status(); installed {
		t.Fatal("copilot should not be installed after uninstall")
	}
}

func TestCopilotTargetReplacesRegularDir(t *testing.T) {
	home := withTempHome(t)
	if err := os.MkdirAll(filepath.Join(home, ".copilot"), 0o755); err != nil {
		t.Fatalf("mkdir copilot dir: %v", err)
	}

	skill := Catalog[0]
	link := filepath.Join(home, ".copilot", "skills", skill.Name)
	if err := os.MkdirAll(link, 0o755); err != nil {
		t.Fatalf("mkdir pre-existing skill dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(link, "stray.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write stray: %v", err)
	}

	canonicalCurrentDir := filepath.Join(home, "canonical", "current")
	if err := os.MkdirAll(canonicalCurrentDir, 0o755); err != nil {
		t.Fatalf("mkdir canonical: %v", err)
	}

	cp := copilotTarget{}
	if _, err := cp.Install(skill, canonicalCurrentDir); err != nil {
		t.Fatalf("copilot install over regular dir: %v", err)
	}

	info, err := os.Lstat(link)
	if err != nil {
		t.Fatalf("lstat after install: %v", err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Fatal("expected symlink after install-over-dir")
	}
}
