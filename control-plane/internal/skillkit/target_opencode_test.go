package skillkit

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenCodeTargetInstallsSkillSymlink(t *testing.T) {
	home := withTempHome(t)
	t.Setenv("USERPROFILE", home)
	canonical := filepath.Join(home, ".agentfield", "skills", "agentfield", "current")
	if err := os.MkdirAll(canonical, 0o755); err != nil {
		t.Fatal(err)
	}
	target := opencodeTarget{}
	installed, err := target.Install(Skill{Name: "agentfield", Version: "1.2.3"}, canonical)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(home, ".config", "opencode", "skills", "agentfield")
	if installed.Method != "symlink" || installed.Path != want {
		t.Fatalf("installed target = %#v", installed)
	}
	got, err := os.Readlink(want)
	if err != nil || got != canonical {
		t.Fatalf("OpenCode link = %q, %v; want %q", got, err, canonical)
	}
	if err := target.Uninstall(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(want); !os.IsNotExist(err) {
		t.Fatalf("skill link still exists after uninstall: %v", err)
	}
}
