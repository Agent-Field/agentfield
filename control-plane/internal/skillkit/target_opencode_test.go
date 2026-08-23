package skillkit

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenCodeTargetInstallsSkillSymlink(t *testing.T) {
	home := withTempHome(t)
	t.Setenv("USERPROFILE", home)
	canonical := filepath.Join(home, ".agentfield", "skills", "agentfield", "1.2.3")
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

func TestOpenCodeTargetReplacesExistingEntryAndReportsManualEntry(t *testing.T) {
	home := withTempHome(t)
	target := opencodeTarget{}
	root, err := target.TargetPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, Catalog[0].Name)
	if err := os.WriteFile(link, []byte("manual"), 0o644); err != nil {
		t.Fatal(err)
	}
	if installed, version, err := target.Status(); err != nil || !installed || version != "manual" {
		t.Fatalf("manual Status = %v %q %v", installed, version, err)
	}
	canonical := filepath.Join(home, ".agentfield", "skills", Catalog[0].Name, Catalog[0].Version)
	if err := os.MkdirAll(canonical, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := target.Install(Catalog[0], canonical); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Readlink(link); err != nil {
		t.Fatalf("replacement is not a symlink: %v", err)
	}
}

func TestOpenCodeTargetUninstallReportsMissingHome(t *testing.T) {
	t.Setenv("HOME", "")
	t.Setenv("USERPROFILE", "")
	t.Setenv("AGENTFIELD_HOME", "")
	if err := (opencodeTarget{}).Uninstall(); err == nil {
		t.Fatal("Uninstall should report an unavailable home directory")
	}
}
