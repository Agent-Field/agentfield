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
	if installed, version, err := target.Status(); err != nil || !installed || version != "1.2.3" {
		t.Fatalf("OpenCode status = %v %q %v", installed, version, err)
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

func TestOpenCodeTargetInstallReportsRootCreationFailure(t *testing.T) {
	home := withTempHome(t)
	if err := os.WriteFile(filepath.Join(home, ".config"), []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := (opencodeTarget{}).Install(Catalog[0], filepath.Join(home, "canonical"))
	if err == nil {
		t.Fatal("Install should report a failure creating the OpenCode skills directory")
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

func TestOpenCodeTargetStatusHandlesMissingAndBrokenLinks(t *testing.T) {
	home := withTempHome(t)
	target := opencodeTarget{}

	installed, version, err := target.Status()
	if err != nil || installed || version != "" {
		t.Fatalf("missing Status = %v %q %v", installed, version, err)
	}

	link, err := target.skillLink(Catalog[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(home, "missing"), link); err != nil {
		t.Fatal(err)
	}
	if installed, version, err := target.Status(); err == nil || installed || version != "" {
		t.Fatalf("broken-link Status = %v %q %v", installed, version, err)
	}
}

func TestOpenCodeTargetStatusResolvesCurrentLink(t *testing.T) {
	home := withTempHome(t)
	target := opencodeTarget{}
	link, err := target.skillLink(Catalog[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	versionDir := filepath.Join(home, "canonical", "1.2.3")
	if err := os.MkdirAll(versionDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(versionDir, filepath.Join(filepath.Dir(link), "current")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("current", link); err != nil {
		t.Fatal(err)
	}
	if installed, version, err := target.Status(); err != nil || !installed || version != "1.2.3" {
		t.Fatalf("current-link Status = %v %q %v", installed, version, err)
	}
}

func TestOpenCodeTargetStatusPreservesVersionFromRemovedDirectLink(t *testing.T) {
	home := withTempHome(t)
	target := opencodeTarget{}
	link, err := target.skillLink(Catalog[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(home, "canonical", "1.2.3"), link); err != nil {
		t.Fatal(err)
	}

	if installed, version, err := target.Status(); err != nil || !installed || version != "1.2.3" {
		t.Fatalf("removed-direct-link Status = %v %q %v", installed, version, err)
	}
}

func TestOpenCodeTargetUninstallRemovesCatalogEntries(t *testing.T) {
	withTempHome(t)
	target := opencodeTarget{}
	root, err := target.TargetPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, skill := range Catalog {
		path, err := target.skillLink(skill)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := target.Uninstall(); err != nil {
		t.Fatal(err)
	}
	for _, skill := range Catalog {
		path, err := target.skillLink(skill)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("catalog entry %q remains after uninstall: %v", skill.Name, err)
		}
	}
}
