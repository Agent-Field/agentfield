package workspace

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, dir, rel, content string, mode os.FileMode) {
	t.Helper()
	full := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(full, []byte(content), mode); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func TestSealHonorsIgnoreRules(t *testing.T) {
	src := t.TempDir()
	cas := NewCAS(t.TempDir())

	writeFile(t, src, "main.go", "package main", 0o644)
	writeFile(t, src, "src/util.py", "print(1)", 0o644)
	writeFile(t, src, ".gitignore", "*.log\nsecret.txt\n", 0o644)
	writeFile(t, src, "app.log", "noise", 0o644)
	writeFile(t, src, "secret.txt", "shh", 0o644)
	writeFile(t, src, "node_modules/dep/index.js", "x", 0o644)
	writeFile(t, src, ".git/config", "gitdata", 0o644)

	m, err := Seal(src, cas)
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}

	got := map[string]bool{}
	for _, f := range m.Files {
		got[f.Path] = true
	}
	// .gitignore itself is a regular file and is sealed (git tracks it too).
	want := []string{"main.go", "src/util.py", ".gitignore"}
	for _, w := range want {
		if !got[w] {
			t.Errorf("expected %q in manifest, missing", w)
		}
	}
	for _, bad := range []string{"app.log", "secret.txt", "node_modules/dep/index.js", ".git/config"} {
		if got[bad] {
			t.Errorf("expected %q to be ignored, but it was sealed", bad)
		}
	}
}

func TestSealMaterializeRoundTrip(t *testing.T) {
	src := t.TempDir()
	cas := NewCAS(t.TempDir())

	writeFile(t, src, "main.go", "package main\n", 0o644)
	writeFile(t, src, "pkg/util.go", "package pkg\n", 0o600)
	writeFile(t, src, "data/nested/deep.txt", "deep content", 0o644)

	m, err := Seal(src, cas)
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}
	id1, err := ManifestID(m)
	if err != nil {
		t.Fatalf("ManifestID: %v", err)
	}

	dst := t.TempDir()
	if err := Materialize(m, dst, cas); err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	// Content must be identical after the round trip.
	for _, f := range m.Files {
		orig, _ := os.ReadFile(filepath.Join(src, filepath.FromSlash(f.Path)))
		got, err := os.ReadFile(filepath.Join(dst, filepath.FromSlash(f.Path)))
		if err != nil {
			t.Fatalf("read materialized %s: %v", f.Path, err)
		}
		if string(orig) != string(got) {
			t.Errorf("content mismatch for %s", f.Path)
		}
		info, err := os.Stat(filepath.Join(dst, filepath.FromSlash(f.Path)))
		if err != nil {
			t.Fatalf("stat %s: %v", f.Path, err)
		}
		if uint32(info.Mode().Perm()) != f.Mode {
			t.Errorf("mode mismatch for %s: got %o want %o", f.Path, info.Mode().Perm(), f.Mode)
		}
	}

	// Re-sealing the materialized tree yields the same content identity
	// (paths + sha256), so the manifest_id matches once mtimes are normalized.
	cas2 := NewCAS(t.TempDir())
	m2, err := Seal(dst, cas2)
	if err != nil {
		t.Fatalf("re-Seal: %v", err)
	}
	if len(m.Files) != len(m2.Files) {
		t.Fatalf("file count changed: %d -> %d", len(m.Files), len(m2.Files))
	}
	for i := range m.Files {
		if m.Files[i].Path != m2.Files[i].Path || m.Files[i].SHA256 != m2.Files[i].SHA256 {
			t.Errorf("round-trip identity mismatch at %d: %+v vs %+v", i, m.Files[i], m2.Files[i])
		}
	}

	// Normalizing mtimes reproduces the exact same manifest_id.
	normalize := func(man *Manifest) {
		for i := range man.Files {
			man.Files[i].MtimeNS = 0
		}
	}
	normalize(m)
	normalize(m2)
	nid1, _ := ManifestID(m)
	nid2, _ := ManifestID(m2)
	if nid1 != nid2 {
		t.Errorf("normalized manifest_id mismatch: %s vs %s", nid1, nid2)
	}
	_ = id1
}
