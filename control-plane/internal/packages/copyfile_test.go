package packages

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCopyFilePreservesExecutableBit guards the property a node relies on when
// it ships an executable: a helper binary or hook script that is 0755 in the
// repo must still be runnable after an install. os.Create would have produced
// 0666&^umask, leaving it non-executable and failing at spawn time with
// "permission denied".
func TestCopyFilePreservesExecutableBit(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "helper")
	if err := os.WriteFile(src, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	dst := filepath.Join(dir, "copied-helper")
	if err := CopyFile(src, dst); err != nil {
		t.Fatalf("CopyFile: %v", err)
	}

	info, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o755 {
		t.Errorf("copied mode = %o, want 755 — an executable a node ships must stay executable", got)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Error("copied file is not executable")
	}
}

// TestCopyFilePreservesNonExecutableMode: the mode is copied, not forced — a
// plain data file must not gain an execute bit.
func TestCopyFilePreservesNonExecutableMode(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(src, []byte("name: demo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	dst := filepath.Join(dir, "copied.yaml")
	if err := CopyFile(src, dst); err != nil {
		t.Fatalf("CopyFile: %v", err)
	}

	info, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o644 {
		t.Errorf("copied mode = %o, want 644", got)
	}
}

// TestCopyFileOverwritesModeOfExistingDest: reinstalling over an existing file
// must correct its mode, not inherit the stale one (O_CREATE alone would leave
// an existing destination's permissions untouched).
func TestCopyFileOverwritesModeOfExistingDest(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "bin")
	if err := os.WriteFile(src, []byte("payload"), 0o755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "stale")
	if err := os.WriteFile(dst, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := CopyFile(src, dst); err != nil {
		t.Fatalf("CopyFile: %v", err)
	}

	info, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o755 {
		t.Errorf("mode after overwrite = %o, want 755", got)
	}
	content, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "payload" {
		t.Errorf("content = %q, want the source content", content)
	}
}

// TestCopyPackagePreservesExecutableBit exercises the property through the
// real install entry point rather than the helper alone.
func TestCopyPackagePreservesExecutableBit(t *testing.T) {
	src := t.TempDir()
	if err := os.MkdirAll(filepath.Join(src, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "bin", "engine"), []byte("ELF"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "agentfield-package.yaml"), []byte("name: demo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	dst := filepath.Join(t.TempDir(), "installed")
	pi := &PackageInstaller{}
	if err := pi.copyPackage(src, dst); err != nil {
		t.Fatalf("copyPackage: %v", err)
	}

	info, err := os.Stat(filepath.Join(dst, "bin", "engine"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Errorf("installed binary mode = %o — a shipped executable must stay executable", info.Mode().Perm())
	}
}
