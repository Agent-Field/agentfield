
package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInstallerCoverage(t *testing.T) {

	t.Run("copyPackage-relpath-fails", func(t *testing.T) {
		pi := &PackageInstaller{}
		// It's very hard to make filepath.Rel fail, so this is mostly for completeness.
		// We can trigger it by having a bad source path.
		err := pi.copyPackage("/../", "/tmp/dest")
		if err == nil {
			t.Fatalf("expected error from copyPackage with bad source")
		}
	})

	t.Run("copyFile-open-fails", func(t *testing.T){
		pi := &PackageInstaller{}
		err := pi.copyFile("/nonexistent", "/tmp/dest")
		if err == nil {
			t.Fatalf("expected file open to fail")
		}
	})

	t.Run("copyFile-create-fails", func(t *testing.T){
		pi := &PackageInstaller{}
		src := filepath.Join(t.TempDir(), "src")
		if err := os.WriteFile(src, []byte{}, 0644); err != nil {
			t.Fatal(err)
		}
		// make dest unwritable
		destDir := t.TempDir()
		if err := os.Chmod(destDir, 0555); err != nil {
			t.Fatal(err)
		}
		err := pi.copyFile(src, filepath.Join(destDir, "dest"))
		if err == nil {
			t.Fatalf("expected file create to fail")
		}
		_ = os.Chmod(destDir, 0755)
	})

	t.Run("installDependencies-python-fails", func(t *testing.T){
		pi := &PackageInstaller{}
		pkgPath := t.TempDir()
		metadata := &PackageMetadata{
			Dependencies: DependencyConfig{
				Python: []string{"non-existent-package-for-sure"},
			},
		}

		// Make python not found
		t.Setenv("PATH", "")

		err := pi.installDependencies(pkgPath, metadata)
		if err == nil || !strings.Contains(err.Error(), "failed to create virtual environment") {
			t.Fatalf("expected python to fail, got %v", err)
		}
	})
	
	t.Run("installDependencies-pip-fails", func(t *testing.T){
		pi := &PackageInstaller{}
		pkgPath := t.TempDir()
		metadata := &PackageMetadata{
			Dependencies: DependencyConfig{
				Python: []string{"non-existent-package-for-sure"},
			},
		}

		// create a fake python that will succeed, but no pip
		fakebin := t.TempDir()
		pythonPath := filepath.Join(fakebin, "python3")
		if err := os.WriteFile(pythonPath, []byte("#!/bin/sh\nexit 0"), 0755); err != nil {
			t.Fatal(err)
		}
		t.Setenv("PATH", fakebin)

		err := pi.installDependencies(pkgPath, metadata)
		if err == nil || !strings.Contains(err.Error(), "failed to install dependency") {
			t.Fatalf("expected pip to fail, got %v", err)
		}
	})
	
	t.Run("InstallPackage-install-deps-fails", func(t *testing.T){
		pi := &PackageInstaller{AgentFieldHome: t.TempDir()}
		sourcePath := t.TempDir()
		writeTestPackage(t, sourcePath, `
name: deps-fail
version: 1.0
dependencies:
  python:
    - hopefully-non-existent-package
`)
		// Make python not found
		t.Setenv("PATH", "")

		err := pi.InstallPackage(sourcePath, true)
		if err == nil || !strings.Contains(err.Error(), "failed to install dependencies") {
			t.Fatalf("expected install deps to fail, got %v", err)
		}
	})
}
