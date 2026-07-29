package packages

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakePythonOnPath installs a stub `python3` whose `-m venv <dir>` creates a
// venv layout with a no-op `pip`, so dependency installation can be exercised
// offline without invoking real Python/pip. It also answers the `-c` version
// probe, since interpreter selection only accepts candidates that actually
// run and report a version (the guard against Windows Store alias stubs).
func fakePythonOnPath(t *testing.T) {
	t.Helper()
	binDir := t.TempDir()
	py := "#!/bin/sh\n" +
		"if [ \"$1\" = \"-c\" ]; then echo \"3.12.0\"; exit 0; fi\n" +
		"if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n" +
		"  mkdir -p \"$3/bin\"\n" +
		"  printf '#!/bin/sh\\nexit 0\\n' > \"$3/bin/pip\"\n" +
		"  chmod +x \"$3/bin/pip\"\n" +
		"fi\n" +
		"exit 0\n"
	if err := os.WriteFile(filepath.Join(binDir, "python3"), []byte(py), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func fakeNPMOnPath(t *testing.T, exitCode int) string {
	t.Helper()
	binDir := t.TempDir()
	record := filepath.Join(t.TempDir(), "npm-record")
	t.Setenv("FAKE_NPM_RECORD", record)
	if runtime.GOOS == "windows" {
		script := "@echo off\r\necho %CD%> \"%FAKE_NPM_RECORD%\"\r\necho %*>> \"%FAKE_NPM_RECORD%\"\r\n"
		if exitCode != 0 {
			script += "echo npm failed 1>&2\r\nexit /b " + string(rune('0'+exitCode)) + "\r\n"
		}
		if err := os.WriteFile(filepath.Join(binDir, "npm.cmd"), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	} else {
		script := "#!/bin/sh\nprintf '%s\\n' \"$PWD\" \"$@\" > \"$FAKE_NPM_RECORD\"\n"
		if exitCode != 0 {
			script += "echo npm failed >&2\nexit " + string(rune('0'+exitCode)) + "\n"
		}
		if err := os.WriteFile(filepath.Join(binDir, "npm"), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return record
}

func TestInstallDependencies_TypeScript(t *testing.T) {
	pkg := t.TempDir()
	if err := os.WriteFile(filepath.Join(pkg, "package.json"), []byte(`{"name":"demo"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	record := fakeNPMOnPath(t, 0)
	if err := InstallDependencies(pkg, &PackageMetadata{Language: " TypeScript "}); err != nil {
		t.Fatalf("InstallDependencies: %v", err)
	}
	got, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), pkg) || !strings.Contains(string(got), "install") {
		t.Fatalf("npm invocation record = %q, want package root and install", got)
	}
	if _, err := os.Stat(filepath.Join(pkg, "venv")); !os.IsNotExist(err) {
		t.Fatalf("TypeScript install created Python venv")
	}
}

// The public installer must run npm only after it has copied the source into
// AgentField's managed package root. This keeps a newly installed scaffold
// runnable without modifying the author's source tree.
func TestPackageInstallerInstallPackage_TypeScriptInstallsManagedCopy(t *testing.T) {
	source := t.TempDir()
	manifest := `config_version: v1
name: typescript-managed-copy
version: 1.0.0
language: typescript
entrypoint:
  start: npm run start
`
	if err := os.WriteFile(filepath.Join(source, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "package.json"), []byte(`{"name":"demo"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	record := fakeNPMOnPath(t, 0)
	home := t.TempDir()
	if err := (&PackageInstaller{AgentFieldHome: home}).InstallPackage(source, false); err != nil {
		t.Fatalf("InstallPackage: %v", err)
	}
	managed := filepath.Join(home, "packages", "typescript-managed-copy")
	got, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), managed) || !strings.Contains(string(got), "install") {
		t.Fatalf("npm invocation record = %q, want npm install in managed package root %q", got, managed)
	}
	if _, err := os.Stat(filepath.Join(managed, "venv")); !os.IsNotExist(err) {
		t.Fatalf("managed TypeScript install created Python venv")
	}
}

func TestInstallTypeScriptDependenciesFailures(t *testing.T) {
	t.Run("missing package json", func(t *testing.T) {
		pkg := t.TempDir()
		err := InstallDependencies(pkg, &PackageMetadata{Language: "typescript"})
		if err == nil || !strings.Contains(err.Error(), "package.json not found") {
			t.Fatalf("error = %v, want actionable missing package.json", err)
		}
		if _, statErr := os.Stat(filepath.Join(pkg, "venv")); !os.IsNotExist(statErr) {
			t.Fatalf("missing package.json fell back to Python venv")
		}
	})
	t.Run("missing npm", func(t *testing.T) {
		pkg := t.TempDir()
		if err := os.WriteFile(filepath.Join(pkg, "package.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Setenv("PATH", t.TempDir())
		err := InstallDependencies(pkg, &PackageMetadata{Language: "typescript"})
		if err == nil || !strings.Contains(err.Error(), "npm executable not found") {
			t.Fatalf("error = %v, want actionable missing npm", err)
		}
		if _, statErr := os.Stat(filepath.Join(pkg, "venv")); !os.IsNotExist(statErr) {
			t.Fatalf("missing npm fell back to Python venv")
		}
	})
	t.Run("npm failure includes output", func(t *testing.T) {
		pkg := t.TempDir()
		if err := os.WriteFile(filepath.Join(pkg, "package.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
		fakeNPMOnPath(t, 2)
		err := InstallDependencies(pkg, &PackageMetadata{Language: "typescript"})
		if err == nil || !strings.Contains(err.Error(), "npm install") || !strings.Contains(err.Error(), "npm failed") {
			t.Fatalf("error = %v, want npm operation and output", err)
		}
		if _, statErr := os.Stat(filepath.Join(pkg, "venv")); !os.IsNotExist(statErr) {
			t.Fatalf("TypeScript npm failure created Python venv")
		}
	})
}

// Contract: a pyproject.toml package gets a venv and is installed via
// `pip install .`, even without a requirements.txt.
func TestInstallPythonDependencies_Pyproject(t *testing.T) {
	fakePythonOnPath(t)
	pkg := t.TempDir()
	if err := os.WriteFile(filepath.Join(pkg, "pyproject.toml"),
		[]byte("[project]\nname = \"demo\"\nversion = \"0.1.0\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := InstallPythonDependencies(pkg, nil, nil); err != nil {
		t.Fatalf("InstallPythonDependencies: %v", err)
	}
	if _, err := os.Stat(filepath.Join(pkg, "venv", "bin", "pip")); err != nil {
		t.Fatalf("expected a venv to be created for a pyproject project: %v", err)
	}
}

// Contract: requirements.txt + manifest-declared deps also trigger a venv.
func TestInstallPythonDependencies_RequirementsAndManifestDeps(t *testing.T) {
	fakePythonOnPath(t)
	pkg := t.TempDir()
	if err := os.WriteFile(filepath.Join(pkg, "requirements.txt"), []byte("httpx\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := InstallPythonDependencies(pkg, []string{"pydantic>=2"}, []string{"libfoo"}); err != nil {
		t.Fatalf("InstallPythonDependencies: %v", err)
	}
	if _, err := os.Stat(filepath.Join(pkg, "venv")); err != nil {
		t.Fatalf("expected venv: %v", err)
	}
}

// Contract: a package with no Python sources needs no venv and is a no-op.
func TestInstallPythonDependencies_NothingToDo(t *testing.T) {
	pkg := t.TempDir()
	if err := InstallPythonDependencies(pkg, nil, nil); err != nil {
		t.Fatalf("expected no-op, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(pkg, "venv")); !os.IsNotExist(err) {
		t.Fatalf("no venv should be created when there are no deps")
	}
}

// Contract: the git installer's findPackageRoot accepts a manifest that declares
// an entrypoint.start and has no main.py (the shape real nodes use).
func TestGitInstaller_FindPackageRoot_AcceptsEntrypoint(t *testing.T) {
	root := t.TempDir()
	pkg := filepath.Join(root, "repo")
	if err := os.MkdirAll(pkg, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: node\nversion: 1.0.0\nentrypoint:\n  start: python -m node.app\n"
	if err := os.WriteFile(filepath.Join(pkg, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	gi := &GitInstaller{AgentFieldHome: t.TempDir()}
	got, err := gi.findPackageRoot(root)
	if err != nil {
		t.Fatalf("findPackageRoot should accept an entrypoint-only manifest: %v", err)
	}
	if !strings.HasSuffix(got, "repo") {
		t.Fatalf("unexpected package root: %s", got)
	}
}
