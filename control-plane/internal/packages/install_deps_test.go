package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakePythonOnPath installs a stub `python3` whose `-m venv <dir>` creates a
// venv layout with a no-op `pip`, so dependency installation can be exercised
// offline without invoking real Python/pip.
func fakePythonOnPath(t *testing.T) {
	t.Helper()
	binDir := t.TempDir()
	py := "#!/bin/sh\n" +
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
