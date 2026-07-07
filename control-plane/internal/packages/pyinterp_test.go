package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParsePyVersion(t *testing.T) {
	cases := []struct {
		in    string
		want  pyVersion
		valid bool
	}{
		{"3.12", pyVersion{3, 12, 0}, true},
		{"3.12.5", pyVersion{3, 12, 5}, true},
		{"3", pyVersion{3, 0, 0}, true},
		{" 3.11 ", pyVersion{3, 11, 0}, true},
		{"3.12rc1", pyVersion{3, 12, 0}, true},
		{"", pyVersion{}, false},
		{"abc", pyVersion{}, false},
	}
	for _, c := range cases {
		got, ok := parsePyVersion(c.in)
		if ok != c.valid || (ok && got != c.want) {
			t.Errorf("parsePyVersion(%q) = %v,%v; want %v,%v", c.in, got, ok, c.want, c.valid)
		}
	}
}

func TestConstraintSatisfied(t *testing.T) {
	cases := []struct {
		ver      pyVersion
		requires string
		want     bool
	}{
		// The SWE-AF repro: 3.10 must NOT satisfy >=3.12, 3.12 must.
		{pyVersion{3, 10, 13}, ">=3.12", false},
		{pyVersion{3, 12, 0}, ">=3.12", true},
		{pyVersion{3, 12, 5}, ">=3.12", true},
		{pyVersion{3, 13, 3}, ">=3.12", true},
		// Ranges.
		{pyVersion{3, 13, 3}, ">=3.12,<3.14", true},
		{pyVersion{3, 14, 0}, ">=3.12,<3.14", false},
		{pyVersion{3, 11, 0}, ">=3.12,<3.14", false},
		// Compatible release (~=).
		{pyVersion{3, 12, 0}, "~=3.11", true},
		{pyVersion{4, 0, 0}, "~=3.11", false},
		{pyVersion{3, 10, 0}, "~=3.11", false},
		{pyVersion{3, 11, 9}, "~=3.11.5", true},
		{pyVersion{3, 12, 0}, "~=3.11.5", false},
		// Wildcards.
		{pyVersion{3, 12, 7}, "==3.12.*", true},
		{pyVersion{3, 13, 0}, "==3.12.*", false},
		{pyVersion{3, 12, 0}, ">=3.10,!=3.11.*", true},
		{pyVersion{3, 11, 4}, ">=3.10,!=3.11.*", false},
		// Exact.
		{pyVersion{3, 12, 5}, "==3.12.5", true},
		{pyVersion{3, 12, 6}, "==3.12.5", false},
		// Permissive: empty / unparseable never blocks.
		{pyVersion{3, 10, 0}, "", true},
		{pyVersion{3, 10, 0}, "not-a-constraint", true},
	}
	for _, c := range cases {
		if got := constraintSatisfied(c.ver, c.requires); got != c.want {
			t.Errorf("constraintSatisfied(%v, %q) = %v; want %v", c.ver, c.requires, got, c.want)
		}
	}
}

func TestUvRequest(t *testing.T) {
	cases := []struct {
		requires string
		want     string
	}{
		{">=3.12", "3.12"},
		{">=3.12,<3.13", "3.12"},
		{"==3.11.*", "3.11"},
		{"~=3.10", "3.10"},
		{">3.12.5", "3.12"},
		{"<3.13", ""}, // no lower bound
		{"", ""},
	}
	for _, c := range cases {
		if got := uvRequest(c.requires); got != c.want {
			t.Errorf("uvRequest(%q) = %q; want %q", c.requires, got, c.want)
		}
	}
}

func TestReadRequiresPython(t *testing.T) {
	t.Run("declared", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "[project]\nname = \"x\"\nrequires-python = \">=3.12\"\n")
		if got := readRequiresPython(dir); got != ">=3.12" {
			t.Fatalf("got %q; want >=3.12", got)
		}
	})
	t.Run("no field", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "[project]\nname = \"x\"\n")
		if got := readRequiresPython(dir); got != "" {
			t.Fatalf("got %q; want empty", got)
		}
	})
	t.Run("no file", func(t *testing.T) {
		if got := readRequiresPython(t.TempDir()); got != "" {
			t.Fatalf("got %q; want empty", got)
		}
	})
	t.Run("malformed", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "this is : not [ valid toml =\n")
		if got := readRequiresPython(dir); got != "" {
			t.Fatalf("got %q; want empty (parse failure must not block)", got)
		}
	})
}

// TestResolveVenvInterpreter exercises the resolution contract with a stubbed
// PATH so no real interpreter/uv/pyenv is involved.
func TestResolveVenvInterpreter(t *testing.T) {
	t.Run("no constraint keeps legacy fallback", func(t *testing.T) {
		dir := t.TempDir() // no pyproject.toml
		interp, err := resolveVenvInterpreter(dir)
		if err != nil || interp != "" {
			t.Fatalf("got (%q,%v); want empty interp, nil err", interp, err)
		}
	})

	t.Run("ambient satisfies", func(t *testing.T) {
		bin := t.TempDir()
		stubPython(t, bin, "python3", "3.12.5")
		t.Setenv("PATH", bin)
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "[project]\nrequires-python = \">=3.12\"\n")

		interp, err := resolveVenvInterpreter(dir)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if interp != "python3" {
			t.Fatalf("got %q; want python3", interp)
		}
	})

	t.Run("incompatible and no provisioner -> actionable error", func(t *testing.T) {
		bin := t.TempDir()
		stubPython(t, bin, "python3", "3.10.13") // too old, no uv/pyenv on PATH
		t.Setenv("PATH", bin)
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "[project]\nrequires-python = \">=3.12\"\n")

		interp, err := resolveVenvInterpreter(dir)
		if interp != "" {
			t.Fatalf("expected no interpreter, got %q", interp)
		}
		if err == nil {
			t.Fatal("expected an error")
		}
		msg := err.Error()
		if !strings.Contains(msg, ">=3.12") || !strings.Contains(msg, "3.10.13") {
			t.Fatalf("error should name required (>=3.12) and found (3.10.13) versions: %q", msg)
		}
	})

	t.Run("provisions via uv when ambient too old", func(t *testing.T) {
		bin := t.TempDir()
		stubPython(t, bin, "python3", "3.10.13")
		// A "real" target interpreter uv will hand back.
		target := stubPython(t, bin, "managed-python", "3.12.10")
		stubUv(t, bin, target)
		t.Setenv("PATH", bin)
		dir := t.TempDir()
		writeFile(t, dir, "pyproject.toml", "[project]\nrequires-python = \">=3.12\"\n")

		interp, err := resolveVenvInterpreter(dir)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if interp != target {
			t.Fatalf("got %q; want uv-provided %q", interp, target)
		}
	})
}

func TestProvisionViaPyenv(t *testing.T) {
	bin := t.TempDir()
	root := t.TempDir()
	// Two installed versions: 3.10.13 (too old) and 3.12.5 (satisfies >=3.12).
	for _, v := range []string{"3.10.13", "3.12.5"} {
		vb := filepath.Join(root, "versions", v, "bin")
		if err := os.MkdirAll(vb, 0o755); err != nil {
			t.Fatal(err)
		}
		writeExecutable(t, filepath.Join(vb, "python"), "#!/bin/sh\nexit 0\n")
	}
	pyenv := "#!/bin/sh\n" +
		"if [ \"$1\" = \"versions\" ]; then printf '3.10.13\\n3.12.5\\n'; exit 0; fi\n" +
		"if [ \"$1\" = \"root\" ]; then echo \"" + root + "\"; exit 0; fi\n" +
		"exit 0\n"
	writeExecutable(t, filepath.Join(bin, "pyenv"), pyenv)
	t.Setenv("PATH", bin)

	got := provisionViaPyenv(">=3.12")
	want := filepath.Join(root, "versions", "3.12.5", "bin", "python")
	if got != want {
		t.Fatalf("provisionViaPyenv(>=3.12) = %q; want the highest satisfying %q", got, want)
	}
	if provisionViaPyenv(">=3.13") != "" {
		t.Fatal("provisionViaPyenv(>=3.13) should find nothing installed")
	}
}

// TestInstallPythonDependencies_ProvisionsForRequiresPython drives the public
// API through the requires-python interpreter-selection branch: a pyproject
// declaring >=3.12 with a satisfying ambient python must build the venv.
func TestInstallPythonDependencies_ProvisionsForRequiresPython(t *testing.T) {
	bin := t.TempDir()
	stubVenvPython(t, bin, "python3", "3.12.5")
	// Keep coreutils on PATH for the stub's mkdir/chmod; ambient already
	// satisfies, so uv/pyenv are never consulted.
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	pkg := t.TempDir()
	writeFile(t, pkg, "pyproject.toml", "[project]\nname = \"demo\"\nrequires-python = \">=3.12\"\n")

	if err := InstallPythonDependencies(pkg, nil, nil); err != nil {
		t.Fatalf("InstallPythonDependencies: %v", err)
	}
	if _, err := os.Stat(filepath.Join(pkg, "venv", "bin", "pip")); err != nil {
		t.Fatalf("expected a venv built via the requires-python-satisfying interpreter: %v", err)
	}
}

// --- test helpers ---

// stubVenvPython writes an interpreter that reports `version` for -c and
// creates a venv (with a no-op pip) for `-m venv`. Returns its full path.
func stubVenvPython(t *testing.T, dir, name, version string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	script := "#!/bin/sh\n" +
		"if [ \"$1\" = \"-c\" ]; then echo \"" + version + "\"; exit 0; fi\n" +
		"if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n" +
		"  mkdir -p \"$3/bin\"\n" +
		"  printf '#!/bin/sh\\nexit 0\\n' > \"$3/bin/pip\"\n" +
		"  chmod +x \"$3/bin/pip\"\n" +
		"fi\n" +
		"exit 0\n"
	writeExecutable(t, p, script)
	return p
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// stubPython writes an executable named `name` in dir that answers
// `-c "import sys;..."` with the given version. Returns its full path.
func stubPython(t *testing.T, dir, name, version string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	script := "#!/bin/sh\n" +
		"if [ \"$1\" = \"-c\" ]; then echo \"" + version + "\"; exit 0; fi\n" +
		"exit 0\n"
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// stubUv writes a fake `uv` that returns findPath for `uv python find ...`.
func stubUv(t *testing.T, dir, findPath string) {
	t.Helper()
	p := filepath.Join(dir, "uv")
	script := "#!/bin/sh\n" +
		"if [ \"$2\" = \"find\" ]; then echo \"" + findPath + "\"; fi\n" +
		"exit 0\n"
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
}
