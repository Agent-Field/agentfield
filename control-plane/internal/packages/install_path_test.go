package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Validation contract for the `--path` subdirectory selector (one repo, multiple
// installable nodes):
//   1. --path <subdir> selects the node whose manifest lives at
//      <root>/<subdir>/agentfield-package.yaml; that subtree is the package root.
//   2. No selector => historical root-first behavior (unchanged).
//   3. A selector pointing where there is no manifest fails, naming the full
//      expected manifest path.
//   4. An absolute or escaping (`..`) selector is rejected syntactically, before
//      any filesystem/clone work.
//   5. The selector is orthogonal to the @ref URL pin — both are honored.
//   6. A Go-language subdir package builds relative to the subdir-as-root.

// writeRootPythonPackage lays down a root Python node (manifest + main.py) and a
// Go node under go/ so one clone directory carries two installable packages.
func writeMultiNodeClone(t *testing.T, cloneDir string) (goSub string) {
	t.Helper()
	writeTestPackage(t, cloneDir, "name: root-node\nversion: 1.0.0\nentrypoint:\n  start: python -m root.app\n")
	goSub = filepath.Join(cloneDir, "go")
	writeGoManifest(t, goSub,
		"name: go-node\nversion: 2.0.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
		"1.21", "")
	return goSub
}

// Contract 4 (syntax): absolute and escaping selectors are rejected without any
// filesystem access; empty and in-tree selectors pass the syntax gate.
func TestValidateSubdirSelector(t *testing.T) {
	t.Parallel()
	valid := []string{"", "go", "go/nested", "./go", "a/../b"}
	for _, s := range valid {
		if err := ValidateSubdirSelector(s); err != nil {
			t.Errorf("ValidateSubdirSelector(%q) unexpected error: %v", s, err)
		}
	}
	invalid := []string{"/abs", "/etc/passwd", "..", "../escape", "../../x", "a/../../b"}
	for _, s := range invalid {
		if err := ValidateSubdirSelector(s); err == nil {
			t.Errorf("ValidateSubdirSelector(%q) = nil; want rejection", s)
		}
	}
}

// Contracts 1, 3, 4 at the resolver level (root selection, manifest existence,
// containment) independent of git.
func TestResolvePackageSubdir(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	goSub := writeMultiNodeClone(t, root)

	t.Run("empty selector returns root unchanged", func(t *testing.T) {
		got, err := ResolvePackageSubdir(root, "")
		if err != nil || got != root {
			t.Fatalf("ResolvePackageSubdir(root, \"\") = %q,%v; want %q,nil", got, err, root)
		}
	})

	t.Run("selects existing subdir", func(t *testing.T) {
		got, err := ResolvePackageSubdir(root, "go")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != goSub {
			t.Fatalf("got %q; want %q", got, goSub)
		}
	})

	t.Run("in-tree traversal that stays inside is allowed", func(t *testing.T) {
		// "go/../go" cleans to "go" and stays within root.
		got, err := ResolvePackageSubdir(root, "go/../go")
		if err != nil || got != goSub {
			t.Fatalf("ResolvePackageSubdir(root, \"go/../go\") = %q,%v; want %q,nil", got, err, goSub)
		}
	})

	t.Run("missing manifest names the expected path", func(t *testing.T) {
		_, err := ResolvePackageSubdir(root, "nope")
		if err == nil {
			t.Fatal("expected error for missing manifest")
		}
		want := filepath.Join(root, "nope", "agentfield-package.yaml")
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q should name expected manifest path %q", err.Error(), want)
		}
	})

	t.Run("absolute path rejected", func(t *testing.T) {
		if _, err := ResolvePackageSubdir(root, "/etc"); err == nil {
			t.Fatal("expected rejection of absolute path")
		}
	})

	t.Run("escaping path rejected", func(t *testing.T) {
		if _, err := ResolvePackageSubdir(root, "../escape"); err == nil {
			t.Fatal("expected rejection of escaping path")
		}
	})
}

// Contracts 1, 2, 3, 4 via the GitInstaller's post-clone resolution against a
// faked clone directory (no network/git needed).
func TestGitInstallerResolvePackageRoot(t *testing.T) {
	t.Parallel()
	clone := t.TempDir()
	goSub := writeMultiNodeClone(t, clone)

	t.Run("no selector installs the root package (unchanged)", func(t *testing.T) {
		gi := &GitInstaller{AgentFieldHome: t.TempDir()}
		root, err := gi.resolvePackageRoot(clone)
		if err != nil {
			t.Fatalf("resolvePackageRoot: %v", err)
		}
		md, err := ParsePackageMetadata(root)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if md.Name != "root-node" {
			t.Fatalf("root install picked %q; want root-node", md.Name)
		}
	})

	t.Run("--path go installs the go subtree package", func(t *testing.T) {
		gi := &GitInstaller{AgentFieldHome: t.TempDir(), Subdir: "go"}
		root, err := gi.resolvePackageRoot(clone)
		if err != nil {
			t.Fatalf("resolvePackageRoot: %v", err)
		}
		if root != goSub {
			t.Fatalf("resolved root %q; want %q", root, goSub)
		}
		md, err := ParsePackageMetadata(root)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if md.Name != "go-node" {
			t.Fatalf("--path go picked %q; want go-node", md.Name)
		}
		if !md.IsGo() {
			t.Fatalf("go subtree package should be a Go node")
		}
	})

	t.Run("--path with no manifest errors, naming the path", func(t *testing.T) {
		gi := &GitInstaller{AgentFieldHome: t.TempDir(), Subdir: "missing"}
		_, err := gi.resolvePackageRoot(clone)
		if err == nil {
			t.Fatal("expected error for a subdir with no manifest")
		}
		want := filepath.Join(clone, "missing", "agentfield-package.yaml")
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q should name %q", err.Error(), want)
		}
	})

	t.Run("absolute and escaping --path rejected before clone work", func(t *testing.T) {
		for _, bad := range []string{"/abs", "../escape"} {
			gi := &GitInstaller{AgentFieldHome: t.TempDir(), Subdir: bad}
			// resolvePackageRoot rejects it...
			if _, err := gi.resolvePackageRoot(clone); err == nil {
				t.Fatalf("resolvePackageRoot(%q) should error", bad)
			}
			// ...and InstallFromGit rejects it up front with no git available,
			// proving the selector is validated before any clone is attempted.
			if err := gi.InstallFromGit("https://github.com/acme/repo", false); err == nil {
				t.Fatalf("InstallFromGit with --path %q should reject before cloning", bad)
			}
		}
	})
}

// Contract 5: the @ref URL pin and the --path selector are independent and both
// honored — ref is parsed from the URL, subdir is resolved after clone.
func TestInstallPathComposesWithRef(t *testing.T) {
	t.Parallel()
	clone := t.TempDir()
	goSub := writeMultiNodeClone(t, clone)

	info, err := ParseGitURL("https://github.com/acme/repo@v1.2.3")
	if err != nil {
		t.Fatalf("ParseGitURL: %v", err)
	}
	if info.Ref != "v1.2.3" {
		t.Fatalf("ref=%q; want v1.2.3 (the @ref must still be parsed with --path in play)", info.Ref)
	}
	if info.CloneURL != "https://github.com/acme/repo" {
		t.Fatalf("cloneURL=%q; want the ref stripped", info.CloneURL)
	}

	gi := &GitInstaller{AgentFieldHome: t.TempDir(), Subdir: "go"}
	root, err := gi.resolvePackageRoot(clone)
	if err != nil {
		t.Fatalf("resolvePackageRoot: %v", err)
	}
	if root != goSub {
		t.Fatalf("subdir resolution %q; want %q — ref and path must compose", root, goSub)
	}
}

// Contract 6: a Go subdir package builds relative to the subdir-as-package-root.
// The stubbed toolchain materializes the -o output; asserting it lands under the
// subdir (and not the repo root) proves the build is rooted at the subdir.
func TestInstallPathGoSubdirBuildsRelativeToSubdir(t *testing.T) {
	clone := t.TempDir()
	goSub := writeMultiNodeClone(t, clone)
	stubGo(t, "1.21.0")

	gi := &GitInstaller{AgentFieldHome: t.TempDir(), Subdir: "go"}
	root, err := gi.resolvePackageRoot(clone)
	if err != nil {
		t.Fatalf("resolvePackageRoot: %v", err)
	}
	md, err := ParsePackageMetadata(root)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if err := InstallDependencies(root, md); err != nil {
		t.Fatalf("InstallDependencies (go subdir): %v", err)
	}
	if _, err := os.Stat(filepath.Join(goSub, "bin", "node")); err != nil {
		t.Fatalf("expected the Go binary at <subdir>/bin/node: %v", err)
	}
	if _, err := os.Stat(filepath.Join(clone, "bin", "node")); !os.IsNotExist(err) {
		t.Fatalf("build must be rooted at the subdir, not the repo root")
	}
}
