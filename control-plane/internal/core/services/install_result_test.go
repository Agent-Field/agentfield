package services

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/stretchr/testify/require"
)

// Validation contract for InstallPackageWithResult — the seam that lets an
// install job report the package that actually landed rather than inferring it
// from a registry diff:
//
//  1. A local install reports the manifest's name.
//  2. A git install reports the name the git installer recorded.
//  3. A git install of a package whose manifest declares `superseded_by:`
//     reports the SUCCESSOR's name — including when the successor takes the
//     predecessor's own name, where a registry diff sees no change at all.
//  4. A failed install reports no name.
//  5. A declared node dependency that cannot be installed does not fail the
//     parent install, which is already in place and usable.

// runGit runs a git command in dir, with a fixed identity so the fixture does
// not depend on the host's git config.
func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, "git %v: %s", args, out)
}

// bareRepoAt commits everything in dir and bare-clones it to bare, which is the
// source `af install` routes to the git installer (the `.git` suffix is what
// IsGitURL keys on for a plain path).
func bareRepoAt(t *testing.T, dir, bare string) string {
	t.Helper()
	runGit(t, dir, "init", "-q")
	runGit(t, dir, "add", "-A")
	runGit(t, dir, "commit", "-qm", "fixture")
	out, err := exec.Command("git", "clone", "-q", "--bare", dir, bare).CombinedOutput()
	require.NoError(t, err, "git clone --bare: %s", out)
	return bare
}

// Contract 1: a local install reports the name in the manifest.
func TestInstallPackageWithResult_LocalReportsManifestName(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	writeNode(t, src, "local-node")

	name, err := newLocalPackageService(t, home).InstallPackageWithResult(src, domain.InstallOptions{})
	require.NoError(t, err)
	require.Equal(t, "local-node", name)
}

// Contract 2: a git install reports what the git installer recorded.
func TestInstallPackageWithResult_GitReportsInstalledName(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	writeNode(t, src, "git-node")

	bare := bareRepoAt(t, src, filepath.Join(t.TempDir(), "fixture.git"))
	name, err := newLocalPackageService(t, home).
		InstallPackageWithResult(bare, domain.InstallOptions{})
	require.NoError(t, err)
	require.Equal(t, "git-node", name)
	require.True(t, installedNamesFromRegistry(t, home)["git-node"])
}

// Contract 3: a redirect reports the successor. The same-name case is the one a
// registry diff cannot see — the set of installed names is identical before and
// after — so it is the reason this seam exists at all.
func TestInstallPackageWithResult_ReportsSupersededSuccessor(t *testing.T) {
	for _, tc := range []struct {
		name          string
		successorName string
	}{
		{name: "successor takes a new name", successorName: "successor-node"},
		{name: "successor takes the same name", successorName: "redirected-node"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			src := filepath.Join(t.TempDir(), "repo")
			writeNode(t, filepath.Join(src, "v2"), tc.successorName)

			// The root redirects into this same repo's v2/ subdirectory. Written
			// after the bare clone location is known, so point at it directly.
			bare := filepath.Join(t.TempDir(), "fixture.git")
			require.NoError(t, os.MkdirAll(src, 0o755))
			require.NoError(t, os.WriteFile(
				filepath.Join(src, "agentfield-package.yaml"),
				[]byte("name: redirected-node\nversion: 1.0.0\nmain: main.py\nsuperseded_by: "+bare+"//v2\n"),
				0o644))
			require.NoError(t, os.WriteFile(filepath.Join(src, "main.py"), []byte("print('ok')\n"), 0o644))

			bareRepoAt(t, src, bare)

			name, err := newLocalPackageService(t, home).
				InstallPackageWithResult(bare, domain.InstallOptions{})
			require.NoError(t, err)
			require.Equal(t, tc.successorName, name,
				"the successor's name is what landed, so it is what must be reported")
			require.True(t, installedNamesFromRegistry(t, home)[tc.successorName])
		})
	}
}

// Contract 4: a failed install reports no name, whatever stage it failed at.
func TestInstallPackageWithResult_FailuresReportNoName(t *testing.T) {
	t.Run("invalid package structure", func(t *testing.T) {
		home := t.TempDir()
		src := filepath.Join(t.TempDir(), "repo")
		// Declares an entrypoint it does not ship, so validation rejects it.
		require.NoError(t, os.MkdirAll(src, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(src, "agentfield-package.yaml"),
			[]byte("name: broken-node\nversion: 1.0.0\nmain: missing.py\n"), 0o644))

		name, err := newLocalPackageService(t, home).InstallPackageWithResult(src, domain.InstallOptions{})
		require.Error(t, err)
		require.Empty(t, name)
		require.Empty(t, installedNamesFromRegistry(t, home),
			"a rejected package must not reach the registry")
	})

	t.Run("dependency build fails", func(t *testing.T) {
		home := t.TempDir()
		src := filepath.Join(t.TempDir(), "repo")
		require.NoError(t, os.MkdirAll(src, 0o755))
		// A Go node whose build cannot succeed: the manifest promises a build
		// entrypoint, and there is no Go module behind it.
		require.NoError(t, os.WriteFile(filepath.Join(src, "agentfield-package.yaml"),
			[]byte("name: broken-go-node\nversion: 1.0.0\nlanguage: go\nentrypoint:\n  build: ./cmd/nope\n  start: bin/nope\n"), 0o644))

		name, err := newLocalPackageService(t, home).InstallPackageWithResult(src, domain.InstallOptions{})
		require.Error(t, err)
		require.Empty(t, name)
	})

	t.Run("registry cannot be written", func(t *testing.T) {
		home := t.TempDir()
		src := filepath.Join(t.TempDir(), "repo")
		writeNode(t, src, "unwritable-node")
		// A directory where the registry file belongs: the write fails, and the
		// install must surface that rather than claim a name.
		require.NoError(t, os.Mkdir(filepath.Join(home, "installed.yaml"), 0o755))

		name, err := newLocalPackageService(t, home).InstallPackageWithResult(src, domain.InstallOptions{})
		require.Error(t, err)
		require.Empty(t, name)
	})
}

// Contract: a dependency cycle terminates. Two packages that declare each other
// as node dependencies, by bare path — the form `resolveNodeRef` cannot name, so
// the already-installed check never fires — and with Force set, which is what
// every update uses, so each install succeeds rather than being refused. Without
// a walk-tracking guard this recurses until the process dies, taking the package
// job manager's `active` latch with it and blocking every later install.
func TestInstallPackageWithResult_DependencyCycleTerminates(t *testing.T) {
	home := t.TempDir()
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a"), filepath.Join(dir, "b")
	for _, n := range []struct{ path, name, dep string }{{a, "cycle-a", b}, {b, "cycle-b", a}} {
		require.NoError(t, os.MkdirAll(n.path, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(n.path, "agentfield-package.yaml"),
			[]byte("name: "+n.name+"\nversion: 1.0.0\nmain: main.py\ndependencies:\n  nodes:\n    - "+n.dep+"\n"), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(n.path, "main.py"), []byte("print('ok')\n"), 0o644))
	}

	done := make(chan struct{})
	var name string
	var err error
	go func() {
		defer close(done)
		name, err = newLocalPackageService(t, home).
			InstallPackageWithResult(a, domain.InstallOptions{Force: true})
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("mutually-dependent packages recursed without terminating")
	}
	require.NoError(t, err)
	require.Equal(t, "cycle-a", name)
	installed := installedNamesFromRegistry(t, home)
	require.True(t, installed["cycle-a"] && installed["cycle-b"], "both sides of the cycle install once")
}

// Contract 5: a node dependency that cannot be installed is reported but does
// not fail the parent — the parent is already installed and usable.
func TestInstallPackageWithResult_UninstallableDependencyDoesNotFailParent(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	require.NoError(t, os.MkdirAll(src, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(src, "agentfield-package.yaml"),
		[]byte("name: parent-node\nversion: 1.0.0\nmain: main.py\ndependencies:\n  nodes:\n    - "+
			filepath.Join(t.TempDir(), "does-not-exist")+"\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(src, "main.py"), []byte("print('ok')\n"), 0o644))

	name, err := newLocalPackageService(t, home).InstallPackageWithResult(src, domain.InstallOptions{})
	require.NoError(t, err, "the parent installed; a bad dependency is not its failure")
	require.Equal(t, "parent-node", name)
	require.True(t, installedNamesFromRegistry(t, home)["parent-node"])
}
