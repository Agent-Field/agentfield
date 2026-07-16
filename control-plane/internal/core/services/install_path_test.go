package services

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// Validation contract for `af install <local-dir> --path <subdir>`:
//   1. --path <subdir> installs the node whose manifest is at <dir>/<subdir>.
//   2. No --path installs the root node (unchanged).
//   3. --path with no manifest there errors, naming the expected path; nothing
//      is installed.
//   4. Absolute / escaping --path is rejected; nothing is installed.
// These mirror the git-path contract for a local source.

// writeNode writes a minimal, dependency-free Python node (so install does no
// venv/network work) at dir with the given package name.
func writeNode(t *testing.T, dir, name string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(dir, 0o755))
	manifest := "name: " + name + "\nversion: 1.0.0\nmain: main.py\n"
	require.NoError(t, os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "main.py"), []byte("print('ok')\n"), 0o644))
}

// installedNamesFromRegistry reads ~/.agentfield/installed.yaml and returns the
// set of installed package names (empty when the registry file is absent).
func installedNamesFromRegistry(t *testing.T, home string) map[string]bool {
	t.Helper()
	names := map[string]bool{}
	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if os.IsNotExist(err) {
		return names
	}
	require.NoError(t, err)
	var reg packages.InstallationRegistry
	require.NoError(t, yaml.Unmarshal(data, &reg))
	for name := range reg.Installed {
		names[name] = true
	}
	return names
}

func newLocalPackageService(t *testing.T, home string) *DefaultPackageService {
	t.Helper()
	return NewPackageService(newMockPackageRegistryStorage(), newMockFileSystemAdapter(), home).(*DefaultPackageService)
}

func TestInstallLocalPackage_PathSelectsSubdir(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	writeNode(t, src, "root-node")
	writeNode(t, filepath.Join(src, "sub"), "sub-node")

	svc := newLocalPackageService(t, home)
	require.NoError(t, svc.InstallPackage(src, domain.InstallOptions{Path: "sub"}))

	names := installedNamesFromRegistry(t, home)
	assert.True(t, names["sub-node"], "the --path subdir node should be installed")
	assert.False(t, names["root-node"], "the root node must NOT be installed when --path selects a subdir")

	// The installed package directory is keyed by the subdir manifest name and
	// contains that subtree at its root.
	pkgDir := filepath.Join(home, "packages", "sub-node")
	if _, err := os.Stat(filepath.Join(pkgDir, "agentfield-package.yaml")); err != nil {
		t.Fatalf("expected sub-node package copied to %s: %v", pkgDir, err)
	}
}

func TestInstallLocalPackage_NoPathInstallsRoot(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	writeNode(t, src, "root-node")
	writeNode(t, filepath.Join(src, "sub"), "sub-node")

	svc := newLocalPackageService(t, home)
	require.NoError(t, svc.InstallPackage(src, domain.InstallOptions{}))

	names := installedNamesFromRegistry(t, home)
	assert.True(t, names["root-node"], "bare install must install the root node")
	assert.False(t, names["sub-node"], "bare install must not reach into subdirectories")
}

func TestInstallLocalPackage_PathMissingManifest(t *testing.T) {
	home := t.TempDir()
	src := filepath.Join(t.TempDir(), "repo")
	writeNode(t, src, "root-node")

	svc := newLocalPackageService(t, home)
	err := svc.InstallPackage(src, domain.InstallOptions{Path: "nope"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), filepath.Join(src, "nope", "agentfield-package.yaml"))
	assert.Empty(t, installedNamesFromRegistry(t, home), "a failed --path install must not mutate the registry")
}

func TestInstallLocalPackage_PathRejectsAbsoluteAndEscape(t *testing.T) {
	for _, bad := range []string{"/etc", "../escape"} {
		t.Run(bad, func(t *testing.T) {
			home := t.TempDir()
			src := filepath.Join(t.TempDir(), "repo")
			writeNode(t, src, "root-node")

			svc := newLocalPackageService(t, home)
			err := svc.InstallPackage(src, domain.InstallOptions{Path: bad})
			require.Error(t, err)
			assert.Empty(t, installedNamesFromRegistry(t, home), "a rejected --path must not install anything")
		})
	}
}
