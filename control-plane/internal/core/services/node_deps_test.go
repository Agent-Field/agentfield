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

// Contract: an af://registry ref resolves to the Agent-Field GitHub repo and
// yields the package name; any other ref is used as-is with an empty name.
func TestResolveNodeRef(t *testing.T) {
	cases := []struct {
		ref        string
		wantSource string
		wantName   string
	}{
		{"af://registry/swe-planner", "https://github.com/Agent-Field/swe-planner", "swe-planner"},
		{"af://registry/pr-af@^1.0", "https://github.com/Agent-Field/pr-af", "pr-af"},
		{"https://github.com/acme/widget", "https://github.com/acme/widget", ""},
	}
	for _, tc := range cases {
		src, name := resolveNodeRef(tc.ref)
		assert.Equal(t, tc.wantSource, src, "source for %s", tc.ref)
		assert.Equal(t, tc.wantName, name, "name for %s", tc.ref)
	}
}

func writeRegistry(t *testing.T, home string, reg *packages.InstallationRegistry) {
	t.Helper()
	data, err := yaml.Marshal(reg)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))
}

func writeManifest(t *testing.T, dir, body string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(body), 0o644))
}

// Contract: installedNames reflects the registry contents.
func TestInstalledNames(t *testing.T) {
	home := t.TempDir()
	ps := &DefaultPackageService{agentfieldHome: home}
	assert.Empty(t, ps.installedNames())

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{"a": {Name: "a"}, "b": {Name: "b"}},
	})
	names := ps.installedNames()
	assert.True(t, names["a"] && names["b"])
}

// Contract: a node dependency that is already installed is not re-installed
// (which also breaks cycles) — installNodeDependencies returns without error.
func TestInstallNodeDependencies_SkipsAlreadyInstalled(t *testing.T) {
	home := t.TempDir()
	ps := &DefaultPackageService{agentfieldHome: home}

	callerDir := filepath.Join(home, "packages", "caller")
	writeManifest(t, callerDir, "name: caller\nversion: 1.0.0\ndependencies:\n  nodes:\n    - af://registry/echo-node\n")
	echoDir := filepath.Join(home, "packages", "echo-node")
	writeManifest(t, echoDir, "name: echo-node\nversion: 1.0.0\n")

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"caller":    {Name: "caller", Path: callerDir},
			"echo-node": {Name: "echo-node", Path: echoDir},
		},
	})

	// `before` is empty, so both packages count as newly installed; the declared
	// dep (echo-node) is already installed and must be skipped — no network call.
	err := ps.installNodeDependencies(map[string]bool{}, domain.InstallOptions{})
	require.NoError(t, err)
}

// Contract: starting a node whose declared dependency is not installed logs a
// warning and continues, without attempting to run anything.
func TestStartNodeDependencies_NotInstalledWarns(t *testing.T) {
	home := t.TempDir()
	as := &DefaultAgentService{agentfieldHome: home}

	nodeDir := filepath.Join(home, "packages", "parent")
	writeManifest(t, nodeDir, "name: parent\nversion: 1.0.0\ndependencies:\n  nodes:\n    - af://registry/absent-node\n")
	writeRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{}})

	node := packages.InstalledPackage{Name: "parent", Path: nodeDir}
	// Must not panic despite a nil process/port manager — the dep is absent so
	// the run path is never reached.
	as.startNodeDependencies(node, map[string]bool{"parent": true}, domain.RunOptions{})
}

// Contract: an already-running dependency is left alone (not restarted).
func TestStartNodeDependencies_SkipsRunningDep(t *testing.T) {
	home := t.TempDir()
	as := &DefaultAgentService{agentfieldHome: home}

	parentDir := filepath.Join(home, "packages", "parent")
	writeManifest(t, parentDir, "name: parent\nversion: 1.0.0\ndependencies:\n  nodes:\n    - af://registry/dep-node\n")
	depDir := filepath.Join(home, "packages", "dep-node")
	writeManifest(t, depDir, "name: dep-node\nversion: 1.0.0\n")

	pid := os.Getpid() // a live process, so reconcile considers it running
	port := 8123
	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"dep-node": {
				Name:    "dep-node",
				Path:    depDir,
				Status:  "running",
				Runtime: packages.RuntimeInfo{PID: &pid, Port: &port},
			},
		},
	})

	node := packages.InstalledPackage{Name: "parent", Path: parentDir}
	// dep-node is running → skipped before the run path; nil managers never used.
	as.startNodeDependencies(node, map[string]bool{"parent": true}, domain.RunOptions{})
}
