package packages

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

// Contract: starting a node whose declared dependency is not installed warns
// and continues without panicking or starting anything.
func TestRunner_StartNodeDependencies_NotInstalled(t *testing.T) {
	home := t.TempDir()
	nodeDir := filepath.Join(home, "packages", "parent")
	if err := os.MkdirAll(nodeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: parent\nversion: 1.0.0\ndependencies:\n  nodes:\n    - af://registry/absent\n"
	if err := os.WriteFile(filepath.Join(nodeDir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := &InstallationRegistry{Installed: map[string]InstalledPackage{}}
	data, _ := yaml.Marshal(reg)
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "parent", Path: nodeDir}
	// Absent dependency → warns and returns; must not panic or recurse into run.
	ar.startNodeDependencies(node, map[string]bool{"parent": true})
}
