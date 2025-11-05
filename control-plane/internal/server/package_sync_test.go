package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestSyncPackagesFromRegistryStoresMissingPackages(t *testing.T) {
	t.Parallel()

	haxenHome := t.TempDir()
	pkgDir := filepath.Join(haxenHome, "example-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))

	installed := `installed:
  example-agent:
    name: Example Agent
    version: 1.0.0
    description: demo agent
    path: ` + pkgDir + `
    source: local
    status: installed
`
	require.NoError(t, os.WriteFile(filepath.Join(haxenHome, "installed.yaml"), []byte(installed), 0o644))

	packageYAML := `name: Example Agent
version: 1.0.0
schema:
  type: object
`
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "haxen-package.yaml"), []byte(packageYAML), 0o644))

	storage := newStubPackageStorage()
	require.NoError(t, SyncPackagesFromRegistry(haxenHome, storage))

	pkg, ok := storage.packages["example-agent"]
	require.True(t, ok)
	require.Equal(t, "Example Agent", pkg.Name)
	require.NotEmpty(t, pkg.ConfigurationSchema)
}

func TestSyncPackagesSkipsExistingEntries(t *testing.T) {
	t.Parallel()

	haxenHome := t.TempDir()
	installed := `installed:
  existing-agent:
    name: Existing
    version: 0.1.0
    description: already present
    path: ` + haxenHome + `
`
	require.NoError(t, os.WriteFile(filepath.Join(haxenHome, "installed.yaml"), []byte(installed), 0o644))

	storage := newStubPackageStorage()
	storage.packages["existing-agent"] = &types.AgentPackage{ID: "existing-agent", Name: "Existing", InstalledAt: time.Now()}

	require.NoError(t, SyncPackagesFromRegistry(haxenHome, storage))

	require.Len(t, storage.packages, 1)
}
