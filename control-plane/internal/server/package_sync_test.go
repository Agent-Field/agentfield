package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestSyncPackagesFromRegistryStoresMissingPackages(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "example-agent")
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
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(installed), 0o644))

	packageYAML := `name: Example Agent
version: 1.0.0
schema:
  type: object
`
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(packageYAML), 0o644))

	storage := newStubPackageStorage()
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))

	pkg, ok := storage.packages["example-agent"]
	require.True(t, ok)
	require.Equal(t, "Example Agent", pkg.Name)
	require.NotEmpty(t, pkg.ConfigurationSchema)
}

func TestSyncPackagesSkipsExistingEntries(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	installed := `installed:
  existing-agent:
    name: Existing
    version: 0.1.0
    description: already present
    path: ` + agentfieldHome + `
`
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(installed), 0o644))

	storage := newStubPackageStorage()
	storage.packages["existing-agent"] = &types.AgentPackage{ID: "existing-agent", Name: "Existing", InstalledAt: time.Now()}

	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))

	require.Len(t, storage.packages, 1)
}

// Reconcile contract 1: a pre-seeded catalog row for a package that IS in the
// registry gets upgraded to installed (status + schema + installed_at).
func TestSyncUpgradesCatalogRowToInstalled(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "cat-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	installed := `installed:
  cat-agent:
    name: Cat Agent
    version: 2.0.0
    description: from registry
    path: ` + pkgDir + `
`
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(installed), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Cat Agent\nversion: 2.0.0\n"), 0o644))

	storage := newStubPackageStorage()
	storage.packages["cat-agent"] = &types.AgentPackage{
		ID: "cat-agent", Name: "Cat Agent", Version: "1.0.0",
		Status: types.PackageStatus("not_configured"),
	}
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))

	pkg := storage.packages["cat-agent"]
	require.Equal(t, types.PackageStatusInstalled, pkg.Status)
	require.Equal(t, "2.0.0", pkg.Version)
	require.Equal(t, pkgDir, pkg.InstallPath)
	require.False(t, pkg.InstalledAt.IsZero())
	require.NotEmpty(t, pkg.ConfigurationSchema)
}

// Reconcile contract 2: a row claiming installed but missing from the
// registry is downgraded to uninstalled.
func TestSyncDowngradesRemovedPackages(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"),
		[]byte("installed: {}\n"), 0o644))

	storage := newStubPackageStorage()
	storage.packages["gone-agent"] = &types.AgentPackage{
		ID: "gone-agent", Name: "Gone", Status: types.PackageStatusInstalled,
		InstalledAt: time.Now(),
	}
	storage.packages["catalog-agent"] = &types.AgentPackage{
		ID: "catalog-agent", Name: "Catalog", Status: types.PackageStatus("not_configured"),
	}
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))

	require.Equal(t, types.PackageStatusUninstalled, storage.packages["gone-agent"].Status)
	// Non-installed rows are untouched by the downgrade pass.
	require.Equal(t, types.PackageStatus("not_configured"), storage.packages["catalog-agent"].Status)
}

// Reconcile contract 3: an absent registry file means nothing is installed —
// the downgrade pass still runs (running/stopped count as installed states).
func TestSyncAbsentRegistryStillDowngrades(t *testing.T) {
	t.Parallel()

	storage := newStubPackageStorage()
	storage.packages["stale-agent"] = &types.AgentPackage{
		ID: "stale-agent", Name: "Stale", Status: types.PackageStatusRunning,
	}
	require.NoError(t, SyncPackagesFromRegistry(t.TempDir(), storage))
	require.Equal(t, types.PackageStatusUninstalled, storage.packages["stale-agent"].Status)
}
