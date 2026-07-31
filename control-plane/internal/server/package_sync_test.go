package server

import (
	"context"
	"errors"
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

func TestSyncPackagesFromRegistryMapsRunningStatus(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "running-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(`installed:
  running-agent:
    name: Running Agent
    version: 1.0.0
    path: `+pkgDir+`
    status: running
`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Running Agent\nversion: 1.0.0\n"), 0o644))

	storage := newStubPackageStorage()
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	require.Equal(t, types.PackageStatusRunning, storage.packages["running-agent"].Status)
}

func TestSyncPackagesFromRegistryUpdatesLifecycleStatus(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "lifecycle-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	runningRegistry := `installed:
  lifecycle-agent:
    name: Lifecycle Agent
    version: 1.0.0
    path: ` + pkgDir + `
    status: running
`
	require.NoError(t, os.WriteFile(registryPath, []byte(runningRegistry), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Lifecycle Agent\nversion: 1.0.0\n"), 0o644))

	storage := newStubPackageStorage()
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	require.Equal(t, types.PackageStatusRunning, storage.packages["lifecycle-agent"].Status)

	stoppedRegistry := `installed:
  lifecycle-agent:
    name: Lifecycle Agent
    version: 1.0.0
    path: ` + pkgDir + `
    status: stopped
`
	require.NoError(t, os.WriteFile(registryPath, []byte(stoppedRegistry), 0o644))
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	require.Equal(t, types.PackageStatusStopped, storage.packages["lifecycle-agent"].Status)
}

func TestSyncPackagesFromRegistryDefaultsUnknownStatusToInstalled(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		status string
	}{
		{name: "missing"},
		{name: "unknown", status: "unknown"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			agentfieldHome := t.TempDir()
			pkgDir := filepath.Join(agentfieldHome, "default-agent")
			require.NoError(t, os.MkdirAll(pkgDir, 0o755))
			installed := `installed:
  default-agent:
    name: Default Agent
    version: 1.0.0
    path: ` + pkgDir + "\n"
			if tt.status != "" {
				installed += "    status: " + tt.status + "\n"
			}
			require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(installed), 0o644))
			require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
				[]byte("name: Default Agent\nversion: 1.0.0\n"), 0o644))

			storage := newStubPackageStorage()
			require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
			require.Equal(t, types.PackageStatusInstalled, storage.packages["default-agent"].Status)
		})
	}
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

func TestSyncPackagesFromRegistryInstalledAtContracts(t *testing.T) {
	t.Parallel()

	knownInstalledAt := time.Date(2026, 7, 30, 21, 48, 53, 0, time.UTC)
	tests := []struct {
		name        string
		installedAt string
		assert      func(*testing.T, time.Time)
	}{
		{
			name:        "uses registry timestamp",
			installedAt: knownInstalledAt.Format(time.RFC3339),
			assert: func(t *testing.T, got time.Time) {
				require.Equal(t, knownInstalledAt, got)
			},
		},
		{
			name: "falls back to current time when absent",
			assert: func(t *testing.T, got time.Time) {
				require.False(t, got.IsZero())
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			agentfieldHome := t.TempDir()
			pkgDir := filepath.Join(agentfieldHome, "timestamp-agent")
			require.NoError(t, os.MkdirAll(pkgDir, 0o755))
			installed := `installed:
  timestamp-agent:
    name: Timestamp Agent
    version: 1.0.0
    path: ` + pkgDir + `
`
			if tt.installedAt != "" {
				installed += `    installed_at: "` + tt.installedAt + `"
`
			}
			require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(installed), 0o644))
			require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
				[]byte("name: Timestamp Agent\nversion: 1.0.0\n"), 0o644))

			storage := newStubPackageStorage()
			require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
			tt.assert(t, storage.packages["timestamp-agent"].InstalledAt)
		})
	}
}

func TestSyncPackagesFromRegistryPreservesExistingFields(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "configured-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(`installed:
  configured-agent:
    name: Configured Agent
    version: 2.0.0
    path: `+pkgDir+`
    installed_at: "2026-07-30T17:48:53-04:00"
`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Configured Agent\nversion: 2.0.0\n"), 0o644))

	priorInstalledAt := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	storage := newStubPackageStorage()
	storage.packages["configured-agent"] = &types.AgentPackage{
		ID:                  "configured-agent",
		Name:                "Configured Agent",
		Version:             "1.0.0",
		Status:              types.PackageStatusUninstalled,
		ConfigurationStatus: types.ConfigurationStatus("configured"),
		InstalledAt:         priorInstalledAt,
	}

	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	got := storage.packages["configured-agent"]
	require.Equal(t, types.PackageStatusInstalled, got.Status)
	require.Equal(t, priorInstalledAt, got.InstalledAt)
	require.Equal(t, types.ConfigurationStatus("configured"), got.ConfigurationStatus)
}

func TestSyncPackagesFromRegistryIsIdempotentForInstalledAt(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "idempotent-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(`installed:
  idempotent-agent:
    name: Idempotent Agent
    version: 1.0.0
    path: `+pkgDir+`
    installed_at: "2026-07-30T17:48:53-04:00"
`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Idempotent Agent\nversion: 1.0.0\n"), 0o644))

	storage := newStubPackageStorage()
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	first := storage.packages["idempotent-agent"].InstalledAt
	require.NoError(t, SyncPackagesFromRegistry(agentfieldHome, storage))
	require.Equal(t, first, storage.packages["idempotent-agent"].InstalledAt)
}

type listingErrorPackageStorage struct {
	*stubPackageStorage
	err error
}

func (s *listingErrorPackageStorage) QueryAgentPackages(context.Context, types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, s.err
}

func TestSyncPackagesFromRegistryIgnoresStorageListingFailure(t *testing.T) {
	t.Parallel()

	storage := &listingErrorPackageStorage{
		stubPackageStorage: newStubPackageStorage(),
		err:                errors.New("storage unavailable"),
	}
	require.NoError(t, SyncPackagesFromRegistry(t.TempDir(), storage))
}
