package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestPackageJobsUninstallReconcilesRegistryToStorage(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "wiring-agent")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: Wiring Agent\nversion: 1.0.0\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), []byte(`installed:
  wiring-agent:
    name: Wiring Agent
    version: 1.0.0
    path: `+pkgDir+`
    installed_at: "2026-07-30T17:48:53-04:00"
    status: stopped
`), 0o644))

	storage := newStubPackageStorage()
	storage.packages["wiring-agent"] = &types.AgentPackage{
		ID:          "wiring-agent",
		Name:        "Wiring Agent",
		Version:     "1.0.0",
		InstallPath: pkgDir,
		Status:      types.PackageStatusInstalled,
		InstalledAt: time.Date(2026, 7, 30, 21, 48, 53, 0, time.UTC),
	}

	fileSystem := infrastorage.NewFileSystemAdapter()
	registryStorage := infrastorage.NewLocalRegistryStorage(
		fileSystem,
		filepath.Join(agentfieldHome, "installed.json"),
	)
	manager := packagejobs.NewManager(registryStorage, agentfieldHome, nil)
	hookCalls := 0
	manager.SetOnRegistryChange(func() {
		hookCalls++
		_ = SyncPackagesFromRegistry(agentfieldHome, storage)
	})

	require.NoError(t, manager.Uninstall("wiring-agent"))
	require.Equal(t, 1, hookCalls)
	require.Equal(t, types.PackageStatusUninstalled, storage.packages["wiring-agent"].Status)
}
