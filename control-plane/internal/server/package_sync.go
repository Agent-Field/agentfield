package server

import (
	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/pkg/types"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"gopkg.in/yaml.v3"
)

type packageStorage interface {
	GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error)
	StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
}

var storePackage = func(storageProvider packageStorage, ctx context.Context, pkg *types.AgentPackage) error {
	return storageProvider.StoreAgentPackage(ctx, pkg)
}

// InstallationRegistry mirrors the structure of installed.yaml
type InstallationRegistry struct {
	Installed map[string]InstalledPackage `yaml:"installed"`
}

type InstalledPackage struct {
	Name        string `yaml:"name"`
	Version     string `yaml:"version"`
	Description string `yaml:"description"`
	Path        string `yaml:"path"`
	Source      string `yaml:"source"`
	SourcePath  string `yaml:"source_path"`
	InstalledAt string `yaml:"installed_at"`
	Status      string `yaml:"status"`
	Runtime     struct {
		Port      int    `yaml:"port"`
		PID       int    `yaml:"pid"`
		StartedAt string `yaml:"started_at"`
		LogFile   string `yaml:"log_file"`
	} `yaml:"runtime"`
}

// SyncPackagesFromRegistry ensures all packages in installed.yaml are present in the database.
func SyncPackagesFromRegistry(haxenHome string, storageProvider packageStorage) error {
	ctx := context.Background()
	registryPath := filepath.Join(haxenHome, "installed.yaml")
	data, err := os.ReadFile(registryPath)
	if err != nil {
		return nil // No registry, nothing to sync
	}
	var registry InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return err
	}
	for pkgName, pkg := range registry.Installed {
		// Check if package exists in DB
		_, err := storageProvider.GetAgentPackage(ctx, pkgName)
		if err == nil {
			continue // Already present
		}
		// Load haxen-package.yaml
		packageYamlPath := filepath.Join(pkg.Path, "haxen-package.yaml")
		packageYamlData, err := os.ReadFile(packageYamlPath)
		if err != nil {
			continue // Skip if missing
		}
		var packageYaml map[string]interface{}
		if err := yaml.Unmarshal(packageYamlData, &packageYaml); err != nil {
			continue
		}
		// Convert schema to JSON for storage
		schemaJson, _ := json.Marshal(packageYaml)
		now := time.Now()
		agentPkg := &types.AgentPackage{
			ID:                  pkgName,
			Name:                pkg.Name,
			Version:             pkg.Version,
			Description:         &pkg.Description,
			InstallPath:         pkg.Path,
			ConfigurationSchema: schemaJson,
			Status:              types.PackageStatusInstalled,
			ConfigurationStatus: types.ConfigurationStatusDraft,
			InstalledAt:         now,
			UpdatedAt:           now,
		}
		_ = storePackage(storageProvider, ctx, agentPkg)
	}
	return nil
}

// StartPackageRegistryWatcher watches the installed.yaml registry and keeps storage in sync.
func StartPackageRegistryWatcher(parentCtx context.Context, haxenHome string, storageProvider packageStorage) (context.CancelFunc, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create registry watcher: %w", err)
	}

	registryDir := haxenHome
	if err := watcher.Add(registryDir); err != nil {
		watcher.Close()
		return nil, fmt.Errorf("failed to watch registry directory %s: %w", registryDir, err)
	}

	ctx, cancel := context.WithCancel(parentCtx)
	syncCh := make(chan struct{}, 1)

	var once sync.Once
	dispatchSync := func() {
		once.Do(func() { logger.Logger.Info().Msg("📦 Package registry watcher started") })
		select {
		case syncCh <- struct{}{}:
		default:
		}
	}

	go func() {
		defer watcher.Close()
		defer close(syncCh)
		registryFile := filepath.Join(haxenHome, "installed.yaml")
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Name == "" {
					continue
				}
				if filepath.Clean(event.Name) != registryFile {
					continue
				}
				if event.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename|fsnotify.Remove) == 0 {
					continue
				}
				dispatchSync()
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				if err != nil {
					logger.Logger.Error().Err(err).Msg("registry watcher error")
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case _, ok := <-syncCh:
				if !ok {
					return
				}
				time.Sleep(250 * time.Millisecond)
				if err := SyncPackagesFromRegistry(haxenHome, storageProvider); err != nil {
					logger.Logger.Error().Err(err).Msg("failed to sync packages from registry")
				} else {
					logger.Logger.Debug().Msg("registry sync completed")
				}
			}
		}
	}()

	return cancel, nil
}
