package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/fsnotify/fsnotify"
	"gopkg.in/yaml.v3"
)

type packageStorage interface {
	GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error)
	StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error)
}

var storePackage = func(storageProvider packageStorage, ctx context.Context, pkg *types.AgentPackage) error {
	return storageProvider.StoreAgentPackage(ctx, pkg)
}

var updatePackage = func(storageProvider packageStorage, ctx context.Context, pkg *types.AgentPackage) error {
	return storageProvider.UpdateAgentPackage(ctx, pkg)
}

// InstallationRegistry mirrors the structure of installed.yaml
type InstallationRegistry struct {
	Installed map[string]InstalledPackage `yaml:"installed"`
}

type InstalledPackage struct {
	Name         string `yaml:"name"`
	Version      string `yaml:"version"`
	Description  string `yaml:"description"`
	Path         string `yaml:"path"`
	Source       string `yaml:"source"`
	SourcePath   string `yaml:"source_path"`
	InstalledAt  string `yaml:"installed_at"`
	Commit       string `yaml:"commit"`
	Ref          string `yaml:"ref"`
	AutoUpdate   *bool  `yaml:"auto_update"`
	UpdatedAt    string `yaml:"updated_at"`
	Status       string `yaml:"status"`
	DesiredState string `yaml:"desired_state"`
	Runtime      struct {
		Port      int    `yaml:"port"`
		PID       int    `yaml:"pid"`
		StartedAt string `yaml:"started_at"`
		LogFile   string `yaml:"log_file"`
		BootID    string `yaml:"boot_id"`
		StartTime string `yaml:"start_time"`
	} `yaml:"runtime"`
}

// SyncPackagesFromRegistry reconciles the database with installed.yaml: every
// registry entry is upserted with an installed status (upgrading pre-seeded
// catalog rows), and previously-installed rows missing from the registry are
// downgraded to uninstalled. An absent registry file means nothing is
// installed and only triggers the downgrade pass.
func SyncPackagesFromRegistry(agentfieldHome string, storageProvider packageStorage) error {
	ctx := context.Background()
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	var registry InstallationRegistry
	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, &registry); err != nil {
			return err
		}
	}

	for pkgName, pkg := range registry.Installed {
		status := packageStatusFromRegistry(pkg.Status)
		existing, err := storageProvider.GetAgentPackage(ctx, pkgName)
		metadata := mirroredPackageMetadata(pkg, existing)
		if err == nil && existing != nil && installedStatus(existing.Status) &&
			existing.Status == status && existing.InstallPath == pkg.Path && existing.Version == pkg.Version &&
			existing.Repository != nil && *existing.Repository == pkg.SourcePath &&
			packageMetadataEqual(existing.Metadata, metadata) {
			continue // Already reconciled
		}
		// Load agentfield-package.yaml
		packageYamlPath := filepath.Join(pkg.Path, "agentfield-package.yaml")
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
		installedAt := now
		if parsed, err := time.Parse(time.RFC3339, pkg.InstalledAt); err == nil && !parsed.IsZero() {
			installedAt = parsed
		}
		recordedSource := pkg.SourcePath
		agentPkg := &types.AgentPackage{
			ID:                  pkgName,
			Name:                pkg.Name,
			Version:             pkg.Version,
			Description:         &pkg.Description,
			Repository:          &recordedSource,
			InstallPath:         pkg.Path,
			ConfigurationSchema: schemaJson,
			Status:              status,
			ConfigurationStatus: types.ConfigurationStatusDraft,
			InstalledAt:         installedAt,
			UpdatedAt:           now,
			Metadata:            metadata,
		}
		if existing != nil {
			// Preserve identity fields the catalog row may carry.
			agentPkg.InstalledAt = existing.InstalledAt
			if agentPkg.InstalledAt.IsZero() {
				agentPkg.InstalledAt = now
			}
			agentPkg.ConfigurationStatus = existing.ConfigurationStatus
			_ = updatePackage(storageProvider, ctx, agentPkg)
			continue
		}
		_ = storePackage(storageProvider, ctx, agentPkg)
	}

	// Downgrade rows that claim to be installed but are gone from the registry.
	all, err := storageProvider.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		return nil // Listing failure must not break startup sync
	}
	for _, row := range all {
		if !installedStatus(row.Status) {
			continue
		}
		if _, present := registry.Installed[row.ID]; present {
			continue
		}
		row.Status = types.PackageStatusUninstalled
		row.UpdatedAt = time.Now()
		_ = updatePackage(storageProvider, ctx, row)
	}
	return nil
}

func packageMetadataEqual(left, right types.PackageMetadata) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func mirroredPackageMetadata(pkg InstalledPackage, existing *types.AgentPackage) types.PackageMetadata {
	metadata := types.PackageMetadata{}
	if existing != nil {
		metadata = existing.Metadata
	}
	custom := make(map[string]interface{}, len(metadata.Custom)+6)
	for key, value := range metadata.Custom {
		custom[key] = value
	}
	autoUpdate := true
	if pkg.AutoUpdate != nil {
		autoUpdate = *pkg.AutoUpdate
	}
	custom["commit"] = pkg.Commit
	custom["ref"] = pkg.Ref
	custom["auto_update"] = autoUpdate
	custom["updated_at"] = pkg.UpdatedAt
	desiredState := pkg.DesiredState
	if desiredState == "" {
		if pkg.Status == "running" {
			desiredState = "running"
		} else {
			desiredState = "stopped"
		}
	}
	custom["desired_state"] = desiredState
	custom["runtime"] = map[string]interface{}{
		"port":       pkg.Runtime.Port,
		"pid":        pkg.Runtime.PID,
		"boot_id":    pkg.Runtime.BootID,
		"start_time": pkg.Runtime.StartTime,
	}
	metadata.Custom = custom
	return metadata
}

func packageStatusFromRegistry(status string) types.PackageStatus {
	switch status {
	case string(types.PackageStatusRunning):
		return types.PackageStatusRunning
	case string(types.PackageStatusStopped):
		return types.PackageStatusStopped
	default:
		return types.PackageStatusInstalled
	}
}

// installedStatus reports whether a package status implies the package is on
// disk (running/stopped are lifecycle refinements of installed).
func installedStatus(status types.PackageStatus) bool {
	switch status {
	case types.PackageStatusInstalled, types.PackageStatusRunning, types.PackageStatusStopped:
		return true
	default:
		return false
	}
}

// StartPackageRegistryWatcher watches the installed.yaml registry and keeps storage in sync.
func StartPackageRegistryWatcher(parentCtx context.Context, agentfieldHome string, storageProvider packageStorage) (context.CancelFunc, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create registry watcher: %w", err)
	}

	registryDir := agentfieldHome
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
		registryFile := filepath.Join(agentfieldHome, "installed.yaml")
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
				if err := SyncPackagesFromRegistry(agentfieldHome, storageProvider); err != nil {
					logger.Logger.Error().Err(err).Msg("failed to sync packages from registry")
				} else {
					logger.Logger.Debug().Msg("registry sync completed")
				}
			}
		}
	}()

	return cancel, nil
}
