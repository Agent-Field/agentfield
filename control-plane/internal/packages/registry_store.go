package packages

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

// installationRegistryMu serializes every in-process installed.yaml
// read-modify-write transaction. The registry is one logical resource even
// when callers use different AgentField homes in tests, and a single lock
// avoids stale whole-file writers losing another package's update.
var installationRegistryMu sync.Mutex

// LoadInstallationRegistry reads one registry under the process-wide lock.
// Callers that will mutate it must use UpdateInstallationRegistry instead.
func LoadInstallationRegistry(path string) (*InstallationRegistry, error) {
	installationRegistryMu.Lock()
	defer installationRegistryMu.Unlock()
	return loadInstallationRegistryUnlocked(path)
}

// WriteInstallationRegistry preserves the direct-write behaviour of legacy
// callers while still participating in the process-wide serialization.
func WriteInstallationRegistry(path string, registry *InstallationRegistry) error {
	installationRegistryMu.Lock()
	defer installationRegistryMu.Unlock()
	return writeInstallationRegistryDirectUnlocked(path, registry)
}

// UpdateInstallationRegistry holds the process-wide registry lock across a
// legacy direct-write transaction. A mutation error leaves the file untouched.
func UpdateInstallationRegistry(path string, mutate func(*InstallationRegistry) error) error {
	installationRegistryMu.Lock()
	defer installationRegistryMu.Unlock()
	registry, err := loadInstallationRegistryUnlocked(path)
	if err != nil {
		return fmt.Errorf("failed to read registry %s: %w", path, err)
	}
	if err := mutate(registry); err != nil {
		return err
	}
	return writeInstallationRegistryDirectUnlocked(path, registry)
}

// WriteInstallationRegistryAtomic and UpdateInstallationRegistryAtomic retain
// the temp-file/rename behaviour used by package maintenance.
func WriteInstallationRegistryAtomic(path string, registry *InstallationRegistry) error {
	installationRegistryMu.Lock()
	defer installationRegistryMu.Unlock()
	return writeInstallationRegistryAtomicUnlocked(path, registry)
}

func UpdateInstallationRegistryAtomic(path string, mutate func(*InstallationRegistry) error) error {
	installationRegistryMu.Lock()
	defer installationRegistryMu.Unlock()
	registry, err := loadInstallationRegistryUnlocked(path)
	if err != nil {
		return fmt.Errorf("failed to read registry %s: %w", path, err)
	}
	if err := mutate(registry); err != nil {
		return err
	}
	return writeInstallationRegistryAtomicUnlocked(path, registry)
}

func loadInstallationRegistryUnlocked(path string) (*InstallationRegistry, error) {
	registry := &InstallationRegistry{Installed: make(map[string]InstalledPackage)}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		parent, parentErr := os.Stat(filepath.Dir(path))
		if errors.Is(parentErr, os.ErrNotExist) || parentErr == nil && parent.IsDir() {
			return registry, nil
		}
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	if err := yaml.Unmarshal(data, registry); err != nil {
		return nil, fmt.Errorf("failed to parse registry: %w", err)
	}
	if registry.Installed == nil {
		registry.Installed = make(map[string]InstalledPackage)
	}
	return registry, nil
}

func marshalInstallationRegistry(registry *InstallationRegistry) ([]byte, error) {
	data, err := yaml.Marshal(registry)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal registry: %w", err)
	}
	return data, nil
}

func writeInstallationRegistryDirectUnlocked(path string, registry *InstallationRegistry) error {
	data, err := marshalInstallationRegistry(registry)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func writeInstallationRegistryAtomicUnlocked(path string, registry *InstallationRegistry) error {
	data, err := marshalInstallationRegistry(registry)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create registry directory: %w", err)
	}
	mode := os.FileMode(0o644)
	if info, statErr := os.Stat(path); statErr == nil {
		mode = info.Mode().Perm()
	}
	temp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Chmod(mode); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	return os.Rename(tempPath, path)
}
