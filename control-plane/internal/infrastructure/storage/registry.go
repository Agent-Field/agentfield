// agentfield/internal/infrastructure/storage/registry.go
package storage

import (
	"encoding/json"
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"os"
	"path/filepath"
)

type LocalRegistryStorage struct {
	fs        interfaces.FileSystemAdapter
	storePath string
}

func NewLocalRegistryStorage(fs interfaces.FileSystemAdapter, path string) interfaces.RegistryStorage {
	return &LocalRegistryStorage{
		fs:        fs,
		storePath: path,
	}
}

func (s *LocalRegistryStorage) LoadRegistry() (*domain.InstallationRegistry, error) {
	if !s.fs.Exists(s.storePath) {
		return &domain.InstallationRegistry{
			Installed: make(map[string]domain.InstalledPackage),
		}, nil
	}

	data, err := s.fs.ReadFile(s.storePath)
	if err != nil {
		return nil, err
	}

	var registry domain.InstallationRegistry
	if err := json.Unmarshal(data, &registry); err != nil {
		return nil, err
	}

	return &registry, nil
}

func (s *LocalRegistryStorage) SaveRegistry(registry *domain.InstallationRegistry) error {
	data, err := json.MarshalIndent(registry, "", "  ")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(s.storePath), 0755); err != nil {
		return err
	}

	return s.fs.WriteFile(s.storePath, data)
}

func (s *LocalRegistryStorage) GetPackage(name string) (*domain.InstalledPackage, error) {
	registry, err := s.LoadRegistry()
	if err != nil {
		return nil, err
	}

	pkg, exists := registry.Installed[name]
	if !exists {
		return nil, os.ErrNotExist
	}

	return &pkg, nil
}

func (s *LocalRegistryStorage) SavePackage(name string, pkg *domain.InstalledPackage) error {
	registry, err := s.LoadRegistry()
	if err != nil {
		return err
	}

	registry.Installed[name] = *pkg
	return s.SaveRegistry(registry)
}
