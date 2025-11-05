// haxen/internal/core/interfaces/storage.go
package interfaces

import "github.com/your-org/haxen/control-plane/internal/core/domain"

type FileSystemAdapter interface {
    ReadFile(path string) ([]byte, error)
    WriteFile(path string, data []byte) error
    Exists(path string) bool
    CreateDirectory(path string) error
    ListDirectory(path string) ([]string, error)
}

type RegistryStorage interface {
    LoadRegistry() (*domain.InstallationRegistry, error)
    SaveRegistry(registry *domain.InstallationRegistry) error
    GetPackage(name string) (*domain.InstalledPackage, error)
    SavePackage(name string, pkg *domain.InstalledPackage) error
}

type ConfigStorage interface {
    LoadHaxenConfig(path string) (*domain.HaxenConfig, error)
    SaveHaxenConfig(path string, config *domain.HaxenConfig) error
}
