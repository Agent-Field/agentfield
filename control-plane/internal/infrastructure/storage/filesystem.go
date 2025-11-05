// agentfield/internal/infrastructure/storage/filesystem.go
package storage

import (
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"os"
)

type DefaultFileSystemAdapter struct{}

func NewFileSystemAdapter() interfaces.FileSystemAdapter {
	return &DefaultFileSystemAdapter{}
}

func (fs *DefaultFileSystemAdapter) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (fs *DefaultFileSystemAdapter) WriteFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0644)
}

func (fs *DefaultFileSystemAdapter) Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func (fs *DefaultFileSystemAdapter) CreateDirectory(path string) error {
	return os.MkdirAll(path, 0755)
}

func (fs *DefaultFileSystemAdapter) ListDirectory(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names, nil
}
