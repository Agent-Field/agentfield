// agentfield/internal/infrastructure/storage/config.go
package storage

import (
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"gopkg.in/yaml.v3"
	"os"
	"path/filepath"
)

type LocalConfigStorage struct {
	fs interfaces.FileSystemAdapter
}

func NewLocalConfigStorage(fs interfaces.FileSystemAdapter) interfaces.ConfigStorage {
	return &LocalConfigStorage{fs: fs}
}

func (s *LocalConfigStorage) LoadAgentFieldConfig(path string) (*domain.AgentFieldConfig, error) {
	if !s.fs.Exists(path) {
		return &domain.AgentFieldConfig{
			HomeDir:     filepath.Dir(path),
			Environment: make(map[string]string),
			MCP: domain.MCPConfig{
				Servers: []domain.MCPServer{},
			},
		}, nil
	}

	data, err := s.fs.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config domain.AgentFieldConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func (s *LocalConfigStorage) SaveAgentFieldConfig(path string, config *domain.AgentFieldConfig) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	return s.fs.WriteFile(path, data)
}
