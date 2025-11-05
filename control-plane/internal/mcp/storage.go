package mcp

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

// ConfigStorage defines the interface for storing and retrieving MCP server configurations.
// This allows for different backend implementations (e.g., YAML file, database).
type ConfigStorage interface {
	// LoadMCPServerConfig retrieves the configuration for a specific MCP server.
	LoadMCPServerConfig(alias string) (*MCPServerConfig, error)
	// SaveMCPServerConfig saves the configuration for a specific MCP server.
	// This should overwrite any existing configuration for the given alias.
	SaveMCPServerConfig(alias string, config *MCPServerConfig) error
	// DeleteMCPServerConfig removes the configuration for a specific MCP server.
	DeleteMCPServerConfig(alias string) error
	// LoadAllMCPServerConfigs retrieves all stored MCP server configurations.
	LoadAllMCPServerConfigs() (map[string]*MCPServerConfig, error)
	// ListMCPServerAliases retrieves a list of all configured MCP server aliases.
	ListMCPServerAliases() ([]string, error)
	// UpdateConfig atomically updates a configuration.
	// The updateFn receives the current config (or nil if it doesn't exist)
	// and should return the new config to be saved.
	// If updateFn returns an error, the transaction is rolled back.
	UpdateConfig(alias string, updateFn func(currentConfig *MCPServerConfig) (*MCPServerConfig, error)) error
}

// YAMLConfigStorage implements ConfigStorage using a YAML file.
// It stores all MCP server configurations in a single agentfield.yaml file
// under the dependencies.mcp_servers key.
type YAMLConfigStorage struct {
	ProjectDir string
	filePath   string
	mu         sync.RWMutex // Protects access to the YAML file
}

// NewYAMLConfigStorage creates a new YAMLConfigStorage.
// projectDir is the root directory of the agentfield project.
func NewYAMLConfigStorage(projectDir string) *YAMLConfigStorage {
	return &YAMLConfigStorage{
		ProjectDir: projectDir,
		filePath:   filepath.Join(projectDir, "agentfield.yaml"),
	}
}

// agentfieldYAML represents the structure of the agentfield.yaml file.
// We only care about the mcp_servers part for this storage.
type agentfieldYAML struct {
	Dependencies struct {
		MCPServers map[string]*MCPServerConfig `yaml:"mcp_servers,omitempty"`
	} `yaml:"dependencies,omitempty"`
	// Other fields in agentfield.yaml are preserved but not directly managed here.
	OtherFields map[string]interface{} `yaml:",inline"`
}

func (s *YAMLConfigStorage) loadAgentFieldYAML() (*agentfieldYAML, error) {
	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			// If agentfield.yaml doesn't exist, return an empty structure
			return &agentfieldYAML{
				Dependencies: struct {
					MCPServers map[string]*MCPServerConfig `yaml:"mcp_servers,omitempty"`
				}{
					MCPServers: make(map[string]*MCPServerConfig),
				},
				OtherFields: make(map[string]interface{}),
			}, nil
		}
		return nil, fmt.Errorf("failed to read agentfield.yaml: %w", err)
	}

	var cfg agentfieldYAML
	// Initialize maps to avoid nil panics if sections are missing
	cfg.Dependencies.MCPServers = make(map[string]*MCPServerConfig)
	cfg.OtherFields = make(map[string]interface{})

	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal agentfield.yaml: %w", err)
	}
	return &cfg, nil
}

func (s *YAMLConfigStorage) saveAgentFieldYAML(cfg *agentfieldYAML) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("failed to marshal agentfield.yaml: %w", err)
	}
	return os.WriteFile(s.filePath, data, 0644)
}

// LoadMCPServerConfig retrieves the configuration for a specific MCP server.
func (s *YAMLConfigStorage) LoadMCPServerConfig(alias string) (*MCPServerConfig, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return nil, err
	}

	serverConfig, ok := cfg.Dependencies.MCPServers[alias]
	if !ok {
		return nil, fmt.Errorf("MCP server config with alias '%s' not found", alias)
	}
	return serverConfig, nil
}

// SaveMCPServerConfig saves the configuration for a specific MCP server.
func (s *YAMLConfigStorage) SaveMCPServerConfig(alias string, config *MCPServerConfig) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return err
	}

	if cfg.Dependencies.MCPServers == nil {
		cfg.Dependencies.MCPServers = make(map[string]*MCPServerConfig)
	}
	cfg.Dependencies.MCPServers[alias] = config

	return s.saveAgentFieldYAML(cfg)
}

// DeleteMCPServerConfig removes the configuration for a specific MCP server.
func (s *YAMLConfigStorage) DeleteMCPServerConfig(alias string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return err
	}

	if _, ok := cfg.Dependencies.MCPServers[alias]; !ok {
		return fmt.Errorf("MCP server config with alias '%s' not found for deletion", alias)
	}
	delete(cfg.Dependencies.MCPServers, alias)

	return s.saveAgentFieldYAML(cfg)
}

// LoadAllMCPServerConfigs retrieves all stored MCP server configurations.
func (s *YAMLConfigStorage) LoadAllMCPServerConfigs() (map[string]*MCPServerConfig, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return nil, err
	}
	// Return a copy to prevent modification of the internal map
	configsCopy := make(map[string]*MCPServerConfig)
	for k, v := range cfg.Dependencies.MCPServers {
		configsCopy[k] = v
	}
	return configsCopy, nil
}

// UpdateConfig atomically updates a configuration.
func (s *YAMLConfigStorage) UpdateConfig(alias string, updateFn func(currentConfig *MCPServerConfig) (*MCPServerConfig, error)) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return err
	}

	currentConfig := cfg.Dependencies.MCPServers[alias] // currentConfig will be nil if not found

	newConfig, err := updateFn(currentConfig)
	if err != nil {
		return fmt.Errorf("update function failed: %w", err) // Rollback: don't save
	}

	if newConfig == nil { // Indicates a desire to delete
		if _, ok := cfg.Dependencies.MCPServers[alias]; ok {
			delete(cfg.Dependencies.MCPServers, alias)
		} else {
			// Nothing to delete, no change needed
			return nil
		}
	} else {
		if cfg.Dependencies.MCPServers == nil {
			cfg.Dependencies.MCPServers = make(map[string]*MCPServerConfig)
		}
		cfg.Dependencies.MCPServers[alias] = newConfig
	}

	return s.saveAgentFieldYAML(cfg)
}

// ListMCPServerAliases retrieves a list of all configured MCP server aliases.
func (s *YAMLConfigStorage) ListMCPServerAliases() ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg, err := s.loadAgentFieldYAML()
	if err != nil {
		return nil, err
	}

	aliases := make([]string, 0, len(cfg.Dependencies.MCPServers))
	for alias := range cfg.Dependencies.MCPServers {
		aliases = append(aliases, alias)
	}
	return aliases, nil
}
