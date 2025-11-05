package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/your-org/haxen/control-plane/internal/config"
	"gopkg.in/yaml.v3"
)

// MCPManager handles MCP server lifecycle and management using simplified command-based architecture
type MCPManager struct {
	projectDir     string
	processes      map[string]*MCPProcess
	template       *TemplateProcessor
	processManager *ProcessManager
	verbose        bool
}

// NewMCPManager creates a new MCP manager instance
func NewMCPManager(cfg *config.Config, projectDir string, verbose bool) *MCPManager {
	template := NewTemplateProcessor(projectDir, verbose)
	return &MCPManager{
		projectDir:     projectDir,
		processes:      make(map[string]*MCPProcess),
		template:       template,
		processManager: NewProcessManager(projectDir, template, verbose),
		verbose:        verbose,
	}
}

// Add adds a new MCP server with the given configuration
func (m *MCPManager) Add(config MCPServerConfig) error {
	// Validate configuration
	if config.Alias == "" {
		return fmt.Errorf("alias is required")
	}
	
	if config.URL == "" && config.RunCmd == "" {
		return fmt.Errorf("either URL or run command is required")
	}
	
	if config.URL != "" && config.RunCmd != "" {
		return fmt.Errorf("URL and run command are mutually exclusive")
	}

	if m.verbose {
		fmt.Printf("Adding MCP server: %s\n", config.Alias)
		if config.URL != "" {
			fmt.Printf("Remote URL: %s\n", config.URL)
		} else {
			fmt.Printf("Run command: %s\n", config.RunCmd)
		}
	}

	// Create server directory
	serverDir := filepath.Join(m.projectDir, "packages", "mcp", config.Alias)
	
	// Handle force flag - remove existing directory if it exists
	if config.Force {
		if _, err := os.Stat(serverDir); err == nil {
			if m.verbose {
				fmt.Printf("Force flag enabled, removing existing directory: %s\n", serverDir)
			}
			if err := os.RemoveAll(serverDir); err != nil {
				return fmt.Errorf("failed to remove existing server directory: %w", err)
			}
		}
	}
	
	if err := os.MkdirAll(serverDir, 0755); err != nil {
		return fmt.Errorf("failed to create server directory: %w", err)
	}

	// Execute setup commands if provided
	if len(config.SetupCmds) > 0 {
		if err := m.processManager.ExecuteSetupCommands(config); err != nil {
			return fmt.Errorf("setup commands failed: %w", err)
		}
	}

	// Store configuration
	configPath := filepath.Join(serverDir, "config.json")
	if err := m.saveConfig(configPath, config); err != nil {
		return fmt.Errorf("failed to save configuration: %w", err)
	}

	// Update haxen.yaml
	if err := m.updateHaxenYAML(config); err != nil {
		return fmt.Errorf("failed to update haxen.yaml: %w", err)
	}

	// Attempt to start and discover capabilities
	_, err := m.Start(config.Alias)
	if err != nil {
		if m.verbose {
			fmt.Printf("Warning: failed to start server for capability discovery: %v\n", err)
		}
	} else {
		// Discover capabilities and generate skills
		cd := NewCapabilityDiscovery(nil, m.projectDir)
		capability, err := cd.discoverServerCapability(config.Alias)
		if err != nil {
			if m.verbose {
				fmt.Printf("Warning: failed to discover capabilities: %v\n", err)
			}
		} else if capability != nil && m.verbose {
			fmt.Printf("Updated config with transport type: %s\n", capability.Transport)
		}
		
		// Stop the server after discovery (it will be started again when needed)
		if err := m.Stop(config.Alias); err != nil {
			if m.verbose {
				fmt.Printf("Warning: failed to stop server after discovery: %v\n", err)
			}
		}
	}

	if m.verbose {
		fmt.Printf("Successfully added MCP server: %s\n", config.Alias)
	}

	return nil
}

// Start starts an MCP server
func (m *MCPManager) Start(alias string) (*MCPProcess, error) {
	// Check if already running
	if process, exists := m.processes[alias]; exists {
		if m.processManager.IsProcessRunning(process) {
			return process, fmt.Errorf("MCP server %s is already running", alias)
		}
		// Remove stale process reference
		delete(m.processes, alias)
	}

	// Load configuration
	config, err := m.loadConfig(alias)
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}

	if m.verbose {
		fmt.Printf("Starting MCP server: %s\n", alias)
	}

	var process *MCPProcess

	if config.URL != "" {
		// Remote MCP - just validate connectivity
		process, err = m.processManager.ConnectRemoteMCP(*config)
	} else {
		// Local MCP - execute run command
		process, err = m.processManager.StartLocalMCP(*config)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to start MCP server: %w", err)
	}

	// Track the process
	m.processes[alias] = process

	// Update configuration with runtime info
	now := time.Now()
	config.PID = process.Config.PID
	config.Status = string(StatusRunning)
	config.StartedAt = &now
	
	if err := m.saveConfig(filepath.Join(m.projectDir, "packages", "mcp", alias, "config.json"), *config); err != nil {
		if m.verbose {
			fmt.Printf("Warning: failed to update configuration: %v\n", err)
		}
	}

	if m.verbose {
		fmt.Printf("Successfully started MCP server: %s\n", alias)
	}

	return process, nil
}

// Stop stops an MCP server
func (m *MCPManager) Stop(alias string) error {
	process, exists := m.processes[alias]
	if !exists {
		return fmt.Errorf("MCP server %s is not running", alias)
	}

	if m.verbose {
		fmt.Printf("Stopping MCP server: %s\n", alias)
	}

	if err := m.processManager.StopProcess(process); err != nil {
		return fmt.Errorf("failed to stop MCP server: %w", err)
	}

	// Remove from tracking
	delete(m.processes, alias)

	// Update configuration
	config, err := m.loadConfig(alias)
	if err == nil {
		config.PID = 0
		config.Status = string(StatusStopped)
		config.StartedAt = nil
		m.saveConfig(filepath.Join(m.projectDir, "packages", "mcp", alias, "config.json"), *config)
	}

	if m.verbose {
		fmt.Printf("Successfully stopped MCP server: %s\n", alias)
	}

	return nil
}

// Remove removes an MCP server
func (m *MCPManager) Remove(alias string) error {
	// Stop if running
	if _, exists := m.processes[alias]; exists {
		if err := m.Stop(alias); err != nil {
			return fmt.Errorf("failed to stop server before removal: %w", err)
		}
	}

	serverDir := filepath.Join(m.projectDir, "packages", "mcp", alias)

	// Check if server exists
	if _, err := os.Stat(serverDir); os.IsNotExist(err) {
		return fmt.Errorf("MCP server %s not found", alias)
	}

	if m.verbose {
		fmt.Printf("Removing MCP server: %s\n", alias)
	}

	// Remove directory
	if err := os.RemoveAll(serverDir); err != nil {
		return fmt.Errorf("failed to remove server directory: %w", err)
	}

	// Update haxen.yaml
	if err := m.removeMCPFromHaxenYAML(alias); err != nil {
		return fmt.Errorf("failed to update haxen.yaml: %w", err)
	}

	if m.verbose {
		fmt.Printf("Successfully removed MCP server: %s\n", alias)
	}

	return nil
}

// Status returns the status of all MCP servers
func (m *MCPManager) Status() ([]MCPServerInfo, error) {
	mcpDir := filepath.Join(m.projectDir, "packages", "mcp")

	var servers []MCPServerInfo

	// Check if MCP directory exists
	if _, err := os.Stat(mcpDir); os.IsNotExist(err) {
		return servers, nil
	}

	// Read all MCP server directories
	entries, err := os.ReadDir(mcpDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read MCP directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		alias := entry.Name()
		serverInfo, err := m.getServerInfo(alias)
		if err != nil {
			if m.verbose {
				fmt.Printf("Warning: failed to get info for server %s: %v\n", alias, err)
			}
			continue
		}

		servers = append(servers, *serverInfo)
	}

	return servers, nil
}

// GetProcess returns the process for a given alias
func (m *MCPManager) GetProcess(alias string) (*MCPProcess, error) {
	process, exists := m.processes[alias]
	if !exists {
		return nil, fmt.Errorf("MCP server %s is not running", alias)
	}
	return process, nil
}

// List returns a list of all installed MCP servers
func (m *MCPManager) List() ([]MCPServerInfo, error) {
	return m.Status()
}

// Restart restarts an MCP server
func (m *MCPManager) Restart(alias string) error {
	// Stop if running
	if _, exists := m.processes[alias]; exists {
		if err := m.Stop(alias); err != nil {
			return fmt.Errorf("failed to stop server: %w", err)
		}
	}

	// Start again
	_, err := m.Start(alias)
	return err
}

// Logs returns a reader for the server logs
func (m *MCPManager) Logs(alias string, follow bool, lines int) (io.ReadCloser, error) {
	logFile := filepath.Join(m.projectDir, "packages", "mcp", alias, fmt.Sprintf("%s.log", alias))
	
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("log file not found for server %s", alias)
	}

	// For now, just return the file reader
	// TODO: Implement follow and lines functionality
	file, err := os.Open(logFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	return file, nil
}

// Helper methods

// saveConfig saves configuration to a file
func (m *MCPManager) saveConfig(path string, config MCPServerConfig) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write configuration: %w", err)
	}

	return nil
}

// loadConfig loads configuration from a file
func (m *MCPManager) loadConfig(alias string) (*MCPServerConfig, error) {
	configPath := filepath.Join(m.projectDir, "packages", "mcp", alias, "config.json")

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read configuration: %w", err)
	}

	var config MCPServerConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse configuration: %w", err)
	}

	return &config, nil
}

// getServerInfo gets detailed server information for status reporting
func (m *MCPManager) getServerInfo(alias string) (*MCPServerInfo, error) {
	config, err := m.loadConfig(alias)
	if err != nil {
		return nil, err
	}

	info := &MCPServerInfo{
		Alias:       config.Alias,
		Description: config.Description,
		Status:      StatusStopped,
		URL:         config.URL,
		RunCmd:      config.RunCmd,
		Port:        config.Port,
		Version:     config.Version,
		Tags:        config.Tags,
	}

	// Check if process is running
	if process, exists := m.processes[alias]; exists && m.processManager.IsProcessRunning(process) {
		info.Status = StatusRunning
		info.PID = process.Config.PID
		info.StartedAt = config.StartedAt
	}

	return info, nil
}

// updateHaxenYAML updates the haxen.yaml file with the new MCP server
func (m *MCPManager) updateHaxenYAML(config MCPServerConfig) error {
	haxenYAMLPath := filepath.Join(m.projectDir, "haxen.yaml")

	// Read existing haxen.yaml
	data, err := os.ReadFile(haxenYAMLPath)
	if err != nil {
		return fmt.Errorf("failed to read haxen.yaml: %w", err)
	}

	// Parse YAML
	var yamlConfig map[string]interface{}
	if err := yaml.Unmarshal(data, &yamlConfig); err != nil {
		return fmt.Errorf("failed to parse haxen.yaml: %w", err)
	}

	// Ensure dependencies section exists
	if yamlConfig["dependencies"] == nil {
		yamlConfig["dependencies"] = make(map[string]interface{})
	}

	dependencies := yamlConfig["dependencies"].(map[string]interface{})

	// Ensure mcp_servers section exists
	if dependencies["mcp_servers"] == nil {
		dependencies["mcp_servers"] = make(map[string]interface{})
	}

	mcpServers := dependencies["mcp_servers"].(map[string]interface{})

	// Build server configuration
	serverConfig := make(map[string]interface{})

	if config.URL != "" {
		serverConfig["url"] = config.URL
	}

	if config.RunCmd != "" {
		serverConfig["run"] = config.RunCmd
	}

	if len(config.SetupCmds) > 0 {
		serverConfig["setup"] = config.SetupCmds
	}

	if config.WorkingDir != "" {
		serverConfig["working_dir"] = config.WorkingDir
	}

	if len(config.Env) > 0 {
		serverConfig["environment"] = config.Env
	}

	if config.HealthCheck != "" {
		serverConfig["health_check"] = config.HealthCheck
	}

	if config.Description != "" {
		serverConfig["description"] = config.Description
	}

	if config.Version != "" {
		serverConfig["version"] = config.Version
	}

	if len(config.Tags) > 0 {
		serverConfig["tags"] = config.Tags
	}

	// Add the new MCP server
	mcpServers[config.Alias] = serverConfig

	// Write back to file
	updatedData, err := yaml.Marshal(yamlConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal haxen.yaml: %w", err)
	}

	if err := os.WriteFile(haxenYAMLPath, updatedData, 0644); err != nil {
		return fmt.Errorf("failed to write haxen.yaml: %w", err)
	}

	return nil
}

// removeMCPFromHaxenYAML removes an MCP server from haxen.yaml
func (m *MCPManager) removeMCPFromHaxenYAML(alias string) error {
	haxenYAMLPath := filepath.Join(m.projectDir, "haxen.yaml")

	// Read existing haxen.yaml
	data, err := os.ReadFile(haxenYAMLPath)
	if err != nil {
		return fmt.Errorf("failed to read haxen.yaml: %w", err)
	}

	// Parse YAML
	var config map[string]interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to parse haxen.yaml: %w", err)
	}

	// Navigate to mcp_servers section
	if dependencies, ok := config["dependencies"].(map[string]interface{}); ok {
		if mcpServers, ok := dependencies["mcp_servers"].(map[string]interface{}); ok {
			delete(mcpServers, alias)
		}
	}

	// Write back to file
	updatedData, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal haxen.yaml: %w", err)
	}

	if err := os.WriteFile(haxenYAMLPath, updatedData, 0644); err != nil {
		return fmt.Errorf("failed to write haxen.yaml: %w", err)
	}

	return nil
}

// loadMCPConfigsFromYAML loads MCP configurations from haxen.yaml
func (m *MCPManager) loadMCPConfigsFromYAML() (map[string]MCPServerConfig, error) {
	haxenYAMLPath := filepath.Join(m.projectDir, "haxen.yaml")

	data, err := os.ReadFile(haxenYAMLPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read haxen.yaml: %w", err)
	}

	var config map[string]interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse haxen.yaml: %w", err)
	}

	configs := make(map[string]MCPServerConfig)

	// Navigate to mcp_servers section
	if dependencies, ok := config["dependencies"].(map[string]interface{}); ok {
		if mcpServers, ok := dependencies["mcp_servers"].(map[string]interface{}); ok {
			for alias, serverData := range mcpServers {
				if serverMap, ok := serverData.(map[string]interface{}); ok {
					serverConfig := MCPServerConfig{
						Alias: alias,
					}

					if url, ok := serverMap["url"].(string); ok {
						serverConfig.URL = url
					}
					if runCmd, ok := serverMap["run"].(string); ok {
						serverConfig.RunCmd = runCmd
					}
					if workingDir, ok := serverMap["working_dir"].(string); ok {
						serverConfig.WorkingDir = workingDir
					}
					if description, ok := serverMap["description"].(string); ok {
						serverConfig.Description = description
					}
					if version, ok := serverMap["version"].(string); ok {
						serverConfig.Version = version
					}
					if healthCheck, ok := serverMap["health_check"].(string); ok {
						serverConfig.HealthCheck = healthCheck
					}

					// Parse setup commands
					if setup, ok := serverMap["setup"].([]interface{}); ok {
						for _, cmd := range setup {
							if cmdStr, ok := cmd.(string); ok {
								serverConfig.SetupCmds = append(serverConfig.SetupCmds, cmdStr)
							}
						}
					}

					// Parse tags
					if tags, ok := serverMap["tags"].([]interface{}); ok {
						for _, tag := range tags {
							if tagStr, ok := tag.(string); ok {
								serverConfig.Tags = append(serverConfig.Tags, tagStr)
							}
						}
					}

					// Parse environment variables
					if env, ok := serverMap["environment"].(map[string]interface{}); ok {
						serverConfig.Env = make(map[string]string)
						for key, value := range env {
							if valueStr, ok := value.(string); ok {
								serverConfig.Env[key] = valueStr
							}
						}
					}

					configs[alias] = serverConfig
				}
			}
		}
	}

	return configs, nil
}

// DiscoverCapabilities discovers capabilities for an MCP server
func (m *MCPManager) DiscoverCapabilities(alias string) (*MCPManifest, error) {
	if m.verbose {
		fmt.Printf("Discovering capabilities for MCP server: %s\n", alias)
	}

	// Load configuration
	config, err := m.loadConfig(alias)
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}

	var manifest *MCPManifest

	if config.URL != "" {
		// URL-based MCP
		manifest, err = m.discoverFromURL(*config)
	} else {
		// Local MCP
		manifest, err = m.discoverFromLocalProcess(*config)
	}

	if err != nil {
		return nil, fmt.Errorf("capability discovery failed: %w", err)
	}

	// Cache discovered capabilities
	if err := m.cacheCapabilities(alias, manifest); err != nil {
		if m.verbose {
			fmt.Printf("Warning: failed to cache capabilities: %v\n", err)
		}
	}

	if m.verbose {
		fmt.Printf("Successfully discovered capabilities for %s: %d tools, %d resources\n", 
			alias, len(manifest.Tools), len(manifest.Resources))
		fmt.Printf("Note: MCP skills will be auto-registered by Haxen SDK when agent starts\n")
	}

	return manifest, nil
}

// discoverFromURL discovers capabilities from a remote MCP server URL
func (m *MCPManager) discoverFromURL(config MCPServerConfig) (*MCPManifest, error) {
	if m.verbose {
		fmt.Printf("Discovering capabilities from URL: %s\n", config.URL)
	}

	// Create MCP protocol client
	client := NewMCPProtocolClient(m.verbose)

	// Discover capabilities from the URL
	tools, resources, err := client.DiscoverCapabilitiesFromURL(config.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to discover from URL: %w", err)
	}

	manifest := &MCPManifest{
		Tools:     tools,
		Resources: resources,
		Version:   config.Version,
	}

	return manifest, nil
}

// discoverFromLocalProcess discovers capabilities from a local MCP process
func (m *MCPManager) discoverFromLocalProcess(config MCPServerConfig) (*MCPManifest, error) {
	if m.verbose {
		fmt.Printf("Discovering capabilities from local process: %s\n", config.Alias)
	}

	// Create capability discovery instance
	cd := NewCapabilityDiscovery(nil, m.projectDir) // Pass nil for config since we don't need it

	// Use the proper discovery logic that tries both stdio and HTTP
	tools, resources, err := cd.discoverFromLocalProcess(config.Alias, config)
	if err != nil {
		return nil, fmt.Errorf("failed to discover capabilities: %w", err)
	}

	manifest := &MCPManifest{
		Tools:     tools,
		Resources: resources,
		Version:   config.Version,
	}

	return manifest, nil
}

// connectAndDiscover connects to an MCP server endpoint and discovers capabilities
func (m *MCPManager) connectAndDiscover(endpoint string) (*MCPManifest, error) {
	if m.verbose {
		fmt.Printf("Connecting to MCP server at: %s\n", endpoint)
	}

	// Create MCP protocol client
	client := NewMCPProtocolClient(m.verbose)

	// Discover capabilities from the endpoint
	tools, resources, err := client.DiscoverCapabilitiesFromURL(endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to connect and discover: %w", err)
	}

	manifest := &MCPManifest{
		Tools:     tools,
		Resources: resources,
	}

	return manifest, nil
}

// parseCapabilityResponse parses a raw capability response into an MCPManifest
func (m *MCPManager) parseCapabilityResponse(response []byte) (*MCPManifest, error) {
	var manifest MCPManifest
	if err := json.Unmarshal(response, &manifest); err != nil {
		return nil, fmt.Errorf("failed to parse capability response: %w", err)
	}
	return &manifest, nil
}

// cacheCapabilities caches discovered capabilities to disk
func (m *MCPManager) cacheCapabilities(alias string, manifest *MCPManifest) error {
	serverDir := filepath.Join(m.projectDir, "packages", "mcp", alias)
	capabilitiesPath := filepath.Join(serverDir, "capabilities.json")

	// Add timestamp to cached data
	cachedData := struct {
		*MCPManifest
		UpdatedAt int64 `json:"updated_at"`
	}{
		MCPManifest: manifest,
		UpdatedAt:   time.Now().Unix(),
	}

	data, err := json.MarshalIndent(cachedData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal capabilities: %w", err)
	}

	if err := os.WriteFile(capabilitiesPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write capabilities cache: %w", err)
	}

	return nil
}

// GenerateSkills generates Python skills based on discovered capabilities
func (m *MCPManager) GenerateSkills(alias string, manifest *MCPManifest) error {
	if m.verbose {
		fmt.Printf("Generating skills for MCP server: %s\n", alias)
	}

	// Use the new SkillGenerator instead of the old template-based approach
	generator := NewSkillGenerator(m.projectDir, m.verbose)
	
	result, err := generator.GenerateSkillsForServer(alias)
	if err != nil {
		return fmt.Errorf("failed to generate skills: %w", err)
	}

	if m.verbose {
		if result.Generated {
			fmt.Printf("Generated consolidated skill file: %s (%d tools)\n", result.FilePath, result.ToolCount)
		} else {
			fmt.Printf("Skill generation result: %s\n", result.Message)
		}
	}

	return nil
}
