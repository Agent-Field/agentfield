package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/your-org/haxen/control-plane/internal/config"
)

// MCPCapability represents a discovered MCP server capability
type MCPCapability struct {
	ServerAlias string     `json:"server_alias"`
	ServerName  string     `json:"server_name"`
	Version     string     `json:"version"`
	Tools       []MCPTool  `json:"tools"`
	Resources   []MCPResource `json:"resources"`
	Endpoint    string     `json:"endpoint"`
	Transport   string     `json:"transport"`
}


// CapabilityDiscovery handles MCP server capability discovery
type CapabilityDiscovery struct {
	projectPath string
	Config      *config.Config // Added to pass to factory
}

// NewCapabilityDiscovery creates a new capability discovery instance
func NewCapabilityDiscovery(cfg *config.Config, projectPath string) *CapabilityDiscovery {
	return &CapabilityDiscovery{
		projectPath: projectPath,
		Config:      cfg,
	}
}

// DiscoverCapabilities discovers capabilities from all installed MCP servers
func (cd *CapabilityDiscovery) DiscoverCapabilities() ([]MCPCapability, error) {
	var capabilities []MCPCapability

	// Read MCP servers from packages/mcp directory
	mcpDir := filepath.Join(cd.projectPath, "packages", "mcp")
	if _, err := os.Stat(mcpDir); os.IsNotExist(err) {
		return capabilities, nil // No MCP servers installed
	}

	entries, err := os.ReadDir(mcpDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read MCP directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		serverAlias := entry.Name()
		capability, err := cd.discoverServerCapability(serverAlias)
		if err != nil {
			fmt.Printf("Warning: Failed to discover capabilities for %s: %v\n", serverAlias, err)
			continue
		}

		if capability != nil {
			capabilities = append(capabilities, *capability)
		}
	}

	return capabilities, nil
}

// migrateOldFormat migrates old mcp.json format to new config.json format
func (cd *CapabilityDiscovery) migrateOldFormat(serverDir string) error {
	oldPath := filepath.Join(serverDir, "mcp.json")
	newPath := filepath.Join(serverDir, "config.json")
	
	// Check if new format already exists
	if _, err := os.Stat(newPath); err == nil {
		return nil // Already migrated
	}
	
	// Read old format
	oldData, err := os.ReadFile(oldPath)
	if err != nil {
		return fmt.Errorf("failed to read mcp.json: %w", err)
	}
	
	var oldFormat map[string]interface{}
	if err := json.Unmarshal(oldData, &oldFormat); err != nil {
		return fmt.Errorf("failed to parse mcp.json: %w", err)
	}
	
	// Convert to new format
	newConfig := MCPServerConfig{}
	
	if alias, ok := oldFormat["alias"].(string); ok {
		newConfig.Alias = alias
	}
	
	if startCmd, ok := oldFormat["start_command"].(string); ok {
		newConfig.RunCmd = startCmd
	}
	
	if source, ok := oldFormat["source"].(string); ok {
		// If source looks like a URL, use it as URL, otherwise as run command
		if strings.HasPrefix(source, "http") {
			newConfig.URL = source
		}
	}
	
	if version, ok := oldFormat["version"].(string); ok {
		newConfig.Version = version
	}
	
	if healthCheck, ok := oldFormat["health_check"].(string); ok {
		newConfig.HealthCheck = healthCheck
	}
	
	// Convert environment variables
	if env, ok := oldFormat["env"].(map[string]interface{}); ok {
		newConfig.Env = make(map[string]string)
		for k, v := range env {
			if vStr, ok := v.(string); ok {
				newConfig.Env[k] = vStr
			}
		}
	}
	
	// Convert port if present
	if port, ok := oldFormat["port"].(float64); ok {
		newConfig.Port = int(port)
	}
	
	// Convert install commands to setup commands
	if installCmds, ok := oldFormat["install_commands"].([]interface{}); ok {
		for _, cmd := range installCmds {
			if cmdStr, ok := cmd.(string); ok {
				newConfig.SetupCmds = append(newConfig.SetupCmds, cmdStr)
			}
		}
	}
	
	// Save in new format
	newData, err := json.MarshalIndent(newConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal new config: %w", err)
	}
	
	if err := os.WriteFile(newPath, newData, 0644); err != nil {
		return fmt.Errorf("failed to write config.json: %w", err)
	}
	
	// Remove old file
	if err := os.Remove(oldPath); err != nil {
		fmt.Printf("Warning: failed to remove old mcp.json: %v\n", err)
	}
	
	fmt.Printf("Migrated %s from mcp.json to config.json format\n", filepath.Base(serverDir))
	return nil
}

// discoverServerCapability discovers capabilities for a specific MCP server
func (cd *CapabilityDiscovery) discoverServerCapability(serverAlias string) (*MCPCapability, error) {
	serverDir := filepath.Join(cd.projectPath, "packages", "mcp", serverAlias)
	
	// Try migration first if config.json doesn't exist
	metadataPath := filepath.Join(serverDir, "config.json")
	if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
		if err := cd.migrateOldFormat(serverDir); err != nil {
			return nil, fmt.Errorf("failed to migrate old format: %w", err)
		}
	}
	
	// Read config.json metadata file
	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json (try running: haxen mcp migrate %s): %w", serverAlias, err)
	}

	var metadata MCPServerConfig
	if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}


	// Initialize capability structure
	capability := &MCPCapability{
		ServerAlias: serverAlias,
		ServerName:  metadata.Alias, // Use alias as server name if no explicit name
		Version:     metadata.Version,
		Transport:   "stdio",
		Tools:       []MCPTool{},
		Resources:   []MCPResource{},
	}

	// Set endpoint based on configuration
	if metadata.URL != "" {
		capability.Endpoint = metadata.URL
		capability.Transport = "http"
		capability.ServerName = metadata.URL
	} else if metadata.RunCmd != "" {
		capability.Endpoint = fmt.Sprintf("stdio://%s", metadata.RunCmd)
	}

	// Try to load cached capabilities first
	capabilitiesPath := filepath.Join(serverDir, "capabilities.json")
	if capabilitiesBytes, err := os.ReadFile(capabilitiesPath); err == nil {
		var cachedCapabilities struct {
			Tools     []struct {
				Name        string                 `json:"name"`
				Description string                 `json:"description"`
				InputSchema map[string]interface{} `json:"inputSchema"`
			} `json:"tools"`
			Resources []struct {
				URI         string `json:"uri"`
				Name        string `json:"name"`
				Description string `json:"description"`
				MimeType    string `json:"mimeType,omitempty"`
			} `json:"resources"`
		}
		
		if err := json.Unmarshal(capabilitiesBytes, &cachedCapabilities); err == nil {
			// Convert cached tools to MCPTool format
			capability.Tools = make([]MCPTool, len(cachedCapabilities.Tools))
			for i, tool := range cachedCapabilities.Tools {
				capability.Tools[i] = MCPTool{
					Name:        tool.Name,
					Description: tool.Description,
					InputSchema: tool.InputSchema,
				}
			}
			
			// Convert cached resources to MCPResource format
			capability.Resources = make([]MCPResource, len(cachedCapabilities.Resources))
			for i, resource := range cachedCapabilities.Resources {
				capability.Resources[i] = MCPResource{
					URI:         resource.URI,
					Name:        resource.Name,
					Description: resource.Description,
					MimeType:    resource.MimeType,
				}
			}
			
			// If we have cached capabilities, return them
			if len(capability.Tools) > 0 || len(capability.Resources) > 0 {
				return capability, nil
			}
		}
	}

	// If no cached capabilities or they're empty, try live discovery
	liveTools, liveResources, err := cd.discoverLiveCapabilities(serverAlias, metadata)
	if err != nil {
		// Create a structured error for better error reporting
		discoveryErr := CapabilityDiscoveryError(serverAlias, "live capability discovery failed", err)
		
		// Log the detailed error information
		fmt.Printf("Warning: Failed to discover live capabilities for %s: %v\n", serverAlias, discoveryErr.Error())
		fmt.Printf("Detailed error: %s\n", discoveryErr.DetailedError())
		fmt.Printf("Suggestion: %s\n", discoveryErr.GetSuggestion())
		fmt.Printf("Falling back to static analysis for %s...\n", serverAlias)
		
		// Try static analysis as fallback
		staticTools, staticResources, staticErr := cd.discoverFromStaticAnalysis(filepath.Join(cd.projectPath, "packages", "mcp", serverAlias), metadata)
		if staticErr != nil {
			staticAnalysisErr := CapabilityDiscoveryError(serverAlias, "static analysis fallback failed", staticErr)
			fmt.Printf("Warning: Static analysis also failed for %s: %v\n", serverAlias, staticAnalysisErr.Error())
			fmt.Printf("Detailed error: %s\n", staticAnalysisErr.DetailedError())
			return capability, nil
		}
		
		// Use static analysis results
		liveTools = staticTools
		liveResources = staticResources
	}

	// Update capability with discovered data
	capability.Tools = liveTools
	capability.Resources = liveResources

	// Cache the discovered capabilities
	if len(liveTools) > 0 || len(liveResources) > 0 {
		if err := cd.CacheCapabilities(serverAlias, liveTools, liveResources); err != nil {
			fmt.Printf("Warning: Failed to cache capabilities for %s: %v\n", serverAlias, err)
		}
	}

	// Update config file with detected transport type
	if capability != nil && capability.Transport != "" {
		if err := cd.updateConfigWithTransport(serverAlias, capability.Transport); err != nil {
			fmt.Printf("Warning: failed to update transport in config for %s: %v\n", serverAlias, err)
		}
	}

	return capability, nil
}

// CacheCapabilities saves discovered capabilities to cache
func (cd *CapabilityDiscovery) CacheCapabilities(serverAlias string, tools []MCPTool, resources []MCPResource) error {
	serverDir := filepath.Join(cd.projectPath, "packages", "mcp", serverAlias)
	
	// Create a structure that matches our expected output format
	type CachedTool struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		InputSchema map[string]interface{} `json:"inputSchema"`
	}
	
	type CachedResource struct {
		URI         string `json:"uri"`
		Name        string `json:"name"`
		Description string `json:"description"`
		MimeType    string `json:"mimeType,omitempty"`
	}
	
	// Convert tools to cached format
	cachedTools := make([]CachedTool, len(tools))
	for i, tool := range tools {
		cachedTools[i] = CachedTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}
	
	// Convert resources to cached format
	cachedResources := make([]CachedResource, len(resources))
	for i, resource := range resources {
		cachedResources[i] = CachedResource{
			URI:         resource.URI,
			Name:        resource.Name,
			Description: resource.Description,
			MimeType:    resource.MimeType,
		}
	}
	
	capabilities := struct {
		Tools     []CachedTool     `json:"tools"`
		Resources []CachedResource `json:"resources"`
		UpdatedAt int64            `json:"updated_at"`
	}{
		Tools:     cachedTools,
		Resources: cachedResources,
		UpdatedAt: time.Now().Unix(),
	}

	capabilitiesBytes, err := json.MarshalIndent(capabilities, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal capabilities: %w", err)
	}

	capabilitiesPath := filepath.Join(serverDir, "capabilities.json")
	if err := os.WriteFile(capabilitiesPath, capabilitiesBytes, 0644); err != nil {
		return fmt.Errorf("failed to write capabilities cache: %w", err)
	}

	return nil
}

// GetServerCapability gets capability for a specific server
func (cd *CapabilityDiscovery) GetServerCapability(serverAlias string) (*MCPCapability, error) {
	return cd.discoverServerCapability(serverAlias)
}

// RefreshCapabilities forces refresh of capabilities for all servers
func (cd *CapabilityDiscovery) RefreshCapabilities() error {
	capabilities, err := cd.DiscoverCapabilities()
	if err != nil {
		return err
	}

	fmt.Printf("Discovered capabilities for %d MCP servers:\n", len(capabilities))
	for _, cap := range capabilities {
		fmt.Printf("- %s: %d tools, %d resources\n", cap.ServerAlias, len(cap.Tools), len(cap.Resources))
	}

	return nil
}

// discoverLiveCapabilities attempts to discover capabilities by running the MCP server
func (cd *CapabilityDiscovery) discoverLiveCapabilities(serverAlias string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Attempting live capability discovery for %s...\n", serverAlias)
	
	if metadata.URL != "" {
		// Remote MCP server - discover from URL
		return cd.discoverFromURL(metadata.URL)
	} else if metadata.RunCmd != "" {
		// Local MCP server - start temporarily and discover
		return cd.discoverFromLocalProcess(serverAlias, metadata)
	}
	
	// No URL or run command - fall back to static analysis
	fmt.Printf("No URL or run command for %s, using static analysis\n", serverAlias)
	return cd.discoverFromStaticAnalysis(filepath.Join(cd.projectPath, "packages", "mcp", serverAlias), metadata)
}

// discoverFromURL discovers capabilities from a remote MCP server URL
func (cd *CapabilityDiscovery) discoverFromURL(url string) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Discovering capabilities from URL: %s\n", url)
	
	// TODO: Implement actual HTTP/WebSocket connection to MCP server
	// For now, return empty capabilities
	// This would involve:
	// 1. Connect to the MCP server at the URL
	// 2. Send MCP protocol messages to list tools and resources
	// 3. Parse the responses
	
	return []MCPTool{}, []MCPResource{}, fmt.Errorf("URL-based discovery not yet implemented")
}

// discoverFromLocalProcess discovers capabilities by temporarily starting a local MCP server
func (cd *CapabilityDiscovery) discoverFromLocalProcess(serverAlias string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Discovering capabilities from local process for %s\n", serverAlias)
	
	// Try stdio discovery first (most common for local MCP servers)
	tools, resources, err := cd.tryStdioDiscovery(serverAlias, metadata)
	if err == nil {
		fmt.Printf("Stdio discovery successful for %s: found %d tools, %d resources\n", serverAlias, len(tools), len(resources))
		return tools, resources, nil
	}
	
	fmt.Printf("Stdio discovery failed for %s: %v\n", serverAlias, err)
	fmt.Printf("Trying HTTP discovery as fallback for %s...\n", serverAlias)
	
	// Fallback to HTTP discovery
	tools, resources, err = cd.tryHTTPDiscovery(serverAlias, metadata)
	if err == nil {
		fmt.Printf("HTTP discovery successful for %s: found %d tools, %d resources\n", serverAlias, len(tools), len(resources))
		return tools, resources, nil
	}
	
	fmt.Printf("Both stdio and HTTP discovery failed for %s: %v\n", serverAlias, err)
	fmt.Printf("Falling back to static analysis for %s...\n", serverAlias)
	
	// Fall back to static analysis
	return cd.discoverFromStaticAnalysis(filepath.Join(cd.projectPath, "packages", "mcp", serverAlias), metadata)
}

// tryStdioDiscovery attempts to discover capabilities using stdio transport with timeout
func (cd *CapabilityDiscovery) tryStdioDiscovery(serverAlias string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Attempting stdio discovery for %s\n", serverAlias)
	
	// Create context with timeout for entire discovery operation (60 seconds)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	
	// Create template processor
	template := NewTemplateProcessor(cd.projectPath, false) // Non-verbose for discovery
	
	// Process template variables in the run command
	vars := template.CreateTemplateVars(metadata, 0) // Port 0 for stdio-based servers
	processedCmd, err := template.ProcessCommand(metadata.RunCmd, vars)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to process run command template: %w", err)
	}
	
	// Set working directory
	workingDir := vars.ServerDir
	if metadata.WorkingDir != "" {
		processedWorkingDir, err := template.ProcessCommand(metadata.WorkingDir, vars)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to process working directory: %w", err)
		}
		workingDir = processedWorkingDir
	}
	
	// Create command with context timeout
	cmd := exec.CommandContext(ctx, "sh", "-c", processedCmd)
	cmd.Dir = workingDir
	
	// Set environment variables
	if len(metadata.Env) > 0 {
		processedEnv, err := template.ProcessEnvironment(metadata.Env, vars)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to process environment variables: %w", err)
		}
		
		cmd.Env = append(cmd.Env, os.Environ()...)
		for key, value := range processedEnv {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}
	
	// Create pipes before starting the process
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return nil, nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	
	// Start the process
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		return nil, nil, fmt.Errorf("failed to start MCP server: %w", err)
	}
	
	// Ensure proper cleanup of all resources
	defer func() {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		
		if cmd.Process != nil {
			// Try graceful shutdown first
			if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
				// Force kill if graceful shutdown fails
				cmd.Process.Kill()
			} else {
				// Wait briefly for graceful shutdown
				time.Sleep(1 * time.Second)
				// Check if still running and force kill if needed
				if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
					cmd.Process.Kill()
				}
			}
			cmd.Wait() // Wait for cleanup
		}
	}()
	
	// Monitor stderr for debugging in a separate goroutine
	var stderrOutput strings.Builder
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			stderrOutput.WriteString(line + "\n")
			fmt.Printf("MCP stderr [%s]: %s\n", serverAlias, line)
		}
	}()
	
	// Wait a moment for the server to start
	select {
	case <-time.After(2 * time.Second):
		// Continue with discovery
	case <-ctx.Done():
		return nil, nil, fmt.Errorf("timeout waiting for server to start: %w", ctx.Err())
	}
	
	// Perform discovery with pipes using timeout-aware implementation
	tools, resources, err := cd.performDiscoveryWithPipes(ctx, stdin, stdout, serverAlias)
	if err != nil {
		// Include stderr output in error for debugging
		stderrStr := stderrOutput.String()
		if stderrStr != "" {
			return nil, nil, fmt.Errorf("stdio discovery failed: %w\nStderr output:\n%s", err, stderrStr)
		}
		return nil, nil, fmt.Errorf("stdio discovery failed: %w", err)
	}
	
	return tools, resources, nil
}

// performDiscoveryWithPipes performs MCP discovery using stdin/stdout pipes with timeout
func (cd *CapabilityDiscovery) performDiscoveryWithPipes(ctx context.Context, stdin io.WriteCloser, stdout io.ReadCloser, serverAlias string) ([]MCPTool, []MCPResource, error) {
	// Create JSON encoder/decoder for line-buffered communication
	encoder := json.NewEncoder(stdin)
	scanner := bufio.NewScanner(stdout)
	
	// Helper function to read JSON response with timeout
	readJSONResponse := func(timeout time.Duration) (*MCPResponse, error) {
		responseCtx, responseCancel := context.WithTimeout(ctx, timeout)
		defer responseCancel()
		
		responseChan := make(chan *MCPResponse, 1)
		errorChan := make(chan error, 1)
		
		go func() {
			if scanner.Scan() {
				var response MCPResponse
				if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
					errorChan <- fmt.Errorf("failed to parse JSON response: %w", err)
					return
				}
				responseChan <- &response
			} else {
				if err := scanner.Err(); err != nil {
					errorChan <- fmt.Errorf("scanner error: %w", err)
				} else {
					errorChan <- fmt.Errorf("no response received")
				}
			}
		}()
		
		select {
		case response := <-responseChan:
			return response, nil
		case err := <-errorChan:
			return nil, err
		case <-responseCtx.Done():
			return nil, fmt.Errorf("timeout waiting for response: %w", responseCtx.Err())
		}
	}
	
	// Step 1: Send initialize request (15 second timeout)
	fmt.Printf("Sending initialize request to %s...\n", serverAlias)
	initRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: InitializeParams{
			ProtocolVersion: "2024-11-05",
			Capabilities: map[string]interface{}{
				"roots": map[string]interface{}{
					"listChanged": true,
				},
			},
			ClientInfo: ClientInfo{
				Name:    "haxen-mcp-client",
				Version: "1.0.0",
			},
		},
	}
	
	if err := encoder.Encode(initRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send initialize request: %w", err)
	}
	
	// Read initialize response
	initResponse, err := readJSONResponse(15 * time.Second)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read initialize response: %w", err)
	}
	
	if initResponse.Error != nil {
		return nil, nil, fmt.Errorf("initialize failed: %s", initResponse.Error.Message)
	}
	
	fmt.Printf("Initialize successful for %s, sending initialized notification...\n", serverAlias)
	
	// Step 2: Send initialized notification (must be a notification, not a request - no ID field)
	initializedNotification := MCPNotification{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
		Params:  map[string]interface{}{},
	}
	
	if err := encoder.Encode(initializedNotification); err != nil {
		return nil, nil, fmt.Errorf("failed to send initialized notification: %w", err)
	}
	
	// Step 3: Request tools list (10 second timeout)
	fmt.Printf("Requesting tools list from %s...\n", serverAlias)
	toolsRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
	}
	
	if err := encoder.Encode(toolsRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send tools/list request: %w", err)
	}
	
	// Read tools response
	toolsResponse, err := readJSONResponse(10 * time.Second)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read tools response: %w", err)
	}
	
	if toolsResponse.Error != nil {
		return nil, nil, fmt.Errorf("tools/list failed: %s", toolsResponse.Error.Message)
	}
	
	// Parse tools from response
	tools, err := cd.parseToolsResponse(toolsResponse.Result)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse tools response: %w", err)
	}
	
	fmt.Printf("Successfully discovered %d tools from %s\n", len(tools), serverAlias)
	
	// Step 4: Request resources list (5 second timeout)
	fmt.Printf("Requesting resources list from %s...\n", serverAlias)
	resourcesRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "resources/list",
	}
	
	if err := encoder.Encode(resourcesRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send resources/list request: %w", err)
	}
	
	// Read resources response
	resourcesResponse, err := readJSONResponse(5 * time.Second)
	if err != nil {
		// Resources might not be supported, that's okay
		fmt.Printf("Resources not supported or failed to read response from %s: %v\n", serverAlias, err)
		return tools, []MCPResource{}, nil
	}
	
	if resourcesResponse.Error != nil {
		// Resources might not be supported, that's okay
		fmt.Printf("Resources not supported by %s: %s\n", serverAlias, resourcesResponse.Error.Message)
		return tools, []MCPResource{}, nil
	}
	
	// Parse resources from response
	resources, err := cd.parseResourcesResponse(resourcesResponse.Result)
	if err != nil {
		fmt.Printf("Failed to parse resources response from %s: %v\n", serverAlias, err)
		return tools, []MCPResource{}, nil
	}
	
	fmt.Printf("Successfully discovered %d resources from %s\n", len(resources), serverAlias)
	
	return tools, resources, nil
}

// parseToolsResponse parses the tools/list response
func (cd *CapabilityDiscovery) parseToolsResponse(result json.RawMessage) ([]MCPTool, error) {
	var resultMap map[string]interface{}
	if err := json.Unmarshal(result, &resultMap); err != nil {
		return nil, fmt.Errorf("invalid tools response format: %w", err)
	}

	toolsInterface, exists := resultMap["tools"]
	if !exists {
		return []MCPTool{}, nil
	}

	toolsList, ok := toolsInterface.([]interface{})
	if !ok {
		return nil, fmt.Errorf("tools is not an array")
	}

	var tools []MCPTool
	for _, toolInterface := range toolsList {
		toolMap, ok := toolInterface.(map[string]interface{})
		if !ok {
			continue
		}

		tool := MCPTool{}

		if name, ok := toolMap["name"].(string); ok {
			tool.Name = name
		}

		if description, ok := toolMap["description"].(string); ok {
			tool.Description = description
		}

		if inputSchema, ok := toolMap["inputSchema"].(map[string]interface{}); ok {
			tool.InputSchema = inputSchema
		}

		tools = append(tools, tool)
	}

	return tools, nil
}

// parseResourcesResponse parses the resources/list response
func (cd *CapabilityDiscovery) parseResourcesResponse(result json.RawMessage) ([]MCPResource, error) {
	var resultMap map[string]interface{}
	if err := json.Unmarshal(result, &resultMap); err != nil {
		return nil, fmt.Errorf("invalid resources response format: %w", err)
	}

	resourcesInterface, exists := resultMap["resources"]
	if !exists {
		return []MCPResource{}, nil
	}

	resourcesList, ok := resourcesInterface.([]interface{})
	if !ok {
		return nil, fmt.Errorf("resources is not an array")
	}

	var resources []MCPResource
	for _, resourceInterface := range resourcesList {
		resourceMap, ok := resourceInterface.(map[string]interface{})
		if !ok {
			continue
		}

		resource := MCPResource{}

		if uri, ok := resourceMap["uri"].(string); ok {
			resource.URI = uri
		}

		if name, ok := resourceMap["name"].(string); ok {
			resource.Name = name
		}

		if description, ok := resourceMap["description"].(string); ok {
			resource.Description = description
		}

		if mimeType, ok := resourceMap["mimeType"].(string); ok {
			resource.MimeType = mimeType
		}

		resources = append(resources, resource)
	}

	return resources, nil
}

// tryHTTPDiscovery attempts to discover capabilities using HTTP transport
func (cd *CapabilityDiscovery) tryHTTPDiscovery(serverAlias string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Attempting HTTP discovery for %s\n", serverAlias)
	
	// Create a temporary manager and process manager for discovery
	template := NewTemplateProcessor(cd.projectPath, false) // Non-verbose for discovery
	processManager := NewProcessManager(cd.projectPath, template, false)
	
	// Start the MCP server temporarily with port assignment
	process, err := processManager.StartLocalMCP(metadata)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to start MCP server for HTTP discovery: %w", err)
	}
	
	// Ensure we clean up the process
	defer func() {
		if err := processManager.StopProcess(process); err != nil {
			fmt.Printf("Warning: failed to stop HTTP discovery process for %s: %v\n", serverAlias, err)
		}
	}()
	
	// Wait a moment for the server to fully start
	time.Sleep(2 * time.Second)
	
	// Connect and discover capabilities via HTTP
	tools, resources, err := cd.connectAndDiscover(process.Config.Port)
	if err != nil {
		return nil, nil, fmt.Errorf("HTTP discovery failed: %w", err)
	}
	
	return tools, resources, nil
}

// connectAndDiscover connects to a running MCP server and discovers its capabilities
func (cd *CapabilityDiscovery) connectAndDiscover(port int) ([]MCPTool, []MCPResource, error) {
	fmt.Printf("Connecting to MCP server on port %d for capability discovery\n", port)
	
	// Create MCP protocol client
	client := NewMCPProtocolClient(false) // Use non-verbose mode for discovery
	
	// Try to discover capabilities via HTTP
	tools, resources, err := client.discoverFromHTTP(port)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to discover capabilities via HTTP: %w", err)
	}
	
	return tools, resources, nil
}

// discoverFromStaticAnalysis attempts static analysis as fallback
func (cd *CapabilityDiscovery) discoverFromStaticAnalysis(serverDir string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	// For Node.js servers, try to run the server and get capabilities
	// Check URL or run command for hints about the server type
	urlOrCmd := metadata.URL
	if metadata.RunCmd != "" {
		urlOrCmd = metadata.RunCmd
	}
	
	if strings.Contains(urlOrCmd, "node") || strings.Contains(urlOrCmd, "npm") || strings.Contains(urlOrCmd, "npx") {
		return cd.discoverNodeJSCapabilities(serverDir, metadata)
	}
	
	// For Python servers
	if strings.Contains(urlOrCmd, "python") || strings.Contains(urlOrCmd, "pip") || strings.Contains(urlOrCmd, "uvx") {
		return cd.discoverPythonCapabilities(serverDir, metadata)
	}
	
	// Default: try to parse from package.json or other metadata
	return cd.discoverFromMetadata(serverDir, metadata)
}

// discoverNodeJSCapabilities discovers capabilities from a Node.js MCP server
func (cd *CapabilityDiscovery) discoverNodeJSCapabilities(serverDir string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	// Try to read from package.json first (in root directory for GitHub cloned servers)
	packageJSONPath := filepath.Join(serverDir, "package.json")
	if _, err := os.Stat(packageJSONPath); err == nil {
		return cd.parseNodeJSPackage(packageJSONPath)
	}
	
	// Try to find main server file and parse it
	serverFiles := []string{
		// GitHub cloned MCP servers - files are in root directory
		filepath.Join(serverDir, "build", "index.js"),    // Built TypeScript (most common)
		filepath.Join(serverDir, "dist", "index.js"),     // Alternative build dir
		filepath.Join(serverDir, "src", "index.js"),      // Source JS
		filepath.Join(serverDir, "src", "index.ts"),      // Source TypeScript
		filepath.Join(serverDir, "index.js"),             // Root JS
		filepath.Join(serverDir, "index.ts"),             // Root TypeScript
		// Legacy server subdirectory support (keep for backward compatibility)
		filepath.Join(serverDir, "server", "index.js"),
		filepath.Join(serverDir, "server", "src", "index.js"),
		filepath.Join(serverDir, "server", "dist", "index.js"),
		filepath.Join(serverDir, "server", "index.ts"),
		filepath.Join(serverDir, "server", "src", "index.ts"),
		filepath.Join(serverDir, "server", "dist", "index.ts"),
	}
	
	for _, serverFile := range serverFiles {
		if _, err := os.Stat(serverFile); err == nil {
			fmt.Printf("Debug: Found server file: %s\n", serverFile)
			return cd.parseNodeJSServerFile(serverFile)
		} else {
			fmt.Printf("Debug: Checked but not found: %s\n", serverFile)
		}
	}
	
	return []MCPTool{}, []MCPResource{}, fmt.Errorf("no server files found")
}

// discoverPythonCapabilities discovers capabilities from a Python MCP server
func (cd *CapabilityDiscovery) discoverPythonCapabilities(serverDir string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	// For Python servers, try to find and parse the main module
	pythonFiles := []string{
		// GitHub cloned Python MCP servers - files are in root directory
		filepath.Join(serverDir, "src", "__main__.py"),       // Source directory
		filepath.Join(serverDir, "src", "main.py"),
		filepath.Join(serverDir, "src", "server.py"),
		filepath.Join(serverDir, "__main__.py"),              // Root directory
		filepath.Join(serverDir, "main.py"),
		filepath.Join(serverDir, "server.py"),
		filepath.Join(serverDir, "app.py"),                   // Common Python entry
		// Legacy server subdirectory support
		filepath.Join(serverDir, "server", "__main__.py"),
		filepath.Join(serverDir, "server", "main.py"),
		filepath.Join(serverDir, "server", "server.py"),
	}
	
	for _, pythonFile := range pythonFiles {
		if _, err := os.Stat(pythonFile); err == nil {
			fmt.Printf("Debug: Found Python server file: %s\n", pythonFile)
			return cd.parsePythonServerFile(pythonFile)
		} else {
			fmt.Printf("Debug: Checked but not found: %s\n", pythonFile)
		}
	}
	
	return []MCPTool{}, []MCPResource{}, fmt.Errorf("no Python server files found")
}

// discoverFromMetadata tries to discover capabilities from metadata files
func (cd *CapabilityDiscovery) discoverFromMetadata(serverDir string, metadata MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	// Check for a manifest or capabilities file
	manifestFiles := []string{
		filepath.Join(serverDir, "manifest.json"),
		filepath.Join(serverDir, "capabilities.json"),
		filepath.Join(serverDir, "server", "manifest.json"),
	}
	
	for _, manifestFile := range manifestFiles {
		if _, err := os.Stat(manifestFile); err == nil {
			return cd.parseManifestFile(manifestFile)
		}
	}
	
	return []MCPTool{}, []MCPResource{}, nil
}

// parseNodeJSPackage parses package.json to extract tool information
func (cd *CapabilityDiscovery) parseNodeJSPackage(packagePath string) ([]MCPTool, []MCPResource, error) {
	data, err := os.ReadFile(packagePath)
	if err != nil {
		return nil, nil, err
	}
	
	var pkg map[string]interface{}
	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil, nil, err
	}
	
	// Look for MCP-specific metadata in package.json
	if mcpData, ok := pkg["mcp"].(map[string]interface{}); ok {
		return cd.parseMCPMetadata(mcpData)
	}
	
	return []MCPTool{}, []MCPResource{}, nil
}

// parseNodeJSServerFile parses a Node.js server file to extract tool definitions
func (cd *CapabilityDiscovery) parseNodeJSServerFile(serverFile string) ([]MCPTool, []MCPResource, error) {
	data, err := os.ReadFile(serverFile)
	if err != nil {
		return nil, nil, err
	}
	
	content := string(data)
	tools := []MCPTool{}
	resources := []MCPResource{}
	
	fmt.Printf("Debug: Parsing server file: %s\n", serverFile)
	fmt.Printf("Debug: File contains ListToolsRequestSchema: %v\n", strings.Contains(content, "ListToolsRequestSchema"))
	fmt.Printf("Debug: File contains 'tools: [': %v\n", strings.Contains(content, "tools: ["))
	
	// Look for tool definitions in the tools array within ListToolsRequestSchema handler
	if strings.Contains(content, "ListToolsRequestSchema") {
		tools = cd.extractToolsFromContent(content)
		fmt.Printf("Debug: Extracted %d tools\n", len(tools))
		for i, tool := range tools {
			fmt.Printf("Debug: Tool %d: %s - %s\n", i+1, tool.Name, tool.Description)
		}
	}
	
	// Look for resource definitions in the resources array within ListResourcesRequestSchema handler
	if strings.Contains(content, "ListResourcesRequestSchema") {
		resources = cd.extractResourcesFromContent(content)
		fmt.Printf("Debug: Extracted %d resources\n", len(resources))
	}
	
	return tools, resources, nil
}

// parsePythonServerFile parses a Python server file to extract tool definitions
func (cd *CapabilityDiscovery) parsePythonServerFile(pythonFile string) ([]MCPTool, []MCPResource, error) {
	data, err := os.ReadFile(pythonFile)
	if err != nil {
		return nil, nil, err
	}
	
	content := string(data)
	tools := []MCPTool{}
	resources := []MCPResource{}
	
	// Simple pattern matching for Python MCP patterns
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		
		// Look for @server.call_tool decorators or similar patterns
		if strings.Contains(line, "@server.call_tool") || strings.Contains(line, "def ") && strings.Contains(line, "_tool") {
			if toolName := cd.extractToolNameFromPython(line); toolName != "" {
				tools = append(tools, MCPTool{
					Name:        toolName,
					Description: fmt.Sprintf("Tool discovered from %s", filepath.Base(pythonFile)),
				})
			}
		}
		
		// Look for resource handlers
		if strings.Contains(line, "@server.list_resources") || strings.Contains(line, "def ") && strings.Contains(line, "_resource") {
			if resourceName := cd.extractResourceNameFromPython(line); resourceName != "" {
				resources = append(resources, MCPResource{
					Name:        resourceName,
					Description: fmt.Sprintf("Resource discovered from %s", filepath.Base(pythonFile)),
				})
			}
		}
	}
	
	return tools, resources, nil
}

// parseManifestFile parses a manifest file for capabilities
func (cd *CapabilityDiscovery) parseManifestFile(manifestFile string) ([]MCPTool, []MCPResource, error) {
	data, err := os.ReadFile(manifestFile)
	if err != nil {
		return nil, nil, err
	}
	
	var manifest struct {
		Tools     []MCPTool     `json:"tools"`
		Resources []MCPResource `json:"resources"`
	}
	
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, nil, err
	}
	
	return manifest.Tools, manifest.Resources, nil
}

// parseMCPMetadata parses MCP metadata from package.json
func (cd *CapabilityDiscovery) parseMCPMetadata(mcpData map[string]interface{}) ([]MCPTool, []MCPResource, error) {
	tools := []MCPTool{}
	resources := []MCPResource{}
	
	if toolsData, ok := mcpData["tools"].([]interface{}); ok {
		for _, toolData := range toolsData {
			if toolMap, ok := toolData.(map[string]interface{}); ok {
				tool := MCPTool{}
				if name, ok := toolMap["name"].(string); ok {
					tool.Name = name
				}
				if desc, ok := toolMap["description"].(string); ok {
					tool.Description = desc
				}
				tools = append(tools, tool)
			}
		}
	}
	
	if resourcesData, ok := mcpData["resources"].([]interface{}); ok {
		for _, resourceData := range resourcesData {
			if resourceMap, ok := resourceData.(map[string]interface{}); ok {
				resource := MCPResource{}
				if name, ok := resourceMap["name"].(string); ok {
					resource.Name = name
				}
				if desc, ok := resourceMap["description"].(string); ok {
					resource.Description = desc
				}
				resources = append(resources, resource)
			}
		}
	}
	
	return tools, resources, nil
}

// Helper functions for extracting names from code patterns
func (cd *CapabilityDiscovery) extractToolNameFromJS(line string) string {
	// Simple extraction - could be enhanced
	if strings.Contains(line, "CallToolRequestSchema") {
		// Try to find tool name in the handler
		return "generic_tool"
	}
	return ""
}

func (cd *CapabilityDiscovery) extractResourceNameFromJS(line string) string {
	// Simple extraction - could be enhanced
	if strings.Contains(line, "ListResourcesRequestSchema") {
		return "generic_resource"
	}
	return ""
}

func (cd *CapabilityDiscovery) extractToolNameFromPython(line string) string {
	// Extract function name from Python def statements
	if strings.Contains(line, "def ") {
		parts := strings.Split(line, "def ")
		if len(parts) > 1 {
			funcPart := strings.Split(parts[1], "(")[0]
			return strings.TrimSpace(funcPart)
		}
	}
	return ""
}

func (cd *CapabilityDiscovery) extractResourceNameFromPython(line string) string {
	// Extract function name from Python def statements
	if strings.Contains(line, "def ") {
		parts := strings.Split(line, "def ")
		if len(parts) > 1 {
			funcPart := strings.Split(parts[1], "(")[0]
			return strings.TrimSpace(funcPart)
		}
	}
	return ""
}

// extractToolsFromContent extracts tool definitions from JavaScript/TypeScript content
func (cd *CapabilityDiscovery) extractToolsFromContent(content string) []MCPTool {
	tools := []MCPTool{}
	
	// Look for tools array in the ListToolsRequestSchema handler
	toolsStart := strings.Index(content, "tools: [")
	if toolsStart == -1 {
		return tools
	}
	
	// Find the end of the tools array
	remaining := content[toolsStart:]
	bracketCount := 0
	toolsEnd := -1
	
	for i, char := range remaining {
		if char == '[' {
			bracketCount++
		} else if char == ']' {
			bracketCount--
			if bracketCount == 0 {
				toolsEnd = i
				break
			}
		}
	}
	
	if toolsEnd == -1 {
		return tools
	}
	
	toolsSection := remaining[:toolsEnd+1]
	
	// Extract individual tool objects
	toolObjects := cd.extractObjectsFromArray(toolsSection)
	
	for _, toolObj := range toolObjects {
		tool := cd.parseToolObject(toolObj)
		if tool.Name != "" {
			tools = append(tools, tool)
		}
	}
	
	return tools
}

// extractResourcesFromContent extracts resource definitions from JavaScript/TypeScript content
func (cd *CapabilityDiscovery) extractResourcesFromContent(content string) []MCPResource {
	resources := []MCPResource{}
	
	// Look for resources array in the ListResourcesRequestSchema handler
	resourcesStart := strings.Index(content, "resources: [")
	if resourcesStart == -1 {
		return resources
	}
	
	// Find the end of the resources array
	remaining := content[resourcesStart:]
	bracketCount := 0
	resourcesEnd := -1
	
	for i, char := range remaining {
		if char == '[' {
			bracketCount++
		} else if char == ']' {
			bracketCount--
			if bracketCount == 0 {
				resourcesEnd = i
				break
			}
		}
	}
	
	if resourcesEnd == -1 {
		return resources
	}
	
	resourcesSection := remaining[:resourcesEnd+1]
	
	// Extract individual resource objects
	resourceObjects := cd.extractObjectsFromArray(resourcesSection)
	
	for _, resourceObj := range resourceObjects {
		resource := cd.parseResourceObject(resourceObj)
		if resource.Name != "" {
			resources = append(resources, resource)
		}
	}
	
	return resources
}

// extractObjectsFromArray extracts individual objects from a JavaScript array string
func (cd *CapabilityDiscovery) extractObjectsFromArray(arrayContent string) []string {
	objects := []string{}
	
	// Simple extraction - look for objects between { and }
	braceCount := 0
	objectStart := -1
	
	for i, char := range arrayContent {
		if char == '{' {
			if braceCount == 0 {
				objectStart = i
			}
			braceCount++
		} else if char == '}' {
			braceCount--
			if braceCount == 0 && objectStart != -1 {
				objects = append(objects, arrayContent[objectStart:i+1])
				objectStart = -1
			}
		}
	}
	
	return objects
}

// parseToolObject parses a JavaScript tool object string to extract tool information
func (cd *CapabilityDiscovery) parseToolObject(objectStr string) MCPTool {
	tool := MCPTool{}
	
	// Extract name
	if nameMatch := cd.extractStringValue(objectStr, "name"); nameMatch != "" {
		tool.Name = nameMatch
	}
	
	// Extract description
	if descMatch := cd.extractStringValue(objectStr, "description"); descMatch != "" {
		tool.Description = descMatch
	}
	
	return tool
}

// parseResourceObject parses a JavaScript resource object string to extract resource information
func (cd *CapabilityDiscovery) parseResourceObject(objectStr string) MCPResource {
	resource := MCPResource{}
	
	// Extract name
	if nameMatch := cd.extractStringValue(objectStr, "name"); nameMatch != "" {
		resource.Name = nameMatch
	}
	
	// Extract description
	if descMatch := cd.extractStringValue(objectStr, "description"); descMatch != "" {
		resource.Description = descMatch
	}
	
	return resource
}

// extractStringValue extracts a string value for a given key from a JavaScript object string
func (cd *CapabilityDiscovery) extractStringValue(objectStr, key string) string {
	// Look for key: "value" or key: 'value'
	patterns := []string{
		key + `: "`,
		key + `: "`,
		key + `: '`,
		key + `:'`,
	}
	
	for _, pattern := range patterns {
		startIdx := strings.Index(objectStr, pattern)
		if startIdx != -1 {
			valueStart := startIdx + len(pattern)
			quote := objectStr[valueStart-1]
			
			// Find the closing quote
			for i := valueStart; i < len(objectStr); i++ {
				if objectStr[i] == quote && (i == 0 || objectStr[i-1] != '\\') {
					return objectStr[valueStart:i]
				}
			}
		}
	}
	
	return ""
}

// updateConfigWithTransport updates the server config file with detected transport type
func (cd *CapabilityDiscovery) updateConfigWithTransport(serverAlias, transport string) error {
	configPath := filepath.Join(cd.projectPath, "packages", "mcp", serverAlias, "config.json")
	
	// Read existing config
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}
	
	// Parse as raw JSON to preserve any extra fields
	var configMap map[string]interface{}
	if err := json.Unmarshal(configData, &configMap); err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}
	
	// Update transport field
	configMap["transport"] = transport
	
	// Save updated config with proper formatting
	updatedData, err := json.MarshalIndent(configMap, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	
	if err := os.WriteFile(configPath, updatedData, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}
	
	return nil
}
