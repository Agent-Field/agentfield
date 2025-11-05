package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"
)

// StdioMCPClient handles communication with stdio-based MCP servers
type StdioMCPClient struct {
	verbose bool
}

// NewStdioMCPClient creates a new stdio MCP client
func NewStdioMCPClient(verbose bool) *StdioMCPClient {
	return &StdioMCPClient{
		verbose: verbose,
	}
}

// DiscoverCapabilitiesFromProcess discovers capabilities from a stdio-based MCP server process
func (c *StdioMCPClient) DiscoverCapabilitiesFromProcess(config MCPServerConfig) ([]MCPTool, []MCPResource, error) {
	if c.verbose {
		fmt.Printf("Starting stdio-based capability discovery for: %s\n", config.Alias)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Start the MCP server process
	cmd := exec.CommandContext(ctx, "sh", "-c", config.RunCmd)

	// Set working directory if specified
	if config.WorkingDir != "" {
		cmd.Dir = config.WorkingDir
	}

	// Set environment variables
	if len(config.Env) > 0 {
		for key, value := range config.Env {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}

	// Create pipes
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	defer stdin.Close()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	defer stdout.Close()

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	defer stderr.Close()

	// Start the process
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("failed to start MCP server process: %w", err)
	}

	// Ensure process cleanup
	defer func() {
		if cmd.Process != nil {
			if killErr := cmd.Process.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				fmt.Printf("WARN: failed to terminate MCP stdio process: %v\n", killErr)
			}
		}
	}()

	if c.verbose {
		fmt.Printf("MCP server process started, performing handshake...\n")
	}

	// Perform MCP handshake and discovery
	tools, resources, err := c.performDiscovery(stdin, stdout, stderr)
	if err != nil {
		return nil, nil, fmt.Errorf("discovery failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("Discovery completed: %d tools, %d resources\n", len(tools), len(resources))
	}

	return tools, resources, nil
}

// performDiscovery performs the MCP handshake and capability discovery
func (c *StdioMCPClient) performDiscovery(stdin io.WriteCloser, stdout io.ReadCloser, stderr io.ReadCloser) ([]MCPTool, []MCPResource, error) {
	// Create JSON-RPC communication channels
	encoder := json.NewEncoder(stdin)
	decoder := json.NewDecoder(stdout)

	// Monitor stderr for debugging
	go func() {
		if c.verbose {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				fmt.Printf("MCP stderr: %s\n", scanner.Text())
			}
		}
	}()

	// Step 1: Send initialize request
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
				Name:    "agentfield-mcp-client",
				Version: "1.0.0",
			},
		},
	}

	if c.verbose {
		fmt.Printf("Sending initialize request...\n")
	}

	if err := encoder.Encode(initRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send initialize request: %w", err)
	}

	// Read initialize response
	var initResponse MCPResponse
	if err := decoder.Decode(&initResponse); err != nil {
		return nil, nil, fmt.Errorf("failed to read initialize response: %w", err)
	}

	if initResponse.Error != nil {
		return nil, nil, fmt.Errorf("initialize failed: %s", initResponse.Error.Message)
	}

	if c.verbose {
		fmt.Printf("Initialize successful, sending initialized notification...\n")
	}

	// Step 2: Send initialized notification (must be a notification, not a request - no ID field)
	initializedNotification := MCPNotification{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
		Params:  map[string]interface{}{},
	}

	if err := encoder.Encode(initializedNotification); err != nil {
		return nil, nil, fmt.Errorf("failed to send initialized notification: %w", err)
	}

	// Step 3: Request tools list
	if c.verbose {
		fmt.Printf("Requesting tools list...\n")
	}

	toolsRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
	}

	if err := encoder.Encode(toolsRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send tools/list request: %w", err)
	}

	// Read tools response
	var toolsResponse MCPResponse
	if err := decoder.Decode(&toolsResponse); err != nil {
		return nil, nil, fmt.Errorf("failed to read tools response: %w", err)
	}

	if toolsResponse.Error != nil {
		return nil, nil, fmt.Errorf("tools/list failed: %s", toolsResponse.Error.Message)
	}

	// Parse tools from response
	tools, err := c.parseToolsResponse(toolsResponse.Result)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse tools response: %w", err)
	}

	// Step 4: Request resources list
	if c.verbose {
		fmt.Printf("Requesting resources list...\n")
	}

	resourcesRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "resources/list",
	}

	if err := encoder.Encode(resourcesRequest); err != nil {
		return nil, nil, fmt.Errorf("failed to send resources/list request: %w", err)
	}

	// Read resources response
	var resourcesResponse MCPResponse
	if err := decoder.Decode(&resourcesResponse); err != nil {
		// Resources might not be supported, that's okay
		if c.verbose {
			fmt.Printf("Resources not supported or failed to read response: %v\n", err)
		}
		return tools, []MCPResource{}, nil
	}

	if resourcesResponse.Error != nil {
		// Resources might not be supported, that's okay
		if c.verbose {
			fmt.Printf("Resources not supported: %s\n", resourcesResponse.Error.Message)
		}
		return tools, []MCPResource{}, nil
	}

	// Parse resources from response
	resources, err := c.parseResourcesResponse(resourcesResponse.Result)
	if err != nil {
		if c.verbose {
			fmt.Printf("Failed to parse resources response: %v\n", err)
		}
		return tools, []MCPResource{}, nil
	}

	return tools, resources, nil
}

// parseToolsResponse parses the tools/list response
func (c *StdioMCPClient) parseToolsResponse(result interface{}) ([]MCPTool, error) {
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid tools response format")
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
func (c *StdioMCPClient) parseResourcesResponse(result interface{}) ([]MCPResource, error) {
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid resources response format")
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
