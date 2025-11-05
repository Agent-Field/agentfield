package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

// MCPProtocolClient handles communication with MCP servers using the MCP protocol
type MCPProtocolClient struct {
	verbose bool
}

// NewMCPProtocolClient creates a new MCP protocol client
func NewMCPProtocolClient(verbose bool) *MCPProtocolClient {
	return &MCPProtocolClient{
		verbose: verbose,
	}
}

// DiscoverCapabilitiesFromProcess discovers capabilities from a running MCP process
func (client *MCPProtocolClient) DiscoverCapabilitiesFromProcess(process *MCPProcess) ([]MCPTool, []MCPResource, error) {
	if client.verbose {
		fmt.Printf("Discovering capabilities from process on port %d\n", process.Config.Port)
	}

	// For stdio-based MCP servers, we need to communicate via stdin/stdout
	if process.Cmd != nil {
		return client.discoverFromStdio(process)
	}

	// For HTTP-based servers, communicate via HTTP
	if process.Config.Port > 0 {
		return client.discoverFromHTTP(process.Config.Port)
	}

	return nil, nil, fmt.Errorf("unsupported MCP server transport")
}

// DiscoverCapabilitiesFromURL discovers capabilities from a remote MCP server URL
func (client *MCPProtocolClient) DiscoverCapabilitiesFromURL(url string) ([]MCPTool, []MCPResource, error) {
	if client.verbose {
		fmt.Printf("Discovering capabilities from URL: %s\n", url)
	}

	// Parse URL to determine transport method
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		// HTTP/WebSocket transport
		return client.discoverFromHTTPURL(url)
	}

	return nil, nil, fmt.Errorf("unsupported URL transport: %s", url)
}

// discoverFromStdio discovers capabilities from a stdio-based MCP server
func (client *MCPProtocolClient) discoverFromStdio(process *MCPProcess) ([]MCPTool, []MCPResource, error) {
	if process.Stdin == nil || process.Stdout == nil {
		return nil, nil, fmt.Errorf("stdin/stdout not available for MCP communication")
	}

	// Send initialize request first
	if err := client.sendInitialize(process.Stdin); err != nil {
		return nil, nil, fmt.Errorf("failed to initialize MCP session: %w", err)
	}

	// Wait for initialize response
	if err := client.waitForInitializeResponse(process.Stdout); err != nil {
		return nil, nil, fmt.Errorf("failed to receive initialize response: %w", err)
	}

	// Discover tools
	tools, err := client.requestToolsList(process.Stdin, process.Stdout)
	if err != nil {
		if client.verbose {
			fmt.Printf("Warning: failed to discover tools: %v\n", err)
		}
		tools = []MCPTool{}
	}

	// Discover resources
	resources, err := client.requestResourcesList(process.Stdin, process.Stdout)
	if err != nil {
		if client.verbose {
			fmt.Printf("Warning: failed to discover resources: %v\n", err)
		}
		resources = []MCPResource{}
	}

	return tools, resources, nil
}

// discoverFromHTTP discovers capabilities from an HTTP-based MCP server
func (client *MCPProtocolClient) discoverFromHTTP(port int) ([]MCPTool, []MCPResource, error) {
	baseURL := fmt.Sprintf("http://localhost:%d", port)
	return client.discoverFromHTTPURL(baseURL)
}

// discoverFromHTTPURL discovers capabilities from an HTTP URL
func (client *MCPProtocolClient) discoverFromHTTPURL(baseURL string) ([]MCPTool, []MCPResource, error) {
	// Try common MCP HTTP endpoints
	endpoints := []string{
		baseURL + "/mcp",
		baseURL + "/api/mcp",
		baseURL,
	}

	var lastErr error
	for _, endpoint := range endpoints {
		tools, resources, err := client.tryHTTPEndpoint(endpoint)
		if err == nil {
			return tools, resources, nil
		}
		lastErr = err
		if client.verbose {
			fmt.Printf("Failed to connect to %s: %v\n", endpoint, err)
		}
	}

	return nil, nil, fmt.Errorf("failed to connect to any HTTP endpoint: %w", lastErr)
}

// tryHTTPEndpoint tries to discover capabilities from a specific HTTP endpoint
func (client *MCPProtocolClient) tryHTTPEndpoint(endpoint string) ([]MCPTool, []MCPResource, error) {
	// Create HTTP client with timeout
	httpClient := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Send tools/list request
	tools, err := client.sendHTTPToolsRequest(httpClient, endpoint)
	if err != nil {
		return nil, nil, err
	}

	// Send resources/list request
	resources, err := client.sendHTTPResourcesRequest(httpClient, endpoint)
	if err != nil {
		// Resources are optional, so don't fail if they're not available
		if client.verbose {
			fmt.Printf("Warning: failed to get resources from %s: %v\n", endpoint, err)
		}
		resources = []MCPResource{}
	}

	return tools, resources, nil
}

// sendInitialize sends the MCP initialize request
func (client *MCPProtocolClient) sendInitialize(stdin io.Writer) error {
	initRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]interface{}{
				"tools":     map[string]interface{}{},
				"resources": map[string]interface{}{},
			},
			"clientInfo": map[string]interface{}{
				"name":    "agentfield-mcp-client",
				"version": "1.0.0",
			},
		},
	}

	return client.sendJSONRPCRequest(stdin, initRequest)
}

// waitForInitializeResponse waits for the initialize response
func (client *MCPProtocolClient) waitForInitializeResponse(stdout io.Reader) error {
	scanner := bufio.NewScanner(stdout)
	scanner.Scan()

	var response MCPResponse
	if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
		return fmt.Errorf("failed to parse initialize response: %w", err)
	}

	if response.Error != nil {
		return fmt.Errorf("initialize error: %s", response.Error.Message)
	}

	return nil
}

// requestToolsList requests the list of tools from MCP server
func (client *MCPProtocolClient) requestToolsList(stdin io.Writer, stdout io.Reader) ([]MCPTool, error) {
	toolsRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
	}

	if err := client.sendJSONRPCRequest(stdin, toolsRequest); err != nil {
		return nil, err
	}

	// Read response
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() {
		return nil, fmt.Errorf("no response received for tools/list")
	}

	var response MCPResponse
	if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
		return nil, fmt.Errorf("failed to parse tools/list response: %w", err)
	}

	if response.Error != nil {
		return nil, fmt.Errorf("tools/list error: %s", response.Error.Message)
	}

	var toolsResponse MCPToolsListResponse
	if err := json.Unmarshal(response.Result, &toolsResponse); err != nil {
		return nil, fmt.Errorf("failed to parse tools list: %w", err)
	}

	// Convert to our internal format
	tools := make([]MCPTool, len(toolsResponse.Tools))
	for i, tool := range toolsResponse.Tools {
		tools[i] = MCPTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}

	return tools, nil
}

// requestResourcesList requests the list of resources from MCP server
func (client *MCPProtocolClient) requestResourcesList(stdin io.Writer, stdout io.Reader) ([]MCPResource, error) {
	resourcesRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "resources/list",
	}

	if err := client.sendJSONRPCRequest(stdin, resourcesRequest); err != nil {
		return nil, err
	}

	// Read response
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() {
		return nil, fmt.Errorf("no response received for resources/list")
	}

	var response MCPResponse
	if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
		return nil, fmt.Errorf("failed to parse resources/list response: %w", err)
	}

	if response.Error != nil {
		return nil, fmt.Errorf("resources/list error: %s", response.Error.Message)
	}

	var resourcesResponse MCPResourcesListResponse
	if err := json.Unmarshal(response.Result, &resourcesResponse); err != nil {
		return nil, fmt.Errorf("failed to parse resources list: %w", err)
	}

	// Convert to our internal format
	resources := make([]MCPResource, len(resourcesResponse.Resources))
	for i, resource := range resourcesResponse.Resources {
		resources[i] = MCPResource{
			URI:         resource.URI,
			Name:        resource.Name,
			Description: resource.Description,
			MimeType:    resource.MimeType,
		}
	}

	return resources, nil
}

// sendHTTPToolsRequest sends tools/list request via HTTP
func (client *MCPProtocolClient) sendHTTPToolsRequest(httpClient *http.Client, endpoint string) ([]MCPTool, error) {
	toolsRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	}

	response, err := client.sendHTTPRequest(httpClient, endpoint, toolsRequest)
	if err != nil {
		return nil, err
	}

	var toolsResponse MCPToolsListResponse
	if err := json.Unmarshal([]byte(response.Result), &toolsResponse); err != nil {
		return nil, fmt.Errorf("failed to parse tools list: %w", err)
	}

	// Convert to our internal format
	tools := make([]MCPTool, len(toolsResponse.Tools))
	for i, tool := range toolsResponse.Tools {
		tools[i] = MCPTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}

	return tools, nil
}

// sendHTTPResourcesRequest sends resources/list request via HTTP
func (client *MCPProtocolClient) sendHTTPResourcesRequest(httpClient *http.Client, endpoint string) ([]MCPResource, error) {
	resourcesRequest := MCPRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "resources/list",
	}

	response, err := client.sendHTTPRequest(httpClient, endpoint, resourcesRequest)
	if err != nil {
		return nil, err
	}

	var resourcesResponse MCPResourcesListResponse
	if err := json.Unmarshal([]byte(response.Result), &resourcesResponse); err != nil {
		return nil, fmt.Errorf("failed to parse resources list: %w", err)
	}

	// Convert to our internal format
	resources := make([]MCPResource, len(resourcesResponse.Resources))
	for i, resource := range resourcesResponse.Resources {
		resources[i] = MCPResource{
			URI:         resource.URI,
			Name:        resource.Name,
			Description: resource.Description,
			MimeType:    resource.MimeType,
		}
	}

	return resources, nil
}

// sendHTTPRequest sends an HTTP request to MCP server
func (client *MCPProtocolClient) sendHTTPRequest(httpClient *http.Client, endpoint string, request MCPRequest) (*MCPResponse, error) {
	requestBytes, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := httpClient.Post(endpoint, "application/json", strings.NewReader(string(requestBytes)))
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
	}

	responseBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var response MCPResponse
	if err := json.Unmarshal(responseBytes, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if response.Error != nil {
		return nil, fmt.Errorf("MCP error: %s", response.Error.Message)
	}

	return &response, nil
}

// sendJSONRPCRequest sends a JSON-RPC request via stdin
func (client *MCPProtocolClient) sendJSONRPCRequest(stdin io.Writer, request MCPRequest) error {
	requestBytes, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Add newline for line-based communication
	requestBytes = append(requestBytes, '\n')

	if _, err := stdin.Write(requestBytes); err != nil {
		return fmt.Errorf("failed to write request: %w", err)
	}

	return nil
}

// StartMCPServerForDiscovery starts an MCP server temporarily for capability discovery
func (client *MCPProtocolClient) StartMCPServerForDiscovery(config MCPServerConfig, template *TemplateProcessor) (*exec.Cmd, error) {
	if config.RunCmd == "" {
		return nil, fmt.Errorf("no run command specified for MCP server")
	}

	// Process template variables in the run command
	vars := template.CreateTemplateVars(config, 0) // Port 0 for stdio-based servers
	processedCmd, err := template.ProcessCommand(config.RunCmd, vars)
	if err != nil {
		return nil, fmt.Errorf("failed to process run command template: %w", err)
	}

	// Set working directory
	workingDir := vars.ServerDir
	if config.WorkingDir != "" {
		processedWorkingDir, err := template.ProcessCommand(config.WorkingDir, vars)
		if err != nil {
			return nil, fmt.Errorf("failed to process working directory: %w", err)
		}
		workingDir = processedWorkingDir
	}

	// Create command using shell to handle complex commands
	cmd := exec.Command("sh", "-c", processedCmd)
	cmd.Dir = workingDir

	// Set environment variables
	if len(config.Env) > 0 {
		processedEnv, err := template.ProcessEnvironment(config.Env, vars)
		if err != nil {
			return nil, fmt.Errorf("failed to process environment variables: %w", err)
		}

		for key, value := range processedEnv {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start MCP server: %w", err)
	}

	// Give the server a moment to start
	time.Sleep(1 * time.Second)

	return cmd, nil
}
