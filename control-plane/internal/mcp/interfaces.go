package mcp

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"time"
)

// MCPServerConfig represents the simplified MCP configuration
type MCPServerConfig struct {
	// Core identification
	Alias       string `yaml:"alias" json:"alias"`
	Description string `yaml:"description,omitempty" json:"description,omitempty"`

	// Connection (mutually exclusive)
	URL    string `yaml:"url,omitempty" json:"url,omitempty"` // For remote MCPs
	RunCmd string `yaml:"run,omitempty" json:"run,omitempty"` // For local MCPs

	// Transport type (stdio or http)
	Transport string `yaml:"transport,omitempty" json:"transport,omitempty"`

	// Setup (optional - runs once during add)
	SetupCmds []string `yaml:"setup,omitempty" json:"setup,omitempty"`

	// Runtime configuration
	WorkingDir string            `yaml:"working_dir,omitempty" json:"working_dir,omitempty"`
	Env        map[string]string `yaml:"environment,omitempty" json:"environment,omitempty"`
	Timeout    time.Duration     `yaml:"timeout,omitempty" json:"timeout,omitempty"`

	// Health & Monitoring
	HealthCheck string `yaml:"health_check,omitempty" json:"health_check,omitempty"`
	Port        int    `yaml:"port,omitempty" json:"port,omitempty"` // Auto-assigned if 0

	// Metadata
	Version string   `yaml:"version,omitempty" json:"version,omitempty"`
	Tags    []string `yaml:"tags,omitempty" json:"tags,omitempty"`

	// Installation options
	Force bool `yaml:"-" json:"-"` // Force reinstall, not persisted

	// Internal runtime fields
	PID       int        `yaml:"-" json:"pid,omitempty"`
	Status    string     `yaml:"-" json:"status,omitempty"`
	StartedAt *time.Time `yaml:"-" json:"started_at,omitempty"`
}

// MCPProcess represents a running MCP server
type MCPProcess struct {
	Config  MCPServerConfig    `json:"config"`
	Cmd     *exec.Cmd          `json:"-"`
	Stdin   io.WriteCloser     `json:"-"`
	Stdout  io.ReadCloser      `json:"-"`
	Stderr  io.ReadCloser      `json:"-"`
	Context context.Context    `json:"-"`
	Cancel  context.CancelFunc `json:"-"`
	LogFile string             `json:"log_file"`
}

// MCPServerStatus represents the status of an MCP server
type MCPServerStatus string

const (
	StatusStopped  MCPServerStatus = "stopped"
	StatusStarting MCPServerStatus = "starting"
	StatusRunning  MCPServerStatus = "running"
	StatusError    MCPServerStatus = "error"
)

// MCPServerInfo represents information about an MCP server for status/list operations
type MCPServerInfo struct {
	Alias       string          `json:"alias"`
	Description string          `json:"description,omitempty"`
	Status      MCPServerStatus `json:"status"`
	URL         string          `json:"url,omitempty"`
	RunCmd      string          `json:"run_cmd,omitempty"`
	Port        int             `json:"port,omitempty"`
	PID         int             `json:"pid,omitempty"`
	StartedAt   *time.Time      `json:"started_at,omitempty"`
	Version     string          `json:"version,omitempty"`
	Tags        []string        `json:"tags,omitempty"`
}

// MCPTool represents an MCP tool definition
type MCPTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"inputSchema"`
	OutputSchema map[string]interface{} `json:"outputSchema,omitempty"`
}

// MCPResource represents an MCP resource definition
type MCPResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mime_type,omitempty"`
}

// MCPManifest represents the capabilities discovered from an MCP server
type MCPManifest struct {
	Tools     []MCPTool     `json:"tools,omitempty"`
	Resources []MCPResource `json:"resources,omitempty"`
	Version   string        `json:"version,omitempty"`
}

// MCPRequest represents a JSON-RPC request to an MCP server
type MCPRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// MCPNotification represents a JSON-RPC notification to an MCP server (no ID field)
type MCPNotification struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// MCPResponse represents a JSON-RPC response from an MCP server
type MCPResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *MCPError       `json:"error,omitempty"`
}

// MCPError represents an error in MCP protocol
type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// InitializeParams represents MCP initialization parameters
type InitializeParams struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ClientInfo      ClientInfo             `json:"clientInfo"`
}

// ClientInfo represents client information
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// MCPToolsListResponse represents the response from tools/list request
type MCPToolsListResponse struct {
	Tools []MCPToolDefinition `json:"tools"`
}

// MCPResourcesListResponse represents the response from resources/list request
type MCPResourcesListResponse struct {
	Resources []MCPResourceDefinition `json:"resources"`
}

// MCPToolDefinition represents a tool definition from MCP protocol
type MCPToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// MCPResourceDefinition represents a resource definition from MCP protocol
type MCPResourceDefinition struct {
	URI         string `json:"uri"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}
