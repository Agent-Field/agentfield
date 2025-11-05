package mcp

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/your-org/haxen/control-plane/internal/config"
)

// SkillGenerator handles the generation of Python skill files from MCP tools
type SkillGenerator struct {
	projectDir string
	verbose    bool
}

// NewSkillGenerator creates a new skill generator instance
func NewSkillGenerator(projectDir string, verbose bool) *SkillGenerator {
	return &SkillGenerator{
		projectDir: projectDir,
		verbose:    verbose,
	}
}

// SkillGenerationResult represents the result of skill generation
type SkillGenerationResult struct {
	Generated bool
	FilePath  string
	ToolCount int
	Message   string
}

// GenerateSkillsForServer generates a Python skill file for an MCP server
func (sg *SkillGenerator) GenerateSkillsForServer(serverAlias string) (*SkillGenerationResult, error) {
	if sg.verbose {
		fmt.Printf("=== DEBUG: Starting skill generation for: %s ===\n", serverAlias)
	}

	// Discover server capabilities using the new simplified architecture
	// Load config for capability discovery
	cfg, err := config.LoadConfig(filepath.Join(sg.projectDir, "haxen.yaml"))
	if err != nil {
		// Fallback to current directory
		cfg, err = config.LoadConfig("haxen.yaml")
		if err != nil {
			return nil, fmt.Errorf("failed to load haxen configuration: %w", err)
		}
	}
	
	discovery := NewCapabilityDiscovery(cfg, sg.projectDir)
	capability, err := discovery.GetServerCapability(serverAlias)
	if err != nil {
		return nil, fmt.Errorf("failed to discover server capabilities: %w", err)
	}

	// DEBUG: Check what we discovered
	if sg.verbose {
		fmt.Printf("=== DEBUG: Discovered capability ===\n")
		fmt.Printf("ServerAlias: %s\n", capability.ServerAlias)
		fmt.Printf("ServerName: %s\n", capability.ServerName)
		fmt.Printf("Version: %s\n", capability.Version)
		fmt.Printf("Transport: %s\n", capability.Transport)
		fmt.Printf("Endpoint: %s\n", capability.Endpoint)
		fmt.Printf("Tools count: %d\n", len(capability.Tools))
		for i, tool := range capability.Tools {
			fmt.Printf("Tool %d: %s - %s\n", i+1, tool.Name, tool.Description)
			if tool.InputSchema != nil {
				fmt.Printf("  InputSchema keys: %v\n", getMapKeys(tool.InputSchema))
			}
		}
		fmt.Printf("Resources count: %d\n", len(capability.Resources))
		for i, resource := range capability.Resources {
			fmt.Printf("Resource %d: %s - %s\n", i+1, resource.Name, resource.Description)
		}
		fmt.Printf("=== DEBUG: End capability info ===\n")
	}

	if len(capability.Tools) == 0 {
		message := fmt.Sprintf("No tools found for server %s, skipping skill generation", serverAlias)
		if sg.verbose {
			fmt.Printf("%s\n", message)
		}
		return &SkillGenerationResult{
			Generated: false,
			FilePath:  "",
			ToolCount: 0,
			Message:   message,
		}, nil
	}

	// Generate the skill file
	if sg.verbose {
		fmt.Printf("=== DEBUG: Generating skill file content ===\n")
	}
	skillContent, err := sg.generateSkillFileContent(capability)
	if err != nil {
		if sg.verbose {
			fmt.Printf("=== DEBUG: Failed to generate skill content: %v ===\n", err)
		}
		return nil, fmt.Errorf("failed to generate skill content: %w", err)
	}

	if sg.verbose {
		fmt.Printf("=== DEBUG: Generated content length: %d ===\n", len(skillContent))
		fmt.Printf("=== DEBUG: First 500 chars of content ===\n")
		if len(skillContent) > 500 {
			fmt.Printf("%s...\n", skillContent[:500])
		} else {
			fmt.Printf("%s\n", skillContent)
		}
	}

	// Write the skill file
	skillsDir := filepath.Join(sg.projectDir, "skills")
	if err := os.MkdirAll(skillsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create skills directory: %w", err)
	}

	skillFileName := fmt.Sprintf("mcp_%s.py", serverAlias)
	skillFilePath := filepath.Join(skillsDir, skillFileName)

	if sg.verbose {
		fmt.Printf("=== DEBUG: Writing skill file to: %s ===\n", skillFilePath)
	}

	if err := os.WriteFile(skillFilePath, []byte(skillContent), 0644); err != nil {
		if sg.verbose {
			fmt.Printf("=== DEBUG: Failed to write skill file: %v ===\n", err)
		}
		return nil, fmt.Errorf("failed to write skill file: %w", err)
	}

	message := fmt.Sprintf("Generated skill file: %s (%d tools)", skillFilePath, len(capability.Tools))
	if sg.verbose {
		fmt.Printf("=== DEBUG: SUCCESS: %s ===\n", message)
	}

	return &SkillGenerationResult{
		Generated: true,
		FilePath:  skillFilePath,
		ToolCount: len(capability.Tools),
		Message:   message,
	}, nil
}

// Helper function to get map keys for debugging
func getMapKeys(m map[string]interface{}) []string {
	if m == nil {
		return []string{}
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// GenerateSkillsForAllServers generates skill files for all installed MCP servers
func (sg *SkillGenerator) GenerateSkillsForAllServers() error {
	// Load config for capability discovery
	cfg, err := config.LoadConfig(filepath.Join(sg.projectDir, "haxen.yaml"))
	if err != nil {
		// Fallback to current directory
		cfg, err = config.LoadConfig("haxen.yaml")
		if err != nil {
			return fmt.Errorf("failed to load haxen configuration: %w", err)
		}
	}
	
	discovery := NewCapabilityDiscovery(cfg, sg.projectDir)
	capabilities, err := discovery.DiscoverCapabilities()
	if err != nil {
		return fmt.Errorf("failed to discover capabilities: %w", err)
	}

	var errors []string
	for _, capability := range capabilities {
		result, err := sg.GenerateSkillsForServer(capability.ServerAlias)
		if err != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", capability.ServerAlias, err))
		} else if sg.verbose && result != nil {
			fmt.Printf("Server %s: %s\n", capability.ServerAlias, result.Message)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to generate skills for some servers: %v", errors)
	}

	if sg.verbose {
		fmt.Printf("Successfully processed skills for %d MCP servers\n", len(capabilities))
	}

	return nil
}

// RemoveSkillsForServer removes the generated skill file for an MCP server
func (sg *SkillGenerator) RemoveSkillsForServer(serverAlias string) error {
	skillFileName := fmt.Sprintf("mcp_%s.py", serverAlias)
	skillFilePath := filepath.Join(sg.projectDir, "skills", skillFileName)

	if _, err := os.Stat(skillFilePath); os.IsNotExist(err) {
		// File doesn't exist, nothing to remove
		return nil
	}

	if err := os.Remove(skillFilePath); err != nil {
		return fmt.Errorf("failed to remove skill file: %w", err)
	}

	if sg.verbose {
		fmt.Printf("Removed skill file: %s\n", skillFilePath)
	}

	return nil
}

// generateSkillFileContent generates the Python content for a skill file
func (sg *SkillGenerator) generateSkillFileContent(capability *MCPCapability) (string, error) {
	// Prepare template data with escaped tools
	escapedTools := make([]map[string]interface{}, len(capability.Tools))
	for i, tool := range capability.Tools {
		escapedTools[i] = map[string]interface{}{
			"Name":        tool.Name,
			"Description": sg.escapeForPython(tool.Description),
			"InputSchema": tool.InputSchema,
		}
	}
	
	templateData := struct {
		ServerAlias    string
		ServerName     string
		Version        string
		Tools          []map[string]interface{}
		GeneratedAt    string
		SkillFunctions []SkillFunction
	}{
		ServerAlias: capability.ServerAlias,
		ServerName:  capability.ServerName,
		Version:     capability.Version,
		Tools:       escapedTools,
		GeneratedAt: time.Now().Format(time.RFC3339),
	}

	// Generate skill functions from tools
	for _, tool := range capability.Tools {
		skillFunc, err := sg.generateSkillFunction(tool, capability.ServerAlias)
		if err != nil {
			return "", fmt.Errorf("failed to generate skill function for tool %s: %w", tool.Name, err)
		}
		templateData.SkillFunctions = append(templateData.SkillFunctions, skillFunc)
	}

	// Execute template with custom functions
	tmpl, err := template.New("skill").Funcs(template.FuncMap{
		"escapeForPython": sg.escapeForPython,
		"jsonFormat": func(v interface{}) string {
			// Simple JSON formatting for input schema
			return fmt.Sprintf("%+v", v)
		},
	}).Parse(skillFileTemplate)
	if err != nil {
		return "", fmt.Errorf("failed to parse skill template: %w", err)
	}

	var content strings.Builder
	if err := tmpl.Execute(&content, templateData); err != nil {
		return "", fmt.Errorf("failed to execute skill template: %w", err)
	}

	return content.String(), nil
}

// SkillFunction represents a generated skill function
type SkillFunction struct {
	Name        string
	ToolName    string
	Description string
	Parameters  []SkillParameter
	DocString   string
}

// SkillParameter represents a skill function parameter
type SkillParameter struct {
	Name        string
	Type        string
	Required    bool
	Description string
	Default     string
}

// generateSkillFunction generates a skill function from an MCP tool
func (sg *SkillGenerator) generateSkillFunction(tool MCPTool, serverAlias string) (SkillFunction, error) {
	// Generate function name: serveralias_toolname
	functionName := sg.generateFunctionName(serverAlias, tool.Name)
	
	// Parse input schema to extract parameters
	parameters, err := sg.parseInputSchema(tool.InputSchema)
	if err != nil {
		return SkillFunction{}, fmt.Errorf("failed to parse input schema: %w", err)
	}

	// Generate docstring
	docString := sg.generateDocString(tool, parameters)

	return SkillFunction{
		Name:        functionName,
		ToolName:    tool.Name,
		Description: tool.Description,
		Parameters:  parameters,
		DocString:   docString,
	}, nil
}

// generateFunctionName generates a valid Python function name
func (sg *SkillGenerator) generateFunctionName(serverAlias, toolName string) string {
	// Convert to snake_case and ensure it's a valid Python identifier
	// First normalize the server alias
	normalizedAlias := strings.ReplaceAll(serverAlias, "-", "_")
	normalizedAlias = strings.ReplaceAll(normalizedAlias, ".", "_")
	normalizedAlias = strings.ReplaceAll(normalizedAlias, "/", "_")
	
	// Then normalize the tool name
	normalizedTool := strings.ReplaceAll(toolName, "-", "_")
	normalizedTool = strings.ReplaceAll(normalizedTool, ".", "_")
	normalizedTool = strings.ReplaceAll(normalizedTool, "/", "_")
	
	name := fmt.Sprintf("%s_%s", normalizedAlias, normalizedTool)
	
	// Ensure it starts with a letter or underscore
	if len(name) > 0 && (name[0] >= '0' && name[0] <= '9') {
		name = "_" + name
	}
	
	return name
}

// parseInputSchema parses JSON schema to extract function parameters
func (sg *SkillGenerator) parseInputSchema(schema map[string]interface{}) ([]SkillParameter, error) {
	var parameters []SkillParameter

	// Handle JSON Schema format
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		required := make(map[string]bool)
		if requiredList, ok := schema["required"].([]interface{}); ok {
			for _, req := range requiredList {
				if reqStr, ok := req.(string); ok {
					required[reqStr] = true
				}
			}
		}

		for paramName, paramDef := range properties {
			if paramDefMap, ok := paramDef.(map[string]interface{}); ok {
				param := SkillParameter{
					Name:     paramName,
					Required: required[paramName],
				}

				// Extract type
				if paramType, ok := paramDefMap["type"].(string); ok {
					param.Type = sg.mapJSONTypeToPython(paramType)
				} else {
					param.Type = "Any"
				}

				// Extract description
				if desc, ok := paramDefMap["description"].(string); ok {
					param.Description = desc
				}

				// Extract default value
				if defaultVal, ok := paramDefMap["default"]; ok {
					param.Default = sg.formatDefaultValue(defaultVal)
				}

				parameters = append(parameters, param)
			}
		}
	}

	return parameters, nil
}

// mapJSONTypeToPython maps JSON Schema types to Python types
func (sg *SkillGenerator) mapJSONTypeToPython(jsonType string) string {
	switch jsonType {
	case "string":
		return "str"
	case "integer":
		return "int"
	case "number":
		return "float"
	case "boolean":
		return "bool"
	case "array":
		return "List[Any]"
	case "object":
		return "Dict[str, Any]"
	default:
		return "Any"
	}
}

// formatDefaultValue formats a default value for Python code
func (sg *SkillGenerator) formatDefaultValue(value interface{}) string {
	switch v := value.(type) {
	case string:
		return fmt.Sprintf(`"%s"`, v)
	case bool:
		if v {
			return "True"
		}
		return "False"
	case nil:
		return "None"
	default:
		// For numbers and other types, convert to string
		return fmt.Sprintf("%v", v)
	}
}

// escapeForPython escapes a string for safe use in Python code
func (sg *SkillGenerator) escapeForPython(s string) string {
	// Replace problematic characters for Python strings
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return s
}

// escapeForDocstring escapes a string for safe use in Python docstrings
func (sg *SkillGenerator) escapeForDocstring(s string) string {
	// For docstrings, we need to handle triple quotes and preserve formatting
	s = strings.ReplaceAll(s, `"""`, `\"\"\"`)
	// Replace any problematic characters but preserve newlines for readability
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return s
}

// generateDocString generates a Python docstring for the skill function
func (sg *SkillGenerator) generateDocString(tool MCPTool, parameters []SkillParameter) string {
	var docString strings.Builder
	
	// Escape the description for safe use in docstring
	escapedDescription := sg.escapeForDocstring(tool.Description)
	
	docString.WriteString(fmt.Sprintf(`"""%s
    
    This is an auto-generated skill function that wraps the MCP tool '%s'.
    
    Args:`, escapedDescription, tool.Name))

	for _, param := range parameters {
		required := ""
		if !param.Required {
			required = ", optional"
		}
		
		defaultInfo := ""
		if param.Default != "" {
			defaultInfo = fmt.Sprintf(", defaults to %s", param.Default)
		}
		
		// Escape parameter description
		escapedParamDesc := sg.escapeForDocstring(param.Description)
		
		docString.WriteString(fmt.Sprintf(`
        %s (%s%s): %s%s`, param.Name, param.Type, required, escapedParamDesc, defaultInfo))
	}

	docString.WriteString(`
        execution_context (ExecutionContext, optional): Haxen execution context for workflow tracking
    
    Returns:
        Any: The result from the MCP tool execution
        
    Raises:
        MCPError: If the MCP server is not available or the tool execution fails
    """`)

	return docString.String()
}

// skillFileTemplate is the template for generating Python skill files
const skillFileTemplate = `"""
Auto-generated MCP skill file for server: {{.ServerAlias}}
Generated at: {{.GeneratedAt}}
Server: {{.ServerName}} ({{.Version}})

This file contains auto-generated skill functions that wrap MCP tools.
Do not modify this file manually - it will be regenerated when the MCP server is updated.
"""

from typing import Any, Dict, List, Optional
from haxen_sdk import app
from haxen_sdk.execution_context import ExecutionContext
from haxen_sdk.mcp.client import MCPClient
from haxen_sdk.mcp.exceptions import (
    MCPError, MCPConnectionError, MCPToolError, MCPTimeoutError
)
from haxen_sdk.agent import Agent

# MCP server configuration
MCP_ALIAS = "{{.ServerAlias}}"
MCP_SERVER_NAME = "{{.ServerName}}"
MCP_VERSION = "{{.Version}}"

# Tool definitions for validation
MCP_TOOLS = [
{{- range .Tools}}
    {
        "name": "{{.Name}}",
        "description": "{{.Description}}",
        "input_schema": {{.InputSchema | jsonFormat}},
    },
{{- end}}
]

async def _get_mcp_client(execution_context: Optional[ExecutionContext] = None) -> MCPClient:
    """Get or create MCP client for this server with execution context."""
    try:
        # Get client from registry
        client = MCPClient.get_or_create(MCP_ALIAS)
        
        # Validate server health
        is_healthy = await client.validate_server_health()
        if not is_healthy:
            raise MCPConnectionError(
                f"MCP server '{MCP_ALIAS}' is not healthy. Please check server status with: haxen mcp status {MCP_ALIAS}",
                endpoint=f"mcp://{MCP_ALIAS}"
            )
        
        # Set execution context for workflow tracking
        if execution_context:
            client.set_execution_context(execution_context)
        elif Agent.get_current():
            # Try to get execution context from current agent
            current_agent = Agent.get_current()
            if hasattr(current_agent, '_current_execution_context') and current_agent._current_execution_context:
                client.set_execution_context(current_agent._current_execution_context)
        
        return client
        
    except ValueError as e:
        # Handle unregistered alias
        raise MCPConnectionError(
            f"MCP server '{MCP_ALIAS}' is not configured. Please install it with: haxen add --mcp {MCP_ALIAS}",
            endpoint=f"mcp://{MCP_ALIAS}"
        ) from e
    except Exception as e:
        # Handle other connection errors
        raise MCPConnectionError(
            f"Failed to connect to MCP server '{MCP_ALIAS}': {str(e)}",
            endpoint=f"mcp://{MCP_ALIAS}"
        ) from e

{{range .SkillFunctions}}
@app.skill(tags=["mcp", "{{$.ServerAlias}}"])
async def {{.Name}}({{range $i, $param := .Parameters}}{{if $i}}, {{end}}{{$param.Name}}: {{$param.Type}}{{if not $param.Required}}{{if $param.Default}} = {{$param.Default}}{{else}} = None{{end}}{{end}}{{end}}{{if .Parameters}}, {{end}}execution_context: Optional[ExecutionContext] = None) -> Any:
    {{.DocString}}
    
    try:
        # Get MCP client with execution context
        client = await _get_mcp_client(execution_context)
        
        # Prepare arguments, filtering out None values for optional parameters
        kwargs = {}
        {{range .Parameters}}
        if {{.Name}} is not None:
            kwargs["{{.Name}}"] = {{.Name}}
        {{end}}
        
        # Call the MCP tool
        result = await client.call_tool("{{.ToolName}}", kwargs)
        return result
        
    except MCPConnectionError:
        # Re-raise connection errors as-is (they have helpful messages)
        raise
    except MCPToolError as e:
        # Re-raise tool errors as-is (they have specific error details)
        raise
    except MCPTimeoutError as e:
        # Re-raise timeout errors with context
        raise MCPTimeoutError(
            f"MCP tool '{{.ToolName}}' timed out after {e.timeout}s",
            timeout=e.timeout
        ) from e
    except Exception as e:
        # Wrap unexpected errors
        raise MCPError(f"Unexpected error calling MCP tool '{{.ToolName}}': {str(e)}") from e

{{end}}

# Register all tools for discovery
def _register_mcp_tools():
    """Register MCP tools for runtime discovery."""
    # This function is called automatically when the module is imported
    # All tools are already registered via @app.skill decorators above
    pass

# Auto-register tools when module is imported
_register_mcp_tools()
`
