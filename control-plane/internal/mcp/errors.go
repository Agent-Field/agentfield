package mcp

import (
	"fmt"
	"strings"
)

// MCPOperationError represents different types of MCP-related errors with context
type MCPOperationError struct {
	Type      MCPOperationErrorType
	Operation string
	ServerID  string
	Message   string
	Cause     error
	Context   map[string]string
	Stdout    string
	Stderr    string
}

// MCPOperationErrorType represents the category of MCP error
type MCPOperationErrorType string

const (
	OpErrorTypeInstallation        MCPOperationErrorType = "installation"
	OpErrorTypeBuild               MCPOperationErrorType = "build"
	OpErrorTypeStartup             MCPOperationErrorType = "startup"
	OpErrorTypeCapabilityDiscovery MCPOperationErrorType = "capability_discovery"
	OpErrorTypeValidation          MCPOperationErrorType = "validation"
	OpErrorTypeConfiguration       MCPOperationErrorType = "configuration"
	OpErrorTypeTemplate            MCPOperationErrorType = "template"
	OpErrorTypeProtocol            MCPOperationErrorType = "protocol"
	OpErrorTypeEnvironment         MCPOperationErrorType = "environment"
)

// Error implements the error interface
func (e *MCPOperationError) Error() string {
	var parts []string

	if e.ServerID != "" {
		parts = append(parts, fmt.Sprintf("server '%s'", e.ServerID))
	}

	if e.Operation != "" {
		parts = append(parts, fmt.Sprintf("operation '%s'", e.Operation))
	}

	parts = append(parts, string(e.Type), "failed")

	if e.Message != "" {
		parts = append(parts, "-", e.Message)
	}

	result := strings.Join(parts, " ")

	if e.Cause != nil {
		result += fmt.Sprintf(": %v", e.Cause)
	}

	return result
}

// Unwrap returns the underlying cause for error unwrapping
func (e *MCPOperationError) Unwrap() error {
	return e.Cause
}

// DetailedError returns a detailed error message including stdout/stderr and context
func (e *MCPOperationError) DetailedError() string {
	var details []string

	details = append(details, e.Error())

	if len(e.Context) > 0 {
		details = append(details, "\nContext:")
		for key, value := range e.Context {
			details = append(details, fmt.Sprintf("  %s: %s", key, value))
		}
	}

	if e.Stdout != "" {
		details = append(details, "\nStdout:")
		details = append(details, e.Stdout)
	}

	if e.Stderr != "" {
		details = append(details, "\nStderr:")
		details = append(details, e.Stderr)
	}

	return strings.Join(details, "\n")
}

// GetSuggestion returns a helpful suggestion based on the error type and content
func (e *MCPOperationError) GetSuggestion() string {
	switch e.Type {
	case OpErrorTypeEnvironment:
		if strings.Contains(e.Message, "ALLOWED_DIR") {
			return "Set the ALLOWED_DIR environment variable to specify the directory the server can access"
		}
		if strings.Contains(e.Message, "NODE_ENV") {
			return "Set NODE_ENV environment variable (e.g., --env NODE_ENV=production)"
		}
		return "Check that all required environment variables are set"

	case OpErrorTypeStartup:
		if strings.Contains(e.Message, "permission denied") {
			return "Check file permissions and ensure the executable is accessible"
		}
		if strings.Contains(e.Message, "command not found") {
			return "Ensure the required runtime (node, python, etc.) is installed and in PATH"
		}
		return "Check server configuration and ensure all dependencies are installed"

	case OpErrorTypeInstallation:
		if strings.Contains(e.Message, "npm install") {
			return "Try running 'npm install' manually in the server directory"
		}
		if strings.Contains(e.Message, "pip install") {
			return "Try running 'pip install' manually in the server directory"
		}
		return "Check network connectivity and package availability"

	case OpErrorTypeBuild:
		if strings.Contains(e.Message, "typescript") || strings.Contains(e.Message, "tsc") {
			return "Ensure TypeScript is installed and tsconfig.json is valid"
		}
		return "Check build configuration and ensure all build dependencies are available"

	case OpErrorTypeCapabilityDiscovery:
		return "Server may require specific environment variables or configuration to start properly"

	default:
		return "Check the detailed error output above for more information"
	}
}

// NewMCPOperationError creates a new MCP error with the given type and message
func NewMCPOperationError(errorType MCPOperationErrorType, operation, serverID, message string) *MCPOperationError {
	return &MCPOperationError{
		Type:      errorType,
		Operation: operation,
		ServerID:  serverID,
		Message:   message,
		Context:   make(map[string]string),
	}
}

// NewMCPOperationErrorWithCause creates a new MCP error wrapping an existing error
func NewMCPOperationErrorWithCause(errorType MCPOperationErrorType, operation, serverID, message string, cause error) *MCPOperationError {
	return &MCPOperationError{
		Type:      errorType,
		Operation: operation,
		ServerID:  serverID,
		Message:   message,
		Cause:     cause,
		Context:   make(map[string]string),
	}
}

// WithContext adds context information to the error
func (e *MCPOperationError) WithContext(key, value string) *MCPOperationError {
	if e.Context == nil {
		e.Context = make(map[string]string)
	}
	e.Context[key] = value
	return e
}

// WithOutput adds command output to the error
func (e *MCPOperationError) WithOutput(stdout, stderr string) *MCPOperationError {
	e.Stdout = stdout
	e.Stderr = stderr
	return e
}

// CommandExecutionError creates an error for command execution failures
func CommandExecutionError(operation, serverID, command string, cause error, stdout, stderr string) *MCPOperationError {
	errorType := OpErrorTypeInstallation
	if strings.Contains(operation, "build") {
		errorType = OpErrorTypeBuild
	} else if strings.Contains(operation, "start") {
		errorType = OpErrorTypeStartup
	}

	return NewMCPOperationErrorWithCause(errorType, operation, serverID, fmt.Sprintf("command failed: %s", command), cause).
		WithContext("command", command).
		WithOutput(stdout, stderr)
}

// EnvironmentError creates an error for environment-related issues
func EnvironmentError(serverID, message string) *MCPOperationError {
	return NewMCPOperationError(OpErrorTypeEnvironment, "environment_check", serverID, message)
}

// CapabilityDiscoveryError creates an error for capability discovery failures
func CapabilityDiscoveryError(serverID, message string, cause error) *MCPOperationError {
	return NewMCPOperationErrorWithCause(OpErrorTypeCapabilityDiscovery, "capability_discovery", serverID, message, cause)
}

// TemplateError creates an error for template processing failures
func TemplateError(serverID, template, message string, cause error) *MCPOperationError {
	return NewMCPOperationErrorWithCause(OpErrorTypeTemplate, "template_processing", serverID, message, cause).
		WithContext("template", template)
}

// MCPValidationError creates an error for validation failures
func MCPValidationError(serverID, field, message string) *MCPOperationError {
	return NewMCPOperationError(OpErrorTypeValidation, "validation", serverID, message).
		WithContext("field", field)
}

// ProtocolError creates an error for MCP protocol communication failures
func ProtocolError(serverID, message string, cause error) *MCPOperationError {
	return NewMCPOperationErrorWithCause(OpErrorTypeProtocol, "protocol_communication", serverID, message, cause)
}

// ErrorFormatter provides different formatting options for errors
type ErrorFormatter struct {
	Verbose bool
	Colors  bool
}

// NewErrorFormatter creates a new error formatter
func NewErrorFormatter(verbose, colors bool) *ErrorFormatter {
	return &ErrorFormatter{
		Verbose: verbose,
		Colors:  colors,
	}
}

// Format formats an error for display
func (f *ErrorFormatter) Format(err error) string {
	mcpErr, ok := err.(*MCPOperationError)
	if !ok {
		return err.Error()
	}

	if f.Verbose {
		result := mcpErr.DetailedError()
		suggestion := mcpErr.GetSuggestion()
		if suggestion != "" {
			result += "\n\nSuggestion: " + suggestion
		}
		return result
	}

	result := mcpErr.Error()
	suggestion := mcpErr.GetSuggestion()
	if suggestion != "" {
		result += "\nSuggestion: " + suggestion
	}
	return result
}

// FormatWithColors formats an error with color codes (if supported)
func (f *ErrorFormatter) FormatWithColors(err error) string {
	// For now, just return the formatted error
	// Color formatting can be added later using a color library
	return f.Format(err)
}
