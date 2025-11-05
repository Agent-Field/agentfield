package interfaces

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// AgentClient defines the interface for communicating with agent nodes
type AgentClient interface {
	// GetMCPHealth retrieves MCP health information from an agent node
	GetMCPHealth(ctx context.Context, nodeID string) (*MCPHealthResponse, error)

	// RestartMCPServer restarts a specific MCP server on an agent node
	RestartMCPServer(ctx context.Context, nodeID, alias string) error

	// GetMCPTools retrieves the list of tools from a specific MCP server
	GetMCPTools(ctx context.Context, nodeID, alias string) (*MCPToolsResponse, error)

	// ShutdownAgent requests graceful shutdown of an agent node via HTTP
	ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*AgentShutdownResponse, error)

	// GetAgentStatus retrieves detailed status information from an agent node
	GetAgentStatus(ctx context.Context, nodeID string) (*AgentStatusResponse, error)
}

// MCPHealthResponse represents the complete MCP health data from an agent node
type MCPHealthResponse struct {
	Servers []MCPServerHealth `json:"servers"`
	Summary MCPSummary        `json:"summary"`
}

// FlexibleTime is a custom time type that can unmarshal timestamps with or without timezone
type FlexibleTime struct {
	time.Time
}

// UnmarshalJSON implements custom JSON unmarshaling for timestamps
func (ft *FlexibleTime) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		return nil
	}

	// Remove quotes from JSON string
	timeStr := strings.Trim(string(data), `"`)

	// Try parsing with different formats
	formats := []string{
		time.RFC3339Nano,             // "2006-01-02T15:04:05.999999999Z07:00"
		time.RFC3339,                 // "2006-01-02T15:04:05Z07:00"
		"2006-01-02T15:04:05.999999", // Without timezone (microseconds)
		"2006-01-02T15:04:05",        // Without timezone (seconds)
	}

	for _, format := range formats {
		if t, err := time.Parse(format, timeStr); err == nil {
			// If no timezone was provided, assume UTC
			if !strings.Contains(timeStr, "Z") && !strings.Contains(timeStr, "+") && !strings.Contains(timeStr, "-") {
				t = t.UTC()
			}
			ft.Time = t
			return nil
		}
	}

	return fmt.Errorf("unable to parse time: %s", timeStr)
}

// MarshalJSON implements custom JSON marshaling for timestamps
func (ft FlexibleTime) MarshalJSON() ([]byte, error) {
	if ft.Time.IsZero() {
		return []byte("null"), nil
	}
	return json.Marshal(ft.Time.Format(time.RFC3339Nano))
}

// MCPServerHealth represents the health status of a single MCP server
type MCPServerHealth struct {
	Alias           string        `json:"alias"`
	Status          string        `json:"status"` // "running", "stopped", "error", "starting"
	ToolCount       int           `json:"tool_count"`
	StartedAt       *FlexibleTime `json:"started_at"`
	LastHealthCheck *FlexibleTime `json:"last_health_check"`
	ErrorMessage    string        `json:"error_message,omitempty"`
	Port            int           `json:"port,omitempty"`
	ProcessID       int           `json:"process_id,omitempty"`
	SuccessRate     float64       `json:"success_rate,omitempty"`
	AvgResponseTime int           `json:"avg_response_time_ms,omitempty"`
}

// MCPSummary represents aggregated MCP health metrics
type MCPSummary struct {
	TotalServers   int     `json:"total_servers"`
	RunningServers int     `json:"running_servers"`
	TotalTools     int     `json:"total_tools"`
	OverallHealth  float64 `json:"overall_health"` // 0.0 to 1.0
}

// MCPToolsResponse represents the tools available from an MCP server
type MCPToolsResponse struct {
	Tools []MCPTool `json:"tools"`
}

// MCPTool represents a single tool from an MCP server
type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// MCPRestartResponse represents the response from restarting an MCP server
type MCPRestartResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// AgentShutdownResponse represents the response from requesting agent shutdown
type AgentShutdownResponse struct {
	Status                string `json:"status"` // "shutting_down", "error"
	Graceful              bool   `json:"graceful"`
	TimeoutSeconds        int    `json:"timeout_seconds,omitempty"`
	EstimatedShutdownTime string `json:"estimated_shutdown_time,omitempty"`
	Message               string `json:"message"`
}

// AgentStatusResponse represents detailed status information from an agent
type AgentStatusResponse struct {
	Status        string                 `json:"status"`                // "running", "stopping", "error"
	Uptime        string                 `json:"uptime"`                // Human-readable uptime
	UptimeSeconds int                    `json:"uptime_seconds"`        // Uptime in seconds
	PID           int                    `json:"pid"`                   // Process ID
	Version       string                 `json:"version"`               // Agent version
	NodeID        string                 `json:"node_id"`               // Agent node ID
	LastActivity  string                 `json:"last_activity"`         // ISO timestamp
	Resources     map[string]interface{} `json:"resources"`             // Resource usage info
	MCPServers    map[string]interface{} `json:"mcp_servers,omitempty"` // MCP server info
	Message       string                 `json:"message,omitempty"`     // Additional status message
}
