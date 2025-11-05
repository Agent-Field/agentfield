package domain

import (
	"fmt"
	"time"
)

// MCPSummaryForUI represents MCP health summary optimized for UI display
type MCPSummaryForUI struct {
	TotalServers   int     `json:"total_servers"`
	RunningServers int     `json:"running_servers"`
	TotalTools     int     `json:"total_tools"`
	OverallHealth  float64 `json:"overall_health"`
	HasIssues      bool    `json:"has_issues"`

	// User mode: simplified capability status
	CapabilitiesAvailable bool   `json:"capabilities_available"`
	ServiceStatus         string `json:"service_status"` // "ready", "degraded", "unavailable"
}

// AgentNodeDetailsForUI represents detailed agent node information including MCP data
type AgentNodeDetailsForUI struct {
	// Embed the base agent node data
	ID            string    `json:"id"`
	TeamID        string    `json:"team_id"`
	BaseURL       string    `json:"base_url"`
	Version       string    `json:"version"`
	HealthStatus  string    `json:"health_status"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	RegisteredAt  time.Time `json:"registered_at"`

	// MCP-specific data (only in developer mode)
	MCPServers []MCPServerHealthForUI `json:"mcp_servers,omitempty"`
	MCPSummary *MCPSummaryForUI       `json:"mcp_summary,omitempty"`
}

// MCPServerHealthForUI represents MCP server health optimized for UI display
type MCPServerHealthForUI struct {
	Alias           string     `json:"alias"`
	Status          string     `json:"status"`
	ToolCount       int        `json:"tool_count"`
	StartedAt       *time.Time `json:"started_at"`
	LastHealthCheck *time.Time `json:"last_health_check"`
	ErrorMessage    string     `json:"error_message,omitempty"`
	Port            int        `json:"port,omitempty"`
	ProcessID       int        `json:"process_id,omitempty"`
	SuccessRate     float64    `json:"success_rate,omitempty"`
	AvgResponseTime int        `json:"avg_response_time_ms,omitempty"`

	// UI-specific fields
	StatusIcon      string `json:"status_icon"`      // Icon name for UI
	StatusColor     string `json:"status_color"`     // Color code for UI
	UptimeFormatted string `json:"uptime_formatted"` // Human-readable uptime
}

// CachedMCPHealth represents cached MCP health data with timestamp
type CachedMCPHealth struct {
	Data      *MCPHealthResponseData `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
	NodeID    string                 `json:"node_id"`
}

// MCPHealthResponseData represents the raw MCP health data from agent
type MCPHealthResponseData struct {
	Servers []MCPServerHealthData `json:"servers"`
	Summary MCPSummaryData        `json:"summary"`
}

// MCPServerHealthData represents raw MCP server health data
type MCPServerHealthData struct {
	Alias           string     `json:"alias"`
	Status          string     `json:"status"`
	ToolCount       int        `json:"tool_count"`
	StartedAt       *time.Time `json:"started_at"`
	LastHealthCheck *time.Time `json:"last_health_check"`
	ErrorMessage    string     `json:"error_message,omitempty"`
	Port            int        `json:"port,omitempty"`
	ProcessID       int        `json:"process_id,omitempty"`
	SuccessRate     float64    `json:"success_rate,omitempty"`
	AvgResponseTime int        `json:"avg_response_time_ms,omitempty"`
}

// MCPSummaryData represents raw MCP summary data
type MCPSummaryData struct {
	TotalServers   int     `json:"total_servers"`
	RunningServers int     `json:"running_servers"`
	TotalTools     int     `json:"total_tools"`
	OverallHealth  float64 `json:"overall_health"`
}

// MCPHealthMode represents the mode for MCP health data display
type MCPHealthMode string

const (
	MCPHealthModeUser      MCPHealthMode = "user"
	MCPHealthModeDeveloper MCPHealthMode = "developer"
)

// MCPServerStatus represents the possible statuses of an MCP server
type MCPServerStatus string

const (
	MCPServerStatusRunning  MCPServerStatus = "running"
	MCPServerStatusStopped  MCPServerStatus = "stopped"
	MCPServerStatusError    MCPServerStatus = "error"
	MCPServerStatusStarting MCPServerStatus = "starting"
	MCPServerStatusUnknown  MCPServerStatus = "unknown"
)

// MCPServiceStatus represents the overall service status for user mode
type MCPServiceStatus string

const (
	MCPServiceStatusReady       MCPServiceStatus = "ready"
	MCPServiceStatusDegraded    MCPServiceStatus = "degraded"
	MCPServiceStatusUnavailable MCPServiceStatus = "unavailable"
)

// TransformMCPHealthForMode transforms raw MCP health data based on display mode
func TransformMCPHealthForMode(data *MCPHealthResponseData, mode MCPHealthMode) (*MCPSummaryForUI, []MCPServerHealthForUI) {
	if data == nil {
		return nil, nil
	}

	// Create summary
	summary := &MCPSummaryForUI{
		TotalServers:   data.Summary.TotalServers,
		RunningServers: data.Summary.RunningServers,
		TotalTools:     data.Summary.TotalTools,
		OverallHealth:  data.Summary.OverallHealth,
		HasIssues:      data.Summary.RunningServers < data.Summary.TotalServers || data.Summary.OverallHealth < 0.8,
	}

	// Set user-mode specific fields
	if mode == MCPHealthModeUser {
		summary.CapabilitiesAvailable = data.Summary.RunningServers > 0
		if data.Summary.OverallHealth >= 0.9 {
			summary.ServiceStatus = string(MCPServiceStatusReady)
		} else if data.Summary.OverallHealth >= 0.5 {
			summary.ServiceStatus = string(MCPServiceStatusDegraded)
		} else {
			summary.ServiceStatus = string(MCPServiceStatusUnavailable)
		}
	}

	// Transform server data (only for developer mode)
	var servers []MCPServerHealthForUI
	if mode == MCPHealthModeDeveloper {
		servers = make([]MCPServerHealthForUI, len(data.Servers))
		for i, server := range data.Servers {
			servers[i] = MCPServerHealthForUI{
				Alias:           server.Alias,
				Status:          server.Status,
				ToolCount:       server.ToolCount,
				StartedAt:       server.StartedAt,
				LastHealthCheck: server.LastHealthCheck,
				ErrorMessage:    server.ErrorMessage,
				Port:            server.Port,
				ProcessID:       server.ProcessID,
				SuccessRate:     server.SuccessRate,
				AvgResponseTime: server.AvgResponseTime,
				StatusIcon:      getStatusIcon(server.Status),
				StatusColor:     getStatusColor(server.Status),
				UptimeFormatted: formatUptime(server.StartedAt),
			}
		}
	}

	return summary, servers
}

// getStatusIcon returns the appropriate icon for a server status
func getStatusIcon(status string) string {
	switch MCPServerStatus(status) {
	case MCPServerStatusRunning:
		return "check-circle"
	case MCPServerStatusStopped:
		return "stop-circle"
	case MCPServerStatusError:
		return "x-circle"
	case MCPServerStatusStarting:
		return "play-circle"
	default:
		return "help-circle"
	}
}

// getStatusColor returns the appropriate color for a server status
func getStatusColor(status string) string {
	switch MCPServerStatus(status) {
	case MCPServerStatusRunning:
		return "green"
	case MCPServerStatusStopped:
		return "gray"
	case MCPServerStatusError:
		return "red"
	case MCPServerStatusStarting:
		return "yellow"
	default:
		return "gray"
	}
}

// formatUptime formats the uptime duration for display
func formatUptime(startedAt *time.Time) string {
	if startedAt == nil {
		return "N/A"
	}

	duration := time.Since(*startedAt)
	if duration < time.Minute {
		return "< 1m"
	} else if duration < time.Hour {
		return duration.Truncate(time.Minute).String()
	} else if duration < 24*time.Hour {
		return duration.Truncate(time.Hour).String()
	} else {
		days := int(duration.Hours() / 24)
		hours := int(duration.Hours()) % 24
		return fmt.Sprintf("%dd %dh", days, hours)
	}
}
