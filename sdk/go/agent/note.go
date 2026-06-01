package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// notePayload represents the JSON payload sent to the AgentField server.
type notePayload struct {
	Message     string   `json:"message"`
	Tags        []string `json:"tags"`
	Timestamp   float64  `json:"timestamp"`
	AgentNodeID string   `json:"agent_node_id"`
}

// Note sends a progress/status message to the AgentField server.
// This is useful for debugging and tracking agent execution progress
// in the AgentField UI.
//
// Notes are sent asynchronously (fire-and-forget) and will not block
// the handler or raise errors that interrupt the workflow.
//
// Example usage:
//
//	agent.Note(ctx, "Starting data processing", "debug", "processing")
//	// ... do work ...
//	agent.Note(ctx, "Processing completed", "info")
func (a *Agent) Note(ctx context.Context, message string, tags ...string) {
	if tags == nil {
		tags = []string{}
	}

	// Fire-and-forget: send note in a goroutine
	go a.sendNote(ctx, message, tags)
}

// Notef sends a formatted progress/status message to the AgentField server.
// This is a convenience method that formats the message using fmt.Sprintf.
//
// Example usage:
//
//	agent.Notef(ctx, "Processing %d items...", len(items))
func (a *Agent) Notef(ctx context.Context, format string, args ...any) {
	a.Note(ctx, fmt.Sprintf(format, args...))
}

// sendNote performs the actual HTTP request to send the note.
func (a *Agent) sendNote(ctx context.Context, message string, tags []string) {
	// Check if AgentField URL is configured
	baseURL := strings.TrimSpace(a.cfg.AgentFieldURL)
	if baseURL == "" {
		// No server configured, silently skip
		return
	}

	// Get execution context from the provided context
	execCtx := ExecutionContextFrom(ctx)

	// Build UI API URL (notes go to /api/ui/v1, not /api/v1)
	uiAPIURL := strings.Replace(baseURL, "/api/v1", "/api/ui/v1", 1)
	if !strings.Contains(uiAPIURL, "/api/ui/v1") {
		// If no /api/v1 was found, append /api/ui/v1
		uiAPIURL = strings.TrimSuffix(baseURL, "/") + "/api/ui/v1"
	}
	noteURL := uiAPIURL + "/executions/note"

	// Build payload
	payload := notePayload{
		Message:     message,
		Tags:        tags,
		Timestamp:   float64(time.Now().UnixNano()) / 1e9, // Unix timestamp as float
		AgentNodeID: a.cfg.NodeID,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		a.logger.Printf("note: failed to marshal payload: %v", err)
		return
	}

	// Build request with execution context headers
	reqCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, noteURL, bytes.NewReader(body))
	if err != nil {
		a.logger.Printf("note: failed to create request: %v", err)
		return
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if a.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+a.cfg.Token)
	}

	// Add execution context headers
	if execCtx.RunID != "" {
		req.Header.Set("X-Run-ID", execCtx.RunID)
	}
	if execCtx.ExecutionID != "" {
		req.Header.Set("X-Execution-ID", execCtx.ExecutionID)
	}
	if execCtx.SessionID != "" {
		req.Header.Set("X-Session-ID", execCtx.SessionID)
	}
	if execCtx.ActorID != "" {
		req.Header.Set("X-Actor-ID", execCtx.ActorID)
	}
	if execCtx.WorkflowID != "" {
		req.Header.Set("X-Workflow-ID", execCtx.WorkflowID)
	}
	req.Header.Set("X-Agent-Node-ID", a.cfg.NodeID)

	// Send request
	resp, err := a.httpClient.Do(req)
	if err != nil {
		// Silently fail - notes should not interrupt workflow
		return
	}
	defer resp.Body.Close()

	// We don't care about the response for fire-and-forget notes
	// but we could log errors for debugging
	if resp.StatusCode >= 400 {
		a.logger.Printf("note: server returned status %d", resp.StatusCode)
	}
}
