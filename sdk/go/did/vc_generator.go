package did

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"sync"
	"time"
)

// VCGenerator handles Verifiable Credential generation for agent executions.
// After a reasoner completes, the generator sends execution metadata to the
// control plane which creates and stores a W3C-compliant VC for the audit trail.
type VCGenerator struct {
	client  *Client
	manager *Manager
	enabled bool
	mu      sync.RWMutex
	logger  *log.Logger
}

// NewVCGenerator creates a VC generator. Generation is disabled by default;
// call SetEnabled(true) after DID registration succeeds.
func NewVCGenerator(client *Client, manager *Manager, logger *log.Logger) *VCGenerator {
	if logger == nil {
		logger = log.Default()
	}
	return &VCGenerator{
		client:  client,
		manager: manager,
		logger:  logger,
	}
}

// SetEnabled enables or disables VC generation.
func (g *VCGenerator) SetEnabled(enabled bool) {
	g.mu.Lock()
	g.enabled = enabled
	g.mu.Unlock()
}

// IsEnabled returns true if VC generation is active.
func (g *VCGenerator) IsEnabled() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.enabled
}

// GenerateExecutionVC creates a Verifiable Credential for a completed execution.
// The input and output are serialized to JSON and base64-encoded before being
// sent to the control plane, matching the Python and TypeScript SDK behavior.
func (g *VCGenerator) GenerateExecutionVC(
	ctx context.Context,
	execCtx ExecutionContext,
	input any,
	output any,
	status string,
	errMsg string,
	durationMS int64,
) (*ExecutionVC, error) {
	if !g.IsEnabled() {
		return nil, nil
	}

	// Fill in agent's own DID if not already set from headers.
	// CallerDID and TargetDID come from X-Caller-DID / X-Target-DID headers
	// forwarded by the control plane — we must NOT overwrite them with the
	// agent's own DID.
	if execCtx.AgentNodeDID == "" && g.manager != nil {
		execCtx.AgentNodeDID = g.manager.GetAgentDID()
	}
	if execCtx.Timestamp == "" {
		execCtx.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	inputData := encodeData(input)
	outputData := encodeData(output)

	req := VCGenerationRequest{
		ExecutionContext: execCtx,
		InputData:        inputData,
		OutputData:       outputData,
		Status:           status,
		ErrorMessage:     errMsg,
		DurationMS:       durationMS,
	}

	vc, err := g.client.GenerateExecutionVC(ctx, req)
	if err != nil {
		return nil, err
	}

	g.logger.Printf("generated VC %s for execution %s", vc.VCID, execCtx.ExecutionID)
	return vc, nil
}

// ExportWorkflowVCChain retrieves the complete VC chain for a workflow.
func (g *VCGenerator) ExportWorkflowVCChain(ctx context.Context, workflowID string) (*WorkflowVCChain, error) {
	return g.client.ExportWorkflowVCChain(ctx, workflowID)
}

// encodeData serializes a value to a base64-encoded JSON string,
// matching the Python and TypeScript SDK convention.
func encodeData(v any) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(b)
}
