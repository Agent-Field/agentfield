package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/did"
)

// DIDManager returns the agent's DID manager, or nil if DID is not enabled.
func (a *Agent) DIDManager() *did.Manager {
	return a.didManager
}

// VCGenerator returns the agent's VC generator, or nil if VC generation is not enabled.
func (a *Agent) VCGenerator() *did.VCGenerator {
	return a.vcGenerator
}

// initializeDIDSystem sets up DID registration and VC generation.
// If DID/PrivateKeyJWK are already configured, it skips auto-registration
// but still sets up the DID manager and VC generator.
func (a *Agent) initializeDIDSystem(ctx context.Context) error {
	// Create DID HTTP client for DID endpoints.
	didClientOpts := []did.ClientOption{did.WithHTTPClient(a.httpClient)}
	if a.cfg.Token != "" {
		didClientOpts = append(didClientOpts, did.WithToken(a.cfg.Token))
	}
	didClient := did.NewClient(a.cfg.AgentFieldURL, didClientOpts...)

	// Create DID manager.
	mgr := did.NewManager(didClient, a.logger)

	if a.cfg.DID != "" && a.cfg.PrivateKeyJWK != "" {
		// Agent already has credentials — skip registration, just populate the manager.
		mgr.SetIdentityFromCredentials(a.cfg.DID, a.cfg.PrivateKeyJWK)
	} else {
		// Auto-register with the control plane's DID service.
		reasonerNames := make([]string, 0, len(a.reasoners))
		for name := range a.reasoners {
			reasonerNames = append(reasonerNames, name)
		}

		if err := mgr.RegisterAgent(ctx, a.cfg.NodeID, reasonerNames, nil); err != nil {
			return fmt.Errorf("DID registration: %w", err)
		}

		// Wire the new credentials into the HTTP client.
		agentDID := mgr.GetAgentDID()
		privateKey := mgr.GetAgentPrivateKeyJWK()
		if agentDID != "" && privateKey != "" {
			if err := a.client.SetDIDCredentials(agentDID, privateKey); err != nil {
				return fmt.Errorf("set DID credentials: %w", err)
			}
			// Update config so Call() and other paths can see the DID.
			a.cfg.DID = agentDID
			a.cfg.PrivateKeyJWK = privateKey
		}
	}

	a.didManager = mgr

	// Wire the sign function on the DID client so VC generation requests are DID-signed.
	didClient.SetSignFunc(func(body []byte) map[string]string {
		if a.client == nil {
			return nil
		}
		return a.client.SignBody(body)
	})

	// Set up VC generator if enabled and DID auth is configured.
	if a.cfg.VCEnabled && a.client != nil && a.client.DIDAuthConfigured() {
		gen := did.NewVCGenerator(didClient, mgr, a.logger)
		gen.SetEnabled(true)
		a.vcGenerator = gen
		a.logger.Printf("VC generation enabled")
	}

	return nil
}

// fillDIDContext populates DID fields on an execution context from the agent's
// DID manager, if available and not already set from headers.
func (a *Agent) fillDIDContext(ec *ExecutionContext) {
	if a.didManager == nil || !a.didManager.IsRegistered() {
		return
	}
	if ec.AgentNodeDID == "" {
		ec.AgentNodeDID = a.didManager.GetAgentDID()
	}
}

// maybeGenerateVC fires a background VC generation request if the agent and
// reasoner configuration allow it.
func (a *Agent) maybeGenerateVC(
	execCtx ExecutionContext,
	input any,
	output any,
	status string,
	errMsg string,
	durationMS int64,
	reasoner *Reasoner,
) {
	if !a.shouldGenerateVC(reasoner) {
		return
	}

	if execCtx.CallerDID == "" {
		a.logger.Printf("⚠️ VC generation for %s: CallerDID is empty (anonymous caller?), control plane will use fallback DID", execCtx.ExecutionID)
	}

	didExecCtx := did.ExecutionContext{
		ExecutionID:  execCtx.ExecutionID,
		WorkflowID:   execCtx.WorkflowID,
		SessionID:    execCtx.SessionID,
		CallerDID:    execCtx.CallerDID,
		TargetDID:    execCtx.TargetDID,
		AgentNodeDID: execCtx.AgentNodeDID,
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := a.vcGenerator.GenerateExecutionVC(ctx, didExecCtx, input, output, status, errMsg, durationMS); err != nil {
			a.logger.Printf("VC generation failed for %s: %v", execCtx.ExecutionID, err)
		}
	}()
}

// shouldGenerateVC checks agent-level and reasoner-level VC settings.
func (a *Agent) shouldGenerateVC(reasoner *Reasoner) bool {
	if a.vcGenerator == nil || !a.vcGenerator.IsEnabled() {
		return false
	}
	if a.didManager == nil || !a.didManager.IsRegistered() {
		return false
	}
	// Per-reasoner override takes precedence.
	if reasoner != nil && reasoner.VCEnabled != nil {
		return *reasoner.VCEnabled
	}
	return true
}
