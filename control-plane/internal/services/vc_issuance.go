package services

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// GenerateExecutionVC generates a verifiable credential for an execution.
func (s *VCService) GenerateExecutionVC(ctx *types.ExecutionContext, inputData, outputData []byte, status string, errorMessage *string, durationMS int) (*types.ExecutionVC, error) {

	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	if !s.config.VCRequirements.RequireVCForExecution {
		// VC generation is disabled by configuration - return nil without error
		return nil, nil
	}

	// Basic validation with consistent null handling
	processedInputData := marshalDataOrNull(inputData)
	processedOutputData := marshalDataOrNull(outputData)

	// Simple error message handling
	var processedErrorMessage *string
	if errorMessage != nil {
		// Basic length limit for error messages
		msg := *errorMessage
		if len(msg) > 500 {
			msg = msg[:500] + "...[truncated]"
		}
		processedErrorMessage = &msg
	}

	// Resolve caller DID — fall back to agent's own DID for anonymous/external callers
	callerDID := ctx.CallerDID
	if callerDID == "" {
		callerDID = ctx.AgentNodeDID
	}
	if callerDID == "" {
		// No DID available at all — skip VC generation gracefully
		return nil, nil
	}
	callerIdentity, err := s.didService.ResolveDID(callerDID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve caller DID: %w", err)
	}

	// Handle optional target DID
	var targetIdentity *types.DIDIdentity
	if ctx.TargetDID != "" && ctx.TargetDID != "did:key:" {
		targetIdentity, err = s.didService.ResolveDID(ctx.TargetDID)
		if err != nil {
			// Target DID resolution failure is not critical - continue without target identity
			targetIdentity = nil
		}
	}

	// Generate hashes for processed data
	inputHash := s.hashData(processedInputData)
	outputHash := s.hashData(processedOutputData)

	// Create VC document with processed data
	vcDoc := s.createVCDocument(ctx, callerIdentity, targetIdentity, inputHash, outputHash, status, processedErrorMessage, durationMS)

	// Sign the VC
	signature, err := s.signVC(vcDoc, callerIdentity)
	if err != nil {
		return nil, fmt.Errorf("failed to sign VC: %w", err)
	}

	// Add proof to VC document
	vcDoc.Proof = types.VCProof{
		Type:               "Ed25519Signature2020",
		Created:            time.Now().UTC().Format(time.RFC3339),
		VerificationMethod: fmt.Sprintf("%s#key-1", callerDID),
		ProofPurpose:       "assertionMethod",
		ProofValue:         signature,
	}

	// Simple VC document serialization
	vcDocBytes, err := json.Marshal(vcDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal VC document: %w", err)
	}

	// Persist canonical execution status for VC metadata
	dbStatus := types.NormalizeExecutionStatus(status)

	// Create execution VC. ParentVCID propagates a chain pointer when the
	// caller (dispatcher, or an SDK that read X-Parent-VC-ID) passed one,
	// so audit chains extend across system boundaries (e.g. trigger event VC
	// → first reasoner's execution VC → downstream call VCs).
	var parentPtr *string
	if ctx.ParentVCID != "" {
		p := ctx.ParentVCID
		parentPtr = &p
	}
	executionVC := &types.ExecutionVC{
		VCID:         s.generateVCID(),
		ExecutionID:  ctx.ExecutionID,
		WorkflowID:   ctx.WorkflowID,
		SessionID:    ctx.SessionID,
		IssuerDID:    callerDID,
		TargetDID:    ctx.TargetDID,
		CallerDID:    callerDID,
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    signature,
		StorageURI:   "",
		DocumentSize: int64(len(vcDocBytes)),
		InputHash:    inputHash,
		OutputHash:   outputHash,
		Status:       dbStatus,
		CreatedAt:    time.Now(),
		Kind:         types.ExecutionVCKindExecution,
		ParentVCID:   parentPtr,
	}

	// Store VC
	if s.ShouldPersistExecutionVC() {
		ctxBg := context.Background()
		if err := s.vcStorage.StoreExecutionVC(ctxBg, executionVC); err != nil {
			return nil, fmt.Errorf("failed to store execution VC: %w", err)
		}
	} else {
		logger.Logger.Debug().Str("execution_id", ctx.ExecutionID).Msg("Execution VC persistence skipped by policy")
	}

	return executionVC, nil
}

// CreateWorkflowVC creates a workflow-level VC that aggregates execution VCs.
func (s *VCService) CreateWorkflowVC(workflowID, sessionID string, executionVCIDs []string) (*types.WorkflowVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Derive start time from the first execution VC if available.
	startTime := time.Now()
	if len(executionVCIDs) > 0 {
		if firstVC, err := s.vcStorage.GetExecutionVC(executionVCIDs[0]); err == nil {
			startTime = firstVC.CreatedAt
		}
	}

	workflowVC := &types.WorkflowVC{
		WorkflowID:     workflowID,
		SessionID:      sessionID,
		ComponentVCs:   executionVCIDs,
		WorkflowVCID:   s.generateVCID(),
		Status:         string(types.ExecutionStatusSucceeded),
		StartTime:      startTime,
		EndTime:        &[]time.Time{time.Now()}[0],
		TotalSteps:     len(executionVCIDs),
		CompletedSteps: len(executionVCIDs),
		StorageURI:     "",
		DocumentSize:   0,
	}

	// Store workflow VC
	if s.ShouldPersistExecutionVC() {
		ctx := context.Background()
		if err := s.vcStorage.StoreWorkflowVC(ctx, workflowVC); err != nil {
			return nil, fmt.Errorf("failed to store workflow VC: %w", err)
		}
	} else {
		logger.Logger.Debug().Str("workflow_id", workflowID).Msg("Workflow VC persistence skipped by policy")
	}

	return workflowVC, nil
}

// createVCDocument creates a VC document for an execution.
func (s *VCService) createVCDocument(ctx *types.ExecutionContext, callerIdentity, targetIdentity *types.DIDIdentity, inputHash, outputHash, status string, errorMessage *string, durationMS int) *types.VCDocument {
	vcID := s.generateVCID()

	credentialSubject := types.VCCredentialSubject{
		ExecutionID: ctx.ExecutionID,
		WorkflowID:  ctx.WorkflowID,
		SessionID:   ctx.SessionID,
		Caller: types.VCCaller{
			DID:          ctx.CallerDID,
			Type:         callerIdentity.ComponentType,
			AgentNodeDID: ctx.AgentNodeDID,
		},
		Target: types.VCTarget{
			DID:          ctx.TargetDID,
			AgentNodeDID: ctx.AgentNodeDID,
			FunctionName: func() string {
				if targetIdentity != nil {
					return targetIdentity.FunctionName
				}
				return "" // No target for standalone/root/leaf executions
			}(),
		},
		Execution: types.VCExecution{
			InputHash:  inputHash,
			OutputHash: outputHash,
			Timestamp:  ctx.Timestamp.UTC().Format(time.RFC3339),
			DurationMS: durationMS,
			Status:     status,
		},
		Audit: types.VCAudit{
			InputDataHash:  inputHash,
			OutputDataHash: outputHash,
			Metadata: map[string]interface{}{
				"agentfield_version": "1.0.0",
				"vc_version":         "1.0",
			},
		},
	}

	if errorMessage != nil {
		credentialSubject.Execution.ErrorMessage = *errorMessage
	}

	return &types.VCDocument{
		Context: []string{
			"https://www.w3.org/2018/credentials/v1",
			"https://agentfield.example.com/contexts/execution/v1",
		},
		Type: []string{
			"VerifiableCredential",
			"AgentFieldExecutionCredential",
		},
		ID:                fmt.Sprintf("urn:agentfield:vc:%s", vcID),
		Issuer:            ctx.CallerDID,
		IssuanceDate:      time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: credentialSubject,
	}
}

// signVC signs a VC document using the caller's private key.
func (s *VCService) signVC(vcDoc *types.VCDocument, callerIdentity *types.DIDIdentity) (string, error) {
	// Create canonical representation for signing
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for signing

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return "", fmt.Errorf("failed to marshal VC for signing: %w", err)
	}

	// Parse private key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(callerIdentity.PrivateKeyJWK), &jwk); err != nil {
		return "", fmt.Errorf("failed to parse private key JWK: %w", err)
	}

	dValue, ok := jwk["d"].(string)
	if !ok {
		return "", fmt.Errorf("invalid private key JWK: missing 'd' parameter")
	}

	privateKeySeed, err := base64.RawURLEncoding.DecodeString(dValue)
	if err != nil {
		return "", fmt.Errorf("failed to decode private key seed: %w", err)
	}

	if len(privateKeySeed) != ed25519.SeedSize {
		return "", fmt.Errorf("invalid private key seed length: got %d, want %d", len(privateKeySeed), ed25519.SeedSize)
	}

	privateKey := ed25519.NewKeyFromSeed(privateKeySeed)

	// Sign the canonical representation
	signature := ed25519.Sign(privateKey, canonicalBytes)

	return base64.RawURLEncoding.EncodeToString(signature), nil
}

// SignAgentTagVC signs an AgentTagVCDocument using the control plane's issuer DID.
// Returns the signed proof to be set on the VC document.
func (s *VCService) SignAgentTagVC(vc *types.AgentTagVCDocument) (*types.VCProof, error) {
	// Resolve the issuer's identity (control plane DID)
	issuerIdentity, err := s.didService.ResolveDID(vc.Issuer)
	if err != nil {
		return nil, fmt.Errorf("cannot resolve issuer DID %s for agent tag VC signing: %w", vc.Issuer, err)
	}

	// Create canonical representation (without proof) for signing
	vcCopy := *vc
	vcCopy.Proof = nil
	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal agent tag VC for signing: %w", err)
	}

	// Parse private key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PrivateKeyJWK), &jwk); err != nil {
		return nil, fmt.Errorf("failed to parse issuer private key JWK: %w", err)
	}

	dValue, ok := jwk["d"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid issuer private key JWK: missing 'd' parameter")
	}

	privateKeySeed, err := base64.RawURLEncoding.DecodeString(dValue)
	if err != nil {
		return nil, fmt.Errorf("failed to decode issuer private key seed: %w", err)
	}

	if len(privateKeySeed) != ed25519.SeedSize {
		return nil, fmt.Errorf("invalid issuer private key seed length: got %d, want %d", len(privateKeySeed), ed25519.SeedSize)
	}

	privateKey := ed25519.NewKeyFromSeed(privateKeySeed)
	signature := ed25519.Sign(privateKey, canonicalBytes)

	return &types.VCProof{
		Type:               "Ed25519Signature2020",
		Created:            time.Now().UTC().Format(time.RFC3339),
		VerificationMethod: fmt.Sprintf("%s#key-1", vc.Issuer),
		ProofPurpose:       "assertionMethod",
		ProofValue:         base64.RawURLEncoding.EncodeToString(signature),
	}, nil
}

// generateWorkflowVCDocument creates a WorkflowVC document on-demand.
func (s *VCService) generateWorkflowVCDocument(workflowID string, executionVCs []types.ExecutionVC) (*types.WorkflowVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Determine workflow status based on execution VCs
	status := s.determineWorkflowStatus(executionVCs)

	// Extract component VC IDs
	componentVCIDs := make([]string, len(executionVCs))
	for i, vc := range executionVCs {
		componentVCIDs[i] = vc.VCID
	}

	// Determine session ID from first execution VC
	sessionID := ""
	if len(executionVCs) > 0 {
		sessionID = executionVCs[0].SessionID
	}

	// Calculate start and end times
	var startTime time.Time
	var endTime *time.Time
	if len(executionVCs) > 0 {
		startTime = executionVCs[0].CreatedAt
		latestTime := executionVCs[0].CreatedAt
		for _, vc := range executionVCs {
			if vc.CreatedAt.Before(startTime) {
				startTime = vc.CreatedAt
			}
			if vc.CreatedAt.After(latestTime) {
				latestTime = vc.CreatedAt
			}
		}
		if types.IsTerminalExecutionStatus(status) {
			endTime = &latestTime
		}
	} else {
		startTime = time.Now()
	}

	// Get af server DID as issuer using dynamic resolution
	agentfieldServerID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("failed to get af server ID: %w", err)
	}

	registry, err := s.didService.GetRegistry(agentfieldServerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get af server DID: %w", err)
	}

	issuerDID := registry.RootDID
	if len(executionVCs) > 0 {
		// Use the issuer from the first execution VC if available
		issuerDID = executionVCs[0].IssuerDID
	}

	// Create WorkflowVC document
	workflowVCDoc := s.createWorkflowVCDocument(workflowID, sessionID, componentVCIDs, status, startTime, endTime, issuerDID)

	// Sign the WorkflowVC
	issuerIdentity, err := s.didService.ResolveDID(issuerDID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve issuer DID: %w", err)
	}

	signature, err := s.signWorkflowVC(workflowVCDoc, issuerIdentity)
	if err != nil {
		return nil, fmt.Errorf("failed to sign workflow VC: %w", err)
	}

	// Add proof to VC document
	workflowVCDoc.Proof = types.VCProof{
		Type:               "Ed25519Signature2020",
		Created:            time.Now().UTC().Format(time.RFC3339),
		VerificationMethod: fmt.Sprintf("%s#key-1", issuerDID),
		ProofPurpose:       "assertionMethod",
		ProofValue:         signature,
	}

	// Serialize VC document
	vcDocBytes, err := json.Marshal(workflowVCDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal workflow VC document: %w", err)
	}

	// Create WorkflowVC
	workflowVC := &types.WorkflowVC{
		WorkflowID:     workflowID,
		SessionID:      sessionID,
		ComponentVCs:   componentVCIDs,
		WorkflowVCID:   s.generateVCID(),
		Status:         status,
		StartTime:      startTime,
		EndTime:        endTime,
		TotalSteps:     len(executionVCs),
		CompletedSteps: s.countCompletedSteps(executionVCs),
		VCDocument:     json.RawMessage(vcDocBytes),
		Signature:      signature,
		IssuerDID:      issuerDID,
		SnapshotTime:   time.Now(),
		StorageURI:     "",
		DocumentSize:   int64(len(vcDocBytes)),
	}

	return workflowVC, nil
}

// createWorkflowVCDocument creates a WorkflowVC document.
func (s *VCService) createWorkflowVCDocument(workflowID, sessionID string, componentVCIDs []string, status string, startTime time.Time, endTime *time.Time, issuerDID string) *types.WorkflowVCDocument {
	vcID := s.generateVCID()

	credentialSubject := types.WorkflowVCCredentialSubject{
		WorkflowID:     workflowID,
		SessionID:      sessionID,
		ComponentVCIDs: componentVCIDs,
		TotalSteps:     len(componentVCIDs),
		CompletedSteps: len(componentVCIDs), // For now, assume all are completed
		Status:         status,
		StartTime:      startTime.UTC().Format(time.RFC3339),
		SnapshotTime:   time.Now().UTC().Format(time.RFC3339),
		Orchestrator: types.VCCaller{
			DID:          issuerDID,
			Type:         "agentfield_server",
			AgentNodeDID: issuerDID,
		},
		Audit: types.VCAudit{
			InputDataHash:  "", // Workflow-level doesn't have specific input/output
			OutputDataHash: "",
			Metadata: map[string]interface{}{
				"agentfield_version": "1.0.0",
				"vc_version":         "1.0",
				"workflow_type":      "agent_execution_chain",
				"total_executions":   len(componentVCIDs),
			},
		},
	}

	if endTime != nil {
		endTimeStr := endTime.UTC().Format(time.RFC3339)
		credentialSubject.EndTime = &endTimeStr
	}

	return &types.WorkflowVCDocument{
		Context: []string{
			"https://www.w3.org/2018/credentials/v1",
			"https://agentfield.example.com/contexts/workflow/v1",
		},
		Type: []string{
			"VerifiableCredential",
			"AgentFieldWorkflowCredential",
		},
		ID:                fmt.Sprintf("urn:agentfield:workflow-vc:%s", vcID),
		Issuer:            issuerDID,
		IssuanceDate:      time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: credentialSubject,
	}
}

// signWorkflowVC signs a WorkflowVC document.
func (s *VCService) signWorkflowVC(vcDoc *types.WorkflowVCDocument, issuerIdentity *types.DIDIdentity) (string, error) {
	// Create canonical representation for signing
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for signing

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return "", fmt.Errorf("failed to marshal workflow VC for signing: %w", err)
	}

	// Parse private key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PrivateKeyJWK), &jwk); err != nil {
		return "", fmt.Errorf("failed to parse private key JWK: %w", err)
	}

	dValue, ok := jwk["d"].(string)
	if !ok {
		return "", fmt.Errorf("invalid private key JWK: missing 'd' parameter")
	}

	privateKeySeed, err := base64.RawURLEncoding.DecodeString(dValue)
	if err != nil {
		return "", fmt.Errorf("failed to decode private key seed: %w", err)
	}

	if len(privateKeySeed) != ed25519.SeedSize {
		return "", fmt.Errorf("invalid private key seed length: got %d, want %d", len(privateKeySeed), ed25519.SeedSize)
	}

	privateKey := ed25519.NewKeyFromSeed(privateKeySeed)

	// Sign the canonical representation
	signature := ed25519.Sign(privateKey, canonicalBytes)

	return base64.RawURLEncoding.EncodeToString(signature), nil
}

// determineWorkflowStatus determines the overall status of a workflow based on execution VCs.
func (s *VCService) determineWorkflowStatus(executionVCs []types.ExecutionVC) string {
	if len(executionVCs) == 0 {
		return string(types.ExecutionStatusPending)
	}

	hasRunning := false
	hasQueued := false
	hasPending := false
	hasFailed := false
	hasCancelled := false
	hasTimeout := false
	hasUnknown := false

	for _, vc := range executionVCs {
		normalized := types.NormalizeExecutionStatus(vc.Status)
		switch normalized {
		case string(types.ExecutionStatusFailed):
			hasFailed = true
		case string(types.ExecutionStatusCancelled):
			hasCancelled = true
		case string(types.ExecutionStatusTimeout):
			hasTimeout = true
		case string(types.ExecutionStatusRunning):
			hasRunning = true
		case string(types.ExecutionStatusQueued):
			hasQueued = true
		case string(types.ExecutionStatusPending):
			hasPending = true
		case string(types.ExecutionStatusUnknown):
			hasUnknown = true
		}
	}

	switch {
	case hasFailed:
		return string(types.ExecutionStatusFailed)
	case hasTimeout:
		return string(types.ExecutionStatusTimeout)
	case hasCancelled:
		return string(types.ExecutionStatusCancelled)
	case hasRunning:
		return string(types.ExecutionStatusRunning)
	case hasQueued:
		return string(types.ExecutionStatusQueued)
	case hasPending:
		return string(types.ExecutionStatusPending)
	case hasUnknown:
		return string(types.ExecutionStatusUnknown)
	default:
		return string(types.ExecutionStatusSucceeded)
	}
}

// countCompletedSteps counts the number of completed execution VCs.
func (s *VCService) countCompletedSteps(executionVCs []types.ExecutionVC) int {
	count := 0
	for _, vc := range executionVCs {
		if types.NormalizeExecutionStatus(vc.Status) == string(types.ExecutionStatusSucceeded) {
			count++
		}
	}
	return count
}
