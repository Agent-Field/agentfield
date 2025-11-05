package services

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/config"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"
)

// VCService handles verifiable credential generation, verification, and management.
type VCService struct {
	config     *config.DIDConfig
	didService *DIDService
	vcStorage  *VCStorage
}

// NewVCService creates a new VC service instance with database storage.
func NewVCService(cfg *config.DIDConfig, didService *DIDService, storageProvider storage.StorageProvider) *VCService {
	return &VCService{
		config:     cfg,
		didService: didService,
		vcStorage:  NewVCStorageWithStorage(storageProvider),
	}
}

// Initialize initializes the VC service.
func (s *VCService) Initialize() error {
	if !s.config.Enabled {
		return nil
	}

	return s.vcStorage.Initialize()
}

// GetDIDService returns the DID service instance for DID resolution operations.
func (s *VCService) GetDIDService() *DIDService {
	return s.didService
}

// IsExecutionVCEnabled reports whether execution VC generation should run
// based on DID being enabled and the execution VC requirement flag.
func (s *VCService) IsExecutionVCEnabled() bool {
	if s == nil || s.config == nil {
		return false
	}
	if !s.config.Enabled {
		return false
	}
	return s.config.VCRequirements.RequireVCForExecution
}

// ShouldPersistExecutionVC reports whether execution VCs should be persisted after generation.
func (s *VCService) ShouldPersistExecutionVC() bool {
	if s == nil || s.config == nil {
		return false
	}
	if !s.config.Enabled {
		return false
	}
	return s.config.VCRequirements.PersistExecutionVC
}

// GenerateExecutionVC generates a verifiable credential for an execution.
func (s *VCService) GenerateExecutionVC(ctx *types.ExecutionContext, inputData, outputData []byte, status string, errorMessage *string, durationMS int) (*types.ExecutionVC, error) {

	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	if !s.config.VCRequirements.RequireVCForExecution {
		return nil, fmt.Errorf("execution VC generation disabled by configuration")
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

	// Resolve caller DID
	callerIdentity, err := s.didService.ResolveDID(ctx.CallerDID)
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
		VerificationMethod: fmt.Sprintf("%s#key-1", ctx.CallerDID),
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

	// Create execution VC
	executionVC := &types.ExecutionVC{
		VCID:         s.generateVCID(),
		ExecutionID:  ctx.ExecutionID,
		WorkflowID:   ctx.WorkflowID,
		SessionID:    ctx.SessionID,
		IssuerDID:    ctx.CallerDID,
		TargetDID:    ctx.TargetDID,
		CallerDID:    ctx.CallerDID,
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    signature,
		StorageURI:   "",
		DocumentSize: int64(len(vcDocBytes)),
		InputHash:    inputHash,
		OutputHash:   outputHash,
		Status:       dbStatus,
		CreatedAt:    time.Now(),
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

// VerifyVC verifies a verifiable credential.
func (s *VCService) VerifyVC(vcDocument json.RawMessage) (*types.VCVerificationResponse, error) {
	if !s.config.Enabled {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: "DID system is disabled",
		}, nil
	}

	var vcDoc types.VCDocument
	if err := json.Unmarshal(vcDocument, &vcDoc); err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to parse VC document: %v", err),
		}, nil
	}

	// Resolve issuer DID
	issuerIdentity, err := s.didService.ResolveDID(vcDoc.Issuer)
	if err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to resolve issuer DID: %v", err),
		}, nil
	}

	// Verify signature
	valid, err := s.verifyVCSignature(&vcDoc, issuerIdentity)
	if err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to verify signature: %v", err),
		}, nil
	}

	if !valid {
		return &types.VCVerificationResponse{
			Valid:   false,
			Message: "Invalid signature",
		}, nil
	}

	return &types.VCVerificationResponse{
		Valid:     true,
		IssuerDID: vcDoc.Issuer,
		IssuedAt:  vcDoc.IssuanceDate,
		Message:   "VC verified successfully",
	}, nil
}

// GetWorkflowVCChain retrieves the complete VC chain for a workflow.
func (s *VCService) GetWorkflowVCChain(workflowID string) (*types.WorkflowVCChainResponse, error) {
	logger.Logger.Debug().Msgf("🔍 GetWorkflowVCChain called for workflow: %s", workflowID)
	logger.Logger.Debug().Msgf("🔍 DID system enabled: %v", s.config.Enabled)

	if !s.config.Enabled {
		logger.Logger.Debug().Msg("🔍 DID system is disabled")
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Get all execution VCs for the workflow
	logger.Logger.Debug().Msgf("🔍 Querying execution VCs for workflow: %s", workflowID)
	executionVCs, err := s.vcStorage.GetExecutionVCsByWorkflow(workflowID)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to get execution VCs")
		return nil, fmt.Errorf("failed to get execution VCs: %w", err)
	}
	logger.Logger.Debug().Msgf("🔍 Found %d execution VCs for workflow %s", len(executionVCs), workflowID)

	// Generate WorkflowVC on-demand with current state
	logger.Logger.Debug().Msgf("🔍 Generating WorkflowVC on-demand for workflow: %s", workflowID)
	workflowVC, err := s.generateWorkflowVCDocument(workflowID, executionVCs)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to generate WorkflowVC")
		return nil, fmt.Errorf("failed to generate workflow VC: %w", err)
	}
	logger.Logger.Debug().Msgf("🔍 Generated WorkflowVC with ID: %s, status: %s", workflowVC.WorkflowVCID, workflowVC.Status)

	// Collect DID resolution bundle for offline verification
	logger.Logger.Debug().Msgf("🔍 Collecting DID resolution bundle for workflow: %s", workflowID)
	didResolutionBundle, err := s.collectDIDResolutionBundle(executionVCs, workflowVC)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to collect DID resolution bundle")
		// Don't fail the entire request if DID resolution fails - just log and continue without bundle
		didResolutionBundle = make(map[string]types.DIDResolutionEntry)
	}
	logger.Logger.Debug().Msgf("🔍 Collected %d DID resolution entries", len(didResolutionBundle))

	logger.Logger.Debug().Msgf("🔍 Returning VC chain with %d execution VCs and workflow status: %s", len(executionVCs), workflowVC.Status)
	return &types.WorkflowVCChainResponse{
		WorkflowID:          workflowID,
		ComponentVCs:        executionVCs,
		WorkflowVC:          *workflowVC,
		TotalSteps:          len(executionVCs),
		Status:              workflowVC.Status,
		DIDResolutionBundle: didResolutionBundle,
	}, nil
}

// CreateWorkflowVC creates a workflow-level VC that aggregates execution VCs.
func (s *VCService) CreateWorkflowVC(workflowID, sessionID string, executionVCIDs []string) (*types.WorkflowVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}

	workflowVC := &types.WorkflowVC{
		WorkflowID:     workflowID,
		SessionID:      sessionID,
		ComponentVCs:   executionVCIDs,
		WorkflowVCID:   s.generateVCID(),
		Status:         string(types.ExecutionStatusSucceeded),
		StartTime:      time.Now(), // TODO: Get actual start time from first execution
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

	privateKey := ed25519.NewKeyFromSeed(privateKeySeed)

	// Sign the canonical representation
	signature := ed25519.Sign(privateKey, canonicalBytes)

	return base64.RawURLEncoding.EncodeToString(signature), nil
}

// verifyVCSignature verifies the signature of a VC document.
func (s *VCService) verifyVCSignature(vcDoc *types.VCDocument, issuerIdentity *types.DIDIdentity) (bool, error) {
	// Create canonical representation for verification
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal VC for verification: %w", err)
	}

	// Parse public key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("failed to parse public key JWK: %w", err)
	}

	xValue, ok := jwk["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)

	// Decode signature
	signatureBytes, err := base64.RawURLEncoding.DecodeString(vcDoc.Proof.ProofValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Verify signature
	return ed25519.Verify(publicKey, canonicalBytes, signatureBytes), nil
}

// hashData creates a SHA-256 hash of data.
func (s *VCService) hashData(data []byte) string {
	if !s.config.VCRequirements.HashSensitiveData {
		return ""
	}

	hash := sha256.Sum256(data)
	return base64.RawURLEncoding.EncodeToString(hash[:])
}

// generateVCID generates a unique VC ID.
func (s *VCService) generateVCID() string {
	// Simple UUID-like generation for now
	// In production, use proper UUID library
	return fmt.Sprintf("vc-%d", time.Now().UnixNano())
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

// QueryExecutionVCs queries execution VCs with filters.
func (s *VCService) QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.QueryExecutionVCs(filters)
}

// GetExecutionVCByExecutionID retrieves a single execution VC by execution identifier.
func (s *VCService) GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.GetExecutionVCByExecutionID(executionID)
}

// ListWorkflowVCs lists all workflow VCs.
func (s *VCService) ListWorkflowVCs() ([]*types.WorkflowVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.ListWorkflowVCs()
}

// collectDIDResolutionBundle collects all unique DIDs from the VC chain and resolves their public keys.
func (s *VCService) collectDIDResolutionBundle(executionVCs []types.ExecutionVC, workflowVC *types.WorkflowVC) (map[string]types.DIDResolutionEntry, error) {
	bundle := make(map[string]types.DIDResolutionEntry)
	resolvedAt := time.Now().UTC().Format(time.RFC3339)

	// Collect unique DIDs from execution VCs
	uniqueDIDs := make(map[string]bool)

	for _, vc := range executionVCs {
		if vc.IssuerDID != "" && vc.IssuerDID != "did:key:" {
			uniqueDIDs[vc.IssuerDID] = true
		}
		if vc.CallerDID != "" && vc.CallerDID != "did:key:" {
			uniqueDIDs[vc.CallerDID] = true
		}
		if vc.TargetDID != "" && vc.TargetDID != "did:key:" {
			uniqueDIDs[vc.TargetDID] = true
		}
	}

	// Add workflow VC issuer DID
	if workflowVC.IssuerDID != "" && workflowVC.IssuerDID != "did:key:" {
		uniqueDIDs[workflowVC.IssuerDID] = true
	}

	// Resolve each unique DID and collect public keys
	for did := range uniqueDIDs {
		if did == "" || did == "did:key:" || len(strings.TrimSpace(did)) == 0 {
			continue // Skip empty or incomplete DIDs
		}

		identity, err := s.didService.ResolveDID(did)
		if err != nil {
			continue // Skip DIDs that can't be resolved
		}

		// Determine DID method from the DID string
		method := "key" // Default to "key" method
		if len(did) > 4 && did[:4] == "did:" {
			parts := strings.Split(did, ":")
			if len(parts) >= 2 {
				method = parts[1]
			}
		}

		// Parse the public key JWK string into a JSON object
		var publicKeyJWK map[string]interface{}
		if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
			continue // Skip DIDs with invalid public key JWK
		}

		// Create resolution entry with properly parsed public key JWK
		bundle[did] = types.DIDResolutionEntry{
			Method:       method,
			PublicKeyJWK: json.RawMessage(identity.PublicKeyJWK), // Keep as raw JSON
			ResolvedFrom: "bundled",
			ResolvedAt:   resolvedAt,
		}

	}
	return bundle, nil
}

// VerificationIssue represents a specific verification problem
type VerificationIssue struct {
	Type        string `json:"type"`
	Severity    string `json:"severity"` // "critical", "warning", "info"
	Component   string `json:"component"`
	Field       string `json:"field"`
	Expected    string `json:"expected"`
	Actual      string `json:"actual"`
	Description string `json:"description"`
}

// ComprehensiveVCVerificationResult provides detailed verification results
type ComprehensiveVCVerificationResult struct {
	Valid                 bool                  `json:"valid"`
	OverallScore          float64               `json:"overall_score"` // 0-100
	CriticalIssues        []VerificationIssue   `json:"critical_issues"`
	Warnings              []VerificationIssue   `json:"warnings"`
	IntegrityChecks       IntegrityCheckResults `json:"integrity_checks"`
	SecurityAnalysis      SecurityAnalysis      `json:"security_analysis"`
	ComplianceChecks      ComplianceChecks      `json:"compliance_checks"`
	VerificationTimestamp string                `json:"verification_timestamp"`
}

// IntegrityCheckResults represents various integrity verification results
type IntegrityCheckResults struct {
	MetadataConsistency bool                `json:"metadata_consistency"`
	FieldConsistency    bool                `json:"field_consistency"`
	TimestampValidation bool                `json:"timestamp_validation"`
	HashValidation      bool                `json:"hash_validation"`
	StructuralIntegrity bool                `json:"structural_integrity"`
	Issues              []VerificationIssue `json:"issues"`
}

// SecurityAnalysis represents security-focused verification results
type SecurityAnalysis struct {
	SignatureStrength string              `json:"signature_strength"`
	KeyValidation     bool                `json:"key_validation"`
	DIDAuthenticity   bool                `json:"did_authenticity"`
	ReplayProtection  bool                `json:"replay_protection"`
	TamperEvidence    []string            `json:"tamper_evidence"`
	SecurityScore     float64             `json:"security_score"`
	Issues            []VerificationIssue `json:"issues"`
}

// ComplianceChecks represents compliance and audit verification results
type ComplianceChecks struct {
	W3CCompliance                bool                `json:"w3c_compliance"`
	AgentFieldStandardCompliance bool                `json:"agentfield_standard_compliance"`
	AuditTrailIntegrity          bool                `json:"audit_trail_integrity"`
	DataIntegrityChecks          bool                `json:"data_integrity_checks"`
	Issues                       []VerificationIssue `json:"issues"`
}

// VerifyExecutionVCComprehensive performs comprehensive verification of an execution VC
func (s *VCService) VerifyExecutionVCComprehensive(executionID string) (*ComprehensiveVCVerificationResult, error) {
	if !s.config.Enabled {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "system_disabled", Severity: "critical", Description: "DID system is disabled"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	// Get the execution VC
	filters := &types.VCFilters{Limit: 1000}
	executionVCs, err := s.vcStorage.QueryExecutionVCs(filters)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution VCs: %w", err)
	}

	var executionVC *types.ExecutionVC
	for _, vc := range executionVCs {
		if vc.ExecutionID == executionID {
			executionVC = &vc
			break
		}
	}

	if executionVC == nil {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "vc_not_found", Severity: "critical", Description: "VC not found for execution"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	result := &ComprehensiveVCVerificationResult{
		VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		CriticalIssues:        []VerificationIssue{},
		Warnings:              []VerificationIssue{},
	}

	// Parse VC document
	var vcDoc types.VCDocument
	if err := json.Unmarshal(executionVC.VCDocument, &vcDoc); err != nil {
		result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
			Type:        "parse_error",
			Severity:    "critical",
			Component:   executionVC.VCID,
			Description: fmt.Sprintf("Failed to parse VC document: %v", err),
		})
		result.Valid = false
		result.OverallScore = 0
		return result, nil
	}

	// Perform comprehensive verification checks
	result.IntegrityChecks = s.performIntegrityChecks(executionVC, &vcDoc)
	result.SecurityAnalysis = s.performSecurityAnalysis(executionVC, &vcDoc)
	result.ComplianceChecks = s.performComplianceChecks(&vcDoc)

	// Collect all issues
	allIssues := []VerificationIssue{}
	allIssues = append(allIssues, result.IntegrityChecks.Issues...)
	allIssues = append(allIssues, result.SecurityAnalysis.Issues...)
	allIssues = append(allIssues, result.ComplianceChecks.Issues...)

	// Separate critical issues and warnings
	for _, issue := range allIssues {
		if issue.Severity == "critical" {
			result.CriticalIssues = append(result.CriticalIssues, issue)
		} else if issue.Severity == "warning" {
			result.Warnings = append(result.Warnings, issue)
		}
	}

	// Calculate overall validity and score
	result.Valid = len(result.CriticalIssues) == 0
	result.OverallScore = s.calculateOverallScore(result)

	return result, nil
}

// performIntegrityChecks performs various integrity checks on the VC
func (s *VCService) performIntegrityChecks(execVC *types.ExecutionVC, vcDoc *types.VCDocument) IntegrityCheckResults {
	result := IntegrityCheckResults{
		MetadataConsistency: true,
		FieldConsistency:    true,
		TimestampValidation: true,
		HashValidation:      true,
		StructuralIntegrity: true,
		Issues:              []VerificationIssue{},
	}

	// CRITICAL CHECK 1: Metadata consistency between top-level and VC document
	if execVC.IssuerDID != vcDoc.Issuer {
		result.MetadataConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "issuer_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "issuer_did",
			Expected:    execVC.IssuerDID,
			Actual:      vcDoc.Issuer,
			Description: "Issuer DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 2: Execution ID consistency
	if execVC.ExecutionID != vcDoc.CredentialSubject.ExecutionID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "execution_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "execution_id",
			Expected:    execVC.ExecutionID,
			Actual:      vcDoc.CredentialSubject.ExecutionID,
			Description: "Execution ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 3: Workflow ID consistency
	if execVC.WorkflowID != vcDoc.CredentialSubject.WorkflowID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "workflow_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "workflow_id",
			Expected:    execVC.WorkflowID,
			Actual:      vcDoc.CredentialSubject.WorkflowID,
			Description: "Workflow ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 4: Session ID consistency
	if execVC.SessionID != vcDoc.CredentialSubject.SessionID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "session_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "session_id",
			Expected:    execVC.SessionID,
			Actual:      vcDoc.CredentialSubject.SessionID,
			Description: "Session ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 5: Caller DID consistency
	if execVC.CallerDID != vcDoc.CredentialSubject.Caller.DID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "caller_did_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "caller_did",
			Expected:    execVC.CallerDID,
			Actual:      vcDoc.CredentialSubject.Caller.DID,
			Description: "Caller DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 6: Target DID consistency
	if execVC.TargetDID != vcDoc.CredentialSubject.Target.DID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "target_did_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "target_did",
			Expected:    execVC.TargetDID,
			Actual:      vcDoc.CredentialSubject.Target.DID,
			Description: "Target DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 7: Status consistency (with AgentField system status mapping)
	if !s.isStatusConsistent(execVC.Status, vcDoc.CredentialSubject.Execution.Status) {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "status_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "status",
			Expected:    execVC.Status,
			Actual:      vcDoc.CredentialSubject.Execution.Status,
			Description: "Status mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 8: Hash consistency
	if execVC.InputHash != vcDoc.CredentialSubject.Execution.InputHash {
		result.HashValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "input_hash_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "input_hash",
			Expected:    execVC.InputHash,
			Actual:      vcDoc.CredentialSubject.Execution.InputHash,
			Description: "Input hash mismatch between metadata and VC document",
		})
	}

	if execVC.OutputHash != vcDoc.CredentialSubject.Execution.OutputHash {
		result.HashValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "output_hash_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "output_hash",
			Expected:    execVC.OutputHash,
			Actual:      vcDoc.CredentialSubject.Execution.OutputHash,
			Description: "Output hash mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 9: Signature consistency
	if execVC.Signature != vcDoc.Proof.ProofValue {
		result.StructuralIntegrity = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "signature_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "signature",
			Expected:    execVC.Signature,
			Actual:      vcDoc.Proof.ProofValue,
			Description: "Signature mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 10: Timestamp validation
	if err := s.validateTimestamp(vcDoc.IssuanceDate); err != nil {
		result.TimestampValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "invalid_timestamp",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "issuance_date",
			Description: fmt.Sprintf("Invalid timestamp: %v", err),
		})
	}

	// CRITICAL CHECK 11: VC structure validation
	if err := s.validateVCStructure(vcDoc); err != nil {
		result.StructuralIntegrity = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "invalid_structure",
			Severity:    "critical",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Invalid VC structure: %v", err),
		})
	}

	return result
}

// performSecurityAnalysis performs security-focused analysis
func (s *VCService) performSecurityAnalysis(execVC *types.ExecutionVC, vcDoc *types.VCDocument) SecurityAnalysis {
	result := SecurityAnalysis{
		SignatureStrength: "Ed25519",
		KeyValidation:     true,
		DIDAuthenticity:   true,
		ReplayProtection:  true,
		TamperEvidence:    []string{},
		SecurityScore:     100.0,
		Issues:            []VerificationIssue{},
	}

	// CRITICAL CHECK: Cryptographic signature verification
	issuerIdentity, err := s.didService.ResolveDID(vcDoc.Issuer)
	if err != nil {
		result.DIDAuthenticity = false
		result.SecurityScore -= 50.0
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "did_resolution_failed",
			Severity:    "critical",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Failed to resolve issuer DID: %v", err),
		})
	} else {
		valid, err := s.verifyVCSignature(vcDoc, issuerIdentity)
		if err != nil || !valid {
			result.KeyValidation = false
			result.SecurityScore -= 40.0
			result.Issues = append(result.Issues, VerificationIssue{
				Type:        "signature_verification_failed",
				Severity:    "critical",
				Component:   execVC.VCID,
				Description: fmt.Sprintf("Signature verification failed: %v", err),
			})
		}
	}

	// Check for tamper evidence
	if evidence := s.detectTamperEvidence(execVC, vcDoc); len(evidence) > 0 {
		result.TamperEvidence = evidence
		result.SecurityScore -= 20.0
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "tamper_evidence",
			Severity:    "warning",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Tamper evidence detected: %v", evidence),
		})
	}

	return result
}

// performComplianceChecks performs compliance verification
func (s *VCService) performComplianceChecks(vcDoc *types.VCDocument) ComplianceChecks {
	result := ComplianceChecks{
		W3CCompliance:                true,
		AgentFieldStandardCompliance: true,
		AuditTrailIntegrity:          true,
		DataIntegrityChecks:          true,
		Issues:                       []VerificationIssue{},
	}

	// Check W3C compliance
	if !s.checkW3CCompliance(vcDoc) {
		result.W3CCompliance = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "w3c_compliance_failure",
			Severity:    "warning",
			Component:   vcDoc.ID,
			Description: "VC does not meet W3C standards",
		})
	}

	// Check AgentField standard compliance
	if !s.checkAgentFieldStandardCompliance(vcDoc) {
		result.AgentFieldStandardCompliance = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "agentfield_compliance_failure",
			Severity:    "warning",
			Component:   vcDoc.ID,
			Description: "VC does not meet AgentField standard requirements",
		})
	}

	return result
}

// Helper methods for verification checks

func (s *VCService) validateTimestamp(timestamp string) error {
	_, err := time.Parse(time.RFC3339, timestamp)
	return err
}

func (s *VCService) validateVCStructure(vcDoc *types.VCDocument) error {
	// Check required fields
	if len(vcDoc.Context) == 0 {
		return fmt.Errorf("missing @context")
	}
	if len(vcDoc.Type) == 0 {
		return fmt.Errorf("missing type")
	}
	if vcDoc.ID == "" {
		return fmt.Errorf("missing id")
	}
	if vcDoc.Issuer == "" {
		return fmt.Errorf("missing issuer")
	}
	if vcDoc.IssuanceDate == "" {
		return fmt.Errorf("missing issuanceDate")
	}
	return nil
}

// marshalDataOrNull marshals data to JSON or returns null JSON if nil/error
func marshalDataOrNull(data interface{}) []byte {
	if data == nil {
		return []byte("null")
	}
	if jsonData, err := json.Marshal(data); err == nil {
		return jsonData
	}
	return []byte("null")
}

func (s *VCService) detectTamperEvidence(execVC *types.ExecutionVC, vcDoc *types.VCDocument) []string {
	evidence := []string{}

	// Check for inconsistencies that indicate tampering
	if execVC.IssuerDID != vcDoc.Issuer {
		evidence = append(evidence, "issuer_did_inconsistency")
	}
	if execVC.ExecutionID != vcDoc.CredentialSubject.ExecutionID {
		evidence = append(evidence, "execution_id_inconsistency")
	}
	if execVC.Signature != vcDoc.Proof.ProofValue {
		evidence = append(evidence, "signature_inconsistency")
	}

	return evidence
}

func (s *VCService) checkW3CCompliance(vcDoc *types.VCDocument) bool {
	// Check W3C VC standard compliance
	requiredContexts := []string{"https://www.w3.org/2018/credentials/v1"}
	for _, required := range requiredContexts {
		found := false
		for _, context := range vcDoc.Context {
			if context == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (s *VCService) checkAgentFieldStandardCompliance(vcDoc *types.VCDocument) bool {
	// Check AgentField-specific compliance requirements
	requiredTypes := []string{"VerifiableCredential", "AgentFieldExecutionCredential"}
	for _, required := range requiredTypes {
		found := false
		for _, vcType := range vcDoc.Type {
			if vcType == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (s *VCService) calculateOverallScore(result *ComprehensiveVCVerificationResult) float64 {
	score := 100.0

	// Deduct points for critical issues
	score -= float64(len(result.CriticalIssues)) * 25.0

	// Deduct points for warnings
	score -= float64(len(result.Warnings)) * 5.0

	// Factor in security score
	score = (score + result.SecurityAnalysis.SecurityScore) / 2.0

	if score < 0 {
		score = 0
	}

	return score
}

// isStatusConsistent checks if status values are consistent, accounting for AgentField system status mapping
func (s *VCService) isStatusConsistent(metadataStatus, vcDocStatus string) bool {
	return types.NormalizeExecutionStatus(metadataStatus) == types.NormalizeExecutionStatus(vcDocStatus)
}

// VerifyWorkflowVCComprehensive performs comprehensive verification of a workflow VC chain
func (s *VCService) VerifyWorkflowVCComprehensive(workflowID string) (*ComprehensiveVCVerificationResult, error) {
	if !s.config.Enabled {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "system_disabled", Severity: "critical", Description: "DID system is disabled"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	// Get the workflow VC chain
	vcChain, err := s.GetWorkflowVCChain(workflowID)
	if err != nil {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "workflow_chain_error", Severity: "critical", Description: fmt.Sprintf("Failed to get workflow VC chain: %v", err)}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	result := &ComprehensiveVCVerificationResult{
		VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		CriticalIssues:        []VerificationIssue{},
		Warnings:              []VerificationIssue{},
	}

	// Verify each execution VC in the workflow
	allIntegrityChecks := IntegrityCheckResults{
		MetadataConsistency: true,
		FieldConsistency:    true,
		TimestampValidation: true,
		HashValidation:      true,
		StructuralIntegrity: true,
		Issues:              []VerificationIssue{},
	}

	allSecurityAnalysis := SecurityAnalysis{
		SignatureStrength: "Ed25519",
		KeyValidation:     true,
		DIDAuthenticity:   true,
		ReplayProtection:  true,
		TamperEvidence:    []string{},
		SecurityScore:     100.0,
		Issues:            []VerificationIssue{},
	}

	allComplianceChecks := ComplianceChecks{
		W3CCompliance:                true,
		AgentFieldStandardCompliance: true,
		AuditTrailIntegrity:          true,
		DataIntegrityChecks:          true,
		Issues:                       []VerificationIssue{},
	}

	// Verify each execution VC in the workflow
	for _, execVC := range vcChain.ComponentVCs {
		// Parse VC document
		var vcDoc types.VCDocument
		if err := json.Unmarshal(execVC.VCDocument, &vcDoc); err != nil {
			result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
				Type:        "parse_error",
				Severity:    "critical",
				Component:   execVC.VCID,
				Description: fmt.Sprintf("Failed to parse VC document: %v", err),
			})
			continue
		}

		// Perform verification checks for this execution VC
		integrityChecks := s.performIntegrityChecks(&execVC, &vcDoc)
		securityAnalysis := s.performSecurityAnalysis(&execVC, &vcDoc)
		complianceChecks := s.performComplianceChecks(&vcDoc)

		// Aggregate results
		if !integrityChecks.MetadataConsistency {
			allIntegrityChecks.MetadataConsistency = false
		}
		if !integrityChecks.FieldConsistency {
			allIntegrityChecks.FieldConsistency = false
		}
		if !integrityChecks.TimestampValidation {
			allIntegrityChecks.TimestampValidation = false
		}
		if !integrityChecks.HashValidation {
			allIntegrityChecks.HashValidation = false
		}
		if !integrityChecks.StructuralIntegrity {
			allIntegrityChecks.StructuralIntegrity = false
		}

		if !securityAnalysis.KeyValidation {
			allSecurityAnalysis.KeyValidation = false
		}
		if !securityAnalysis.DIDAuthenticity {
			allSecurityAnalysis.DIDAuthenticity = false
		}
		if !securityAnalysis.ReplayProtection {
			allSecurityAnalysis.ReplayProtection = false
		}

		if !complianceChecks.W3CCompliance {
			allComplianceChecks.W3CCompliance = false
		}
		if !complianceChecks.AgentFieldStandardCompliance {
			allComplianceChecks.AgentFieldStandardCompliance = false
		}
		if !complianceChecks.AuditTrailIntegrity {
			allComplianceChecks.AuditTrailIntegrity = false
		}
		if !complianceChecks.DataIntegrityChecks {
			allComplianceChecks.DataIntegrityChecks = false
		}

		// Collect all issues
		allIntegrityChecks.Issues = append(allIntegrityChecks.Issues, integrityChecks.Issues...)
		allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, securityAnalysis.Issues...)
		allComplianceChecks.Issues = append(allComplianceChecks.Issues, complianceChecks.Issues...)

		// Collect tamper evidence
		allSecurityAnalysis.TamperEvidence = append(allSecurityAnalysis.TamperEvidence, securityAnalysis.TamperEvidence...)

		// Update security score (take minimum)
		if securityAnalysis.SecurityScore < allSecurityAnalysis.SecurityScore {
			allSecurityAnalysis.SecurityScore = securityAnalysis.SecurityScore
		}
	}

	// Verify workflow VC itself if it exists
	if vcChain.WorkflowVC.VCDocument != nil {
		var workflowVCDoc types.WorkflowVCDocument
		if err := json.Unmarshal(vcChain.WorkflowVC.VCDocument, &workflowVCDoc); err != nil {
			result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
				Type:        "workflow_vc_parse_error",
				Severity:    "critical",
				Component:   vcChain.WorkflowVC.WorkflowVCID,
				Description: fmt.Sprintf("Failed to parse workflow VC document: %v", err),
			})
		} else {
			// Verify workflow VC signature
			issuerIdentity, err := s.didService.ResolveDID(workflowVCDoc.Issuer)
			if err != nil {
				allSecurityAnalysis.DIDAuthenticity = false
				allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, VerificationIssue{
					Type:        "workflow_did_resolution_failed",
					Severity:    "critical",
					Component:   vcChain.WorkflowVC.WorkflowVCID,
					Description: fmt.Sprintf("Failed to resolve workflow VC issuer DID: %v", err),
				})
			} else {
				valid, err := s.verifyWorkflowVCSignature(&workflowVCDoc, issuerIdentity)
				if err != nil || !valid {
					allSecurityAnalysis.KeyValidation = false
					allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, VerificationIssue{
						Type:        "workflow_signature_verification_failed",
						Severity:    "critical",
						Component:   vcChain.WorkflowVC.WorkflowVCID,
						Description: fmt.Sprintf("Workflow VC signature verification failed: %v", err),
					})
				}
			}

			// Check workflow VC compliance
			if !s.checkWorkflowVCCompliance(&workflowVCDoc) {
				allComplianceChecks.AgentFieldStandardCompliance = false
				allComplianceChecks.Issues = append(allComplianceChecks.Issues, VerificationIssue{
					Type:        "workflow_compliance_failure",
					Severity:    "warning",
					Component:   vcChain.WorkflowVC.WorkflowVCID,
					Description: "Workflow VC does not meet AgentField standard requirements",
				})
			}
		}
	}

	// Set aggregated results
	result.IntegrityChecks = allIntegrityChecks
	result.SecurityAnalysis = allSecurityAnalysis
	result.ComplianceChecks = allComplianceChecks

	// Collect all issues
	allIssues := []VerificationIssue{}
	allIssues = append(allIssues, result.IntegrityChecks.Issues...)
	allIssues = append(allIssues, result.SecurityAnalysis.Issues...)
	allIssues = append(allIssues, result.ComplianceChecks.Issues...)

	// Separate critical issues and warnings
	for _, issue := range allIssues {
		if issue.Severity == "critical" {
			result.CriticalIssues = append(result.CriticalIssues, issue)
		} else if issue.Severity == "warning" {
			result.Warnings = append(result.Warnings, issue)
		}
	}

	// Calculate overall validity and score
	result.Valid = len(result.CriticalIssues) == 0
	result.OverallScore = s.calculateOverallScore(result)

	return result, nil
}

// verifyWorkflowVCSignature verifies the signature of a WorkflowVC document
func (s *VCService) verifyWorkflowVCSignature(vcDoc *types.WorkflowVCDocument, issuerIdentity *types.DIDIdentity) (bool, error) {
	// Create canonical representation for verification
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal workflow VC for verification: %w", err)
	}

	// Parse public key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("failed to parse public key JWK: %w", err)
	}

	xValue, ok := jwk["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)

	// Decode signature
	signatureBytes, err := base64.RawURLEncoding.DecodeString(vcDoc.Proof.ProofValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Verify signature
	return ed25519.Verify(publicKey, canonicalBytes, signatureBytes), nil
}

// checkWorkflowVCCompliance checks if a workflow VC meets AgentField standard compliance
func (s *VCService) checkWorkflowVCCompliance(vcDoc *types.WorkflowVCDocument) bool {
	// Check AgentField-specific compliance requirements for workflow VCs
	requiredTypes := []string{"VerifiableCredential", "AgentFieldWorkflowCredential"}
	for _, required := range requiredTypes {
		found := false
		for _, vcType := range vcDoc.Type {
			if vcType == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
