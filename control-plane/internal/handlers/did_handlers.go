package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/pkg/types"
)

// DIDService defines the DID operations required by handlers.
type DIDService interface {
	RegisterAgent(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error)
	ResolveDID(did string) (*types.DIDIdentity, error)
	ListAllAgentDIDs() ([]string, error)
}

// VCService defines the VC operations required by handlers.
type VCService interface {
	VerifyVC(vcDocument json.RawMessage) (*types.VCVerificationResponse, error)
	GetWorkflowVCChain(workflowID string) (*types.WorkflowVCChainResponse, error)
	CreateWorkflowVC(workflowID, sessionID string, executionVCIDs []string) (*types.WorkflowVC, error)
	GenerateExecutionVC(ctx *types.ExecutionContext, inputData, outputData []byte, status string, errorMessage *string, durationMS int) (*types.ExecutionVC, error)
	QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error)
	ListWorkflowVCs() ([]*types.WorkflowVC, error)
	GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error)
}

// DIDHandlers handles DID-related HTTP requests.
type DIDHandlers struct {
	didService DIDService
	vcService  VCService
}

// NewDIDHandlers creates a new DID handlers instance.
func NewDIDHandlers(didService DIDService, vcService VCService) *DIDHandlers {
	return &DIDHandlers{
		didService: didService,
		vcService:  vcService,
	}
}

// RegisterAgent handles agent DID registration requests.
// POST /api/v1/did/register
func (h *DIDHandlers) RegisterAgent(c *gin.Context) {
	logger.Logger.Debug().Msg("🔍 DID registration endpoint called")

	var req types.DIDRegistrationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	if req.AgentNodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent_node_id is required"})
		return
	}

	response, err := h.didService.RegisterAgent(&req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to register agent"})
		return
	}

	c.JSON(http.StatusOK, response)
}

// ResolveDID handles DID resolution requests.
// GET /api/v1/did/resolve/:did
func (h *DIDHandlers) ResolveDID(c *gin.Context) {
	did := c.Param("did")
	if did == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "DID parameter is required"})
		return
	}

	identity, err := h.didService.ResolveDID(did)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "DID not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"did":             identity.DID,
		"public_key_jwk":  identity.PublicKeyJWK,
		"component_type":  identity.ComponentType,
		"function_name":   identity.FunctionName,
		"derivation_path": identity.DerivationPath,
	})
}

// VerifyVC handles VC verification requests.
// POST /api/v1/did/verify
func (h *DIDHandlers) VerifyVC(c *gin.Context) {
	var req types.VCVerificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	response, err := h.vcService.VerifyVC(req.VCDocument)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to verify VC"})
		return
	}

	c.JSON(http.StatusOK, response)
}

// GetWorkflowVCChain handles workflow VC chain requests.
// GET /api/v1/did/workflow/:workflow_id/vc-chain
func (h *DIDHandlers) GetWorkflowVCChain(c *gin.Context) {
	workflowID := c.Param("workflow_id")
	logger.Logger.Debug().Msgf("🔍 GetWorkflowVCChain endpoint called for workflow: %s", workflowID)

	if workflowID == "" {
		logger.Logger.Debug().Msg("🔍 Missing workflow_id parameter")
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "workflow_id parameter is required",
		})
		return
	}

	logger.Logger.Debug().Msgf("🔍 Calling VCService.GetWorkflowVCChain for workflow: %s", workflowID)
	// Get workflow VC chain
	response, err := h.vcService.GetWorkflowVCChain(workflowID)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 VCService.GetWorkflowVCChain failed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to get workflow VC chain",
			"details": err.Error(),
		})
		return
	}

	logger.Logger.Debug().Msg("🔍 Successfully retrieved workflow VC chain, returning response")
	c.JSON(http.StatusOK, response)
}

// CreateWorkflowVC handles workflow VC creation requests.
// POST /api/v1/did/workflow/:workflow_id/vc
func (h *DIDHandlers) CreateWorkflowVC(c *gin.Context) {
	workflowID := c.Param("workflow_id")
	if workflowID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "workflow_id parameter is required",
		})
		return
	}

	var req struct {
		SessionID      string   `json:"session_id"`
		ExecutionVCIDs []string `json:"execution_vc_ids"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request body",
			"details": err.Error(),
		})
		return
	}

	// Create workflow VC
	workflowVC, err := h.vcService.CreateWorkflowVC(workflowID, req.SessionID, req.ExecutionVCIDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to create workflow VC",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, workflowVC)
}

// GetDIDStatus handles DID system status requests.
// GET /api/v1/did/status
func (h *DIDHandlers) GetDIDStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "active",
		"message":   "DID system is operational",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

// CreateExecutionVC handles execution VC creation requests from Python SDK.
// POST /api/v1/execution/vc
func (h *DIDHandlers) CreateExecutionVC(c *gin.Context) {
	logger.Logger.Debug().Msg("🔍 DEBUG: CreateExecutionVC handler called")

	var req struct {
		ExecutionContext struct {
			ExecutionID  string `json:"execution_id"`
			WorkflowID   string `json:"workflow_id"`
			SessionID    string `json:"session_id"`
			CallerDID    string `json:"caller_did"`
			TargetDID    string `json:"target_did"`
			AgentNodeDID string `json:"agent_node_did"`
			Timestamp    string `json:"timestamp"`
		} `json:"execution_context"`
		InputData    []byte  `json:"input_data"`
		OutputData   []byte  `json:"output_data"`
		Status       string  `json:"status"`
		ErrorMessage *string `json:"error_message"`
		DurationMS   int     `json:"duration_ms"`
	}

	logger.Logger.Debug().Msg("🔍 DEBUG: About to parse JSON request body")
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Logger.Error().Err(err).Msg("🔍 DEBUG: JSON binding failed in CreateExecutionVC")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request body",
			"details": err.Error(),
		})
		return
	}

	logger.Logger.Debug().
		Str("execution_id", req.ExecutionContext.ExecutionID).
		Str("workflow_id", req.ExecutionContext.WorkflowID).
		Str("caller_did", req.ExecutionContext.CallerDID).
		Str("target_did", req.ExecutionContext.TargetDID).
		Str("status", req.Status).
		Int("duration_ms", req.DurationMS).
		Msg("🔍 DEBUG: Successfully parsed VC creation request")

	// Parse timestamp
	timestamp, err := time.Parse(time.RFC3339, req.ExecutionContext.Timestamp)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid timestamp format",
			"details": err.Error(),
		})
		return
	}

	// Create execution context
	execCtx := &types.ExecutionContext{
		ExecutionID:  req.ExecutionContext.ExecutionID,
		WorkflowID:   req.ExecutionContext.WorkflowID,
		SessionID:    req.ExecutionContext.SessionID,
		CallerDID:    req.ExecutionContext.CallerDID,
		TargetDID:    req.ExecutionContext.TargetDID,
		AgentNodeDID: req.ExecutionContext.AgentNodeDID,
		Timestamp:    timestamp,
	}

	// Generate execution VC
	executionVC, err := h.vcService.GenerateExecutionVC(
		execCtx,
		req.InputData,
		req.OutputData,
		req.Status,
		req.ErrorMessage,
		req.DurationMS,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to generate execution VC",
			"details": err.Error(),
		})
		return
	}

	// Return the VC data
	c.JSON(http.StatusOK, gin.H{
		"vc_id":        executionVC.VCID,
		"execution_id": executionVC.ExecutionID,
		"workflow_id":  executionVC.WorkflowID,
		"session_id":   executionVC.SessionID,
		"issuer_did":   executionVC.IssuerDID,
		"target_did":   executionVC.TargetDID,
		"caller_did":   executionVC.CallerDID,
		"vc_document":  executionVC.VCDocument,
		"signature":    executionVC.Signature,
		"input_hash":   executionVC.InputHash,
		"output_hash":  executionVC.OutputHash,
		"status":       executionVC.Status,
		"created_at":   executionVC.CreatedAt.Format(time.RFC3339),
	})
}

// ExportVCs handles VC export requests for external verification.
// GET /api/v1/did/export/vcs
func (h *DIDHandlers) ExportVCs(c *gin.Context) {

	// Parse query parameters for filtering
	var filters types.VCFilters

	if workflowID := c.Query("workflow_id"); workflowID != "" {
		filters.WorkflowID = &workflowID
	}
	if sessionID := c.Query("session_id"); sessionID != "" {
		filters.SessionID = &sessionID
	}
	if issuerDID := c.Query("issuer_did"); issuerDID != "" {
		filters.IssuerDID = &issuerDID
	}
	if status := c.Query("status"); status != "" {
		filters.Status = &status
	}

	// Set default limit if not provided
	if filters.Limit == 0 {
		filters.Limit = 100
	}

	// Query execution VCs
	executionVCs, err := h.vcService.QueryExecutionVCs(&filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to query execution VCs",
			"details": err.Error(),
		})
		return
	}

	// Query workflow VCs
	workflowVCs, err := h.vcService.ListWorkflowVCs()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to query workflow VCs",
			"details": err.Error(),
		})
		return
	}

	// Get all registered agent DIDs from the DID service
	agentDIDs, err := h.didService.ListAllAgentDIDs()
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to list agent DIDs")
		// Continue with empty list rather than failing the entire request
		agentDIDs = []string{}
	}

	// Convert execution VCs to export format
	var executionVCsExport []map[string]interface{}
	for _, vc := range executionVCs {
		// Convert execution VC to export format
		executionVCsExport = append(executionVCsExport, map[string]interface{}{
			"vc_id":        vc.VCID,
			"execution_id": vc.ExecutionID,
			"workflow_id":  vc.WorkflowID,
			"session_id":   vc.SessionID,
			"issuer_did":   vc.IssuerDID,
			"target_did":   vc.TargetDID,
			"caller_did":   vc.CallerDID,
			"status":       vc.Status,
			"created_at":   vc.CreatedAt.Format(time.RFC3339),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"agent_dids":      agentDIDs,
		"execution_vcs":   executionVCsExport,
		"workflow_vcs":    workflowVCs,
		"total_count":     len(executionVCs) + len(workflowVCs),
		"filters_applied": filters,
	})
}

// GetDIDDocument handles DID document requests (W3C DID standard).
// GET /api/v1/did/document/:did
func (h *DIDHandlers) GetDIDDocument(c *gin.Context) {
	did := c.Param("did")
	if did == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "DID parameter is required",
		})
		return
	}

	// Resolve DID to get identity information
	identity, err := h.didService.ResolveDID(did)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "DID not found",
			"details": err.Error(),
		})
		return
	}

	// Parse public key JWK
	var publicKeyJWK map[string]interface{}
	if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to parse public key",
			"details": err.Error(),
		})
		return
	}

	// Create W3C DID Document
	didDocument := map[string]interface{}{
		"@context": []string{
			"https://www.w3.org/ns/did/v1",
			"https://w3id.org/security/suites/ed25519-2020/v1",
		},
		"id": did,
		"verificationMethod": []map[string]interface{}{
			{
				"id":           did + "#key-1",
				"type":         "Ed25519VerificationKey2020",
				"controller":   did,
				"publicKeyJwk": publicKeyJWK,
			},
		},
		"authentication": []string{
			did + "#key-1",
		},
		"assertionMethod": []string{
			did + "#key-1",
		},
		"service": []map[string]interface{}{
			{
				"id":              did + "#haxen-service",
				"type":            "HaxenAgentService",
				"serviceEndpoint": "https://haxen.example.com/api/v1",
				"description":     "Haxen Agent Platform Service",
			},
		},
	}

	c.JSON(http.StatusOK, didDocument)
}

// RegisterRoutes registers all DID-related routes.
func (h *DIDHandlers) RegisterRoutes(router *gin.RouterGroup) {
	didGroup := router.Group("/did")
	{
		didGroup.POST("/register", h.RegisterAgent)
		didGroup.GET("/resolve/:did", h.ResolveDID)
		didGroup.POST("/verify", h.VerifyVC)
		didGroup.GET("/workflow/:workflow_id/vc-chain", h.GetWorkflowVCChain)
		didGroup.POST("/workflow/:workflow_id/vc", h.CreateWorkflowVC)
		didGroup.GET("/status", h.GetDIDStatus)
		didGroup.GET("/export/vcs", h.ExportVCs)
		didGroup.GET("/document/:did", h.GetDIDDocument)
	}

	// Execution VC endpoint (separate from DID group to match Python SDK expectations)
	router.POST("/execution/vc", h.CreateExecutionVC)
}
