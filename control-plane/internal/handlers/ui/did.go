package ui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// DIDHandler provides handlers for UI-related DID operations.
type DIDHandler struct {
	storage    storage.StorageProvider
	didService *services.DIDService
	vcService  *services.VCService
}

// NewDIDHandler creates a new DIDHandler.
func NewDIDHandler(storage storage.StorageProvider, didService *services.DIDService, vcService *services.VCService) *DIDHandler {
	return &DIDHandler{
		storage:    storage,
		didService: didService,
		vcService:  vcService,
	}
}

// GetNodeDIDHandler handles requests for DID information about a specific node.
// GET /api/ui/v1/nodes/:nodeId/did
func (h *DIDHandler) GetNodeDIDHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}

	// If DID service is not available, return empty response
	if h.didService == nil {
		c.JSON(http.StatusOK, gin.H{
			"has_did":        false,
			"did_status":     "inactive",
			"reasoner_count": 0,
			"skill_count":    0,
			"last_updated":   "",
		})
		return
	}

	// Get af server ID dynamically
	agentfieldServerID, err := h.didService.GetAgentFieldServerID()
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"has_did":        false,
			"did_status":     "inactive",
			"reasoner_count": 0,
			"skill_count":    0,
			"error":          fmt.Sprintf("Failed to get af server ID: %v", err),
		})
		return
	}

	// Get DID registry for the af server (not the node)
	registry, err := h.didService.GetRegistry(agentfieldServerID)
	if err != nil || registry == nil {
		c.JSON(http.StatusOK, gin.H{
			"has_did":        false,
			"did_status":     "inactive",
			"reasoner_count": 0,
			"skill_count":    0,
			"last_updated":   "",
		})
		return
	}

	// Get agent info for this node
	agentInfo, exists := registry.AgentNodes[nodeID]
	if !exists {
		c.JSON(http.StatusOK, gin.H{
			"has_did":        false,
			"did_status":     "inactive",
			"reasoner_count": 0,
			"skill_count":    0,
			"last_updated":   "",
		})
		return
	}

	// Determine status
	status := string(agentInfo.Status)
	if agentInfo.DID == "" {
		status = "inactive"
	}

	c.JSON(http.StatusOK, gin.H{
		"did":                  agentInfo.DID,
		"agent_node_id":        nodeID,
		"agentfield_server_id": registry.AgentFieldServerID,
		"public_key_jwk":       agentInfo.PublicKeyJWK,
		"derivation_path":      agentInfo.DerivationPath,
		"reasoners":            agentInfo.Reasoners,
		"skills":               agentInfo.Skills,
		"status":               status,
		"registered_at":        agentInfo.RegisteredAt.Format(time.RFC3339),
	})
}

// GetNodeVCStatusHandler handles requests for VC status information about a specific node.
// GET /api/ui/v1/nodes/:nodeId/vc-status
func (h *DIDHandler) GetNodeVCStatusHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}

	// If VC service is not available, return empty response
	if h.vcService == nil {
		c.JSON(http.StatusOK, gin.H{
			"has_vcs":             false,
			"vc_count":            0,
			"verified_count":      0,
			"last_vc_created":     "",
			"verification_status": "none",
		})
		return
	}

	// Get VCs for this node (filter by issuer DID or target DID)
	filters := &types.VCFilters{
		Limit: 1000, // Get all VCs for this node
	}

	executionVCs, err := h.vcService.QueryExecutionVCs(filters)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"has_vcs":             false,
			"vc_count":            0,
			"verified_count":      0,
			"last_vc_created":     "",
			"verification_status": "none",
		})
		return
	}

	// Filter VCs related to this node
	var nodeVCs []types.ExecutionVC
	for _, vc := range executionVCs {
		// Check if this VC is related to the node (by issuer DID or target DID)
		if vc.IssuerDID != "" && contains(vc.IssuerDID, nodeID) ||
			vc.TargetDID != "" && contains(vc.TargetDID, nodeID) {
			nodeVCs = append(nodeVCs, vc)
		}
	}

	vcCount := len(nodeVCs)
	verifiedCount := 0
	var lastVCCreated string

	for _, vc := range nodeVCs {
		normalized := types.NormalizeExecutionStatus(vc.Status)
		if normalized == string(types.ExecutionStatusSucceeded) || vc.Status == "verified" {
			verifiedCount++
		}
		if lastVCCreated == "" || vc.CreatedAt.After(parseTime(lastVCCreated)) {
			lastVCCreated = vc.CreatedAt.Format(time.RFC3339)
		}
	}

	// Determine verification status
	var verificationStatus string
	if vcCount == 0 {
		verificationStatus = "none"
	} else if verifiedCount == vcCount {
		verificationStatus = "verified"
	} else if hasFailedVCs(nodeVCs) {
		verificationStatus = "failed"
	} else {
		verificationStatus = "pending"
	}

	c.JSON(http.StatusOK, gin.H{
		"has_vcs":             vcCount > 0,
		"vc_count":            vcCount,
		"verified_count":      verifiedCount,
		"last_vc_created":     lastVCCreated,
		"verification_status": verificationStatus,
	})
}

// GetExecutionVCStatusHandler handles requests for VC status information about a specific execution.
// GET /api/ui/v1/executions/:executionId/vc-status
func (h *DIDHandler) GetExecutionVCStatusHandler(c *gin.Context) {
	// Try both parameter names for compatibility with UI and Agent API routes
	executionID := c.Param("executionId")
	if executionID == "" {
		executionID = c.Param("execution_id")
	}
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "executionId or execution_id is required"})
		return
	}

	// DEBUG: Log the execution ID being requested
	fmt.Printf("DEBUG: GetExecutionVCStatusHandler called for execution_id: %s\n", executionID)

	// If VC service is not available, return empty response
	if h.vcService == nil {
		fmt.Printf("DEBUG: VC service is nil for execution_id: %s\n", executionID)
		c.JSON(http.StatusOK, gin.H{
			"has_vc":     false,
			"status":     "none",
			"created_at": "",
		})
		return
	}

	executionVC, err := h.vcService.GetExecutionVCByExecutionID(executionID)
	if err != nil {
		fmt.Printf("DEBUG: Execution VC lookup failed for %s: %v\n", executionID, err)
		c.JSON(http.StatusOK, gin.H{
			"has_vc":     false,
			"status":     "none",
			"created_at": "",
		})
		return
	}

	fmt.Printf("DEBUG: Found VC for execution_id %s: vc_id=%s, status=%s, vc_document_type=%T\n",
		executionID, executionVC.VCID, executionVC.Status, executionVC.VCDocument)

	var vcDocumentForResponse interface{}
	documentStatus := executionVC.Status

	if len(executionVC.VCDocument) > 0 {
		var parsed interface{}
		if err := json.Unmarshal(executionVC.VCDocument, &parsed); err != nil {
			fmt.Printf("DEBUG: VC document parsing failed for %s: %v\n", executionID, err)
			vcDocumentForResponse = map[string]interface{}{
				"parse_error": true,
				"error":       err.Error(),
				"raw_length":  len(executionVC.VCDocument),
				"note":        "VC document could not be parsed as valid JSON",
			}
			documentStatus = "malformed"
		} else {
			vcDocumentForResponse = parsed
			fmt.Printf("DEBUG: VC document is valid JSON (%d bytes)\n", len(executionVC.VCDocument))
		}
	} else if executionVC.StorageURI != "" {
		vcDocumentForResponse = map[string]interface{}{
			"storage_uri":         executionVC.StorageURI,
			"document_size_bytes": executionVC.DocumentSize,
			"note":                "VC document stored via external URI",
		}
		documentStatus = "external"
	} else {
		documentStatus = "empty"
	}

	c.JSON(http.StatusOK, gin.H{
		"has_vc":              true,
		"vc_id":               executionVC.VCID,
		"status":              documentStatus,
		"original_status":     executionVC.Status,
		"created_at":          executionVC.CreatedAt.Format(time.RFC3339),
		"storage_uri":         executionVC.StorageURI,
		"document_size_bytes": executionVC.DocumentSize,
		"vc_document":         vcDocumentForResponse,
	})
}

// GetExecutionVCHandler handles requests for VC information about a specific execution.
// GET /api/ui/v1/executions/:executionId/vc
func (h *DIDHandler) GetExecutionVCHandler(c *gin.Context) {
	// Try both parameter names for compatibility with UI and Agent API routes
	executionID := c.Param("executionId")
	if executionID == "" {
		executionID = c.Param("execution_id")
	}
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "executionId or execution_id is required"})
		return
	}

	// DEBUG: Log the execution ID being requested
	fmt.Printf("DEBUG: GetExecutionVCHandler called for execution_id: %s\n", executionID)

	// If VC service is not available, return error
	if h.vcService == nil {
		fmt.Printf("DEBUG: VC service is nil for execution_id: %s\n", executionID)
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "VC service not available"})
		return
	}

	executionVC, err := h.vcService.GetExecutionVCByExecutionID(executionID)
	if err != nil {
		fmt.Printf("DEBUG: No VC found for execution_id: %s (err=%v)\n", executionID, err)
		c.JSON(http.StatusNotFound, gin.H{"error": "VC not found for this execution"})
		return
	}

	if len(executionVC.VCDocument) == 0 {
		if executionVC.StorageURI == "" {
			fmt.Printf("DEBUG: VC document is empty for execution_id: %s\n", executionID)
			c.JSON(http.StatusNotFound, gin.H{"error": "VC document not found or empty"})
			return
		}
	}

	c.JSON(http.StatusOK, executionVC)
}

// GetWorkflowVCChainHandler handles requests for workflow VC chain information.
// GET /api/ui/v1/workflows/:workflowId/vc-chain
func (h *DIDHandler) GetWorkflowVCChainHandler(c *gin.Context) {
	workflowID := c.Param("workflowId")
	if workflowID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "workflowId is required"})
		return
	}

	// If VC service is not available, return empty response
	if h.vcService == nil {
		c.JSON(http.StatusOK, gin.H{
			"workflow_id":     workflowID,
			"total_steps":     0,
			"completed_steps": 0,
			"status":          "none",
			"component_vcs":   []interface{}{},
		})
		return
	}

	// Get workflow VC chain
	response, err := h.vcService.GetWorkflowVCChain(workflowID)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"workflow_id":     workflowID,
			"total_steps":     0,
			"completed_steps": 0,
			"status":          "error",
			"component_vcs":   []interface{}{},
		})
		return
	}

	c.JSON(http.StatusOK, response)
}

// DownloadVCHandler handles requests to download a VC document.
// GET /api/ui/v1/vc/:vc_id/download
func (h *DIDHandler) DownloadVCHandler(c *gin.Context) {
	vcID := c.Param("vc_id")
	if vcID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "vc_id is required"})
		return
	}

	// If VC service is not available, return error
	if h.vcService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "VC service not available"})
		return
	}

	// Get VC document
	filters := &types.VCFilters{
		Limit: 1000,
	}

	executionVCs, err := h.vcService.QueryExecutionVCs(filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query VCs"})
		return
	}

	// Find the VC
	for _, vc := range executionVCs {
		if vc.VCID == vcID {
			// Set headers for file download
			c.Header("Content-Type", "application/json")
			c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=vc-%s.json", vcID))

			// Return the VC document
			c.JSON(http.StatusOK, vc.VCDocument)
			return
		}
	}

	c.JSON(http.StatusNotFound, gin.H{"error": "VC not found"})
}

// VerifyVCHandler handles requests to verify a VC.
// POST /api/ui/v1/vc/verify
func (h *DIDHandler) VerifyVCHandler(c *gin.Context) {
	var req types.VCVerificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request body",
			"details": err.Error(),
		})
		return
	}

	// If VC service is not available, return error
	if h.vcService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "VC service not available"})
		return
	}

	// Verify VC
	response, err := h.vcService.VerifyVC(req.VCDocument)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to verify VC",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, response)
}

// VerifyExecutionVCComprehensiveHandler handles requests for comprehensive VC verification.
// POST /api/ui/v1/executions/:executionId/verify-vc
func (h *DIDHandler) VerifyExecutionVCComprehensiveHandler(c *gin.Context) {
	// Try both parameter names for compatibility with UI and Agent API routes
	executionID := c.Param("executionId")
	if executionID == "" {
		executionID = c.Param("execution_id")
	}
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "executionId or execution_id is required"})
		return
	}

	// If VC service is not available, return error
	if h.vcService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "VC service not available"})
		return
	}

	// Perform comprehensive VC verification
	result, err := h.vcService.VerifyExecutionVCComprehensive(executionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to verify VC",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, result)
}

// VerifyWorkflowVCComprehensiveHandler handles requests for comprehensive workflow VC verification.
// POST /api/ui/v1/workflows/:workflowId/verify-vc
func (h *DIDHandler) VerifyWorkflowVCComprehensiveHandler(c *gin.Context) {
	workflowID := c.Param("workflowId")
	if workflowID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "workflowId is required"})
		return
	}

	// If VC service is not available, return error
	if h.vcService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "VC service not available"})
		return
	}

	// Perform comprehensive workflow VC verification
	result, err := h.vcService.VerifyWorkflowVCComprehensive(workflowID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to verify workflow VC",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, result)
}

// ExportVCsHandler handles requests to export VCs with optional filtering.
// GET /api/ui/v1/did/export/vcs
func (h *DIDHandler) ExportVCsHandler(c *gin.Context) {
	// If VC service is not available, return empty response
	if h.vcService == nil {
		c.JSON(http.StatusOK, gin.H{
			"execution_vcs": []interface{}{},
			"workflow_vcs":  []interface{}{},
			"total_count":   0,
		})
		return
	}

	// Parse query parameters for filtering
	filters := &types.VCFilters{}

	filters.Limit = 100 // default
	if limit := c.Query("limit"); limit != "" {
		if parsedLimit, err := strconv.Atoi(limit); err == nil {
			filters.Limit = parsedLimit
		}
	}

	if offset := c.Query("offset"); offset != "" {
		if parsedOffset, err := strconv.Atoi(offset); err == nil {
			filters.Offset = parsedOffset
		}
	}

	if status := c.Query("status"); status != "" {
		filters.Status = &status
	}

	if workflowID := c.Query("workflow_id"); workflowID != "" {
		filters.WorkflowID = &workflowID
	}
	if executionID := c.Query("execution_id"); executionID != "" {
		filters.ExecutionID = &executionID
	}

	if sessionID := c.Query("session_id"); sessionID != "" {
		filters.SessionID = &sessionID
	}

	// Get execution VCs
	executionVCs, err := h.vcService.QueryExecutionVCs(filters)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"execution_vcs": []interface{}{},
			"workflow_vcs":  []interface{}{},
			"total_count":   0,
		})
		return
	}

	// Get workflow VCs (if available)
	workflowVCs := []interface{}{} // Placeholder - implement if needed

	c.JSON(http.StatusOK, gin.H{
		"execution_vcs": executionVCs,
		"workflow_vcs":  workflowVCs,
		"total_count":   len(executionVCs),
	})
}

// GetDIDSystemStatusHandler handles requests for DID system status.
// GET /api/ui/v1/did/status
func (h *DIDHandler) GetDIDSystemStatusHandler(c *gin.Context) {
	if h.didService == nil {
		c.JSON(http.StatusOK, gin.H{
			"status":    "inactive",
			"message":   "DID system is not enabled",
			"timestamp": time.Now().Format(time.RFC3339),
		})
		return
	}

	// TODO: Implement actual DID system health check
	c.JSON(http.StatusOK, gin.H{
		"status":    "active",
		"message":   "DID system is operational",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

// GetDIDResolutionBundleHandler handles requests for DID resolution bundle information.
// GET /api/ui/v1/did/:did/resolution-bundle
func (h *DIDHandler) GetDIDResolutionBundleHandler(c *gin.Context) {
	did := c.Param("did")
	if did == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "did is required"})
		return
	}

	// If DID service is not available, return empty response
	if h.didService == nil {
		c.JSON(http.StatusOK, gin.H{
			"did":               did,
			"resolution_status": "inactive",
			"did_document":      nil,
			"verification_keys": []interface{}{},
			"service_endpoints": []interface{}{},
			"related_vcs":       []interface{}{},
			"component_dids":    []interface{}{},
			"resolution_metadata": gin.H{
				"resolved_at": time.Now().Format(time.RFC3339),
				"resolver":    "agentfield-server",
				"status":      "inactive",
			},
		})
		return
	}

	// Get DID registry to find the DID
	var didDocument interface{}
	var verificationKeys []interface{}
	var serviceEndpoints []interface{}
	var componentDIDs []interface{}
	var resolutionStatus string = "not_found"

	// Try to find the DID in agent DIDs
	agentDIDs, err := h.storage.ListAgentDIDs(c.Request.Context())
	if err == nil {
		for _, agentDID := range agentDIDs {
			if agentDID.DID == did {
				resolutionStatus = "resolved"

				// Build DID document
				didDocument = gin.H{
					"@context": []string{
						"https://www.w3.org/ns/did/v1",
						"https://w3id.org/security/suites/jws-2020/v1",
					},
					"id": did,
					"verificationMethod": []gin.H{
						{
							"id":           did + "#key-1",
							"type":         "JsonWebKey2020",
							"controller":   did,
							"publicKeyJwk": agentDID.PublicKeyJWK,
						},
					},
					"authentication":  []string{did + "#key-1"},
					"assertionMethod": []string{did + "#key-1"},
					"service": []gin.H{
						{
							"id":              did + "#agent-service",
							"type":            "AgentService",
							"serviceEndpoint": fmt.Sprintf("https://agentfield-server/agents/%s", agentDID.AgentNodeID),
						},
					},
				}

				// Add verification keys
				verificationKeys = append(verificationKeys, gin.H{
					"id":           did + "#key-1",
					"type":         "JsonWebKey2020",
					"controller":   did,
					"publicKeyJwk": agentDID.PublicKeyJWK,
					"purpose":      []string{"authentication", "assertionMethod"},
				})

				// Add service endpoints
				serviceEndpoints = append(serviceEndpoints, gin.H{
					"id":              did + "#agent-service",
					"type":            "AgentService",
					"serviceEndpoint": fmt.Sprintf("https://agentfield-server/agents/%s", agentDID.AgentNodeID),
				})

				// Add component DIDs (reasoners and skills)
				for reasonerName, reasonerInfo := range agentDID.Reasoners {
					componentDIDs = append(componentDIDs, gin.H{
						"did":            reasonerInfo.DID,
						"type":           "reasoner",
						"name":           reasonerName,
						"capabilities":   reasonerInfo.Capabilities,
						"exposure_level": reasonerInfo.ExposureLevel,
					})
				}

				for skillName, skillInfo := range agentDID.Skills {
					componentDIDs = append(componentDIDs, gin.H{
						"did":            skillInfo.DID,
						"type":           "skill",
						"name":           skillName,
						"tags":           skillInfo.Tags,
						"exposure_level": skillInfo.ExposureLevel,
					})
				}

				break
			}
		}
	}

	// If not found in agent DIDs, try component DIDs
	if resolutionStatus == "not_found" {
		componentDIDInfos, err := h.storage.ListComponentDIDs(c.Request.Context(), "")
		if err == nil {
			for _, componentDID := range componentDIDInfos {
				if componentDID.ComponentDID == did {
					resolutionStatus = "resolved"

					// Build DID document for component
					didDocument = gin.H{
						"@context": []string{
							"https://www.w3.org/ns/did/v1",
							"https://w3id.org/security/suites/jws-2020/v1",
						},
						"id": did,
						"verificationMethod": []gin.H{
							{
								"id":         did + "#key-1",
								"type":       "JsonWebKey2020",
								"controller": componentDID.AgentDID,
							},
						},
						"authentication":  []string{did + "#key-1"},
						"assertionMethod": []string{did + "#key-1"},
						"service": []gin.H{
							{
								"id":              did + "#component-service",
								"type":            fmt.Sprintf("%sService", componentDID.ComponentType),
								"serviceEndpoint": fmt.Sprintf("https://agentfield-server/components/%s", componentDID.ComponentID),
							},
						},
					}

					// Add service endpoints
					serviceEndpoints = append(serviceEndpoints, gin.H{
						"id":              did + "#component-service",
						"type":            fmt.Sprintf("%sService", componentDID.ComponentType),
						"serviceEndpoint": fmt.Sprintf("https://agentfield-server/components/%s", componentDID.ComponentID),
					})

					break
				}
			}
		}
	}

	// Get related VCs for this DID
	var relatedVCs []interface{}
	if h.vcService != nil {
		filters := &types.VCFilters{
			Limit: 100,
		}

		executionVCs, err := h.vcService.QueryExecutionVCs(filters)
		if err == nil {
			for _, vc := range executionVCs {
				if vc.IssuerDID == did || vc.TargetDID == did || vc.CallerDID == did {
					relatedVCs = append(relatedVCs, gin.H{
						"vc_id":        vc.VCID,
						"execution_id": vc.ExecutionID,
						"workflow_id":  vc.WorkflowID,
						"status":       vc.Status,
						"role":         getDIDRole(did, vc),
						"created_at":   vc.CreatedAt.Format(time.RFC3339),
					})
				}
			}
		}
	}

	// Build resolution metadata
	resolutionMetadata := gin.H{
		"resolved_at": time.Now().Format(time.RFC3339),
		"resolver":    "agentfield-server",
		"status":      resolutionStatus,
		"method":      "agentfield",
	}

	if resolutionStatus == "resolved" {
		resolutionMetadata["content_type"] = "application/did+ld+json"
	}

	c.JSON(http.StatusOK, gin.H{
		"did":                 did,
		"resolution_status":   resolutionStatus,
		"did_document":        didDocument,
		"verification_keys":   verificationKeys,
		"service_endpoints":   serviceEndpoints,
		"related_vcs":         relatedVCs,
		"component_dids":      componentDIDs,
		"resolution_metadata": resolutionMetadata,
	})
}

// DownloadDIDResolutionBundleHandler handles requests to download a DID resolution bundle.
// GET /api/ui/v1/did/:did/resolution-bundle/download
func (h *DIDHandler) DownloadDIDResolutionBundleHandler(c *gin.Context) {
	did := c.Param("did")
	if did == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "did is required"})
		return
	}

	// Get the resolution bundle data (reuse the logic from GetDIDResolutionBundleHandler)
	// This is a simplified version - in production, you might want to extract this to a shared function

	// Set headers for file download
	filename := fmt.Sprintf("did-resolution-bundle-%s.json", sanitizeDIDForFilename(did))
	c.Header("Content-Type", "application/json")
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename))

	// Get the bundle data by calling the resolution logic
	// For now, we'll create a minimal bundle structure
	bundle := gin.H{
		"@context": []string{
			"https://www.w3.org/ns/did/v1",
			"https://w3id.org/security/suites/jws-2020/v1",
		},
		"did": did,
		"resolution_metadata": gin.H{
			"resolved_at": time.Now().Format(time.RFC3339),
			"resolver":    "agentfield-server",
			"method":      "agentfield",
		},
		"bundle_type":  "did_resolution",
		"generated_at": time.Now().Format(time.RFC3339),
	}

	// If DID service is available, get more detailed information
	if h.didService != nil {
		// Add more detailed resolution data here
		bundle["status"] = "resolved"
	} else {
		bundle["status"] = "service_unavailable"
	}

	c.JSON(http.StatusOK, bundle)
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 &&
		(s == substr ||
			(len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr)))
}

func parseTime(timeStr string) time.Time {
	if timeStr == "" {
		return time.Time{}
	}
	t, _ := time.Parse(time.RFC3339, timeStr)
	return t
}

func hasFailedVCs(vcs []types.ExecutionVC) bool {
	for _, vc := range vcs {
		if types.NormalizeExecutionStatus(vc.Status) == string(types.ExecutionStatusFailed) {
			return true
		}
	}
	return false
}

// getDIDRole determines the role of a DID in a VC (issuer, target, or caller)
func getDIDRole(did string, vc types.ExecutionVC) string {
	if vc.IssuerDID == did {
		return "issuer"
	}
	if vc.TargetDID == did {
		return "target"
	}
	if vc.CallerDID == did {
		return "caller"
	}
	return "unknown"
}

// sanitizeDIDForFilename sanitizes a DID string to be safe for use in filenames
func sanitizeDIDForFilename(did string) string {
	replacer := strings.NewReplacer(
		":", "_",
		"/", "_",
		"\\", "_",
		"?", "_",
		"*", "_",
		"<", "_",
		">", "_",
		"|", "_",
		"\"", "_",
		" ", "_",
	)

	sanitized := replacer.Replace(did)
	if len(sanitized) > 100 {
		sanitized = sanitized[:100]
	}

	return sanitized
}
