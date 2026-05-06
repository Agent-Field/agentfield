package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// registerDIDWellKnownRoutes installs the W3C did:web resolution endpoints at
// the router root. Per the did:web spec:
//
//	did:web:{domain}                   → GET /.well-known/did.json
//	did:web:{domain}:agents:{agentID}  → GET /agents/{agentID}/did.json
//
// See: https://w3c-ccg.github.io/did-method-web/
func (s *AgentFieldServer) registerDIDWellKnownRoutes() {
	if !s.config.Features.DID.Enabled || s.didWebService == nil {
		return
	}
	s.Router.GET("/.well-known/did.json", s.handleDIDWebServerDocument)
	s.Router.GET("/agents/:agentID/did.json", s.handleDIDWebAgentDocument)
}

// registerDIDRoutes installs agent-facing DID/VC endpoints under /api/v1:
// DID CRUD, tag VCs, policy distribution, revocation lists, and the issuer
// public key.
func (s *AgentFieldServer) registerDIDRoutes(agentAPI *gin.RouterGroup) {
	logger.Logger.Debug().
		Bool("did_enabled", s.config.Features.DID.Enabled).
		Bool("did_service_available", s.didService != nil).
		Bool("vc_service_available", s.vcService != nil).
		Msg("DID Route Registration Check")

	if s.config.Features.DID.Enabled && s.didService != nil && s.vcService != nil {
		logger.Logger.Debug().Msg("Registering DID routes - all conditions met")
		didHandlers := handlers.NewDIDHandlers(s.didService, s.vcService)
		if s.didWebService != nil {
			didHandlers.SetDIDWebService(s.didWebService)
		}

		didHandlers.RegisterRoutes(agentAPI)

		// Add af server DID endpoint
		agentAPI.GET("/did/agentfield-server", func(c *gin.Context) {
			agentfieldServerID, err := s.didService.GetAgentFieldServerID()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "Failed to get af server ID",
					"details": fmt.Sprintf("AgentField server ID error: %v", err),
				})
				return
			}

			registry, err := s.didService.GetRegistry(agentfieldServerID)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "Failed to get af server DID",
					"details": fmt.Sprintf("Registry error: %v", err),
				})
				return
			}

			if registry == nil {
				c.JSON(http.StatusNotFound, gin.H{
					"error":   "AgentField server DID not found",
					"details": "No DID registry exists for af server 'default'. The DID system may not be properly initialized.",
				})
				return
			}

			if registry.RootDID == "" {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "AgentField server DID is empty",
					"details": "Registry exists but root DID is empty. The DID system may be corrupted.",
				})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"agentfield_server_id":  "default",
				"agentfield_server_did": registry.RootDID,
				"message":               "AgentField server DID retrieved successfully",
			})
		})
	} else {
		logger.Logger.Warn().
			Bool("did_enabled", s.config.Features.DID.Enabled).
			Bool("did_service_available", s.didService != nil).
			Bool("vc_service_available", s.vcService != nil).
			Msg("DID routes NOT registered - conditions not met")
	}

	// Agent Tag VC endpoint (for agents to download their own verified tag credential)
	if s.tagVCVerifier != nil {
		agentAPI.GET("/agents/:node_id/tag-vc", func(c *gin.Context) {
			agentID := c.Param("node_id")
			if agentID == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "agent_id is required"})
				return
			}
			record, err := s.storage.GetAgentTagVC(c.Request.Context(), agentID)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{
					"error":   "tag_vc_not_found",
					"message": fmt.Sprintf("No tag VC found for agent %s", agentID),
				})
				return
			}
			if record.RevokedAt != nil {
				c.JSON(http.StatusGone, gin.H{
					"error":   "tag_vc_revoked",
					"message": "Agent tag VC has been revoked",
				})
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"agent_id":    record.AgentID,
				"agent_did":   record.AgentDID,
				"vc_id":       record.VCID,
				"vc_document": json.RawMessage(record.VCDocument),
				"issued_at":   record.IssuedAt,
				"expires_at":  record.ExpiresAt,
			})
		})
		logger.Logger.Info().Msg("🔐 Agent tag VC endpoint registered")
	}

	// Policy distribution endpoint — agents cache these for local policy evaluation
	if s.accessPolicyService != nil {
		agentAPI.GET("/policies", func(c *gin.Context) {
			policies, err := s.accessPolicyService.ListPolicies(c.Request.Context())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "failed_to_list_policies",
					"message": "Failed to list policies",
				})
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"policies":   policies,
				"total":      len(policies),
				"fetched_at": time.Now().UTC().Format(time.RFC3339),
			})
		})
		logger.Logger.Info().Msg("📋 Policy distribution endpoint registered (GET /api/v1/policies)")
	}

	// Revocation list endpoint — agents cache revoked DIDs for local verification
	if s.didWebService != nil {
		agentAPI.GET("/revocations", func(c *gin.Context) {
			docs, err := s.storage.ListDIDDocuments(c.Request.Context())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "failed_to_list_revocations",
					"message": "Failed to list revocations",
				})
				return
			}
			revokedDIDs := make([]string, 0)
			for _, doc := range docs {
				if doc.IsRevoked() {
					revokedDIDs = append(revokedDIDs, doc.DID)
				}
			}
			c.JSON(http.StatusOK, gin.H{
				"revoked_dids": revokedDIDs,
				"total":        len(revokedDIDs),
				"fetched_at":   time.Now().UTC().Format(time.RFC3339),
			})
		})
		logger.Logger.Info().Msg("🚫 Revocation list endpoint registered (GET /api/v1/revocations)")
	}

	// Registered DIDs endpoint — agents cache this set for local verification
	// to ensure only known/registered DIDs are accepted on direct calls.
	agentAPI.GET("/registered-dids", func(c *gin.Context) {
		agentDIDs, err := s.storage.ListAgentDIDs(c.Request.Context())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "failed_to_list_registered_dids",
				"message": "Failed to list registered DIDs",
			})
			return
		}
		registeredDIDs := make([]string, 0, len(agentDIDs))
		for _, info := range agentDIDs {
			if info.Status == types.AgentDIDStatusActive {
				registeredDIDs = append(registeredDIDs, info.DID)
			}
		}
		c.JSON(http.StatusOK, gin.H{
			"registered_dids": registeredDIDs,
			"total":           len(registeredDIDs),
			"fetched_at":      time.Now().UTC().Format(time.RFC3339),
		})
	})
	logger.Logger.Info().Msg("✅ Registered DIDs endpoint registered (GET /api/v1/registered-dids)")

	// Issuer public key endpoint — agents use this for offline VC signature verification.
	// Registered at /did/issuer-public-key (public, semantic path) and
	// /admin/public-key (legacy alias for backward compatibility).
	if s.didService != nil {
		publicKeyHandler := func(c *gin.Context) {
			issuerDID, err := s.didService.GetControlPlaneIssuerDID()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "issuer_did_unavailable",
					"message": "Issuer DID unavailable",
				})
				return
			}
			identity, err := s.didService.ResolveDID(issuerDID)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "public_key_unavailable",
					"message": "Public key unavailable",
				})
				return
			}
			var publicKeyJWK map[string]interface{}
			if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "public_key_parse_error",
					"message": "Failed to parse public key JWK",
				})
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"issuer_did":     issuerDID,
				"public_key_jwk": publicKeyJWK,
				"fetched_at":     time.Now().UTC().Format(time.RFC3339),
			})
		}
		agentAPI.GET("/did/issuer-public-key", publicKeyHandler)
		agentAPI.GET("/admin/public-key", publicKeyHandler) // legacy alias
		logger.Logger.Info().Msg("🔑 Issuer public key endpoint registered (GET /api/v1/did/issuer-public-key)")
	}
}

// handleDIDWebServerDocument serves the server's root DID document per W3C did:web spec.
// GET /.well-known/did.json -> resolves did:web:{domain}
func (s *AgentFieldServer) handleDIDWebServerDocument(c *gin.Context) {
	serverID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "server DID not available"})
		return
	}
	registry, err := s.didService.GetRegistry(serverID)
	if err != nil || registry == nil || registry.RootDID == "" {
		c.JSON(http.StatusNotFound, gin.H{"error": "server DID not found"})
		return
	}
	s.serveDIDDocument(c, registry.RootDID)
}

// handleDIDWebAgentDocument serves an agent's DID document per W3C did:web spec.
// GET /agents/:agentID/did.json -> resolves did:web:{domain}:agents:{agentID}
func (s *AgentFieldServer) handleDIDWebAgentDocument(c *gin.Context) {
	agentID := c.Param("agentID")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID is required"})
		return
	}
	did := s.didWebService.GenerateDIDWeb(agentID)
	s.serveDIDDocument(c, did)
}

// serveDIDDocument resolves a DID and returns a W3C-compliant DID document.
// It tries did:web resolution (database) first, then falls back to did:key (in-memory).
func (s *AgentFieldServer) serveDIDDocument(c *gin.Context, did string) {
	// Try did:web resolution via DIDWebService (stored in database)
	if s.didWebService != nil && strings.HasPrefix(did, "did:web:") {
		result, err := s.didWebService.ResolveDID(c.Request.Context(), did)
		if err == nil && result.DIDDocument != nil {
			c.JSON(http.StatusOK, result.DIDDocument)
			return
		}
		if err == nil && result.DIDResolutionMetadata.Error == "deactivated" {
			c.JSON(http.StatusGone, gin.H{"error": "DID has been revoked"})
			return
		}
	}

	// Fall back to did:key resolution via DIDService (in-memory registry)
	identity, err := s.didService.ResolveDID(did)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "DID not found"})
		return
	}

	var publicKeyJWK map[string]interface{}
	if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to parse public key"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"@context": []string{
			"https://www.w3.org/ns/did/v1",
			"https://w3id.org/security/suites/ed25519-2020/v1",
		},
		"id": did,
		"verificationMethod": []gin.H{{
			"id":           did + "#key-1",
			"type":         "Ed25519VerificationKey2020",
			"controller":   did,
			"publicKeyJwk": publicKeyJWK,
		}},
		"authentication":  []string{did + "#key-1"},
		"assertionMethod": []string{did + "#key-1"},
	})
}
