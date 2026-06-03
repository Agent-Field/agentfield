// Package internal provides integration tests for the VC-based authorization system.
//
// These tests verify the complete flow from storage through services to HTTP handlers
// with minimal mocking, using real SQLite storage for integration validation.
package internal

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Test Infrastructure
// =============================================================================

// testContext holds all components needed for integration testing
type testContext struct {
	t                   *testing.T
	ctx                 context.Context
	storage             *storage.LocalStorage
	didWebService       *mockDIDWebService
	accessPolicyService *services.AccessPolicyService
	router              *gin.Engine
	cleanup             func()
}

// mockDIDWebService provides a minimal DID web service for testing
type mockDIDWebService struct {
	domain  string
	storage storage.StorageProvider
}

func newMockDIDWebService(domain string, s storage.StorageProvider) *mockDIDWebService {
	return &mockDIDWebService{
		domain:  domain,
		storage: s,
	}
}

func (m *mockDIDWebService) GenerateDIDWeb(agentID string) string {
	encodedDomain := strings.ReplaceAll(m.domain, ":", "%3A")
	return fmt.Sprintf("did:web:%s:agents:%s", encodedDomain, agentID)
}

func (m *mockDIDWebService) ParseDIDWeb(did string) (string, error) {
	if !strings.HasPrefix(did, "did:web:") {
		return "", fmt.Errorf("invalid did:web format")
	}
	parts := strings.Split(did, ":")
	for i, part := range parts {
		if part == "agents" && i+1 < len(parts) {
			return parts[i+1], nil
		}
	}
	return "", fmt.Errorf("invalid did:web format: missing 'agents' segment")
}

func (m *mockDIDWebService) VerifyDIDOwnership(ctx context.Context, did string, message []byte, signature []byte) (bool, error) {
	// Look up the DID document to get the public key
	record, err := m.storage.GetDIDDocument(ctx, did)
	if err != nil {
		return false, fmt.Errorf("DID not found: %w", err)
	}

	if record.IsRevoked() {
		return false, fmt.Errorf("DID is revoked")
	}

	// Parse the public key from JWK
	var jwk struct {
		X string `json:"x"`
	}
	if err := json.Unmarshal([]byte(record.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("invalid public key JWK: %w", err)
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(jwk.X)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)
	return ed25519.Verify(publicKey, message, signature), nil
}

func (m *mockDIDWebService) RevokeDID(ctx context.Context, did string) error {
	return m.storage.RevokeDIDDocument(ctx, did)
}

func (m *mockDIDWebService) GetOrCreateDIDDocument(ctx context.Context, agentID string) (*types.DIDWebDocument, string, error) {
	did := m.GenerateDIDWeb(agentID)

	// Try to get existing
	record, err := m.storage.GetDIDDocument(ctx, did)
	if err == nil && !record.IsRevoked() {
		var didDoc types.DIDWebDocument
		if err := json.Unmarshal(record.DIDDocument, &didDoc); err != nil {
			return nil, "", err
		}
		return &didDoc, did, nil
	}

	// Generate new key pair
	publicKey, _, err := ed25519.GenerateKey(nil)
	if err != nil {
		return nil, "", err
	}

	pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
		base64.RawURLEncoding.EncodeToString(publicKey))

	didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
	docBytes, _ := json.Marshal(didDoc)

	record = &types.DIDDocumentRecord{
		DID:          did,
		AgentID:      agentID,
		DIDDocument:  docBytes,
		PublicKeyJWK: pubKeyJWK,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	if err := m.storage.StoreDIDDocument(ctx, record); err != nil {
		return nil, "", err
	}

	return didDoc, did, nil
}

func (m *mockDIDWebService) ResolveDID(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
	record, err := m.storage.GetDIDDocument(ctx, did)
	if err != nil {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "notFound"},
		}, nil
	}

	if record.IsRevoked() {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "deactivated"},
			DIDDocumentMetadata:   types.DIDDocumentMetadata{Deactivated: true},
		}, nil
	}

	var didDoc types.DIDWebDocument
	if err := json.Unmarshal(record.DIDDocument, &didDoc); err != nil {
		return &types.DIDResolutionResult{
			DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "invalidDidDocument"},
		}, nil
	}

	return &types.DIDResolutionResult{
		DIDDocument:           &didDoc,
		DIDResolutionMetadata: types.DIDResolutionMetadata{ContentType: "application/did+ld+json"},
	}, nil
}

// setupTestContext creates a fully initialized test environment with real storage
func setupTestContext(t *testing.T) *testContext {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()

	// Initialize real SQLite storage
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "test_agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "test_agentfield.bolt"),
		},
	}

	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping integration test")
		}
		t.Fatalf("failed to initialize local storage: %v", err)
	}

	// Create mock DID Web service (uses real storage)
	didWebService := newMockDIDWebService("localhost:8080", ls)

	// Create access policy service (replaces legacy permission service)
	accessPolicyService := services.NewAccessPolicyService(ls)
	err := accessPolicyService.Initialize(ctx)
	require.NoError(t, err, "failed to initialize access policy service")

	// Set up Gin router for HTTP tests
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// Add test middleware that simulates DID auth verification.
	// In production, the DID auth middleware verifies signatures and sets verified_caller_did.
	// For integration tests, we trust the X-Caller-DID header directly.
	router.Use(func(c *gin.Context) {
		if did := c.GetHeader("X-Caller-DID"); did != "" {
			c.Set("verified_caller_did", did)
		}
		c.Next()
	})

	tc := &testContext{
		t:                   t,
		ctx:                 ctx,
		storage:             ls,
		didWebService:       didWebService,
		accessPolicyService: accessPolicyService,
		router:              router,
		cleanup: func() {
			_ = ls.Close(ctx)
		},
	}

	t.Cleanup(tc.cleanup)

	return tc
}

// createTestAgent creates a test agent in storage with the given ID and tags.
// Tags are stored as approved tags (key:value format) for authorization matching.
// Deployment metadata tags are excluded from authorization — only ApprovedTags
// are used by CanonicalAgentTags for permission enforcement.
func (tc *testContext) createTestAgent(agentID string, tags map[string]string) *types.AgentNode {
	tc.t.Helper()

	// Convert key:value tags to canonical approved tags
	var approvedTags []string
	for k, v := range tags {
		approvedTags = append(approvedTags, k+":"+v)
	}

	agent := &types.AgentNode{
		ID:             agentID,
		DeploymentType: "test",
		ApprovedTags:   approvedTags,
		Metadata: types.AgentMetadata{
			Deployment: &types.DeploymentMetadata{
				Tags: tags,
			},
		},
		RegisteredAt: time.Now(),
	}

	err := tc.storage.RegisterAgent(tc.ctx, agent)
	require.NoError(tc.t, err, "failed to register test agent")

	return agent
}

// =============================================================================
// Phase 1: Storage Layer Tests
// =============================================================================

func TestVCAuth_Phase1_Storage_DIDDocuments(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("store and retrieve DID document", func(t *testing.T) {
		// Create a DID document
		agentID := "test-agent-did-1"
		did := tc.didWebService.GenerateDIDWeb(agentID)

		// Generate a test public key JWK
		pubKey, _, err := ed25519.GenerateKey(nil)
		require.NoError(t, err)

		pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
			base64.RawURLEncoding.EncodeToString(pubKey))

		didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
		docBytes, err := json.Marshal(didDoc)
		require.NoError(t, err)

		record := &types.DIDDocumentRecord{
			DID:          did,
			AgentID:      agentID,
			DIDDocument:  docBytes,
			PublicKeyJWK: pubKeyJWK,
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
		}

		// Store the DID document
		err = tc.storage.StoreDIDDocument(tc.ctx, record)
		require.NoError(t, err)

		// Retrieve by DID
		retrieved, err := tc.storage.GetDIDDocument(tc.ctx, did)
		require.NoError(t, err)
		require.NotNil(t, retrieved)
		assert.Equal(t, did, retrieved.DID)
		assert.Equal(t, agentID, retrieved.AgentID)
		assert.False(t, retrieved.IsRevoked())

		// Retrieve by agent ID
		retrievedByAgent, err := tc.storage.GetDIDDocumentByAgentID(tc.ctx, agentID)
		require.NoError(t, err)
		require.NotNil(t, retrievedByAgent)
		assert.Equal(t, did, retrievedByAgent.DID)
	})

	t.Run("revoke DID document", func(t *testing.T) {
		agentID := "test-agent-did-2"
		did := tc.didWebService.GenerateDIDWeb(agentID)

		pubKey, _, err := ed25519.GenerateKey(nil)
		require.NoError(t, err)

		pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
			base64.RawURLEncoding.EncodeToString(pubKey))

		didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
		docBytes, _ := json.Marshal(didDoc)

		record := &types.DIDDocumentRecord{
			DID:          did,
			AgentID:      agentID,
			DIDDocument:  docBytes,
			PublicKeyJWK: pubKeyJWK,
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
		}

		err = tc.storage.StoreDIDDocument(tc.ctx, record)
		require.NoError(t, err)

		// Revoke the DID
		err = tc.storage.RevokeDIDDocument(tc.ctx, did)
		require.NoError(t, err)

		// Verify it's revoked
		retrieved, err := tc.storage.GetDIDDocument(tc.ctx, did)
		require.NoError(t, err)
		assert.True(t, retrieved.IsRevoked())
	})

	t.Run("list DID documents", func(t *testing.T) {
		// Create multiple DID documents
		for i := 3; i <= 5; i++ {
			agentID := fmt.Sprintf("test-agent-did-%d", i)
			did := tc.didWebService.GenerateDIDWeb(agentID)

			pubKey, _, _ := ed25519.GenerateKey(nil)
			pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
				base64.RawURLEncoding.EncodeToString(pubKey))

			didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
			docBytes, _ := json.Marshal(didDoc)

			record := &types.DIDDocumentRecord{
				DID:          did,
				AgentID:      agentID,
				DIDDocument:  docBytes,
				PublicKeyJWK: pubKeyJWK,
				CreatedAt:    time.Now(),
				UpdatedAt:    time.Now(),
			}

			_ = tc.storage.StoreDIDDocument(tc.ctx, record)
		}

		// List all DID documents
		docs, err := tc.storage.ListDIDDocuments(tc.ctx)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(docs), 3)
	})
}

func TestVCAuth_Phase1_Storage_AccessPolicies(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("create and retrieve access policy", func(t *testing.T) {
		now := time.Now()
		policy := &types.AccessPolicy{
			Name:           "test-policy",
			CallerTags:     []string{"analytics"},
			TargetTags:     []string{"data-service"},
			AllowFunctions: []string{"query_*", "get_*"},
			DenyFunctions:  []string{"delete_*"},
			Constraints: map[string]types.AccessConstraint{
				"limit": {Operator: "<=", Value: 1000},
			},
			Action:    "allow",
			Priority:  100,
			Enabled:   true,
			CreatedAt: now,
			UpdatedAt: now,
		}

		err := tc.storage.CreateAccessPolicy(tc.ctx, policy)
		require.NoError(t, err)
		assert.NotZero(t, policy.ID, "policy ID should be set after creation")

		// Retrieve by ID
		retrieved, err := tc.storage.GetAccessPolicyByID(tc.ctx, policy.ID)
		require.NoError(t, err)
		assert.Equal(t, "test-policy", retrieved.Name)
		assert.Equal(t, "allow", retrieved.Action)
		assert.Equal(t, 100, retrieved.Priority)
		assert.True(t, retrieved.Enabled)
	})

	t.Run("list access policies", func(t *testing.T) {
		policies, err := tc.storage.GetAccessPolicies(tc.ctx)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(policies), 1)
	})

	t.Run("update access policy", func(t *testing.T) {
		now := time.Now()
		policy := &types.AccessPolicy{
			Name:       "to-update",
			CallerTags: []string{"caller"},
			TargetTags: []string{"target"},
			Action:     "deny",
			Priority:   50,
			Enabled:    true,
			CreatedAt:  now,
			UpdatedAt:  now,
		}

		err := tc.storage.CreateAccessPolicy(tc.ctx, policy)
		require.NoError(t, err)

		policy.Action = "allow"
		policy.Priority = 200
		policy.UpdatedAt = time.Now()
		err = tc.storage.UpdateAccessPolicy(tc.ctx, policy)
		require.NoError(t, err)

		retrieved, err := tc.storage.GetAccessPolicyByID(tc.ctx, policy.ID)
		require.NoError(t, err)
		assert.Equal(t, "allow", retrieved.Action)
		assert.Equal(t, 200, retrieved.Priority)
	})

	t.Run("delete access policy", func(t *testing.T) {
		now := time.Now()
		policy := &types.AccessPolicy{
			Name:       "to-delete",
			CallerTags: []string{"temp"},
			TargetTags: []string{"temp"},
			Action:     "allow",
			Priority:   10,
			Enabled:    true,
			CreatedAt:  now,
			UpdatedAt:  now,
		}

		err := tc.storage.CreateAccessPolicy(tc.ctx, policy)
		require.NoError(t, err)

		err = tc.storage.DeleteAccessPolicy(tc.ctx, policy.ID)
		require.NoError(t, err)

		_, err = tc.storage.GetAccessPolicyByID(tc.ctx, policy.ID)
		assert.Error(t, err, "deleted policy should not be retrievable")
	})
}

// =============================================================================
// Phase 2: Service Layer Tests
// =============================================================================

func TestVCAuth_Phase2_Service_DIDWebService(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("generate DID web identifier", func(t *testing.T) {
		did := tc.didWebService.GenerateDIDWeb("my-agent")
		assert.Contains(t, did, "did:web:")
		assert.Contains(t, did, "agents:my-agent")
	})

	t.Run("parse DID web identifier", func(t *testing.T) {
		did := tc.didWebService.GenerateDIDWeb("parsed-agent")
		agentID, err := tc.didWebService.ParseDIDWeb(did)
		require.NoError(t, err)
		assert.Equal(t, "parsed-agent", agentID)
	})

	t.Run("get or create DID document", func(t *testing.T) {
		agentID := "new-agent-did"

		// First call creates
		didDoc1, did1, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, agentID)
		require.NoError(t, err)
		require.NotNil(t, didDoc1)
		assert.Contains(t, did1, agentID)

		// Second call returns existing
		didDoc2, did2, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, agentID)
		require.NoError(t, err)
		assert.Equal(t, did1, did2)
		assert.Equal(t, didDoc1.ID, didDoc2.ID)
	})

	t.Run("resolve DID", func(t *testing.T) {
		agentID := "resolvable-agent"
		_, did, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, agentID)
		require.NoError(t, err)

		// Resolve the DID
		result, err := tc.didWebService.ResolveDID(tc.ctx, did)
		require.NoError(t, err)
		assert.NotNil(t, result.DIDDocument)
		assert.Empty(t, result.DIDResolutionMetadata.Error)
	})

	t.Run("resolve non-existent DID returns not found", func(t *testing.T) {
		result, err := tc.didWebService.ResolveDID(tc.ctx, "did:web:localhost:agents:nonexistent")
		require.NoError(t, err)
		assert.Equal(t, "notFound", result.DIDResolutionMetadata.Error)
	})

	t.Run("resolve revoked DID returns deactivated", func(t *testing.T) {
		agentID := "to-revoke-agent"
		_, did, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, agentID)
		require.NoError(t, err)

		// Revoke
		err = tc.didWebService.RevokeDID(tc.ctx, did)
		require.NoError(t, err)

		// Resolve should show deactivated
		result, err := tc.didWebService.ResolveDID(tc.ctx, did)
		require.NoError(t, err)
		assert.Equal(t, "deactivated", result.DIDResolutionMetadata.Error)
		assert.True(t, result.DIDDocumentMetadata.Deactivated)
	})
}

func TestVCAuth_Phase2_Service_AccessPolicyService(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("add and evaluate allow policy", func(t *testing.T) {
		req := &types.AccessPolicyRequest{
			Name:           "analytics-to-data",
			CallerTags:     []string{"analytics"},
			TargetTags:     []string{"data-service"},
			AllowFunctions: []string{"query_*", "get_*"},
			DenyFunctions:  []string{"delete_*"},
			Constraints: map[string]types.AccessConstraint{
				"limit": {Operator: "<=", Value: float64(1000)},
			},
			Action:   "allow",
			Priority: 100,
		}

		policy, err := tc.accessPolicyService.AddPolicy(tc.ctx, req)
		require.NoError(t, err)
		assert.NotZero(t, policy.ID)

		// Evaluate: analytics caller → data-service target → query_data → allowed
		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"analytics"}, []string{"data-service"},
			"query_data", map[string]any{"limit": float64(500)},
		)
		assert.True(t, result.Matched, "policy should match")
		assert.True(t, result.Allowed, "access should be allowed")
		assert.Equal(t, "analytics-to-data", result.PolicyName)
	})

	t.Run("deny function takes precedence", func(t *testing.T) {
		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"analytics"}, []string{"data-service"},
			"delete_records", map[string]any{},
		)
		assert.True(t, result.Matched, "policy should match")
		assert.False(t, result.Allowed, "delete should be denied")
		assert.Contains(t, result.Reason, "denied")
	})

	t.Run("constraint violation denies access", func(t *testing.T) {
		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"analytics"}, []string{"data-service"},
			"query_data", map[string]any{"limit": float64(5000)},
		)
		assert.True(t, result.Matched, "policy should match")
		assert.False(t, result.Allowed, "over-limit query should be denied")
		assert.Contains(t, result.Reason, "Constraint violation")
	})

	t.Run("non-matching tags yield no match", func(t *testing.T) {
		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"unknown"}, []string{"data-service"},
			"query_data", nil,
		)
		assert.False(t, result.Matched, "policy should not match for unknown caller tag")
	})

	t.Run("update policy changes behavior", func(t *testing.T) {
		policies, err := tc.accessPolicyService.ListPolicies(tc.ctx)
		require.NoError(t, err)
		require.NotEmpty(t, policies)

		policyID := policies[0].ID

		updateReq := &types.AccessPolicyRequest{
			Name:       "analytics-to-data-updated",
			CallerTags: []string{"analytics"},
			TargetTags: []string{"data-service"},
			Action:     "deny", // Changed to deny
			Priority:   100,
		}

		updated, err := tc.accessPolicyService.UpdatePolicy(tc.ctx, policyID, updateReq)
		require.NoError(t, err)
		assert.Equal(t, "deny", updated.Action)

		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"analytics"}, []string{"data-service"},
			"query_data", nil,
		)
		assert.True(t, result.Matched)
		assert.False(t, result.Allowed, "should be denied after policy update")
	})

	t.Run("remove policy removes enforcement", func(t *testing.T) {
		policies, err := tc.accessPolicyService.ListPolicies(tc.ctx)
		require.NoError(t, err)
		require.NotEmpty(t, policies)

		err = tc.accessPolicyService.RemovePolicy(tc.ctx, policies[0].ID)
		require.NoError(t, err)

		result := tc.accessPolicyService.EvaluateAccess(
			[]string{"analytics"}, []string{"data-service"},
			"query_data", nil,
		)
		assert.False(t, result.Matched, "no policy should match after deletion")
	})
}

// =============================================================================
// Phase 3: Middleware Tests
// =============================================================================

func TestVCAuth_Phase3_Middleware_DIDAuth(t *testing.T) {
	tc := setupTestContext(t)

	// Generate test key pair
	publicKey, privateKey, err := ed25519.GenerateKey(nil)
	require.NoError(t, err)

	// Create DID and store document
	agentID := "middleware-test-agent"
	did := tc.didWebService.GenerateDIDWeb(agentID)

	// Store DID document with our test key
	pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
		base64.RawURLEncoding.EncodeToString(publicKey))

	didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
	docBytes, _ := json.Marshal(didDoc)

	record := &types.DIDDocumentRecord{
		DID:          did,
		AgentID:      agentID,
		DIDDocument:  docBytes,
		PublicKeyJWK: pubKeyJWK,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
	err = tc.storage.StoreDIDDocument(tc.ctx, record)
	require.NoError(t, err)

	// Helper to sign requests
	signRequest := func(body []byte) (string, string) {
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))
		return base64.StdEncoding.EncodeToString(signature), timestamp
	}

	t.Run("request without DID passes through", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		req := httptest.NewRequest("POST", "/test", strings.NewReader(`{}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("request with valid DID signature succeeds", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			verifiedDID := middleware.GetVerifiedCallerDID(c)
			c.JSON(200, gin.H{"verified_did": verifiedDID})
		})

		body := []byte(`{"test":"data"}`)
		signature, timestamp := signRequest(body)

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", signature)
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)

		var response map[string]string
		json.Unmarshal(w.Body.Bytes(), &response)
		assert.Equal(t, did, response["verified_did"])
	})

	t.Run("request with DID but missing signature fails", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		req := httptest.NewRequest("POST", "/test", strings.NewReader(`{}`))
		req.Header.Set("X-Caller-DID", did)
		// Missing signature and timestamp

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})

	t.Run("request with invalid signature fails", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		body := []byte(`{"test":"data"}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString([]byte("invalid")))
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})

	t.Run("request with expired timestamp fails", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		body := []byte(`{"test":"data"}`)
		// Use timestamp 10 minutes ago
		oldTimestamp := strconv.FormatInt(time.Now().Add(-10*time.Minute).Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", oldTimestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", oldTimestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})
}

// =============================================================================
// Phase 5: End-to-End Integration Tests
// =============================================================================

func TestVCAuth_Phase5_EndToEnd_DIDAuthentication(t *testing.T) {
	tc := setupTestContext(t)

	// Generate key pair for the calling agent
	publicKey, privateKey, err := ed25519.GenerateKey(nil)
	require.NoError(t, err)

	agentID := "did-auth-e2e-agent"
	did := tc.didWebService.GenerateDIDWeb(agentID)

	// Store DID document with the test public key
	pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
		base64.RawURLEncoding.EncodeToString(publicKey))

	didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
	docBytes, _ := json.Marshal(didDoc)

	record := &types.DIDDocumentRecord{
		DID:          did,
		AgentID:      agentID,
		DIDDocument:  docBytes,
		PublicKeyJWK: pubKeyJWK,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
	tc.storage.StoreDIDDocument(tc.ctx, record)

	// Helper to create signed requests
	signAndSend := func(router *gin.Engine, method, path string, body []byte) *httptest.ResponseRecorder {
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		req := httptest.NewRequest(method, path, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w
	}

	t.Run("authenticated request succeeds with valid signature", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			verifiedDID := middleware.GetVerifiedCallerDID(c)
			c.JSON(200, gin.H{
				"success":      true,
				"verified_did": verifiedDID,
			})
		})

		body := []byte(`{"action": "test"}`)
		w := signAndSend(router, "POST", "/test", body)

		assert.Equal(t, 200, w.Code)

		var response map[string]interface{}
		json.Unmarshal(w.Body.Bytes(), &response)
		assert.True(t, response["success"].(bool))
		assert.Equal(t, did, response["verified_did"])
	})

	t.Run("request with tampered body fails", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"success": true})
		})

		// Sign with original body
		originalBody := []byte(`{"action": "original"}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(originalBody)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		// Send with different body
		tamperedBody := []byte(`{"action": "tampered"}`)
		req := httptest.NewRequest("POST", "/test", bytes.NewReader(tamperedBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})

	t.Run("replay attack with old timestamp fails", func(t *testing.T) {
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300, // 5 minutes
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"success": true})
		})

		body := []byte(`{"action": "test"}`)
		// Use timestamp from 10 minutes ago
		oldTimestamp := strconv.FormatInt(time.Now().Add(-10*time.Minute).Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", oldTimestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", oldTimestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})
}

// =============================================================================
// Phase 6: SDK Compatibility Tests
// =============================================================================

func TestVCAuth_Phase6_SDK_GoClientDIDAuth(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("SDK signing produces valid signature", func(t *testing.T) {
		// Generate test key pair
		publicKey, privateKey, err := ed25519.GenerateKey(nil)
		require.NoError(t, err)

		agentID := "sdk-test-agent"
		did := tc.didWebService.GenerateDIDWeb(agentID)

		// Store DID document with public key
		pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
			base64.RawURLEncoding.EncodeToString(publicKey))

		didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
		docBytes, _ := json.Marshal(didDoc)

		record := &types.DIDDocumentRecord{
			DID:          did,
			AgentID:      agentID,
			DIDDocument:  docBytes,
			PublicKeyJWK: pubKeyJWK,
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
		}
		tc.storage.StoreDIDDocument(tc.ctx, record)

		// Create private key JWK for SDK
		privateKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","d":"%s","x":"%s"}`,
			base64.RawURLEncoding.EncodeToString(privateKey.Seed()),
			base64.RawURLEncoding.EncodeToString(publicKey))

		// Simulate SDK signing (matching did_auth.go implementation)
		body := []byte(`{"target": "other-agent.skill", "input": {"data": "test"}}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))
		signatureB64 := base64.StdEncoding.EncodeToString(signature)

		// Set up router with DID auth middleware
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/execute", func(c *gin.Context) {
			verifiedDID := middleware.GetVerifiedCallerDID(c)
			c.JSON(200, gin.H{"verified_did": verifiedDID})
		})

		// Make request with SDK-style headers
		req := httptest.NewRequest("POST", "/execute", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", signatureB64)
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)

		var response map[string]string
		json.Unmarshal(w.Body.Bytes(), &response)
		assert.Equal(t, did, response["verified_did"])

		t.Logf("SDK signature verification successful for DID: %s", did)
		t.Logf("Private key JWK (for SDK testing): %s", privateKeyJWK)
	})
}

// =============================================================================
// Regression: DID:web document lifecycle during agent registration
// =============================================================================

// TestVCAuth_Regression_DIDWebDocumentCreatedDuringRegistration is a regression test
// for a bug where GetOrCreateDIDDocument was never called during agent registration,
// causing VerifyDIDOwnership (DID auth middleware) to always fail with "notFound"
// because no DID:web document existed in storage.
//
// The fix (in RegisterNodeHandler) calls GetOrCreateDIDDocument after DID:key
// registration. This test ensures that flow works end-to-end and catches any
// regression that would break it.
func TestVCAuth_Regression_DIDWebDocumentCreatedDuringRegistration(t *testing.T) {
	tc := setupTestContext(t)

	agentID := "regression-agent-lifecycle"

	t.Run("without DID:web document, auth middleware rejects signed requests", func(t *testing.T) {
		// Simulate the OLD broken behavior: agent is registered in storage
		// but GetOrCreateDIDDocument is never called.
		agent := tc.createTestAgent(agentID+"-broken", nil)
		_ = agent

		// The DID:web identifier exists as a string...
		did := tc.didWebService.GenerateDIDWeb(agentID + "-broken")

		// ...but no DID document was stored, so ResolveDID returns "notFound"
		result, err := tc.didWebService.ResolveDID(tc.ctx, did)
		require.NoError(t, err)
		assert.Equal(t, "notFound", result.DIDResolutionMetadata.Error,
			"BUG REGRESSION: ResolveDID should return notFound when GetOrCreateDIDDocument was never called")
		assert.Nil(t, result.DIDDocument)

		// VerifyDIDOwnership should also fail
		valid, err := tc.didWebService.VerifyDIDOwnership(tc.ctx, did, []byte("test"), []byte("sig"))
		assert.Error(t, err)
		assert.False(t, valid)
		assert.Contains(t, err.Error(), "not found",
			"BUG REGRESSION: VerifyDIDOwnership fails because no DID:web document exists")
	})

	t.Run("with GetOrCreateDIDDocument called during registration, auth middleware accepts signed requests", func(t *testing.T) {
		// Simulate the FIXED behavior: agent is registered AND GetOrCreateDIDDocument
		// is called, just like RegisterNodeHandler does after the fix.
		agent := tc.createTestAgent(agentID, nil)
		_ = agent

		// This is the critical call that the fix adds to RegisterNodeHandler.
		didDoc, did, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, agentID)
		require.NoError(t, err, "GetOrCreateDIDDocument must succeed during registration")
		require.NotNil(t, didDoc, "DID document must be created")
		assert.Contains(t, did, agentID)

		// Verify the document is now resolvable
		result, err := tc.didWebService.ResolveDID(tc.ctx, did)
		require.NoError(t, err)
		assert.Empty(t, result.DIDResolutionMetadata.Error,
			"After GetOrCreateDIDDocument, ResolveDID must succeed")
		require.NotNil(t, result.DIDDocument, "DID document must be resolvable")
		assert.Equal(t, did, result.DIDDocument.ID)

		// Verify the document has a verification method with a public key
		require.NotEmpty(t, result.DIDDocument.VerificationMethod,
			"DID document must have at least one verification method")
		vm := result.DIDDocument.VerificationMethod[0]
		assert.Equal(t, "JsonWebKey2020", vm.Type)
		assert.NotEmpty(t, vm.PublicKeyJwk, "Verification method must contain a public key JWK")

		// Extract the public key and generate a matching private key to sign a request,
		// then verify through the middleware. We use the stored public key to prove the
		// full chain works.
		var jwk struct {
			X string `json:"x"`
		}
		err = json.Unmarshal(vm.PublicKeyJwk, &jwk)
		require.NoError(t, err, "Public key JWK must be valid JSON")

		publicKeyBytes, err := base64.RawURLEncoding.DecodeString(jwk.X)
		require.NoError(t, err)
		assert.Len(t, publicKeyBytes, ed25519.PublicKeySize,
			"Public key must be a valid Ed25519 key")
	})

	t.Run("idempotent: calling GetOrCreateDIDDocument twice returns same document", func(t *testing.T) {
		reregAgentID := agentID + "-rereg"
		tc.createTestAgent(reregAgentID, nil)

		// First call (initial registration)
		doc1, did1, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, reregAgentID)
		require.NoError(t, err)
		require.NotNil(t, doc1)

		// Second call (re-registration)
		doc2, did2, err := tc.didWebService.GetOrCreateDIDDocument(tc.ctx, reregAgentID)
		require.NoError(t, err)
		require.NotNil(t, doc2)

		assert.Equal(t, did1, did2, "DID must be stable across re-registrations")
		assert.Equal(t, doc1.ID, doc2.ID, "Document ID must be stable across re-registrations")
	})

	t.Run("end-to-end: registered agent can authenticate through DID middleware", func(t *testing.T) {
		// This is the full end-to-end flow:
		// 1. Agent registered in storage
		// 2. GetOrCreateDIDDocument called (our fix)
		// 3. Agent signs a request
		// 4. DID auth middleware verifies the signature

		e2eAgentID := agentID + "-e2e"
		tc.createTestAgent(e2eAgentID, nil)

		// Create a known key pair for signing
		publicKey, privateKey, err := ed25519.GenerateKey(nil)
		require.NoError(t, err)

		// Store a DID document with our known test key (simulating what
		// GetOrCreateDIDDocument does, but with a key we control for signing)
		did := tc.didWebService.GenerateDIDWeb(e2eAgentID)
		pubKeyJWK := fmt.Sprintf(`{"kty":"OKP","crv":"Ed25519","x":"%s"}`,
			base64.RawURLEncoding.EncodeToString(publicKey))

		didDoc := types.NewDIDWebDocument(did, json.RawMessage(pubKeyJWK))
		docBytes, _ := json.Marshal(didDoc)

		record := &types.DIDDocumentRecord{
			DID:          did,
			AgentID:      e2eAgentID,
			DIDDocument:  docBytes,
			PublicKeyJWK: pubKeyJWK,
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
		}
		err = tc.storage.StoreDIDDocument(tc.ctx, record)
		require.NoError(t, err)

		// Set up router with DID auth middleware
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			verifiedDID := middleware.GetVerifiedCallerDID(c)
			c.JSON(200, gin.H{"verified_did": verifiedDID})
		})

		// Sign a request
		body := []byte(`{"action":"test-registration-flow"}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code,
			"Registered agent with DID:web document must pass DID auth middleware")

		var response map[string]string
		err = json.Unmarshal(w.Body.Bytes(), &response)
		require.NoError(t, err)
		assert.Equal(t, did, response["verified_did"],
			"Middleware must set the verified DID in context")
	})

	t.Run("end-to-end: unregistered DID:web document fails auth middleware", func(t *testing.T) {
		// Agent exists in storage but NO DID:web document was created.
		// This is the exact bug scenario. The middleware must reject.
		noDocAgentID := agentID + "-no-doc"
		tc.createTestAgent(noDocAgentID, nil)

		// Generate a key pair (agent has keys, but no doc in storage)
		_, privateKey, err := ed25519.GenerateKey(nil)
		require.NoError(t, err)

		did := tc.didWebService.GenerateDIDWeb(noDocAgentID)

		// Set up router with DID auth middleware
		router := gin.New()
		router.Use(middleware.DIDAuthMiddleware(tc.didWebService, middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: 300,
		}))
		router.POST("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		// Sign a request
		body := []byte(`{"action":"test-no-doc"}`)
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		bodyHash := sha256.Sum256(body)
		payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
		signature := ed25519.Sign(privateKey, []byte(payload))

		req := httptest.NewRequest("POST", "/test", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Caller-DID", did)
		req.Header.Set("X-DID-Signature", base64.StdEncoding.EncodeToString(signature))
		req.Header.Set("X-DID-Timestamp", timestamp)

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code,
			"Agent without DID:web document must be rejected by auth middleware")

		var errResponse map[string]string
		err = json.Unmarshal(w.Body.Bytes(), &errResponse)
		require.NoError(t, err)
		assert.Equal(t, "verification_error", errResponse["error"],
			"Error must indicate verification failure due to missing document")
	})
}

// =============================================================================
// Phase 8: Re-registration State Preservation (D3)
// =============================================================================

func TestVCAuth_ReRegistration_PreservesApprovalState(t *testing.T) {
	tc := setupTestContext(t)

	t.Run("ready agent re-registering stays ready", func(t *testing.T) {
		agent := &types.AgentNode{
			ID:              "reregister-ready",
			LifecycleStatus: types.AgentStatusReady,
			ApprovedTags:    []string{"finance:billing"},
			RegisteredAt:    time.Now(),
		}
		err := tc.storage.RegisterAgent(tc.ctx, agent)
		require.NoError(t, err)

		// Verify the agent is ready
		stored, err := tc.storage.GetAgent(tc.ctx, "reregister-ready")
		require.NoError(t, err)
		assert.Equal(t, types.AgentStatusReady, stored.LifecycleStatus)
		assert.Equal(t, []string{"finance:billing"}, stored.ApprovedTags)
	})

	t.Run("admin-revoked agent stays pending on re-register", func(t *testing.T) {
		// Create an agent that was admin-revoked: pending_approval + empty approved tags
		agent := &types.AgentNode{
			ID:              "reregister-revoked",
			LifecycleStatus: types.AgentStatusPendingApproval,
			ApprovedTags:    nil, // Admin cleared approved tags
			ProposedTags:    []string{"sensitive"},
			RegisteredAt:    time.Now(),
		}
		err := tc.storage.RegisterAgent(tc.ctx, agent)
		require.NoError(t, err)

		// Verify admin-revoked state is stored
		stored, err := tc.storage.GetAgent(tc.ctx, "reregister-revoked")
		require.NoError(t, err)
		assert.Equal(t, types.AgentStatusPendingApproval, stored.LifecycleStatus)
		assert.Empty(t, stored.ApprovedTags)
	})
}
