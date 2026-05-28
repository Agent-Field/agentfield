package middleware

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// --- Test Mocks ---

type testAgentResolver struct {
	agents map[string]*types.AgentNode
}

func (r *testAgentResolver) GetAgent(_ context.Context, agentID string) (*types.AgentNode, error) {
	if a, ok := r.agents[agentID]; ok {
		return a, nil
	}
	return &types.AgentNode{ID: agentID, ApprovedTags: []string{"public"}}, nil
}

type failingAgentResolver struct{}

func (r *failingAgentResolver) GetAgent(_ context.Context, _ string) (*types.AgentNode, error) {
	return nil, fmt.Errorf("storage unavailable")
}

type testDIDResolver struct{}

func (r *testDIDResolver) GenerateDIDWeb(agentID string) string {
	return "did:web:localhost%3A8080:agents:" + agentID
}

func (r *testDIDResolver) ResolveAgentIDByDID(_ context.Context, _ string) string {
	return ""
}

type testDIDWebService struct {
	publicKeys map[string]ed25519.PublicKey
}

func (s *testDIDWebService) VerifyDIDOwnership(_ context.Context, did string, message []byte, signature []byte) (bool, error) {
	pub, ok := s.publicKeys[did]
	if !ok {
		return false, fmt.Errorf("did not found")
	}
	return ed25519.Verify(pub, message, signature), nil
}

type testPolicyService struct {
	result *types.PolicyEvaluationResult
}

func (s *testPolicyService) EvaluateAccess(callerTags, targetTags []string, functionName string, inputParams map[string]any) *types.PolicyEvaluationResult {
	if s.result != nil {
		return s.result
	}
	return &types.PolicyEvaluationResult{Matched: false}
}

func signRequestBody(body []byte, did string, privateKey ed25519.PrivateKey, ts time.Time) (map[string]string, error) {
	timestamp := fmt.Sprintf("%d", ts.Unix())
	bodyHash := sha256.Sum256(body)
	payload := fmt.Sprintf("%s:%x", timestamp, bodyHash)
	signature := ed25519.Sign(privateKey, []byte(payload))
	return map[string]string{
		"X-Caller-DID":    did,
		"X-DID-Signature": base64.StdEncoding.EncodeToString(signature),
		"X-DID-Timestamp": timestamp,
	}, nil
}

// --- Test Helpers ---

func setupTestRoute(policyService AccessPolicyServiceInterface, didService DIDWebServiceInterface, resolver AgentResolverInterface) *gin.Engine {
	return setupTestRouteWithConfig(policyService, didService, resolver, PermissionConfig{Enabled: true})
}

func setupTestRouteWithConfig(policyService AccessPolicyServiceInterface, didService DIDWebServiceInterface, resolver AgentResolverInterface, config PermissionConfig) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(DIDAuthMiddleware(didService, DIDAuthConfig{
		Enabled:                true,
		TimestampWindowSeconds: 300,
	}))
	if resolver == nil {
		resolver = &testAgentResolver{}
	}
	router.Use(PermissionCheckMiddleware(
		policyService,
		nil, // tagVCVerifier
		resolver,
		&testDIDResolver{},
		config,
	))
	router.POST("/api/v1/execute/:target", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	return router
}

// --- Tests ---

func TestPermission_PolicyAllows(t *testing.T) {
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{
		Matched:    true,
		Allowed:    true,
		PolicyName: "allow-analytics",
	}}
	router := setupTestRoute(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/target-agent.query", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestPermission_PolicyDenies(t *testing.T) {
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{
		Matched:    true,
		Allowed:    false,
		PolicyName: "deny-delete",
		Reason:     "delete_* functions denied",
	}}
	router := setupTestRoute(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/target-agent.delete_records", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Contains(t, w.Body.String(), "access_denied")
}

func TestPermission_NoPolicyMatchAllows(t *testing.T) {
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{Matched: false}}
	router := setupTestRoute(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/open-agent.reasoner", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestPermission_NilPolicyServiceAllows(t *testing.T) {
	router := setupTestRoute(nil, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/any-agent.reasoner", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestPermission_TargetResolutionErrorDenied(t *testing.T) {
	policy := &testPolicyService{}
	router := setupTestRoute(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, &failingAgentResolver{})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/some-agent.reasoner", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Contains(t, w.Body.String(), "target_resolution_failed")
}

func TestPermission_PendingApprovalBlocked(t *testing.T) {
	resolver := &testAgentResolver{agents: map[string]*types.AgentNode{
		"pending-agent": {ID: "pending-agent", LifecycleStatus: types.AgentStatusPendingApproval},
	}}
	policy := &testPolicyService{}
	router := setupTestRoute(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, resolver)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/pending-agent.reasoner", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), "agent_pending_approval")
}

func TestPermission_AnonymousAllowedWhenNoPolicyMatches(t *testing.T) {
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{Matched: false}}
	router := setupTestRouteWithConfig(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil,
		PermissionConfig{Enabled: true})

	// Request without any caller identity — allowed when no policy matches
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/target-agent.query", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestPermission_DisabledAllowsAll(t *testing.T) {
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{Matched: true, Allowed: false}}
	router := setupTestRouteWithConfig(policy, &testDIDWebService{publicKeys: map[string]ed25519.PublicKey{}}, nil, PermissionConfig{Enabled: false})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/any-agent.reasoner", bytes.NewReader([]byte(`{"x":1}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}
