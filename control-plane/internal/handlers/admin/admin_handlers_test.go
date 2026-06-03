package admin

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ============================================================================
// Mock storage for AccessPolicyService
// ============================================================================

type mockPolicyStorage struct {
	policies  []*types.AccessPolicy
	nextID    int64
	createErr error
}

func (m *mockPolicyStorage) GetAccessPolicies(_ context.Context) ([]*types.AccessPolicy, error) {
	result := make([]*types.AccessPolicy, len(m.policies))
	copy(result, m.policies)
	return result, nil
}

func (m *mockPolicyStorage) GetAccessPolicyByID(_ context.Context, id int64) (*types.AccessPolicy, error) {
	for _, p := range m.policies {
		if p.ID == id {
			return p, nil
		}
	}
	return nil, fmt.Errorf("policy %d not found", id)
}

func (m *mockPolicyStorage) CreateAccessPolicy(_ context.Context, policy *types.AccessPolicy) error {
	if m.createErr != nil {
		return m.createErr
	}
	m.nextID++
	policy.ID = m.nextID
	m.policies = append(m.policies, policy)
	return nil
}

func (m *mockPolicyStorage) UpdateAccessPolicy(_ context.Context, policy *types.AccessPolicy) error {
	for i, p := range m.policies {
		if p.ID == policy.ID {
			m.policies[i] = policy
			return nil
		}
	}
	return fmt.Errorf("policy %d not found", policy.ID)
}

func (m *mockPolicyStorage) DeleteAccessPolicy(_ context.Context, id int64) error {
	for i, p := range m.policies {
		if p.ID == id {
			m.policies = append(m.policies[:i], m.policies[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("policy %d not found", id)
}

// ============================================================================
// Mock storage for TagApprovalService
// ============================================================================

type mockTagStorage struct {
	agents   map[string]*types.AgentNode
	agentDID map[string]*types.AgentDIDInfo
}

func newMockTagStorage() *mockTagStorage {
	return &mockTagStorage{
		agents:   make(map[string]*types.AgentNode),
		agentDID: make(map[string]*types.AgentDIDInfo),
	}
}

func (m *mockTagStorage) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	a, ok := m.agents[id]
	if !ok {
		return nil, fmt.Errorf("agent %s not found", id)
	}
	return a, nil
}

func (m *mockTagStorage) RegisterAgent(_ context.Context, node *types.AgentNode) error {
	m.agents[node.ID] = node
	return nil
}

func (m *mockTagStorage) ListAgentsByLifecycleStatus(_ context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	var result []*types.AgentNode
	for _, a := range m.agents {
		if a.LifecycleStatus == status {
			result = append(result, a)
		}
	}
	return result, nil
}

func (m *mockTagStorage) GetAgentDID(_ context.Context, agentID string) (*types.AgentDIDInfo, error) {
	info, ok := m.agentDID[agentID]
	if !ok {
		return nil, fmt.Errorf("DID not found")
	}
	return info, nil
}

func (m *mockTagStorage) StoreAgentTagVC(_ context.Context, _, _, _, _, _ string, _ time.Time, _ *time.Time) error {
	return nil
}

func (m *mockTagStorage) RevokeAgentTagVC(_ context.Context, _ string) error {
	return nil
}

// ============================================================================
// Access Policy Handler Tests
// ============================================================================

func setupPolicyRouter(storage *mockPolicyStorage) (*gin.Engine, *services.AccessPolicyService) {
	svc := services.NewAccessPolicyService(storage)
	_ = svc.Initialize(context.Background())
	handlers := NewAccessPolicyHandlers(svc)

	r := gin.New()
	api := r.Group("/api/v1")
	handlers.RegisterRoutes(api)
	return r, svc
}

func TestAccessPolicyHandlers_ListPolicies_Empty(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/policies", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp types.AccessPolicyListResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, 0, resp.Total)
	assert.Empty(t, resp.Policies)
}

func TestAccessPolicyHandlers_CreatePolicy_Success(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	body := `{"name":"finance_to_billing","caller_tags":["finance"],"target_tags":["billing"],"action":"allow","priority":10}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/policies", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusCreated, w.Code)

	var policy types.AccessPolicy
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &policy))
	assert.Equal(t, "finance_to_billing", policy.Name)
	assert.True(t, policy.Enabled)
}

func TestAccessPolicyHandlers_CreatePolicy_InvalidAction(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	body := `{"name":"bad","caller_tags":["a"],"target_tags":["b"],"action":"maybe"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/policies", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_action")
}

func TestAccessPolicyHandlers_CreatePolicy_InvalidJSON(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/policies", bytes.NewBufferString("{invalid"))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_request")
}

func TestAccessPolicyHandlers_GetPolicy_Success(t *testing.T) {
	now := time.Now()
	storage := &mockPolicyStorage{
		policies: []*types.AccessPolicy{
			{ID: 1, Name: "test", CallerTags: []string{"a"}, TargetTags: []string{"b"},
				Action: "allow", Enabled: true, CreatedAt: now, UpdatedAt: now},
		},
	}
	router, _ := setupPolicyRouter(storage)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/policies/1", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var policy types.AccessPolicy
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &policy))
	assert.Equal(t, "test", policy.Name)
}

func TestAccessPolicyHandlers_GetPolicy_NotFound(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/policies/999", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestAccessPolicyHandlers_GetPolicy_InvalidID(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/policies/abc", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_id")
}

func TestAccessPolicyHandlers_DeletePolicy_Success(t *testing.T) {
	now := time.Now()
	storage := &mockPolicyStorage{
		policies: []*types.AccessPolicy{
			{ID: 1, Name: "to_delete", CallerTags: []string{"a"}, TargetTags: []string{"b"},
				Action: "allow", Enabled: true, CreatedAt: now, UpdatedAt: now},
		},
	}
	router, _ := setupPolicyRouter(storage)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/admin/policies/1", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "success")
}

func TestAccessPolicyHandlers_DeletePolicy_NotFound(t *testing.T) {
	router, _ := setupPolicyRouter(&mockPolicyStorage{})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/admin/policies/999", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "delete_failed")
}

func TestAccessPolicyHandlers_UpdatePolicy_Success(t *testing.T) {
	now := time.Now()
	storage := &mockPolicyStorage{
		policies: []*types.AccessPolicy{
			{ID: 1, Name: "original", CallerTags: []string{"a"}, TargetTags: []string{"b"},
				Action: "allow", Priority: 5, Enabled: true, CreatedAt: now, UpdatedAt: now},
		},
	}
	router, _ := setupPolicyRouter(storage)

	body := `{"name":"updated","caller_tags":["x"],"target_tags":["y"],"action":"deny","priority":20}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPut, "/api/v1/admin/policies/1", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var policy types.AccessPolicy
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &policy))
	assert.Equal(t, "updated", policy.Name)
	assert.Equal(t, "deny", policy.Action)
}

// ============================================================================
// Tag Approval Handler Tests
// ============================================================================

func setupTagApprovalRouter(storage *mockTagStorage) *gin.Engine {
	cfg := config.TagApprovalRulesConfig{DefaultMode: "manual"}
	svc := services.NewTagApprovalService(cfg, storage)
	handlers := NewTagApprovalHandlers(svc, nil)

	r := gin.New()
	api := r.Group("/api/v1")
	handlers.RegisterRoutes(api)
	return r
}

func TestTagApprovalHandlers_ListPendingAgents_Empty(t *testing.T) {
	router := setupTagApprovalRouter(newMockTagStorage())

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/agents/pending", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, float64(0), resp["total"])
}

func TestTagApprovalHandlers_ListPendingAgents_ReturnsPending(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["pending-1"] = &types.AgentNode{
		ID:              "pending-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"finance"},
		RegisteredAt:    time.Now(),
	}
	storage.agents["ready-1"] = &types.AgentNode{
		ID:              "ready-1",
		LifecycleStatus: types.AgentStatusReady,
	}

	router := setupTagApprovalRouter(storage)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/agents/pending", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, float64(1), resp["total"])
}

func TestTagApprovalHandlers_ApproveAgentTags_Success(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"finance"},
	}

	router := setupTagApprovalRouter(storage)

	body := `{"approved_tags":["finance"]}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/approve-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "success")

	// Verify the agent was updated
	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusStarting, agent.LifecycleStatus)
	assert.Equal(t, []string{"finance"}, agent.ApprovedTags)
}

func TestTagApprovalHandlers_ApproveAgentTags_InvalidJSON(t *testing.T) {
	storage := newMockTagStorage()
	router := setupTagApprovalRouter(storage)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/approve-tags", bytes.NewBufferString("{bad"))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestTagApprovalHandlers_ApproveAgentTags_NonPendingReturns409(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
	}

	router := setupTagApprovalRouter(storage)

	body := `{"approved_tags":["finance"]}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/approve-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusConflict, w.Code)
	assert.Contains(t, w.Body.String(), "not_pending_approval")
}

func TestTagApprovalHandlers_RejectAgentTags_NonPendingReturns409(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
	}

	router := setupTagApprovalRouter(storage)

	body := `{"reason":"revoke access"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/reject-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusConflict, w.Code)
	assert.Contains(t, w.Body.String(), "not_pending_approval")
}

func TestTagApprovalHandlers_ApproveAgentTags_PerSkill(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"payment"}},
		},
	}

	router := setupTagApprovalRouter(storage)

	body := `{"approved_tags":[],"skill_tags":{"s1":["payment"]}}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/approve-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusStarting, agent.LifecycleStatus)
	assert.Equal(t, []string{"payment"}, agent.Skills[0].ApprovedTags)
}

func TestTagApprovalHandlers_RejectAgentTags_Success(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"root"},
	}

	router := setupTagApprovalRouter(storage)

	body := `{"reason":"Forbidden tag"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/reject-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "success")

	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusOffline, agent.LifecycleStatus)
}

func TestTagApprovalHandlers_RejectAgentTags_EmptyBody(t *testing.T) {
	// Rejection should work even without a body (reason is optional)
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
	}

	router := setupTagApprovalRouter(storage)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/reject-tags", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestTagApprovalHandlers_RejectAgentTags_NotFoundFails(t *testing.T) {
	router := setupTagApprovalRouter(newMockTagStorage())

	body := `{"reason":"test"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/nonexistent/reject-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "rejection_failed")
}

func TestTagApprovalHandlers_RevokeAgentTags_ReadyAgent(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
		ApprovedTags:    []string{"finance", "billing"},
	}

	router := setupTagApprovalRouter(storage)

	body := `{"reason":"security review"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/revoke-tags", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "success")

	// Verify agent was transitioned to pending_approval with cleared tags
	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
	assert.Nil(t, agent.ApprovedTags)
}

func TestTagApprovalHandlers_RevokeAgentTags_EmptyBody(t *testing.T) {
	storage := newMockTagStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
		ApprovedTags:    []string{"finance"},
	}

	router := setupTagApprovalRouter(storage)

	// Revocation without reason should work
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-1/revoke-tags", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestTagApprovalHandlers_RevokeAgentTags_NotFoundFails(t *testing.T) {
	router := setupTagApprovalRouter(newMockTagStorage())

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/nonexistent/revoke-tags", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "revocation_failed")
}

// Trigger plugin system stubs — interface fillers for the test mock; not exercised.
func (m *mockTagStorage) CreateTrigger(context.Context, *types.Trigger) error { return nil }
func (m *mockTagStorage) GetTrigger(context.Context, string) (*types.Trigger, error) { return nil, nil }
func (m *mockTagStorage) ListTriggers(context.Context, string, string) ([]*types.Trigger, error) { return nil, nil }
func (m *mockTagStorage) UpdateTrigger(context.Context, *types.Trigger) error { return nil }
func (m *mockTagStorage) DeleteTrigger(context.Context, string) error { return nil }
func (m *mockTagStorage) UpsertCodeManagedTrigger(context.Context, *types.Trigger) (string, error) { return "", nil }
func (m *mockTagStorage) MarkOrphanedTriggers(context.Context, string, []string) error { return nil }
func (m *mockTagStorage) SetTriggerOverride(context.Context, string, bool, bool) error { return nil }
func (m *mockTagStorage) ConvertTriggerToUIManaged(context.Context, string) error { return nil }
func (m *mockTagStorage) InsertInboundEvent(context.Context, *types.InboundEvent) error { return nil }
func (m *mockTagStorage) InboundEventExistsByIdempotency(context.Context, string, string) (bool, error) { return false, nil }
func (m *mockTagStorage) GetInboundEvent(context.Context, string) (*types.InboundEvent, error) { return nil, nil }
func (m *mockTagStorage) ListInboundEvents(context.Context, string, int) ([]*types.InboundEvent, error) { return nil, nil }
func (m *mockTagStorage) MarkInboundEventProcessed(context.Context, string, string, string, string) error { return nil }
func (m *mockTagStorage) SetInboundEventDispatchedWorkflow(context.Context, string, string) error { return nil }
func (m *mockTagStorage) GetInboundEventByWorkflowID(context.Context, string) (*types.InboundEvent, error) { return nil, nil }
func (m *mockTagStorage) TriggerMetrics(context.Context) (*types.TriggerMetrics, error) { return &types.TriggerMetrics{}, nil }
