package admin

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type failingPolicyStorage struct {
	mockPolicyStorage
	getPoliciesErr error
	getPolicyErr   error
	updateErr      error
	deleteErr      error
}

func (m *failingPolicyStorage) GetAccessPolicies(_ context.Context) ([]*types.AccessPolicy, error) {
	if m.getPoliciesErr != nil {
		return nil, m.getPoliciesErr
	}
	return m.mockPolicyStorage.GetAccessPolicies(context.Background())
}

func (m *failingPolicyStorage) GetAccessPolicyByID(_ context.Context, id int64) (*types.AccessPolicy, error) {
	if m.getPolicyErr != nil {
		return nil, m.getPolicyErr
	}
	return m.mockPolicyStorage.GetAccessPolicyByID(context.Background(), id)
}

func (m *failingPolicyStorage) UpdateAccessPolicy(_ context.Context, policy *types.AccessPolicy) error {
	if m.updateErr != nil {
		return m.updateErr
	}
	return m.mockPolicyStorage.UpdateAccessPolicy(context.Background(), policy)
}

func (m *failingPolicyStorage) DeleteAccessPolicy(_ context.Context, id int64) error {
	if m.deleteErr != nil {
		return m.deleteErr
	}
	return m.mockPolicyStorage.DeleteAccessPolicy(context.Background(), id)
}

type adminStorageMock struct {
	storage.StorageProvider
	agents             map[string]*types.AgentNode
	listAgentsErr      error
	registerAgentErr   error
	getAgentErr        error
	revokeAgentTagErr  error
	pendingAgentsErr   error
}

func newAdminStorageMock() *adminStorageMock {
	return &adminStorageMock{
		agents: make(map[string]*types.AgentNode),
	}
}

func (m *adminStorageMock) ListAgents(_ context.Context, _ types.AgentFilters) ([]*types.AgentNode, error) {
	if m.listAgentsErr != nil {
		return nil, m.listAgentsErr
	}
	result := make([]*types.AgentNode, 0, len(m.agents))
	for _, agent := range m.agents {
		result = append(result, agent)
	}
	return result, nil
}

func (m *adminStorageMock) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	if m.getAgentErr != nil {
		return nil, m.getAgentErr
	}
	agent, ok := m.agents[id]
	if !ok {
		return nil, errors.New("agent not found")
	}
	return agent, nil
}

func (m *adminStorageMock) RegisterAgent(_ context.Context, agent *types.AgentNode) error {
	if m.registerAgentErr != nil {
		return m.registerAgentErr
	}
	m.agents[agent.ID] = agent
	return nil
}

func (m *adminStorageMock) ListAgentsByLifecycleStatus(_ context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	if m.pendingAgentsErr != nil {
		return nil, m.pendingAgentsErr
	}
	var result []*types.AgentNode
	for _, agent := range m.agents {
		if agent.LifecycleStatus == status {
			result = append(result, agent)
		}
	}
	return result, nil
}

func (m *adminStorageMock) RevokeAgentTagVC(_ context.Context, _ string) error {
	return m.revokeAgentTagErr
}

func setupPolicyRouterWithStorage(storage services.AccessPolicyStorage) *gin.Engine {
	svc := services.NewAccessPolicyService(storage)
	_ = svc.Initialize(context.Background())
	handlers := NewAccessPolicyHandlers(svc)

	r := gin.New()
	api := r.Group("/api/v1")
	handlers.RegisterRoutes(api)
	return r
}

func setupTagApprovalRouterWithStorage(storage *adminStorageMock) *gin.Engine {
	cfg := config.TagApprovalRulesConfig{DefaultMode: "manual"}
	svc := services.NewTagApprovalService(cfg, storage)
	handlers := NewTagApprovalHandlers(svc, storage)

	r := gin.New()
	api := r.Group("/api/v1")
	handlers.RegisterRoutes(api)
	return r
}

func TestAccessPolicyHandlers_ErrorBranches(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name         string
		method       string
		path         string
		body         string
		storage      services.AccessPolicyStorage
		wantCode     int
		wantContains string
	}{
		{
			name: "list policies failure",
			method: http.MethodGet,
			path: "/api/v1/admin/policies",
			storage: &failingPolicyStorage{
				getPoliciesErr: errors.New("boom"),
			},
			wantCode:     http.StatusInternalServerError,
			wantContains: "list_failed",
		},
		{
			name: "create policy storage failure",
			method: http.MethodPost,
			path: "/api/v1/admin/policies",
			body: `{"name":"p1","caller_tags":["a"],"target_tags":["b"],"action":"allow"}`,
			storage: &failingPolicyStorage{
				mockPolicyStorage: mockPolicyStorage{
					createErr: errors.New("create failed"),
				},
			},
			wantCode:     http.StatusInternalServerError,
			wantContains: "create_failed",
		},
		{
			name: "update policy invalid id",
			method: http.MethodPut,
			path: "/api/v1/admin/policies/not-a-number",
			body: `{"name":"updated","caller_tags":["a"],"target_tags":["b"],"action":"allow"}`,
			storage: &failingPolicyStorage{},
			wantCode:     http.StatusBadRequest,
			wantContains: "invalid_id",
		},
		{
			name: "update policy invalid json",
			method: http.MethodPut,
			path: "/api/v1/admin/policies/1",
			body: `{"name":`,
			storage: &failingPolicyStorage{},
			wantCode:     http.StatusBadRequest,
			wantContains: "invalid_request",
		},
		{
			name: "update policy invalid action",
			method: http.MethodPut,
			path: "/api/v1/admin/policies/1",
			body: `{"name":"updated","caller_tags":["a"],"target_tags":["b"],"action":"nope"}`,
			storage: &failingPolicyStorage{},
			wantCode:     http.StatusBadRequest,
			wantContains: "invalid_action",
		},
		{
			name: "update policy get by id failure",
			method: http.MethodPut,
			path: "/api/v1/admin/policies/1",
			body: `{"name":"updated","caller_tags":["a"],"target_tags":["b"],"action":"allow"}`,
			storage: &failingPolicyStorage{
				getPolicyErr: errors.New("missing"),
			},
			wantCode:     http.StatusInternalServerError,
			wantContains: "update_failed",
		},
		{
			name: "delete policy invalid id",
			method: http.MethodDelete,
			path: "/api/v1/admin/policies/not-a-number",
			storage: &failingPolicyStorage{},
			wantCode:     http.StatusBadRequest,
			wantContains: "invalid_id",
		},
		{
			name: "delete policy cache reload failure",
			method: http.MethodDelete,
			path: "/api/v1/admin/policies/1",
			storage: &failingPolicyStorage{
				mockPolicyStorage: mockPolicyStorage{
					policies: []*types.AccessPolicy{{
						ID:        1,
						Name:      "policy",
						CallerTags: []string{"a"},
						TargetTags: []string{"b"},
						Action:    "allow",
						Enabled:   true,
						CreatedAt: now,
						UpdatedAt: now,
					}},
				},
				getPoliciesErr: errors.New("reload failed"),
			},
			wantCode:     http.StatusInternalServerError,
			wantContains: "delete_failed",
		},
		{
			name: "update policy cache reload failure after storage update",
			method: http.MethodPut,
			path: "/api/v1/admin/policies/1",
			body: `{"name":"updated","caller_tags":["a"],"target_tags":["b"],"action":"allow"}`,
			storage: &failingPolicyStorage{
				mockPolicyStorage: mockPolicyStorage{
					policies: []*types.AccessPolicy{{
						ID:         1,
						Name:       "policy",
						CallerTags: []string{"a"},
						TargetTags: []string{"b"},
						Action:     "allow",
						Enabled:    true,
						CreatedAt:  now,
						UpdatedAt:  now,
					}},
				},
				getPoliciesErr: errors.New("reload failed"),
			},
			wantCode:     http.StatusInternalServerError,
			wantContains: "update_failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := setupPolicyRouterWithStorage(tt.storage)

			var body *bytes.Buffer
			if tt.body != "" {
				body = bytes.NewBufferString(tt.body)
			} else {
				body = bytes.NewBuffer(nil)
			}

			w := httptest.NewRecorder()
			req := httptest.NewRequest(tt.method, tt.path, body)
			if tt.body != "" {
				req.Header.Set("Content-Type", "application/json")
			}
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.wantCode, w.Code)
			assert.Contains(t, w.Body.String(), tt.wantContains)
		})
	}
}

func TestTagApprovalHandlers_ListApprovedAgents(t *testing.T) {
	now := time.Date(2026, 4, 8, 10, 11, 12, 0, time.UTC)
	store := newAdminStorageMock()
	store.agents["pending"] = &types.AgentNode{
		ID:              "pending",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ApprovedTags:    []string{"ignored"},
		RegisteredAt:    now,
	}
	store.agents["untagged"] = &types.AgentNode{
		ID:              "untagged",
		LifecycleStatus: types.AgentStatusReady,
		RegisteredAt:    now,
	}
	store.agents["approved"] = &types.AgentNode{
		ID:              "approved",
		LifecycleStatus: types.AgentStatusReady,
		ProposedTags:    []string{"finance"},
		ApprovedTags:    []string{"finance", "billing"},
		RegisteredAt:    now,
	}

	router := setupTagApprovalRouterWithStorage(store)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/agents/approved", nil)
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	var resp struct {
		Agents []types.PendingAgentResponse `json:"agents"`
		Total  int                          `json:"total"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Agents, 1)
	assert.Equal(t, 1, resp.Total)
	assert.Equal(t, "approved", resp.Agents[0].AgentID)
	assert.Equal(t, []string{"finance", "billing"}, resp.Agents[0].ApprovedTags)
	assert.Equal(t, "2026-04-08T10:11:12Z", resp.Agents[0].RegisteredAt)
}

func TestTagApprovalHandlers_ListApprovedAgents_ListError(t *testing.T) {
	store := newAdminStorageMock()
	store.listAgentsErr = errors.New("list failed")

	router := setupTagApprovalRouterWithStorage(store)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/agents/approved", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "list_failed")
}

func TestTagApprovalHandlers_ListKnownTags(t *testing.T) {
	store := newAdminStorageMock()
	store.agents["agent-1"] = &types.AgentNode{
		ID:           "agent-1",
		ProposedTags: []string{"billing", "finance"},
		ApprovedTags: []string{"finance", "ops"},
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"search"}, ProposedTags: []string{"reasoner-tag"}},
		},
		Skills: []types.SkillDefinition{
			{Tags: []string{"summarize"}, ProposedTags: []string{"skill-tag", "billing"}},
		},
	}

	router := setupTagApprovalRouterWithStorage(store)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/tags", nil)
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	var resp struct {
		Tags  []string `json:"tags"`
		Total int      `json:"total"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, []string{"billing", "finance", "ops", "reasoner-tag", "search", "skill-tag", "summarize"}, resp.Tags)
	assert.Equal(t, len(resp.Tags), resp.Total)
}

func TestTagApprovalHandlers_ListKnownTags_ListError(t *testing.T) {
	store := newAdminStorageMock()
	store.listAgentsErr = errors.New("list failed")

	router := setupTagApprovalRouterWithStorage(store)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/tags", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "list_failed")
}

func TestTagApprovalHandlers_ApproveAndRevokeValidationBranches(t *testing.T) {
	store := newAdminStorageMock()
	store.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
	}

	t.Run("approve missing agent id", func(t *testing.T) {
		handler := NewTagApprovalHandlers(services.NewTagApprovalService(config.TagApprovalRulesConfig{DefaultMode: "manual"}, store), store)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/approve-tags", bytes.NewBufferString(`{"approved_tags":["finance"]}`))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.ApproveAgentTags(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "missing_agent_id")
	})

	t.Run("approve not found returns approval failed", func(t *testing.T) {
		router := setupTagApprovalRouterWithStorage(newAdminStorageMock())

		w := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/missing/approve-tags", bytes.NewBufferString(`{"approved_tags":["finance"]}`))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusInternalServerError, w.Code)
		assert.Contains(t, w.Body.String(), "approval_failed")
	})

	t.Run("revoke missing agent id", func(t *testing.T) {
		handler := NewTagApprovalHandlers(services.NewTagApprovalService(config.TagApprovalRulesConfig{DefaultMode: "manual"}, store), store)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/revoke-tags", bytes.NewBufferString(`{"reason":"test"}`))
		c.Request.Header.Set("Content-Type", "application/json")

		handler.RevokeAgentTags(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "missing_agent_id")
	})

	t.Run("revoke already revoked returns conflict", func(t *testing.T) {
		alreadyRevoked := newAdminStorageMock()
		alreadyRevoked.agents["agent-2"] = &types.AgentNode{
			ID:              "agent-2",
			LifecycleStatus: types.AgentStatusPendingApproval,
		}
		router := setupTagApprovalRouterWithStorage(alreadyRevoked)

		w := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/agents/agent-2/revoke-tags", bytes.NewBufferString(`{"reason":"repeat"}`))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusConflict, w.Code)
		assert.Contains(t, w.Body.String(), "already_revoked")
	})
}
