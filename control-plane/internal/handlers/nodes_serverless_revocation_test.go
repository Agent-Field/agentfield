package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegisterServerlessAgentHandler_BlocksReregisterOfRevokedAgent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/discover", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"node_id":"serverless-revoked",
			"version":"2026.04.09",
			"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"},"tags":["tag-a"]}],
			"skills":[]
		}`))
	}))
	defer discoveryServer.Close()

	store := &nodeRESTStorageStub{
		agent: &types.AgentNode{
			ID:              "serverless-revoked",
			LifecycleStatus: types.AgentStatusPendingApproval,
			ApprovedTags:    nil,
			DeploymentType:  "serverless",
		},
	}

	router := gin.New()
	router.POST("/serverless/register", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

	req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)

	var body map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	assert.Equal(t, "agent_pending_approval", body["error"])
	assert.Contains(t, body["message"], "awaiting tag approval")
	assert.Nil(t, store.registeredAgent)

	stored, err := store.GetAgent(req.Context(), "serverless-revoked")
	require.NoError(t, err)
	require.NotNil(t, stored)
	assert.Equal(t, types.AgentStatusPendingApproval, stored.LifecycleStatus)
	assert.Nil(t, stored.ApprovedTags)
}

func TestRegisterServerlessAgentHandler_PreservesApprovedTagsOnReregister(t *testing.T) {
	gin.SetMode(gin.TestMode)

	discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/discover", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		// Reasoners and skills both carry tags — some approved, some not.
		// The re-register preservation path in RegisterServerlessAgentHandler
		// must filter each reasoner's and each skill's Tags down to only
		// the approved set, which is what the assertions below verify.
		_, _ = w.Write([]byte(`{
			"node_id":"serverless-approved",
			"version":"2026.04.09",
			"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"},"tags":["tag-a","tag-b"]}],
			"skills":[{"id":"s1","input_schema":{"type":"object"},"tags":["tag-a","tag-c"]}]
		}`))
	}))
	defer discoveryServer.Close()

	store := &nodeRESTStorageStub{
		agent: &types.AgentNode{
			ID:              "serverless-approved",
			LifecycleStatus: types.AgentStatusReady,
			ApprovedTags:    []string{"tag-a"},
			DeploymentType:  "serverless",
		},
	}

	router := gin.New()
	router.POST("/serverless/register", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

	req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusCreated, resp.Code)
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, types.AgentStatusReady, store.registeredAgent.LifecycleStatus)
	assert.Equal(t, []string{"tag-a"}, store.registeredAgent.ApprovedTags)

	// Per-reasoner filtering: tag-a is approved, tag-b is not.
	require.Len(t, store.registeredAgent.Reasoners, 1)
	assert.Equal(t, []string{"tag-a"}, store.registeredAgent.Reasoners[0].ApprovedTags)

	// Per-skill filtering: tag-a is approved, tag-c is not. This assertion
	// exercises the Skills loop at nodes.go ~1473-1480, which was otherwise
	// uncovered (the previous version of this test sent skills:[]).
	require.Len(t, store.registeredAgent.Skills, 1)
	assert.Equal(t, []string{"tag-a"}, store.registeredAgent.Skills[0].ApprovedTags)

	stored, err := store.GetAgent(req.Context(), "serverless-approved")
	require.NoError(t, err)
	require.NotNil(t, stored)
	assert.Equal(t, types.AgentStatusReady, stored.LifecycleStatus)
	assert.Equal(t, []string{"tag-a"}, stored.ApprovedTags)
}

func TestRegisterServerlessAgentHandler_FirstRegistrationUnchanged(t *testing.T) {
	gin.SetMode(gin.TestMode)

	discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/discover", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"node_id":"serverless-new",
			"version":"2026.04.09",
			"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"},"tags":["tag-a"]}],
			"skills":[]
		}`))
	}))
	defer discoveryServer.Close()

	store := &nodeRESTStorageStub{}

	router := gin.New()
	router.POST("/serverless/register", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

	req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusCreated, resp.Code)
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, types.AgentStatusReady, store.registeredAgent.LifecycleStatus)
	assert.Nil(t, store.registeredAgent.ApprovedTags)
}
