package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestIdentityHandlersDIDEndpoints(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := setupTestStorage(t)
	ctx := context.Background()
	now := time.Now().UTC()
	require.NoError(t, store.StoreAgentFieldServerDID(ctx, "did:af:server", "did:root:server", []byte("encrypted-seed"), now, now))
	require.NoError(t, store.StoreAgentDID(ctx, "agent-alpha", "did:af:agent-alpha", "did:af:server", "{}", 1))
	require.NoError(t, store.StoreAgentDID(ctx, "agent-beta", "did:af:agent-beta", "did:af:server", "{}", 2))
	require.NoError(t, store.StoreComponentDID(ctx, "reasoner.summarizer", "did:af:reasoner-summarizer", "did:af:agent-alpha", "reasoner", "Summarizer", 11))
	require.NoError(t, store.StoreComponentDID(ctx, "skill.deploy", "did:af:skill-deploy", "did:af:agent-alpha", "skill", "Deploy", 12))

	handler := NewIdentityHandlers(store, nil)
	router := gin.New()
	handler.RegisterRoutes(router.Group("/api/ui/v1"))

	statsRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/dids/stats")
	require.Equal(t, http.StatusOK, statsRecorder.Code)
	var stats DIDStatsResponse
	decodeResponseBody(t, statsRecorder, &stats)
	require.Equal(t, 2, stats.TotalAgents)
	require.Equal(t, 1, stats.TotalReasoners)
	require.Equal(t, 1, stats.TotalSkills)
	require.Equal(t, 4, stats.TotalDIDs)

	searchRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/dids/search?q=summarizer&type=reasoner&limit=5&offset=0")
	require.Equal(t, http.StatusOK, searchRecorder.Code)
	var searchResponse struct {
		Results []DIDSearchResult `json:"results"`
		Total   int               `json:"total"`
	}
	decodeResponseBody(t, searchRecorder, &searchResponse)
	require.Len(t, searchResponse.Results, 1)
	require.Equal(t, 1, searchResponse.Total)
	require.Equal(t, "reasoner", searchResponse.Results[0].Type)
	require.Equal(t, "Summarizer", searchResponse.Results[0].Name)
	require.Equal(t, "11", searchResponse.Results[0].DerivationPath)

	listRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/agents?limit=10&offset=0")
	require.Equal(t, http.StatusOK, listRecorder.Code)
	var listResponse struct {
		Agents []AgentDIDResponse `json:"agents"`
		Total  int                `json:"total"`
	}
	decodeResponseBody(t, listRecorder, &listResponse)
	require.Len(t, listResponse.Agents, 2)
	require.Equal(t, 2, listResponse.Total)
	agentsByNodeID := make(map[string]AgentDIDResponse, len(listResponse.Agents))
	for _, agent := range listResponse.Agents {
		agentsByNodeID[agent.AgentNodeID] = agent
	}
	require.Contains(t, agentsByNodeID, "agent-alpha")
	require.Equal(t, 1, agentsByNodeID["agent-alpha"].ReasonerCount)
	require.Equal(t, 1, agentsByNodeID["agent-alpha"].SkillCount)
	require.Empty(t, agentsByNodeID["agent-alpha"].DIDWeb)

	detailsRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/agents/agent-alpha/details?limit=1&offset=0")
	require.Equal(t, http.StatusOK, detailsRecorder.Code)
	var detailsResponse struct {
		Agent            AgentDIDResponse `json:"agent"`
		TotalReasoners   int              `json:"total_reasoners"`
		ReasonersHasMore bool             `json:"reasoners_has_more"`
	}
	decodeResponseBody(t, detailsRecorder, &detailsResponse)
	require.Equal(t, "agent-alpha", detailsResponse.Agent.AgentNodeID)
	require.Len(t, detailsResponse.Agent.Reasoners, 1)
	require.Len(t, detailsResponse.Agent.Skills, 1)
	require.Equal(t, 1, detailsResponse.TotalReasoners)
	require.False(t, detailsResponse.ReasonersHasMore)

	notFoundRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/agents/agent-missing/details")
	require.Equal(t, http.StatusNotFound, notFoundRecorder.Code)
	var notFoundBody map[string]string
	decodeResponseBody(t, notFoundRecorder, &notFoundBody)
	require.Equal(t, "Agent not found", notFoundBody["error"])
}

func TestIdentityHandlersSearchCredentials(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := setupTestStorage(t)
	ctx := context.Background()
	require.NoError(t, store.StoreExecutionVC(ctx,
		"vc-1", "exec-1", "wf-1", "session-1",
		"did:af:agent-alpha", "did:target:one", "did:caller:one",
		"input-hash-1", "output-hash-1", "completed",
		[]byte(`{"id":"vc-1"}`), "sig-1", "", 16,
	))
	require.NoError(t, store.StoreExecutionVC(ctx,
		"vc-2", "exec-2", "wf-2", "session-2",
		"did:af:agent-beta", "did:target:two", "did:caller:two",
		"input-hash-2", "output-hash-2", "failed",
		[]byte(`{"id":"vc-2"}`), "sig-2", "", 16,
	))

	handler := NewIdentityHandlers(store, nil)
	router := gin.New()
	handler.RegisterRoutes(router.Group("/api/ui/v1"))

	verifiedRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/credentials/search?status=verified&workflow_id=wf-1&q=exec-1&start_time=bad-time&limit=10&offset=0")
	require.Equal(t, http.StatusOK, verifiedRecorder.Code)
	var verifiedResponse struct {
		Credentials []VCSearchResult `json:"credentials"`
		Total       int              `json:"total"`
		HasMore     bool             `json:"has_more"`
	}
	decodeResponseBody(t, verifiedRecorder, &verifiedResponse)
	require.Len(t, verifiedResponse.Credentials, 1)
	require.Equal(t, 1, verifiedResponse.Total)
	require.False(t, verifiedResponse.HasMore)
	require.Equal(t, "vc-1", verifiedResponse.Credentials[0].VCID)
	require.True(t, verifiedResponse.Credentials[0].Verified)

	failedRecorder := performIdentityRequest(t, router, http.MethodGet, "/api/ui/v1/identity/credentials/search?status=failed&execution_id=exec-2&limit=10&offset=0")
	require.Equal(t, http.StatusOK, failedRecorder.Code)
	var failedResponse struct {
		Credentials []VCSearchResult `json:"credentials"`
		Total       int              `json:"total"`
	}
	decodeResponseBody(t, failedRecorder, &failedResponse)
	require.Len(t, failedResponse.Credentials, 1)
	require.Equal(t, 1, failedResponse.Total)
	require.Equal(t, "vc-2", failedResponse.Credentials[0].VCID)
	require.False(t, failedResponse.Credentials[0].Verified)
	}

func performIdentityRequest(t *testing.T, router *gin.Engine, method, path string) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(method, path, nil)
	router.ServeHTTP(recorder, request)
	return recorder
}

func decodeResponseBody(t *testing.T, recorder *httptest.ResponseRecorder, target interface{}) {
	t.Helper()
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), target))
}
