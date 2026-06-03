package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubAgentLister struct {
	agents []*types.AgentNode
	err    error
	calls  int
}

func (s *stubAgentLister) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return s.agents, nil
}

func TestDiscoveryCapabilities_Defaults(t *testing.T) {
	gin.SetMode(gin.TestMode)
	InvalidateDiscoveryCache()

	lister := &stubAgentLister{agents: buildDiscoveryAgents()}
	router := gin.New()
	router.GET("/api/v1/discovery/capabilities", DiscoveryCapabilitiesHandler(lister))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp DiscoveryResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, 2, resp.TotalAgents)
	assert.Equal(t, 2, resp.TotalReasoners)
	assert.Equal(t, 2, resp.TotalSkills)
	assert.Equal(t, 100, resp.Pagination.Limit)
	assert.Equal(t, 0, resp.Pagination.Offset)
	assert.False(t, resp.Pagination.HasMore)

	if assert.Len(t, resp.Capabilities, 2) {
		assert.Equal(t, "agent-alpha", resp.Capabilities[0].AgentID)
		assert.NotEmpty(t, resp.Capabilities[0].Reasoners[0].InvocationTarget)
		assert.NotNil(t, resp.Capabilities[0].Reasoners[0].Description)
		assert.Nil(t, resp.Capabilities[0].Reasoners[0].InputSchema)
	}
}

func TestDiscoveryCapabilities_WithFiltersAndSchemas(t *testing.T) {
	gin.SetMode(gin.TestMode)
	InvalidateDiscoveryCache()

	lister := &stubAgentLister{agents: buildDiscoveryAgents()}
	router := gin.New()
	router.GET("/api/v1/discovery/capabilities", DiscoveryCapabilitiesHandler(lister))

	url := "/api/v1/discovery/capabilities?node_id=agent-beta&reasoner=*research*&tags=ml*&include_input_schema=true&include_output_schema=true&include_examples=true&include_descriptions=false&health_status=active&limit=1&offset=0"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp DiscoveryResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, 1, resp.TotalAgents)
	assert.Equal(t, 1, resp.TotalReasoners)
	assert.Equal(t, 0, resp.TotalSkills)
	assert.Equal(t, 1, len(resp.Capabilities))

	reasoners := resp.Capabilities[0].Reasoners
	if assert.Len(t, reasoners, 1) {
		assert.Equal(t, "deep_research", reasoners[0].ID)
		assert.Nil(t, reasoners[0].Description)
		assert.NotNil(t, reasoners[0].InputSchema)
		assert.NotNil(t, reasoners[0].OutputSchema)
		assert.NotNil(t, reasoners[0].Examples)
		assert.Equal(t, "agent-beta:deep_research", reasoners[0].InvocationTarget)
	}

	assert.Empty(t, resp.Capabilities[0].Skills)
}

func TestDiscoveryCapabilities_Formats(t *testing.T) {
	gin.SetMode(gin.TestMode)
	InvalidateDiscoveryCache()

	lister := &stubAgentLister{agents: buildDiscoveryAgents()}
	router := gin.New()
	router.GET("/api/v1/discovery/capabilities", DiscoveryCapabilitiesHandler(lister))

	xmlReq := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities?format=xml", nil)
	xmlRec := httptest.NewRecorder()
	router.ServeHTTP(xmlRec, xmlReq)

	require.Equal(t, http.StatusOK, xmlRec.Code)
	assert.Contains(t, xmlRec.Body.String(), "<discovery")
	assert.Contains(t, xmlRec.Body.String(), `id="deep_research"`)

	compactReq := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities?format=compact&skill=web_*", nil)
	compactRec := httptest.NewRecorder()
	router.ServeHTTP(compactRec, compactReq)

	require.Equal(t, http.StatusOK, compactRec.Code)
	var compactResp CompactDiscoveryResponse
	require.NoError(t, json.Unmarshal(compactRec.Body.Bytes(), &compactResp))
	assert.Equal(t, 2, len(compactResp.Reasoners)) // skill filter does not affect reasoners
	assert.Equal(t, 1, len(compactResp.Skills))
}

func TestDiscoveryCapabilities_ValidationAndCaching(t *testing.T) {
	gin.SetMode(gin.TestMode)
	InvalidateDiscoveryCache()

	lister := &stubAgentLister{agents: buildDiscoveryAgents()}
	router := gin.New()
	router.GET("/api/v1/discovery/capabilities", DiscoveryCapabilitiesHandler(lister))

	badReq := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities?format=yaml", nil)
	badRec := httptest.NewRecorder()
	router.ServeHTTP(badRec, badReq)

	require.Equal(t, http.StatusBadRequest, badRec.Code)
	var badResp map[string]interface{}
	require.NoError(t, json.Unmarshal(badRec.Body.Bytes(), &badResp))
	assert.Equal(t, "invalid_parameter", badResp["error"])

	first := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities", nil)
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, first)

	second := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities", nil)
	secondRec := httptest.NewRecorder()
	router.ServeHTTP(secondRec, second)

	assert.Equal(t, 1, lister.calls, "expected cached agents to be reused")
}

func buildDiscoveryAgents() []*types.AgentNode {
	now := time.Now().UTC()

	deepInput := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)
	deepOutput := json.RawMessage(`{"type":"object","properties":{"findings":{"type":"array"}}}`)
	webInput := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"num_results":{"type":"integer","default":10}},"required":["query"]}`)

	return []*types.AgentNode{
		{
			ID:             "agent-alpha",
			BaseURL:        "http://alpha:8080",
			Version:        "1.0.0",
			DeploymentType: "long_running",
			HealthStatus:   types.HealthStatusActive,
			LastHeartbeat:  now,
			Reasoners: []types.ReasonerDefinition{
				{
					ID:           "summarize",
					InputSchema:  deepInput,
					OutputSchema: deepOutput,
					Tags:         []string{"summarization"},
				},
			},
			Skills: []types.SkillDefinition{
				{
					ID:          "math",
					InputSchema: webInput,
					Tags:        []string{"math", "utility"},
				},
			},
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{
					"descriptions": map[string]interface{}{
						"summarize": "Summarize content quickly",
						"math":      "Math helper",
					},
				},
			},
		},
		{
			ID:             "agent-beta",
			BaseURL:        "http://beta:8080",
			Version:        "2.0.0",
			DeploymentType: "long_running",
			HealthStatus:   types.HealthStatusActive,
			LastHeartbeat:  now,
			Reasoners: []types.ReasonerDefinition{
				{
					ID:           "deep_research",
					InputSchema:  deepInput,
					OutputSchema: deepOutput,
					Tags:         []string{"ml", "research"},
				},
			},
			Skills: []types.SkillDefinition{
				{
					ID:          "web_search",
					InputSchema: webInput,
					Tags:        []string{"web", "search"},
				},
			},
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{
					"descriptions": map[string]interface{}{
						"deep_research": "Performs comprehensive research",
						"web_search":    "Search the web using multiple engines",
					},
					"examples": map[string]interface{}{
						"deep_research": []interface{}{
							map[string]interface{}{
								"name":        "basic",
								"description": "basic example",
								"input": map[string]interface{}{
									"query": "what is AgentField?",
								},
							},
						},
					},
				},
			},
		},
	}
}
