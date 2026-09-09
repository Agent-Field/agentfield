package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ginContextWithQuery builds a gin.Context whose request carries the given raw
// query string, for exercising the query-parsing helpers directly.
func ginContextWithQuery(rawQuery string) *gin.Context {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/discovery?"+rawQuery, nil)
	return c
}

func TestParseDiscoveryFiltersInvalidParams(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cases := []struct {
		name      string
		query     string
		wantParam string
		wantAllow []string
	}{
		{"invalid include_examples", "include_examples=maybe", "include_examples", []string{"true", "false"}},
		{"invalid include_descriptions", "include_descriptions=huh", "include_descriptions", []string{"true", "false"}},
		{"invalid include_input_schema", "include_input_schema=x", "include_input_schema", []string{"true", "false"}},
		{"invalid include_output_schema", "include_output_schema=x", "include_output_schema", []string{"true", "false"}},
		{"negative limit rejected", "limit=-1", "limit", []string{"0-500"}},
		{"non-numeric offset rejected", "offset=abc", "offset", []string{"0-1000000"}},
		{"invalid format rejected", "format=yaml", "format", []string{"json", "xml", "compact"}},
		{"invalid health_status rejected", "health_status=broken", "health_status", nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseDiscoveryFilters(ginContextWithQuery(tc.query))
			require.Error(t, err)
			pe, ok := err.(*parameterError)
			require.True(t, ok, "expected *parameterError, got %T", err)
			assert.Equal(t, tc.wantParam, pe.Parameter, "error should name the offending parameter")
			if tc.wantAllow != nil {
				assert.Equal(t, tc.wantAllow, pe.Allowed, "error should list allowed values")
			}
		})
	}
}

func TestParseDiscoveryFiltersHappyPath(t *testing.T) {
	gin.SetMode(gin.TestMode)

	filters, err := parseDiscoveryFilters(ginContextWithQuery(
		"format=compact&include_examples=true&include_input_schema=1&include_output_schema=yes" +
			"&include_descriptions=false&limit=50&offset=5&reasoner=sum*&skill=web_*&tags=ml,research&health_status=active",
	))
	require.NoError(t, err)

	assert.Equal(t, "compact", filters.Format)
	assert.True(t, filters.IncludeExamples)
	assert.True(t, filters.IncludeInputSchema)
	assert.True(t, filters.IncludeOutputSchema)
	assert.False(t, filters.IncludeDescriptions)
	assert.Equal(t, 50, filters.Limit)
	assert.Equal(t, 5, filters.Offset)
	require.NotNil(t, filters.ReasonerPattern)
	assert.Equal(t, "sum*", *filters.ReasonerPattern)
	require.NotNil(t, filters.SkillPattern)
	assert.Equal(t, "web_*", *filters.SkillPattern)
	assert.Equal(t, []string{"ml", "research"}, filters.Tags)
	require.NotNil(t, filters.HealthStatus)
	assert.Equal(t, types.HealthStatusActive, *filters.HealthStatus)
}

func TestParseDiscoveryFiltersDefaults(t *testing.T) {
	gin.SetMode(gin.TestMode)

	filters, err := parseDiscoveryFilters(ginContextWithQuery(""))
	require.NoError(t, err)

	assert.Equal(t, "json", filters.Format)
	assert.True(t, filters.IncludeDescriptions, "descriptions default on")
	assert.False(t, filters.IncludeExamples, "examples default off")
	assert.Equal(t, 100, filters.Limit)
	assert.Equal(t, 0, filters.Offset)
	assert.Nil(t, filters.HealthStatus)
	assert.Empty(t, filters.AgentIDs)
}

func TestCollectAgentIDsMergesAliasesTrimmedDedupedSorted(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// agent + agent_ids + node_ids mixed, with whitespace and duplicates.
	filters, err := parseDiscoveryFilters(ginContextWithQuery(
		"agent=zeta&agent_ids=%20alpha%20,beta,alpha&node_ids=beta,gamma",
	))
	require.NoError(t, err)
	// parseDiscoveryFilters dedupes + sorts the collected ids.
	assert.Equal(t, []string{"alpha", "beta", "gamma", "zeta"}, filters.AgentIDs)
}

func TestCollectAgentIDsFallsBackToNodeID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// When `agent` is absent, the single id comes from `node_id`.
	got := collectAgentIDs(ginContextWithQuery("node_id=node-7"))
	assert.Equal(t, []string{"node-7"}, got)

	// `agent` wins over `node_id` when both are present.
	got = collectAgentIDs(ginContextWithQuery("agent=picked&node_id=ignored"))
	assert.Equal(t, []string{"picked"}, got)

	// No ids at all yields an empty slice.
	assert.Empty(t, collectAgentIDs(ginContextWithQuery("")))
}

func TestBuildDiscoveryResponsePaginatesAfterFiltering(t *testing.T) {
	agents := buildDiscoveryAgents()

	// No agent filter, no health filter: both agents match. Offset past the
	// first, limit 1, so totals reflect the full filtered set (2) while the
	// returned page carries only the second agent.
	filters := DiscoveryFilters{
		Limit:  1,
		Offset: 1,
	}
	resp := buildDiscoveryResponse(agents, filters)

	assert.Equal(t, 2, resp.TotalAgents, "totals reflect the full filtered set")
	require.Len(t, resp.Capabilities, 1, "page trimmed to limit")
	assert.False(t, resp.Pagination.HasMore, "no more after the last page")
	assert.Equal(t, 1, resp.Pagination.Offset)
	assert.Equal(t, 1, resp.Pagination.Limit)
}

func TestBuildDiscoveryResponseFiltersByAgentIDAndTags(t *testing.T) {
	agents := buildDiscoveryAgents()

	tagPattern := DiscoveryFilters{
		AgentIDs:        []string{"agent-beta"},
		ReasonerPattern: optionalString("*research*"),
		Tags:            []string{"ml*"},
		Limit:           100,
	}
	resp := buildDiscoveryResponse(agents, tagPattern)

	require.Len(t, resp.Capabilities, 1)
	assert.Equal(t, "agent-beta", resp.Capabilities[0].AgentID)
	require.Len(t, resp.Capabilities[0].Reasoners, 1)
	assert.Equal(t, "deep_research", resp.Capabilities[0].Reasoners[0].ID)
}

func TestDecodeSchemaHandlesMissingAndMalformed(t *testing.T) {
	assert.Nil(t, decodeSchema(nil), "empty schema decodes to nil")
	assert.Nil(t, decodeSchema(json.RawMessage(`not json`)), "malformed schema decodes to nil, not an error")

	schema := decodeSchema(json.RawMessage(`{"type":"object"}`))
	require.NotNil(t, schema)
	assert.Equal(t, "object", schema["type"])
}

func TestExtractDescriptionHandlesBlankAndMissing(t *testing.T) {
	assert.Nil(t, extractDescription(types.AgentMetadata{}, "id"), "nil custom metadata")

	blank := types.AgentMetadata{Custom: map[string]interface{}{
		"descriptions": map[string]interface{}{"id": "  "},
	}}
	assert.Nil(t, extractDescription(blank, "id"), "blank description is ignored")

	present := types.AgentMetadata{Custom: map[string]interface{}{
		"descriptions": map[string]interface{}{"id": "real description"},
	}}
	got := extractDescription(present, "id")
	require.NotNil(t, got)
	assert.Equal(t, "real description", *got)
}

func TestExtractExamplesHandlesTypedListAndMalformed(t *testing.T) {
	// Typed []map slice.
	typed := types.AgentMetadata{Custom: map[string]interface{}{
		"examples": map[string]interface{}{
			"id": []map[string]interface{}{{"name": "a"}},
		},
	}}
	assert.Len(t, extractExamples(typed, "id"), 1)

	// []interface{} slice with a non-map element that must be skipped.
	mixed := types.AgentMetadata{Custom: map[string]interface{}{
		"examples": map[string]interface{}{
			"id": []interface{}{map[string]interface{}{"name": "a"}, "skip-me"},
		},
	}}
	assert.Len(t, extractExamples(mixed, "id"), 1)

	// Malformed examples container decodes to nil, not an error.
	assert.Nil(t, extractExamples(types.AgentMetadata{Custom: map[string]interface{}{"examples": "bad"}}, "id"))
	assert.Nil(t, extractExamples(types.AgentMetadata{}, "id"))
}

func TestMatchesPatternAndTags(t *testing.T) {
	assert.True(t, matchesPattern("anything", "*"), "star matches all")
	assert.True(t, matchesPattern("anything", ""), "empty matches all")
	assert.True(t, matchesPattern("research_deep", "*research*"))
	assert.False(t, matchesPattern("summary", "*research*"))

	assert.True(t, matchesTags([]string{"ml", "research"}, nil), "no patterns matches all")
	assert.True(t, matchesTags([]string{"ml"}, []string{"ml*"}))
	assert.False(t, matchesTags([]string{"ops"}, []string{"ml*"}))
}

func TestDedupeStrings(t *testing.T) {
	assert.Equal(t, []string{"a", "b"}, dedupeStrings([]string{"", "a", "a", "b", ""}))
	assert.Empty(t, dedupeStrings(nil))
}
