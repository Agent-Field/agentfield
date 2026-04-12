package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoverJSON(t *testing.T) {
	body := `{
		"discovered_at": "2025-01-01T00:00:00Z",
		"total_agents": 1,
		"total_reasoners": 1,
		"total_skills": 0,
		"pagination": {"limit": 50, "offset": 0, "has_more": false},
		"capabilities": [{
			"agent_id": "agent-1",
			"base_url": "http://agent",
			"version": "1.0.0",
			"health_status": "active",
			"deployment_type": "long_running",
			"last_heartbeat": "2025-01-01T00:00:00Z",
			"reasoners": [{"id": "deep_research", "invocation_target": "agent-1:deep_research"}],
			"skills": []
		}]
	}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query()
		assert.Equal(t, "agent-1", query.Get("agent"))
		assert.Equal(t, "true", query.Get("include_input_schema"))
		assert.Equal(t, "json", query.Get("format"))
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
	})
	require.NoError(t, err)

	result, err := a.Discover(context.Background(), WithAgent("agent-1"), WithDiscoveryInputSchema(true), WithLimit(50))
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.JSON)

	assert.Equal(t, "json", result.Format)
	assert.Equal(t, 1, result.JSON.TotalAgents)
	assert.Equal(t, "agent-1", result.JSON.Capabilities[0].AgentID)
	assert.Equal(t, "agent-1:deep_research", result.JSON.Capabilities[0].Reasoners[0].InvocationTarget)
	assert.Equal(t, body, result.Raw)
}

func TestDiscoverCompactAndXML(t *testing.T) {
	compact := `{"discovered_at":"2025-01-01T00:00:00Z","reasoners":[{"id":"r1","agent_id":"a1","target":"a1:r1"}],"skills":[]}`
	xml := "<discovery><summary total_agents=\"0\" /></discovery>"

	compactServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "compact", r.URL.Query().Get("format"))
		fmt.Fprint(w, compact)
	}))
	defer compactServer.Close()

	xmlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "xml", r.URL.Query().Get("format"))
		w.Header().Set("Content-Type", "application/xml")
		fmt.Fprint(w, xml)
	}))
	defer xmlServer.Close()

	aCompact, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: compactServer.URL,
	})
	require.NoError(t, err)

	compactResult, err := aCompact.Discover(context.Background(), WithFormat("compact"))
	require.NoError(t, err)
	require.NotNil(t, compactResult.Compact)
	assert.Equal(t, "compact", compactResult.Format)
	assert.Equal(t, "a1", compactResult.Compact.Reasoners[0].AgentID)

	aXML, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: xmlServer.URL,
	})
	require.NoError(t, err)

	xmlResult, err := aXML.Discover(context.Background(), WithFormat("xml"), WithTags([]string{"ml"}))
	require.NoError(t, err)
	assert.Equal(t, "xml", xmlResult.Format)
	assert.Equal(t, xml, xmlResult.XML)
	assert.Nil(t, xmlResult.JSON)
	assert.Nil(t, xmlResult.Compact)
}

func TestDiscoverRejectsInvalidFormat(t *testing.T) {
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "http://localhost:8080",
	})
	require.NoError(t, err)

	_, err = a.Discover(context.Background(), WithFormat("yaml"))
	require.Error(t, err)
}

func TestDedupeHelper(t *testing.T) {
	values := []string{"a", "b", "a", "", "c"}
	assert.Equal(t, []string{"a", "b", "c"}, dedupe(values))
}

func TestDiscoverSupportsExtendedFilters(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query()
		assert.Equal(t, "agent-1,agent-2", query.Get("agent_ids"))
		assert.Equal(t, "agent-*", query.Get("reasoner"))
		assert.Equal(t, "skill-*", query.Get("skill"))
		assert.Equal(t, "tag-a,tag-b", query.Get("tags"))
		assert.Equal(t, "true", query.Get("include_output_schema"))
		assert.Equal(t, "true", query.Get("include_descriptions"))
		assert.Equal(t, "true", query.Get("include_examples"))
		assert.Equal(t, "active", query.Get("health_status"))
		assert.Equal(t, "10", query.Get("limit"))
		assert.Equal(t, "20", query.Get("offset"))
		assert.Equal(t, "Bearer token-123", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Accept"))
		fmt.Fprint(w, `{"discovered_at":"2025-01-01T00:00:00Z","total_agents":0,"total_reasoners":0,"total_skills":0,"pagination":{"limit":10,"offset":20,"has_more":false},"capabilities":[]}`)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Token:         "token-123",
	})
	require.NoError(t, err)

	result, err := a.Discover(
		context.Background(),
		WithNodeID(""),
		WithAgentIDs([]string{"agent-1", "agent-2", "agent-1"}),
		WithNodeIDs(nil),
		WithReasonerPattern("agent-*"),
		WithSkillPattern("skill-*"),
		WithTags([]string{"tag-a", "tag-b", "tag-a"}),
		WithDiscoveryOutputSchema(true),
		WithDiscoveryDescriptions(true),
		WithDiscoveryExamples(true),
		WithHealthStatus("ACTIVE"),
		WithLimit(10),
		WithOffset(20),
	)
	require.NoError(t, err)
	assert.Equal(t, "json", result.Format)
	assert.NotNil(t, result.JSON)
}
