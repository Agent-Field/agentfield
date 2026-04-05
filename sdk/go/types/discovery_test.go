package types

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoveryPaginationJSONRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		original DiscoveryPagination
	}{
		{
			name: "with more results",
			original: DiscoveryPagination{
				Limit:   20,
				Offset:  40,
				HasMore: true,
			},
		},
		{
			name: "last page",
			original: DiscoveryPagination{
				Limit:   20,
				Offset:  80,
				HasMore: false,
			},
		},
		{
			name: "zero values",
			original: DiscoveryPagination{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.original)
			require.NoError(t, err)

			var decoded DiscoveryPagination
			err = json.Unmarshal(data, &decoded)
			require.NoError(t, err)

			assert.Equal(t, tt.original, decoded)
		})
	}
}

func TestDiscoveryResponseJSONRoundTrip(t *testing.T) {
	desc := "A test reasoner"
	now := time.Now().UTC().Truncate(time.Second)

	original := DiscoveryResponse{
		DiscoveredAt:   now,
		TotalAgents:    2,
		TotalReasoners: 3,
		TotalSkills:    1,
		Pagination: DiscoveryPagination{
			Limit:   10,
			Offset:  0,
			HasMore: false,
		},
		Capabilities: []AgentCapability{
			{
				AgentID:        "agent-1",
				BaseURL:        "http://localhost:8001",
				Version:        "1.0.0",
				HealthStatus:   "healthy",
				DeploymentType: "k8s",
				LastHeartbeat:  now,
				Reasoners: []ReasonerCapability{
					{
						ID:               "r1",
						Description:      &desc,
						Tags:             []string{"tag-a"},
						InputSchema:      map[string]interface{}{"type": "object"},
						OutputSchema:     map[string]interface{}{"type": "object"},
						InvocationTarget: "http://localhost:8001/reasoners/r1",
					},
				},
				Skills: []SkillCapability{
					{
						ID:               "s1",
						Tags:             []string{"skill-tag"},
						InputSchema:      map[string]interface{}{"type": "object"},
						InvocationTarget: "http://localhost:8001/skills/s1",
					},
				},
			},
		},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded DiscoveryResponse
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.TotalAgents, decoded.TotalAgents)
	assert.Equal(t, original.TotalReasoners, decoded.TotalReasoners)
	assert.Equal(t, original.TotalSkills, decoded.TotalSkills)
	assert.Equal(t, original.Pagination, decoded.Pagination)
	require.Len(t, decoded.Capabilities, 1)

	cap := decoded.Capabilities[0]
	assert.Equal(t, "agent-1", cap.AgentID)
	assert.Equal(t, "http://localhost:8001", cap.BaseURL)
	assert.Equal(t, "1.0.0", cap.Version)
	assert.Equal(t, "healthy", cap.HealthStatus)
	assert.Equal(t, "k8s", cap.DeploymentType)

	require.Len(t, cap.Reasoners, 1)
	r := cap.Reasoners[0]
	assert.Equal(t, "r1", r.ID)
	require.NotNil(t, r.Description)
	assert.Equal(t, desc, *r.Description)
	assert.Equal(t, []string{"tag-a"}, r.Tags)
	assert.Equal(t, "http://localhost:8001/reasoners/r1", r.InvocationTarget)

	require.Len(t, cap.Skills, 1)
	s := cap.Skills[0]
	assert.Equal(t, "s1", s.ID)
	assert.Equal(t, []string{"skill-tag"}, s.Tags)
	assert.Equal(t, "http://localhost:8001/skills/s1", s.InvocationTarget)
}

func TestAgentCapabilityOmitEmptyFields(t *testing.T) {
	cap := AgentCapability{
		AgentID: "agent-1",
		BaseURL: "http://localhost:8001",
		Version: "1.0.0",
		// No Reasoners, no Skills — omitempty fields
	}

	data, err := json.Marshal(cap)
	require.NoError(t, err)

	// Unmarshal back and check slices
	var decoded AgentCapability
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, cap.AgentID, decoded.AgentID)
}

func TestReasonerCapabilityOptionalDescription(t *testing.T) {
	// With nil description — should be omitted from JSON
	r := ReasonerCapability{
		ID:               "r1",
		InvocationTarget: "http://localhost/r1",
	}

	data, err := json.Marshal(r)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasDesc := m["description"]
	assert.False(t, hasDesc, "description should be omitted when nil")

	// With non-nil description
	desc := "my reasoner"
	r2 := ReasonerCapability{
		ID:               "r2",
		Description:      &desc,
		InvocationTarget: "http://localhost/r2",
	}

	data2, err := json.Marshal(r2)
	require.NoError(t, err)

	var decoded ReasonerCapability
	require.NoError(t, json.Unmarshal(data2, &decoded))
	require.NotNil(t, decoded.Description)
	assert.Equal(t, desc, *decoded.Description)
}

func TestSkillCapabilityOptionalDescription(t *testing.T) {
	// With nil description — should be omitted
	s := SkillCapability{
		ID:               "s1",
		InvocationTarget: "http://localhost/s1",
	}

	data, err := json.Marshal(s)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasDesc := m["description"]
	assert.False(t, hasDesc, "description should be omitted when nil")
}

func TestCompactDiscoveryResponseJSONRoundTrip(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)

	original := CompactDiscoveryResponse{
		DiscoveredAt: now,
		Reasoners: []CompactCapability{
			{
				ID:      "r1",
				AgentID: "agent-1",
				Target:  "http://localhost:8001/reasoners/r1",
				Tags:    []string{"tag1"},
			},
		},
		Skills: []CompactCapability{
			{
				ID:      "s1",
				AgentID: "agent-1",
				Target:  "http://localhost:8001/skills/s1",
			},
		},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded CompactDiscoveryResponse
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.DiscoveredAt, decoded.DiscoveredAt)
	require.Len(t, decoded.Reasoners, 1)
	assert.Equal(t, "r1", decoded.Reasoners[0].ID)
	assert.Equal(t, "agent-1", decoded.Reasoners[0].AgentID)
	assert.Equal(t, "http://localhost:8001/reasoners/r1", decoded.Reasoners[0].Target)
	assert.Equal(t, []string{"tag1"}, decoded.Reasoners[0].Tags)

	require.Len(t, decoded.Skills, 1)
	assert.Equal(t, "s1", decoded.Skills[0].ID)
	assert.Equal(t, "agent-1", decoded.Skills[0].AgentID)
}

func TestCompactCapabilityTagsOmitEmpty(t *testing.T) {
	c := CompactCapability{
		ID:      "c1",
		AgentID: "a1",
		Target:  "http://example.com",
		// No Tags — omitempty
	}

	data, err := json.Marshal(c)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasTags := m["tags"]
	assert.False(t, hasTags, "tags should be omitted when nil")
}

func TestDiscoveryResultFields(t *testing.T) {
	// DiscoveryResult is a wrapper struct — verify fields can be set and read
	jsonResp := &DiscoveryResponse{TotalAgents: 5}
	compactResp := &CompactDiscoveryResponse{}

	r := DiscoveryResult{
		Format:  "json",
		JSON:    jsonResp,
		Compact: compactResp,
		XML:     "<agents/>",
		Raw:     `{"agents":[]}`,
	}

	assert.Equal(t, "json", r.Format)
	assert.Equal(t, 5, r.JSON.TotalAgents)
	assert.NotNil(t, r.Compact)
	assert.Equal(t, "<agents/>", r.XML)
	assert.Equal(t, `{"agents":[]}`, r.Raw)
}
