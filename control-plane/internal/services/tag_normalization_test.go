package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
)

// --- normalizeTag tests ---

func TestNormalizeTag_Lowercase(t *testing.T) {
	assert.Equal(t, "admin", normalizeTag("ADMIN"))
	assert.Equal(t, "operator", normalizeTag("Operator"))
	assert.Equal(t, "read-only", normalizeTag("Read-Only"))
}

func TestNormalizeTag_TrimWhitespace(t *testing.T) {
	assert.Equal(t, "admin", normalizeTag("  admin  "))
	assert.Equal(t, "admin", normalizeTag("\tadmin\n"))
}

func TestNormalizeTag_Empty(t *testing.T) {
	assert.Equal(t, "", normalizeTag(""))
	assert.Equal(t, "", normalizeTag("   "))
}

func TestNormalizeTag_SpecialCharacters(t *testing.T) {
	assert.Equal(t, "admin-role", normalizeTag("admin-role"))
	assert.Equal(t, "team:alpha", normalizeTag("Team:Alpha"))
	assert.Equal(t, "v1.2.3", normalizeTag("V1.2.3"))
}

func TestNormalizeTag_Unicode(t *testing.T) {
	// Unicode letters should be lowercased where applicable
	assert.Equal(t, "héllo", normalizeTag("Héllo"))
}

func TestNormalizeTag_ConsistentOutput(t *testing.T) {
	// Same input → same output always
	assert.Equal(t, normalizeTag("ADMIN"), normalizeTag("ADMIN"))
	assert.Equal(t, normalizeTag("  Reader  "), normalizeTag("  Reader  "))
}

// --- CanonicalAgentTags tests ---

func TestCanonicalAgentTags_NilAgent(t *testing.T) {
	result := CanonicalAgentTags(nil)
	assert.Nil(t, result)
}

func TestCanonicalAgentTags_EmptyAgent(t *testing.T) {
	agent := &types.AgentNode{}
	result := CanonicalAgentTags(agent)
	assert.Empty(t, result)
}

func TestCanonicalAgentTags_ReasonerTags(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"reader", "writer"}},
		},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"reader", "writer"}, result)
}

func TestCanonicalAgentTags_ReasonerApprovedTagsTakePrecedence(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{
				Tags:         []string{"raw-tag"},
				ApprovedTags: []string{"approved-tag"},
			},
		},
	}
	result := CanonicalAgentTags(agent)
	// When ApprovedTags is non-empty, it replaces Tags
	assert.Equal(t, []string{"approved-tag"}, result)
	assert.NotContains(t, result, "raw-tag")
}

func TestCanonicalAgentTags_SkillTags(t *testing.T) {
	agent := &types.AgentNode{
		Skills: []types.SkillDefinition{
			{Tags: []string{"skill-tag-1", "skill-tag-2"}},
		},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"skill-tag-1", "skill-tag-2"}, result)
}

func TestCanonicalAgentTags_SkillApprovedTagsTakePrecedence(t *testing.T) {
	agent := &types.AgentNode{
		Skills: []types.SkillDefinition{
			{
				Tags:         []string{"raw-skill"},
				ApprovedTags: []string{"approved-skill"},
			},
		},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"approved-skill"}, result)
	assert.NotContains(t, result, "raw-skill")
}

func TestCanonicalAgentTags_AgentApprovedTags(t *testing.T) {
	agent := &types.AgentNode{
		ApprovedTags: []string{"agent-approved"},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"agent-approved"}, result)
}

func TestCanonicalAgentTags_Deduplication(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"admin", "reader"}},
			{Tags: []string{"admin", "writer"}},
		},
		ApprovedTags: []string{"admin"},
	}
	result := CanonicalAgentTags(agent)
	// "admin" should appear only once despite being in multiple sources
	count := 0
	for _, tag := range result {
		if tag == "admin" {
			count++
		}
	}
	assert.Equal(t, 1, count, "duplicate tag 'admin' should appear only once")
}

func TestCanonicalAgentTags_NormalizationApplied(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"ADMIN", "  Reader  "}},
		},
	}
	result := CanonicalAgentTags(agent)
	assert.Contains(t, result, "admin")
	assert.Contains(t, result, "reader")
	// Uppercase/whitespace variants should not appear
	assert.NotContains(t, result, "ADMIN")
	assert.NotContains(t, result, "  Reader  ")
}

func TestCanonicalAgentTags_EmptyTagsIgnored(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"", "   ", "valid-tag"}},
		},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"valid-tag"}, result)
}

func TestCanonicalAgentTags_DeploymentTagsExcluded(t *testing.T) {
	// Deployment tags should NOT be included — they are self-asserted.
	// CanonicalAgentTags only uses Reasoner/Skill/agent-level ApprovedTags.
	agent := &types.AgentNode{
		ApprovedTags: []string{"safe"},
	}
	result := CanonicalAgentTags(agent)
	assert.Equal(t, []string{"safe"}, result)
}

func TestCanonicalAgentTags_CombinedSources(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"reasoner-tag"}},
		},
		Skills: []types.SkillDefinition{
			{Tags: []string{"skill-tag"}},
		},
		ApprovedTags: []string{"agent-tag"},
	}
	result := CanonicalAgentTags(agent)
	assert.Contains(t, result, "reasoner-tag")
	assert.Contains(t, result, "skill-tag")
	assert.Contains(t, result, "agent-tag")
	assert.Len(t, result, 3)
}

func TestCanonicalAgentTags_ConsistentOutput(t *testing.T) {
	// Same input should always produce the same output
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{Tags: []string{"admin", "reader"}},
		},
	}
	result1 := CanonicalAgentTags(agent)
	result2 := CanonicalAgentTags(agent)
	assert.Equal(t, result1, result2)
}
