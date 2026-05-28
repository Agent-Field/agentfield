package services

import (
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// CanonicalAgentTags returns normalized plain tags for permission matching.
// Canonical tags are lowercased, trimmed, plain values (e.g. "admin").
func CanonicalAgentTags(agent *types.AgentNode) []string {
	if agent == nil {
		return nil
	}

	seen := make(map[string]struct{})
	tags := make([]string, 0)

	add := func(tag string) {
		normalized := normalizeTag(tag)
		if normalized == "" {
			return
		}
		if _, exists := seen[normalized]; exists {
			return
		}
		seen[normalized] = struct{}{}
		tags = append(tags, normalized)
	}

	// NOTE: Deployment metadata tags (agent.Metadata.Deployment.Tags) and
	// agent.DeploymentType are excluded from canonical authorization tags because
	// they are self-asserted at registration time and NOT subject to the tag
	// approval workflow. Including them would allow agents to self-assign
	// authorization-relevant tags.

	for _, reasoner := range agent.Reasoners {
		// Prefer approved tags over raw tags for canonical matching
		sourceTags := reasoner.Tags
		if len(reasoner.ApprovedTags) > 0 {
			sourceTags = reasoner.ApprovedTags
		}
		for _, tag := range sourceTags {
			add(tag)
		}
	}

	for _, skill := range agent.Skills {
		// Prefer approved tags over raw tags for canonical matching
		sourceTags := skill.Tags
		if len(skill.ApprovedTags) > 0 {
			sourceTags = skill.ApprovedTags
		}
		for _, tag := range sourceTags {
			add(tag)
		}
	}

	// Include agent-level approved tags
	for _, tag := range agent.ApprovedTags {
		add(tag)
	}

	return tags
}

func normalizeTag(tag string) string {
	return strings.ToLower(strings.TrimSpace(tag))
}
