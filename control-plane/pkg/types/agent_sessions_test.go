package types

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHydrateAgentSessionsFromMetadata(t *testing.T) {
	agent := &AgentNode{
		Metadata: AgentMetadata{Custom: map[string]interface{}{
			"sessions": []map[string]interface{}{
				{
					"name":          "voice",
					"provider":      "openai",
					"transport":     "webrtc",
					"proposed_tags": []string{"support"},
				},
			},
		}},
	}

	HydrateAgentSessions(agent)

	require.Len(t, agent.Sessions, 1)
	require.Equal(t, "voice", agent.Sessions[0].Name)
	require.Equal(t, []string{"support"}, agent.Sessions[0].ProposedTags)
}

func TestHydrateAgentSessionsSkipsInvalidAndExistingValues(t *testing.T) {
	withExisting := &AgentNode{
		Sessions: []SessionDefinition{{Name: "existing"}},
		Metadata: AgentMetadata{Custom: map[string]interface{}{
			"sessions": []map[string]interface{}{{"name": "metadata"}},
		}},
	}
	HydrateAgentSessions(withExisting)
	require.Equal(t, "existing", withExisting.Sessions[0].Name)

	invalid := &AgentNode{Metadata: AgentMetadata{Custom: map[string]interface{}{
		"sessions": func() {},
	}}}
	HydrateAgentSessions(invalid)
	require.Empty(t, invalid.Sessions)

	HydrateAgentSessions(nil)
}

func TestSyncAgentSessionsToMetadata(t *testing.T) {
	agent := &AgentNode{
		Sessions: []SessionDefinition{{
			Name:         "voice",
			Provider:     "openai",
			Transport:    "webrtc",
			ApprovedTags: []string{"support"},
		}},
	}

	SyncAgentSessionsToMetadata(agent)

	require.NotNil(t, agent.Metadata.Custom)
	raw := agent.Metadata.Custom["sessions"]
	bytes, err := json.Marshal(raw)
	require.NoError(t, err)
	require.JSONEq(t, `[{"name":"voice","provider":"openai","transport":"webrtc","approved_tags":["support"]}]`, string(bytes))

	empty := &AgentNode{}
	SyncAgentSessionsToMetadata(empty)
	require.Nil(t, empty.Metadata.Custom)
	SyncAgentSessionsToMetadata(nil)
}
