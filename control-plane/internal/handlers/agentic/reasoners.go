package agentic

import (
	"math"
	"net/http"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// Field boosts for reasoner search (BM25F-lite). The reasoner id is the
// strongest signal, then its tags, then the owning agent id, then any
// human-authored description carried in agent metadata.
const (
	reasonerFieldBoostID          = 3.0
	reasonerFieldBoostTags        = 2.0
	reasonerFieldBoostAgentID     = 1.5
	reasonerFieldBoostDescription = 1.0
)

const (
	reasonerSearchDefaultLimit = 10
	reasonerSearchMaxLimit     = 50
)

// ReasonerSearchResult is one ranked reasoner. It carries everything the
// driving agent needs to invoke the reasoner immediately, without a second
// lookup: the invocation target and the owning agent's current health.
type ReasonerSearchResult struct {
	ReasonerID       string   `json:"reasoner_id"`
	AgentID          string   `json:"agent_id"`
	InvocationTarget string   `json:"invocation_target"`
	Tags             []string `json:"tags,omitempty"`
	Score            float64  `json:"score"`
	AgentHealth      string   `json:"agent_health"`
}

// ReasonersHandler ranks installed reasoners against a free-text query using a
// self-contained BM25F-lite ranker. It reads the same registration data the
// discovery surface exposes (store.ListAgents) so search results never drift
// from discovery. The corpus is rebuilt per request — acceptable for the small
// reasoner population and far simpler than a background index.
//
// GET /api/v1/agentic/reasoners?q=<free text>&agent=<optional id>&limit=<1..50>
func ReasonersHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		query := strings.TrimSpace(c.Query("q"))
		if query == "" {
			respondError(c, http.StatusBadRequest, "missing_query",
				"q is required — pass free text describing the reasoner you need, e.g. ?q=review+pull+request")
			return
		}

		limit := getIntQuery(c, "limit", reasonerSearchDefaultLimit)
		if limit <= 0 {
			limit = reasonerSearchDefaultLimit
		}
		if limit > reasonerSearchMaxLimit {
			limit = reasonerSearchMaxLimit
		}

		agentFilter := strings.TrimSpace(c.Query("agent"))

		agents, err := store.ListAgents(c.Request.Context(), types.AgentFilters{})
		if err != nil {
			respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
			return
		}

		docs, meta := buildReasonerCorpus(agents, agentFilter)
		idx := newBM25Index(docs)
		hits := idx.Search(query)

		// Capacity is the constant maximum, not the request-supplied limit, so
		// the allocation size is provably attacker-independent (CodeQL
		// go/uncontrolled-allocation-size); the loop below still stops at limit.
		results := make([]ReasonerSearchResult, 0, reasonerSearchMaxLimit)
		for _, hit := range hits {
			if len(results) >= limit {
				break
			}
			record, ok := meta[hit.id]
			if !ok {
				continue
			}
			record.Score = roundScore(hit.score)
			results = append(results, record)
		}

		respondOK(c, gin.H{
			"query":         query,
			"results":       results,
			"total_indexed": len(docs),
		})
	}
}

// buildReasonerCorpus turns the registered agents into one searchable document
// per reasoner, keyed by its invocation target (unique across agents). It also
// returns a lookup of the result payload for each key so a ranked hit maps back
// to its source record without re-scanning the agent list.
func buildReasonerCorpus(agents []*types.AgentNode, agentFilter string) ([]searchDoc, map[string]ReasonerSearchResult) {
	docs := make([]searchDoc, 0)
	meta := make(map[string]ReasonerSearchResult)

	for _, agent := range agents {
		if agent == nil {
			continue
		}
		if agentFilter != "" && agent.ID != agentFilter {
			continue
		}
		for _, reasoner := range agent.Reasoners {
			// Matches the discovery surface's invocation_target format
			// (agent:reasoner, colon-delimited). Unique per reasoner.
			target := agent.ID + ":" + reasoner.ID

			fields := []searchField{
				{boost: reasonerFieldBoostID, text: reasoner.ID},
				{boost: reasonerFieldBoostTags, text: strings.Join(reasoner.Tags, " ")},
				{boost: reasonerFieldBoostAgentID, text: agent.ID},
			}
			if desc := reasonerDescription(agent, reasoner.ID); desc != "" {
				fields = append(fields, searchField{boost: reasonerFieldBoostDescription, text: desc})
			}

			docs = append(docs, searchDoc{id: target, fields: fields})
			meta[target] = ReasonerSearchResult{
				ReasonerID:       reasoner.ID,
				AgentID:          agent.ID,
				InvocationTarget: target,
				Tags:             reasoner.Tags,
				AgentHealth:      string(agent.HealthStatus),
			}
		}
	}
	return docs, meta
}

// reasonerDescription pulls an optional human description for a reasoner from
// agent metadata (metadata.custom.descriptions[<reasoner id>]) — the same
// side-channel the discovery handler reads. ReasonerDefinition itself carries no
// description field, so absence is normal and returns an empty string.
func reasonerDescription(agent *types.AgentNode, reasonerID string) string {
	if agent.Metadata.Custom == nil {
		return ""
	}
	raw, ok := agent.Metadata.Custom["descriptions"]
	if !ok {
		return ""
	}
	m, ok := raw.(map[string]interface{})
	if !ok {
		return ""
	}
	desc, ok := m[reasonerID]
	if !ok {
		return ""
	}
	text, ok := desc.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

// roundScore trims BM25 scores to six decimals so the JSON is tidy and stable
// across platforms without affecting ranking.
func roundScore(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
