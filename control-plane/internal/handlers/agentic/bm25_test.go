package agentic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{name: "empty", in: "", want: nil},
		{name: "snake_case", in: "run_pr_resolver", want: []string{"run", "pr", "resolver"}},
		{name: "kebab-case", in: "review-pull-request", want: []string{"review", "pull", "request"}},
		{name: "camelCase", in: "reviewPullRequest", want: []string{"review", "pull", "request"}},
		{name: "PascalCase", in: "PlanTask", want: []string{"plan", "task"}},
		{name: "acronym then word", in: "reviewPRRequest", want: []string{"review", "pr", "request"}},
		{name: "leading acronym", in: "HTTPServer", want: []string{"http", "server"}},
		{name: "digit boundary", in: "v2Handler", want: []string{"v2", "handler"}},
		{name: "mixed separators", in: "get forecast, now!", want: []string{"get", "forecast", "now"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tokenize(tt.in))
		})
	}
}

// buildIndex is a small corpus of reasoner-shaped documents used across the
// ranking tests. It mirrors the field-weighting the handler applies.
func fieldsFor(id, tags, agentID string) []searchField {
	return []searchField{
		{boost: reasonerFieldBoostID, text: id},
		{boost: reasonerFieldBoostTags, text: tags},
		{boost: reasonerFieldBoostAgentID, text: agentID},
	}
}

func rankedIDs(hits []searchHit) []string {
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.id
	}
	return ids
}

func TestBM25Search(t *testing.T) {
	docs := []searchDoc{
		{id: "pr-af:review_pull_request", fields: fieldsFor("review_pull_request", "pr code-review", "pr-af")},
		{id: "swe:plan_task", fields: fieldsFor("plan_task", "planning", "swe")},
		{id: "misc:run_pr_resolver", fields: fieldsFor("run_pr_resolver", "automation", "misc")},
		{id: "sec:audit", fields: fieldsFor("audit", "security vulnerability", "sec")},
		{id: "weather:get_forecast", fields: fieldsFor("get_forecast", "weather", "weather-agent")},
	}
	idx := newBM25Index(docs)

	t.Run("exact id match ranks first", func(t *testing.T) {
		hits := idx.Search("plan_task")
		require.NotEmpty(t, hits)
		assert.Equal(t, "swe:plan_task", hits[0].id)
	})

	t.Run("multi-token query ranks pr-review reasoner first", func(t *testing.T) {
		hits := idx.Search("review pull request")
		require.NotEmpty(t, hits)
		assert.Equal(t, "pr-af:review_pull_request", hits[0].id)
	})

	t.Run("tag-only match is found", func(t *testing.T) {
		hits := idx.Search("vulnerability")
		ids := rankedIDs(hits)
		require.Len(t, ids, 1)
		assert.Equal(t, "sec:audit", ids[0])
	})

	t.Run("snake_case splitting lets pr resolve match run_pr_resolver", func(t *testing.T) {
		hits := idx.Search("pr resolve")
		ids := rankedIDs(hits)
		assert.Contains(t, ids, "misc:run_pr_resolver")
	})

	t.Run("no match returns empty", func(t *testing.T) {
		assert.Empty(t, idx.Search("kubernetes"))
	})

	t.Run("empty query returns empty", func(t *testing.T) {
		assert.Empty(t, idx.Search("   "))
	})
}

func TestBM25DeterministicTieBreak(t *testing.T) {
	// Two documents with byte-identical searchable content: scores tie, so the
	// order must be deterministic — ascending by document id.
	docs := []searchDoc{
		{id: "zeta:same", fields: fieldsFor("same", "", "")},
		{id: "alpha:same", fields: fieldsFor("same", "", "")},
	}
	idx := newBM25Index(docs)

	hits := idx.Search("same")
	require.Len(t, hits, 2)
	assert.Equal(t, hits[0].score, hits[1].score, "scores should tie")
	assert.Equal(t, "alpha:same", hits[0].id)
	assert.Equal(t, "zeta:same", hits[1].id)
}

func TestBM25EmptyCorpus(t *testing.T) {
	idx := newBM25Index(nil)
	assert.Empty(t, idx.Search("anything"))
}
