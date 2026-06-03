package knowledgebase

import (
	"strings"
)

// GuideStep is a single article in a recommended reading path.
type GuideStep struct {
	Order   int            `json:"order"`
	Article ArticleSummary `json:"article"`
	Reason  string         `json:"reason"`
}

// Guide returns a goal-oriented reading path based on a text goal description.
func (kb *KB) Guide(goal string) []GuideStep {
	kb.mu.RLock()
	defer kb.mu.RUnlock()

	goal = strings.ToLower(goal)

	// Score each article's relevance to the goal
	type scored struct {
		article *Article
		score   float64
		reason  string
	}

	var candidates []scored

	for _, a := range kb.articles {
		score, reason := scoreArticleForGoal(a, goal)
		if score > 0 {
			candidates = append(candidates, scored{article: a, score: score, reason: reason})
		}
	}

	// Sort by topic order (building < patterns < sdk < examples < observability < identity)
	// then by score within topics
	topicOrder := map[string]int{
		"building":      0,
		"sdk":           1,
		"patterns":      2,
		"examples":      3,
		"observability": 4,
		"identity":      5,
	}

	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			oi := topicOrder[candidates[i].article.Topic]
			oj := topicOrder[candidates[j].article.Topic]
			if oi > oj || (oi == oj && candidates[i].score < candidates[j].score) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Cap at 10 steps
	if len(candidates) > 10 {
		candidates = candidates[:10]
	}

	steps := make([]GuideStep, len(candidates))
	for i, c := range candidates {
		steps[i] = GuideStep{
			Order:   i + 1,
			Article: Summarize(c.article),
			Reason:  c.reason,
		}
	}

	return steps
}

func scoreArticleForGoal(a *Article, goal string) (float64, string) {
	score := 0.0
	reason := ""

	// Direct tag match
	for _, tag := range a.Tags {
		if strings.Contains(goal, strings.ToLower(tag)) {
			score += 0.3
			if reason == "" {
				reason = "Matches goal keyword: " + tag
			}
		}
	}

	// Title match
	titleLower := strings.ToLower(a.Title)
	words := strings.Fields(goal)
	for _, w := range words {
		if len(w) > 3 && strings.Contains(titleLower, w) {
			score += 0.25
			if reason == "" {
				reason = "Title matches: " + a.Title
			}
		}
	}

	// Summary match
	summaryLower := strings.ToLower(a.Summary)
	for _, w := range words {
		if len(w) > 3 && strings.Contains(summaryLower, w) {
			score += 0.15
			if reason == "" {
				reason = "Relevant: " + a.Summary
			}
		}
	}

	// Topic relevance for common goals
	goalTopics := map[string][]string{
		"build":     {"building", "sdk"},
		"create":    {"building", "sdk"},
		"debug":     {"observability"},
		"monitor":   {"observability"},
		"security":  {"identity", "patterns"},
		"pattern":   {"patterns"},
		"example":   {"examples"},
		"identity":  {"identity"},
		"did":       {"identity"},
		"memory":    {"building"},
		"tool":      {"building"},
		"harness":   {"building", "patterns"},
		"reasoner":  {"building", "sdk"},
		"agent":     {"building", "sdk", "examples"},
		"research":  {"examples", "patterns"},
		"audit":     {"identity", "observability"},
	}

	for keyword, topics := range goalTopics {
		if strings.Contains(goal, keyword) {
			for _, t := range topics {
				if a.Topic == t {
					score += 0.2
					if reason == "" {
						reason = "Relevant to " + keyword + " tasks"
					}
				}
			}
		}
	}

	// Beginner articles get a small boost for common goals
	if a.Difficulty == "beginner" && score > 0 {
		score += 0.05
	}

	return score, reason
}
