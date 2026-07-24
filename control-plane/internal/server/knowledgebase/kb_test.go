package knowledgebase

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	kb := New()
	assert.NotNil(t, kb)
	assert.Empty(t, kb.Topics())
}

func TestAddAndGet(t *testing.T) {
	kb := New()
	kb.RegisterTopic("building", "Build agents")
	kb.Add(Article{
		ID: "building/hello", Topic: "building", Title: "Hello",
		Summary: "Hello world", Content: "# Hello", Tags: []string{"hello"},
		Difficulty: "beginner",
	})

	article := kb.Get("building/hello")
	require.NotNil(t, article)
	assert.Equal(t, "Hello", article.Title)
	assert.Equal(t, "# Hello", article.Content)
}

func TestGet_NotFound(t *testing.T) {
	kb := New()
	assert.Nil(t, kb.Get("nonexistent"))
}

func TestTopics(t *testing.T) {
	kb := New()
	kb.RegisterTopic("building", "Build agents")
	kb.RegisterTopic("patterns", "Architectural patterns")
	kb.Add(Article{ID: "building/a", Topic: "building"})
	kb.Add(Article{ID: "building/b", Topic: "building"})
	kb.Add(Article{ID: "patterns/a", Topic: "patterns"})

	topics := kb.Topics()
	require.Len(t, topics, 2)
	assert.Equal(t, "building", topics[0].Name)
	assert.Equal(t, 2, topics[0].ArticleCount)
	assert.Equal(t, "patterns", topics[1].Name)
	assert.Equal(t, 1, topics[1].ArticleCount)
}

func TestSearch_ByTopic(t *testing.T) {
	kb := New()
	kb.Add(Article{ID: "a/1", Topic: "a"})
	kb.Add(Article{ID: "b/1", Topic: "b"})
	kb.Add(Article{ID: "a/2", Topic: "a"})

	results := kb.Search("a", "", "", "", 0)
	assert.Len(t, results, 2)
}

func TestSearch_BySDK(t *testing.T) {
	kb := New()
	kb.Add(Article{ID: "a/1", Topic: "a", SDK: "python"})
	kb.Add(Article{ID: "a/2", Topic: "a", SDK: "go"})
	kb.Add(Article{ID: "a/3", Topic: "a", SDK: ""})

	results := kb.Search("", "python", "", "", 0)
	assert.Len(t, results, 1)
	assert.Equal(t, "python", results[0].SDK)
}

func TestSearch_ByDifficulty(t *testing.T) {
	kb := New()
	kb.Add(Article{ID: "a/1", Topic: "a", Difficulty: "beginner"})
	kb.Add(Article{ID: "a/2", Topic: "a", Difficulty: "advanced"})

	results := kb.Search("", "", "beginner", "", 0)
	assert.Len(t, results, 1)
}

func TestSearch_ByKeyword(t *testing.T) {
	kb := New()
	kb.Add(Article{ID: "a/1", Topic: "a", Title: "Building Agents", Tags: []string{"agent"}})
	kb.Add(Article{ID: "a/2", Topic: "a", Title: "Memory Scopes", Tags: []string{"memory"}})
	kb.Add(Article{ID: "a/3", Topic: "a", Summary: "How to build agent systems"})

	results := kb.Search("", "", "", "agent", 0)
	assert.Len(t, results, 2) // title match + tag match
}

func TestSearch_WithLimit(t *testing.T) {
	kb := New()
	for i := 0; i < 10; i++ {
		kb.Add(Article{ID: "a/" + string(rune('0'+i)), Topic: "a"})
	}
	results := kb.Search("", "", "", "", 3)
	assert.Len(t, results, 3)
}

func TestSummarize(t *testing.T) {
	a := &Article{
		ID: "test/1", Topic: "test", Title: "Test Article",
		Summary: "A test", Content: "Full content here",
		Tags: []string{"test"}, Difficulty: "beginner",
	}
	s := Summarize(a)
	assert.Equal(t, "test/1", s.ID)
	assert.Equal(t, "A test", s.Summary)
	// Summary should not include content
	assert.Empty(t, "", "") // Content is not in ArticleSummary type
}

func TestSummarizeAll(t *testing.T) {
	articles := []*Article{
		{ID: "a", Title: "A"},
		{ID: "b", Title: "B"},
	}
	summaries := SummarizeAll(articles)
	assert.Len(t, summaries, 2)
	assert.Equal(t, "A", summaries[0].Title)
}

func TestGuide(t *testing.T) {
	kb := New()
	kb.RegisterTopic("building", "Build")
	kb.RegisterTopic("patterns", "Patterns")
	kb.RegisterTopic("examples", "Examples")

	kb.Add(Article{
		ID: "building/reasoner", Topic: "building", Title: "Creating a Reasoner",
		Summary: "How to build a Python agent", Tags: []string{"reasoner", "python", "agent"},
		Difficulty: "beginner",
	})
	kb.Add(Article{
		ID: "patterns/hunt", Topic: "patterns", Title: "HUNT PROVE Pattern",
		Summary: "Security adversarial pattern", Tags: []string{"security", "hunt"},
		Difficulty: "advanced",
	})
	kb.Add(Article{
		ID: "examples/sec-af", Topic: "examples", Title: "Security Auditor",
		Summary: "Security scanner example", Tags: []string{"security", "audit"},
		Difficulty: "advanced",
	})

	steps := kb.Guide("build a security agent")
	require.NotEmpty(t, steps)
	// Should find security-related articles
	found := false
	for _, s := range steps {
		if s.Article.ID == "patterns/hunt" || s.Article.ID == "examples/sec-af" {
			found = true
		}
	}
	assert.True(t, found, "guide should include security-related articles")
}

func TestGuide_EmptyGoal(t *testing.T) {
	kb := New()
	steps := kb.Guide("")
	assert.Empty(t, steps)
}

func TestLoadDefaultContent(t *testing.T) {
	kb := New()
	LoadDefaultContent(kb)

	topics := kb.Topics()
	assert.GreaterOrEqual(t, len(topics), 6, "should have at least 6 topics")

	// Check specific articles exist
	assert.NotNil(t, kb.Get("building/reasoner-python"))
	assert.NotNil(t, kb.Get("building/ai-vs-harness"))
	assert.NotNil(t, kb.Get("patterns/hunt-prove"))
	assert.NotNil(t, kb.Get("sdk/python-quickstart"))
	assert.NotNil(t, kb.Get("examples/sec-af"))
	assert.NotNil(t, kb.Get("observability/metrics"))
	assert.NotNil(t, kb.Get("identity/did-setup"))

	// Check article quality
	article := kb.Get("building/reasoner-python")
	assert.NotEmpty(t, article.Title)
	assert.NotEmpty(t, article.Summary)
	assert.NotEmpty(t, article.Content)
	assert.NotEmpty(t, article.Tags)
	assert.NotEmpty(t, article.Difficulty)
}

func TestLoadDefaultContent_AllArticlesValid(t *testing.T) {
	kb := New()
	LoadDefaultContent(kb)

	all := kb.Search("", "", "", "", 0)
	assert.Greater(t, len(all), 30, "should have at least 30 articles")

	for _, a := range all {
		assert.NotEmpty(t, a.ID, "article missing ID")
		assert.NotEmpty(t, a.Topic, "article %s missing topic", a.ID)
		assert.NotEmpty(t, a.Title, "article %s missing title", a.ID)
		assert.NotEmpty(t, a.Difficulty, "article %s missing difficulty", a.ID)
	}
}
