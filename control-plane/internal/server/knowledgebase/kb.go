package knowledgebase

import (
	"sort"
	"strings"
	"sync"
)

// Article represents a knowledge base article.
type Article struct {
	ID         string   `json:"id"`
	Topic      string   `json:"topic"`
	Title      string   `json:"title"`
	Summary    string   `json:"summary"`
	Content    string   `json:"content"`
	Tags       []string `json:"tags"`
	SDK        string   `json:"sdk,omitempty"`
	Difficulty string   `json:"difficulty"`
	References []string `json:"references,omitempty"`
}

// TopicInfo describes a topic with its article count.
type TopicInfo struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	ArticleCount int    `json:"article_count"`
}

// KB is the in-memory knowledge base.
type KB struct {
	mu       sync.RWMutex
	articles map[string]*Article // id -> article
	byTopic  map[string][]*Article
	topics   map[string]string // topic name -> description
}

// New creates an empty KB.
func New() *KB {
	return &KB{
		articles: make(map[string]*Article),
		byTopic:  make(map[string][]*Article),
		topics:   make(map[string]string),
	}
}

// RegisterTopic adds a topic description.
func (kb *KB) RegisterTopic(name, description string) {
	kb.mu.Lock()
	defer kb.mu.Unlock()
	kb.topics[name] = description
}

// Add adds an article to the KB.
func (kb *KB) Add(a Article) {
	kb.mu.Lock()
	defer kb.mu.Unlock()
	article := a // copy
	kb.articles[a.ID] = &article
	kb.byTopic[a.Topic] = append(kb.byTopic[a.Topic], &article)
}

// AddBatch adds multiple articles.
func (kb *KB) AddBatch(articles []Article) {
	for _, a := range articles {
		kb.Add(a)
	}
}

// Topics returns all topics with article counts.
func (kb *KB) Topics() []TopicInfo {
	kb.mu.RLock()
	defer kb.mu.RUnlock()

	var topics []TopicInfo
	for name, desc := range kb.topics {
		count := len(kb.byTopic[name])
		topics = append(topics, TopicInfo{
			Name:         name,
			Description:  desc,
			ArticleCount: count,
		})
	}

	sort.Slice(topics, func(i, j int) bool {
		return topics[i].Name < topics[j].Name
	})
	return topics
}

// Get retrieves a single article by ID.
func (kb *KB) Get(id string) *Article {
	kb.mu.RLock()
	defer kb.mu.RUnlock()
	return kb.articles[id]
}

// Search filters articles by topic, SDK, difficulty, and/or keyword.
func (kb *KB) Search(topic, sdk, difficulty, query string, limit int) []*Article {
	kb.mu.RLock()
	defer kb.mu.RUnlock()

	query = strings.ToLower(query)
	var results []*Article

	for _, a := range kb.articles {
		if topic != "" && a.Topic != topic {
			continue
		}
		if sdk != "" && a.SDK != sdk {
			continue
		}
		if difficulty != "" && a.Difficulty != difficulty {
			continue
		}
		if query != "" && !matchesArticle(a, query) {
			continue
		}
		results = append(results, a)
	}

	// Sort by topic then ID for deterministic results
	sort.Slice(results, func(i, j int) bool {
		if results[i].Topic != results[j].Topic {
			return results[i].Topic < results[j].Topic
		}
		return results[i].ID < results[j].ID
	})

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results
}

// ArticleSummary returns an article without the full content (for listings).
type ArticleSummary struct {
	ID         string   `json:"id"`
	Topic      string   `json:"topic"`
	Title      string   `json:"title"`
	Summary    string   `json:"summary"`
	Tags       []string `json:"tags"`
	SDK        string   `json:"sdk,omitempty"`
	Difficulty string   `json:"difficulty"`
}

// Summarize converts an article to a summary (no content).
func Summarize(a *Article) ArticleSummary {
	return ArticleSummary{
		ID:         a.ID,
		Topic:      a.Topic,
		Title:      a.Title,
		Summary:    a.Summary,
		Tags:       a.Tags,
		SDK:        a.SDK,
		Difficulty: a.Difficulty,
	}
}

// SummarizeAll converts multiple articles to summaries.
func SummarizeAll(articles []*Article) []ArticleSummary {
	summaries := make([]ArticleSummary, len(articles))
	for i, a := range articles {
		summaries[i] = Summarize(a)
	}
	return summaries
}

func matchesArticle(a *Article, query string) bool {
	if strings.Contains(strings.ToLower(a.Title), query) {
		return true
	}
	if strings.Contains(strings.ToLower(a.Summary), query) {
		return true
	}
	if strings.Contains(strings.ToLower(a.ID), query) {
		return true
	}
	for _, tag := range a.Tags {
		if strings.Contains(strings.ToLower(tag), query) {
			return true
		}
	}
	return false
}
