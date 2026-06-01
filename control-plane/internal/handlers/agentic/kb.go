package agentic

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/knowledgebase"
	"github.com/gin-gonic/gin"
)

// KBTopicsHandler returns all KB topics with article counts.
func KBTopicsHandler(kb *knowledgebase.KB) gin.HandlerFunc {
	return func(c *gin.Context) {
		respondOK(c, gin.H{
			"topics": kb.Topics(),
		})
	}
}

// KBArticlesHandler searches and filters KB articles.
func KBArticlesHandler(kb *knowledgebase.KB) gin.HandlerFunc {
	return func(c *gin.Context) {
		topic := c.Query("topic")
		sdk := c.Query("sdk")
		difficulty := c.Query("difficulty")
		query := c.Query("q")
		limit := getIntQuery(c, "limit", 50)

		if limit > 200 {
			limit = 200
		}

		articles := kb.Search(topic, sdk, difficulty, query, limit)
		respondOK(c, gin.H{
			"articles": knowledgebase.SummarizeAll(articles),
			"total":    len(articles),
			"filters": gin.H{
				"topic":      topic,
				"sdk":        sdk,
				"difficulty": difficulty,
				"q":          query,
			},
		})
	}
}

// KBArticleHandler returns a single article by ID with full content.
func KBArticleHandler(kb *knowledgebase.KB) gin.HandlerFunc {
	return func(c *gin.Context) {
		articleID := c.Param("article_id")
		// Handle nested IDs like "building/reasoner-python" via wildcard
		if sub := c.Param("sub_id"); sub != "" {
			articleID = articleID + "/" + sub
		}

		if articleID == "" {
			respondError(c, http.StatusBadRequest, "missing_article_id", "article_id path parameter is required")
			return
		}

		article := kb.Get(articleID)
		if article == nil {
			respondError(c, http.StatusNotFound, "article_not_found", "article "+articleID+" not found")
			return
		}

		respondOK(c, article)
	}
}

// KBGuideHandler returns a goal-oriented reading path.
func KBGuideHandler(kb *knowledgebase.KB) gin.HandlerFunc {
	return func(c *gin.Context) {
		goal := c.Query("goal")
		if goal == "" {
			respondError(c, http.StatusBadRequest, "missing_goal", "goal query parameter is required (e.g. ?goal=build+security+agent)")
			return
		}

		steps := kb.Guide(goal)
		respondOK(c, gin.H{
			"goal":  goal,
			"steps": steps,
			"total": len(steps),
		})
	}
}
