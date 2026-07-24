package agentic

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/gin-gonic/gin"
)

// DiscoverHandler returns a handler that searches the API catalog.
func DiscoverHandler(catalog *apicatalog.Catalog) gin.HandlerFunc {
	return func(c *gin.Context) {
		query := c.Query("q")
		group := c.Query("group")
		method := c.Query("method")
		limit := getIntQuery(c, "limit", 20)

		if limit > 100 {
			limit = 100
		}

		authLevel := getAuthLevel(c)
		results := catalog.Search(query, method, group, 0)
		results = catalog.FilterByAuth(results, authLevel)

		if len(results) > limit {
			results = results[:limit]
		}

		respondOK(c, gin.H{
			"endpoints": results,
			"total":     len(results),
			"groups":    catalog.Groups(),
			"filters": gin.H{
				"q":      query,
				"group":  group,
				"method": method,
			},
			"see_also": gin.H{
				"live_agents": "GET /api/v1/discovery/capabilities — lists running agents, their reasoners, skills, and invocation targets",
				"kb":          "GET /api/v1/agentic/kb/topics — knowledge base with SDK patterns, architecture guides, and examples",
			},
		})
	}
}

// Smart404Handler returns a NoRoute handler that provides endpoint suggestions.
// It wraps an existing noRouteHandler for UI SPA fallback support.
func Smart404Handler(catalog *apicatalog.Catalog, uiNoRouteHandler gin.HandlerFunc) gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		method := c.Request.Method

		// Delegate UI paths to the original handler
		if len(path) >= 4 && path[:4] == "/ui/" {
			if uiNoRouteHandler != nil {
				uiNoRouteHandler(c)
			} else {
				c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
			}
			return
		}

		authLevel := getAuthLevel(c)
		suggestions := catalog.FindSimilar(method, path, 5)

		type suggestionResponse struct {
			Method  string  `json:"method"`
			Path    string  `json:"path"`
			Summary string  `json:"summary"`
			Score   float64 `json:"score"`
		}

		var sugResp []suggestionResponse
		for _, s := range suggestions {
			// Manual auth filtering on suggestions
			entries := catalog.FilterByAuth([]apicatalog.EndpointEntry{s.EndpointEntry}, authLevel)
			if len(entries) > 0 {
				sugResp = append(sugResp, suggestionResponse{
					Method:  s.Method,
					Path:    s.Path,
					Summary: s.Summary,
					Score:   s.Score,
				})
			}
		}

		c.JSON(http.StatusNotFound, gin.H{
			"error":       "endpoint_not_found",
			"message":     method + " " + path + " does not exist",
			"suggestions": sugResp,
			"help": gin.H{
				"api_search":      "GET /api/v1/agentic/discover?q=<keyword> — search all API endpoints",
				"live_agents":     "GET /api/v1/discovery/capabilities — list running agents and their reasoners/skills",
				"kb":              "GET /api/v1/agentic/kb/topics — knowledge base (public, no auth)",
				"guide":           "GET /api/v1/agentic/kb/guide?goal=<description> — goal-oriented learning path (public)",
			},
		})
	}
}
