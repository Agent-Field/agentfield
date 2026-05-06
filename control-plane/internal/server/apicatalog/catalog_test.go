package apicatalog

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	c := New()
	assert.NotNil(t, c)
	assert.Empty(t, c.All())
	assert.Empty(t, c.Groups())
}

func TestRegisterAndAll(t *testing.T) {
	c := New()
	c.Register(EndpointEntry{Method: "GET", Path: "/api/v1/health", Group: "health", Summary: "Health check"})
	c.Register(EndpointEntry{Method: "POST", Path: "/api/v1/nodes", Group: "nodes", Summary: "Register node"})

	all := c.All()
	assert.Len(t, all, 2)
	assert.Equal(t, "GET", all[0].Method)
	assert.Equal(t, "POST", all[1].Method)
}

func TestRegisterBatch(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "g1"},
		{Method: "POST", Path: "/b", Group: "g2"},
		{Method: "PUT", Path: "/c", Group: "g1"},
	})
	assert.Len(t, c.All(), 3)
}

func TestGroups(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "nodes"},
		{Method: "POST", Path: "/b", Group: "execute"},
		{Method: "PUT", Path: "/c", Group: "nodes"},
	})
	groups := c.Groups()
	assert.Equal(t, []string{"execute", "nodes"}, groups)
}

func TestSearch_ByKeyword(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/api/v1/health", Group: "health", Summary: "Health check", Tags: []string{"health"}},
		{Method: "POST", Path: "/api/v1/execute/:target", Group: "execute", Summary: "Execute a reasoner", Tags: []string{"execute"}},
		{Method: "GET", Path: "/api/v1/nodes", Group: "nodes", Summary: "List nodes", Tags: []string{"nodes"}},
	})

	results := c.Search("execute", "", "", 0)
	require.Len(t, results, 1)
	assert.Equal(t, "/api/v1/execute/:target", results[0].Path)
}

func TestSearch_ByGroup(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "nodes"},
		{Method: "POST", Path: "/b", Group: "nodes"},
		{Method: "GET", Path: "/c", Group: "execute"},
	})

	results := c.Search("", "", "nodes", 0)
	assert.Len(t, results, 2)
}

func TestSearch_ByMethod(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "g1"},
		{Method: "POST", Path: "/b", Group: "g1"},
		{Method: "GET", Path: "/c", Group: "g2"},
	})

	results := c.Search("", "GET", "", 0)
	assert.Len(t, results, 2)
}

func TestSearch_WithLimit(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "g1"},
		{Method: "GET", Path: "/b", Group: "g1"},
		{Method: "GET", Path: "/c", Group: "g1"},
	})

	results := c.Search("", "", "", 2)
	assert.Len(t, results, 2)
}

func TestSearch_EmptyQuery(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/a", Group: "g1"},
		{Method: "POST", Path: "/b", Group: "g2"},
	})

	results := c.Search("", "", "", 0)
	assert.Len(t, results, 2)
}

func TestFindSimilar(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/api/v1/nodes", Group: "nodes", Summary: "List nodes"},
		{Method: "GET", Path: "/api/v1/nodes/:node_id", Group: "nodes", Summary: "Get node"},
		{Method: "POST", Path: "/api/v1/execute/:target", Group: "execute", Summary: "Execute"},
		{Method: "GET", Path: "/api/v1/executions/:id", Group: "executions", Summary: "Get execution"},
	})

	suggestions := c.FindSimilar("GET", "/api/v1/node", 3)
	require.NotEmpty(t, suggestions)
	// /api/v1/nodes should score highest (closest match)
	assert.Contains(t, suggestions[0].Path, "/api/v1/nodes")
}

func TestFindSimilar_MethodBonus(t *testing.T) {
	c := New()
	c.RegisterBatch([]EndpointEntry{
		{Method: "GET", Path: "/api/v1/nodes", Group: "nodes"},
		{Method: "POST", Path: "/api/v1/nodes", Group: "nodes"},
	})

	suggestions := c.FindSimilar("POST", "/api/v1/nodes", 2)
	require.Len(t, suggestions, 2)
	// POST should score higher than GET due to method match
	assert.Equal(t, "POST", suggestions[0].Method)
}

func TestFindSimilar_Limit(t *testing.T) {
	c := New()
	for i := 0; i < 10; i++ {
		c.Register(EndpointEntry{Method: "GET", Path: "/api/v1/test", Group: "test"})
	}

	suggestions := c.FindSimilar("GET", "/api/v1/test", 3)
	assert.LessOrEqual(t, len(suggestions), 3)
}

func TestFilterByAuth(t *testing.T) {
	c := New()
	entries := []EndpointEntry{
		{Method: "GET", Path: "/health", AuthLevel: "public"},
		{Method: "GET", Path: "/api/v1/nodes", AuthLevel: "api_key"},
		{Method: "POST", Path: "/api/v1/admin/tags", AuthLevel: "admin"},
		{Method: "GET", Path: "/api/v1/connector/config", AuthLevel: "connector"},
	}

	// Public caller: only public
	public := c.FilterByAuth(entries, "public")
	assert.Len(t, public, 1)
	assert.Equal(t, "/health", public[0].Path)

	// API key caller: public + api_key
	apiKey := c.FilterByAuth(entries, "api_key")
	assert.Len(t, apiKey, 2)

	// Admin caller: everything
	admin := c.FilterByAuth(entries, "admin")
	assert.Len(t, admin, 4)
}

func TestDefaultEntries_NotEmpty(t *testing.T) {
	entries := DefaultEntries()
	assert.Greater(t, len(entries), 50, "should have at least 50 curated entries")

	// Every entry should have method, path, group
	for _, e := range entries {
		assert.NotEmpty(t, e.Method, "entry missing method: %s", e.Path)
		assert.NotEmpty(t, e.Path, "entry missing path")
		assert.NotEmpty(t, e.Group, "entry missing group: %s", e.Path)
	}
}

func TestLevenshtein(t *testing.T) {
	assert.Equal(t, 0, levenshtein("abc", "abc"))
	assert.Equal(t, 1, levenshtein("abc", "ab"))
	assert.Equal(t, 3, levenshtein("abc", "xyz"))
	assert.Equal(t, 3, levenshtein("", "abc"))
	assert.Equal(t, 0, levenshtein("", ""))
}

func TestIsAccessible(t *testing.T) {
	assert.True(t, isAccessible("public", "public"))
	assert.True(t, isAccessible("webhook", "public"))
	assert.True(t, isAccessible("public", "api_key"))
	assert.True(t, isAccessible("webhook", "api_key"))
	assert.True(t, isAccessible("api_key", "api_key"))
	assert.False(t, isAccessible("api_key", "public"))
	assert.False(t, isAccessible("api_key", "webhook"))
	assert.True(t, isAccessible("admin", "admin"))
	assert.False(t, isAccessible("admin", "api_key"))
}
