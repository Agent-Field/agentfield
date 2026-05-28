package apicatalog

import (
	"sort"
	"strings"
	"sync"
)

// EndpointEntry describes a single API endpoint.
type EndpointEntry struct {
	Method      string       `json:"method"`
	Path        string       `json:"path"`
	Group       string       `json:"group"`
	Summary     string       `json:"summary"`
	AuthLevel   string       `json:"auth_level"` // "public", "api_key", "admin", "connector", "webhook"
	Parameters  []ParamEntry `json:"parameters,omitempty"`
	RequestBody *BodyEntry   `json:"request_body,omitempty"`
	Tags        []string     `json:"tags,omitempty"`
}

// ParamEntry describes a single URL or query parameter.
type ParamEntry struct {
	Name     string `json:"name"`
	In       string `json:"in"` // "path", "query"
	Required bool   `json:"required"`
	Type     string `json:"type"`
	Desc     string `json:"description,omitempty"`
}

// BodyEntry describes the request body schema.
type BodyEntry struct {
	ContentType string            `json:"content_type"`
	Fields      map[string]string `json:"fields,omitempty"` // field name -> type description
	Example     string            `json:"example,omitempty"`
}

// Suggestion is a fuzzy-matched endpoint returned by FindSimilar.
type Suggestion struct {
	EndpointEntry
	Score float64 `json:"score"`
}

// Catalog is a thread-safe in-memory registry of API endpoints.
type Catalog struct {
	mu       sync.RWMutex
	entries  []EndpointEntry
	byGroup  map[string][]EndpointEntry
	byMethod map[string][]EndpointEntry
}

// New creates an empty Catalog.
func New() *Catalog {
	return &Catalog{
		entries:  make([]EndpointEntry, 0, 128),
		byGroup:  make(map[string][]EndpointEntry),
		byMethod: make(map[string][]EndpointEntry),
	}
}

// Register adds an endpoint to the catalog.
func (c *Catalog) Register(e EndpointEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = append(c.entries, e)
	c.byGroup[e.Group] = append(c.byGroup[e.Group], e)
	c.byMethod[e.Method] = append(c.byMethod[e.Method], e)
}

// RegisterBatch adds multiple endpoints at once.
func (c *Catalog) RegisterBatch(entries []EndpointEntry) {
	for _, e := range entries {
		c.Register(e)
	}
}

// All returns every registered endpoint.
func (c *Catalog) All() []EndpointEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]EndpointEntry, len(c.entries))
	copy(out, c.entries)
	return out
}

// Groups returns all unique group names.
func (c *Catalog) Groups() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	groups := make([]string, 0, len(c.byGroup))
	for g := range c.byGroup {
		groups = append(groups, g)
	}
	sort.Strings(groups)
	return groups
}

// Search filters endpoints by keyword (matches path, summary, tags) plus optional method/group.
func (c *Catalog) Search(query, method, group string, limit int) []EndpointEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()

	query = strings.ToLower(query)
	var results []EndpointEntry

	for _, e := range c.entries {
		if method != "" && !strings.EqualFold(e.Method, method) {
			continue
		}
		if group != "" && !strings.EqualFold(e.Group, group) {
			continue
		}
		if query != "" && !matchesKeyword(e, query) {
			continue
		}
		results = append(results, e)
	}

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results
}

// FindSimilar returns the top-N endpoints most similar to the given method+path.
// Used by the smart 404 handler to suggest corrections.
func (c *Catalog) FindSimilar(method, path string, limit int) []Suggestion {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if limit <= 0 {
		limit = 5
	}

	path = strings.ToLower(path)
	pathSegments := strings.Split(strings.Trim(path, "/"), "/")

	var suggestions []Suggestion
	for _, e := range c.entries {
		score := similarity(method, path, pathSegments, e)
		if score > 0.15 {
			suggestions = append(suggestions, Suggestion{EndpointEntry: e, Score: score})
		}
	}

	sort.Slice(suggestions, func(i, j int) bool {
		return suggestions[i].Score > suggestions[j].Score
	})

	if len(suggestions) > limit {
		suggestions = suggestions[:limit]
	}
	return suggestions
}

// FilterByAuth returns only entries accessible at the given auth level.
func (c *Catalog) FilterByAuth(entries []EndpointEntry, authLevel string) []EndpointEntry {
	var filtered []EndpointEntry
	for _, e := range entries {
		if isAccessible(e.AuthLevel, authLevel) {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// --- internal helpers ---

func matchesKeyword(e EndpointEntry, query string) bool {
	if strings.Contains(strings.ToLower(e.Path), query) {
		return true
	}
	if strings.Contains(strings.ToLower(e.Summary), query) {
		return true
	}
	if strings.Contains(strings.ToLower(e.Group), query) {
		return true
	}
	for _, tag := range e.Tags {
		if strings.Contains(strings.ToLower(tag), query) {
			return true
		}
	}
	return false
}

func similarity(method, path string, pathSegments []string, e EndpointEntry) float64 {
	score := 0.0

	// Method match bonus
	if strings.EqualFold(method, e.Method) {
		score += 0.2
	}

	entryPath := strings.ToLower(e.Path)
	entrySegments := strings.Split(strings.Trim(entryPath, "/"), "/")

	// Prefix matching: how many leading segments match?
	prefixMatch := 0
	minLen := len(pathSegments)
	if len(entrySegments) < minLen {
		minLen = len(entrySegments)
	}
	for i := 0; i < minLen; i++ {
		ps := pathSegments[i]
		es := entrySegments[i]
		// Treat :param as wildcard match
		if strings.HasPrefix(es, ":") || ps == es {
			prefixMatch++
		} else {
			break
		}
	}
	if minLen > 0 {
		score += 0.4 * float64(prefixMatch) / float64(minLen)
	}

	// Segment overlap (order-independent)
	overlap := 0
	for _, ps := range pathSegments {
		for _, es := range entrySegments {
			if ps == es || strings.HasPrefix(es, ":") {
				overlap++
				break
			}
		}
	}
	maxSegments := len(pathSegments)
	if len(entrySegments) > maxSegments {
		maxSegments = len(entrySegments)
	}
	if maxSegments > 0 {
		score += 0.25 * float64(overlap) / float64(maxSegments)
	}

	// Levenshtein on last segment (the "resource name")
	if len(pathSegments) > 0 && len(entrySegments) > 0 {
		lastQuery := pathSegments[len(pathSegments)-1]
		lastEntry := entrySegments[len(entrySegments)-1]
		if !strings.HasPrefix(lastEntry, ":") {
			dist := levenshtein(lastQuery, lastEntry)
			maxLen := len(lastQuery)
			if len(lastEntry) > maxLen {
				maxLen = len(lastEntry)
			}
			if maxLen > 0 {
				score += 0.15 * (1.0 - float64(dist)/float64(maxLen))
			}
		}
	}

	return score
}

func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}

	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}

	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min3(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

func isAccessible(endpointAuth, callerAuth string) bool {
	levels := map[string]int{
		"public":    0,
		"webhook":   0,
		"api_key":   1,
		"admin":     2,
		"connector": 2,
	}
	callerLevel, ok := levels[callerAuth]
	if !ok {
		callerLevel = 0
	}
	endpointLevel, ok := levels[endpointAuth]
	if !ok {
		endpointLevel = 1
	}
	return callerLevel >= endpointLevel
}
