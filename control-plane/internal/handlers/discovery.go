package handlers

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// AgentLister is the minimal dependency required for discovery.
type AgentLister interface {
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
}

// DiscoveryFilters captures query parameters for capability discovery.
type DiscoveryFilters struct {
	AgentIDs            []string
	ReasonerPattern     *string
	SkillPattern        *string
	Tags                []string
	IncludeInputSchema  bool
	IncludeOutputSchema bool
	IncludeDescriptions bool
	IncludeExamples     bool
	Format              string
	HealthStatus        *types.HealthStatus
	Limit               int
	Offset              int
}

type parameterError struct {
	Parameter string
	Provided  string
	Allowed   []string
	Reason    string
}

func (e *parameterError) Error() string {
	if e.Reason != "" {
		return e.Reason
	}
	return fmt.Sprintf("invalid %s parameter", e.Parameter)
}

// DiscoveryPagination mirrors the response pagination metadata.
type DiscoveryPagination struct {
	Limit   int  `json:"limit"`
	Offset  int  `json:"offset"`
	HasMore bool `json:"has_more"`
}

// DiscoveryResponse is the default JSON payload.
type DiscoveryResponse struct {
	DiscoveredAt   time.Time           `json:"discovered_at"`
	TotalAgents    int                 `json:"total_agents"`
	TotalReasoners int                 `json:"total_reasoners"`
	TotalSkills    int                 `json:"total_skills"`
	Pagination     DiscoveryPagination `json:"pagination"`
	Capabilities   []AgentCapability   `json:"capabilities"`
}

// AgentCapability describes a single agent and its reasoners/skills.
type AgentCapability struct {
	AgentID        string               `json:"agent_id"`
	GroupID        string               `json:"group_id"`
	BaseURL        string               `json:"base_url"`
	Version        string               `json:"version"`
	HealthStatus   string               `json:"health_status"`
	DeploymentType string               `json:"deployment_type"`
	LastHeartbeat  time.Time            `json:"last_heartbeat"`
	Reasoners      []ReasonerCapability `json:"reasoners"`
	Skills         []SkillCapability    `json:"skills"`
}

// ReasonerCapability captures metadata for a reasoner.
type ReasonerCapability struct {
	ID               string                   `json:"id"`
	Description      *string                  `json:"description,omitempty"`
	Tags             []string                 `json:"tags,omitempty"`
	InputSchema      map[string]interface{}   `json:"input_schema,omitempty"`
	OutputSchema     map[string]interface{}   `json:"output_schema,omitempty"`
	Examples         []map[string]interface{} `json:"examples,omitempty"`
	InvocationTarget string                   `json:"invocation_target"`
}

// SkillCapability captures metadata for a skill.
type SkillCapability struct {
	ID               string                 `json:"id"`
	Description      *string                `json:"description,omitempty"`
	Tags             []string               `json:"tags,omitempty"`
	InputSchema      map[string]interface{} `json:"input_schema,omitempty"`
	InvocationTarget string                 `json:"invocation_target"`
}

// CompactDiscoveryResponse is a lightweight view for LLM/tooling scenarios.
type CompactDiscoveryResponse struct {
	DiscoveredAt time.Time           `json:"discovered_at"`
	Reasoners    []CompactCapability `json:"reasoners"`
	Skills       []CompactCapability `json:"skills"`
}

// CompactCapability is the minimal representation of a capability.
type CompactCapability struct {
	ID      string   `json:"id"`
	AgentID string   `json:"agent_id"`
	Target  string   `json:"target"`
	Tags    []string `json:"tags,omitempty"`
}

var (
	agentCache     []*types.AgentNode
	agentCacheLock sync.RWMutex
	agentCacheTime time.Time
	agentCacheTTL  = 30 * time.Second
)

var (
	discoveryRequestCounter = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "agentfield_discovery_requests_total",
		Help: "Total number of discovery requests processed by the control plane.",
	}, []string{"format", "status"})
	discoveryRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "agentfield_discovery_request_duration_seconds",
		Help:    "Latency distribution for discovery responses.",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2},
	}, []string{"format"})
	discoveryCacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "agentfield_discovery_cache_hits_total",
		Help: "Count of discovery cache hits within the TTL window.",
	})
	discoveryCacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "agentfield_discovery_cache_misses_total",
		Help: "Count of discovery cache refreshes due to TTL expiry or cold starts.",
	})
	discoveryFilterUsage = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "agentfield_discovery_filter_usage_total",
		Help: "Usage count for discovery filters.",
	}, []string{"filter_type"})
)

// InvalidateDiscoveryCache resets the cached agent list.
func InvalidateDiscoveryCache() {
	agentCacheLock.Lock()
	defer agentCacheLock.Unlock()
	agentCache = nil
	agentCacheTime = time.Time{}
}

func getCachedAgents(ctx context.Context, storageProvider AgentLister) ([]*types.AgentNode, bool, error) {
	agentCacheLock.RLock()
	if time.Since(agentCacheTime) < agentCacheTTL && agentCache != nil {
		defer agentCacheLock.RUnlock()
		discoveryCacheHits.Inc()
		logger.Logger.Debug().Int("agents", len(agentCache)).Msg("discovery cache hit")
		return agentCache, true, nil
	}
	agentCacheLock.RUnlock()

	agents, err := storageProvider.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, false, err
	}

	agentCacheLock.Lock()
	agentCache = agents
	agentCacheTime = time.Now()
	agentCacheLock.Unlock()

	discoveryCacheMisses.Inc()
	logger.Logger.Debug().Int("agents", len(agents)).Msg("discovery cache refreshed")

	return agents, false, nil
}

// DiscoveryCapabilitiesHandler exposes the discovery endpoint.
func DiscoveryCapabilitiesHandler(storageProvider AgentLister) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		requestFormat := normalizeDiscoveryFormat(strings.ToLower(strings.TrimSpace(c.DefaultQuery("format", "json"))))
		filters, err := parseDiscoveryFilters(c)
		if err != nil {
			if pErr, ok := err.(*parameterError); ok {
				details := gin.H{
					"parameter": pErr.Parameter,
				}
				if pErr.Provided != "" {
					details["provided"] = pErr.Provided
				}
				if len(pErr.Allowed) > 0 {
					details["allowed"] = pErr.Allowed
				}

				recordDiscoveryMetrics(requestFormat, "error", time.Since(start))
				logDiscoveryError(c, requestFormat, time.Since(start), err)
				c.JSON(http.StatusBadRequest, gin.H{
					"error":   "invalid_parameter",
					"message": pErr.Error(),
					"details": details,
				})
				return
			}
			recordDiscoveryMetrics(requestFormat, "error", time.Since(start))
			logDiscoveryError(c, requestFormat, time.Since(start), err)
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "invalid_parameter",
				"message": err.Error(),
			})
			return
		}

		trackFilterUsage(filters)

		agents, cacheHit, err := getCachedAgents(c.Request.Context(), storageProvider)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to fetch agents for discovery")
			recordDiscoveryMetrics(filters.Format, "error", time.Since(start))
			logDiscoveryError(c, filters.Format, time.Since(start), err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "internal_error",
				"message": "Failed to retrieve agent capabilities",
			})
			return
		}

		response := buildDiscoveryResponse(agents, filters)
		switch filters.Format {
		case "xml":
			xmlBody, err := formatXMLResponse(response)
			if err != nil {
				logger.Logger.Error().Err(err).Msg("failed to render XML discovery response")
				recordDiscoveryMetrics(filters.Format, "error", time.Since(start))
				logDiscoveryError(c, filters.Format, time.Since(start), err)
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "internal_error",
					"message": "Failed to format discovery response",
				})
				return
			}
			duration := time.Since(start)
			recordDiscoveryMetrics(filters.Format, "success", duration)
			logDiscoverySuccess(c, filters, response, cacheHit, duration)
			c.Data(http.StatusOK, "application/xml", []byte(xmlBody))
		case "compact":
			duration := time.Since(start)
			recordDiscoveryMetrics(filters.Format, "success", duration)
			logDiscoverySuccess(c, filters, response, cacheHit, duration)
			c.JSON(http.StatusOK, formatCompactResponse(response))
		default:
			duration := time.Since(start)
			recordDiscoveryMetrics(filters.Format, "success", duration)
			logDiscoverySuccess(c, filters, response, cacheHit, duration)
			c.JSON(http.StatusOK, formatJSONResponse(response))
		}
	}
}

func parseDiscoveryFilters(c *gin.Context) (DiscoveryFilters, error) {
	format := strings.ToLower(strings.TrimSpace(c.DefaultQuery("format", "json")))
	switch format {
	case "json", "xml", "compact":
	default:
		return DiscoveryFilters{}, &parameterError{
			Parameter: "format",
			Provided:  format,
			Allowed:   []string{"json", "xml", "compact"},
			Reason:    "invalid format parameter. Must be one of: json, xml, compact",
		}
	}

	includeDescriptions := true
	if v := c.Query("include_descriptions"); v != "" {
		parsed, err := parseBool(v)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "include_descriptions",
				Provided:  v,
				Allowed:   []string{"true", "false"},
				Reason:    "invalid include_descriptions parameter",
			}
		}
		includeDescriptions = parsed
	}

	includeExamples := false
	if v := c.Query("include_examples"); v != "" {
		parsed, err := parseBool(v)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "include_examples",
				Provided:  v,
				Allowed:   []string{"true", "false"},
				Reason:    "invalid include_examples parameter",
			}
		}
		includeExamples = parsed
	}

	includeInputSchema := false
	if v := c.Query("include_input_schema"); v != "" {
		parsed, err := parseBool(v)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "include_input_schema",
				Provided:  v,
				Allowed:   []string{"true", "false"},
				Reason:    "invalid include_input_schema parameter",
			}
		}
		includeInputSchema = parsed
	}

	includeOutputSchema := false
	if v := c.Query("include_output_schema"); v != "" {
		parsed, err := parseBool(v)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "include_output_schema",
				Provided:  v,
				Allowed:   []string{"true", "false"},
				Reason:    "invalid include_output_schema parameter",
			}
		}
		includeOutputSchema = parsed
	}

	limit := 100
	if v := c.Query("limit"); v != "" {
		parsed, err := parseInt(v, 0, 500)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "limit",
				Provided:  v,
				Allowed:   []string{"0-500"},
				Reason:    "invalid limit parameter",
			}
		}
		limit = parsed
	}

	offset := 0
	if v := c.Query("offset"); v != "" {
		parsed, err := parseInt(v, 0, 1_000_000)
		if err != nil {
			return DiscoveryFilters{}, &parameterError{
				Parameter: "offset",
				Provided:  v,
				Allowed:   []string{"0-1000000"},
				Reason:    "invalid offset parameter",
			}
		}
		offset = parsed
	}

	var healthStatus *types.HealthStatus
	if v := strings.TrimSpace(c.Query("health_status")); v != "" {
		normalized := types.HealthStatus(strings.ToLower(v))
		switch normalized {
		case types.HealthStatusActive, types.HealthStatusInactive, types.HealthStatusDegraded, types.HealthStatusUnknown:
			healthStatus = &normalized
		default:
			return DiscoveryFilters{}, &parameterError{
				Parameter: "health_status",
				Provided:  v,
				Allowed:   []string{string(types.HealthStatusActive), string(types.HealthStatusInactive), string(types.HealthStatusDegraded), string(types.HealthStatusUnknown)},
				Reason:    "invalid health_status parameter",
			}
		}
	}

	agentIDs := dedupeStrings(collectAgentIDs(c))
	if len(agentIDs) > 0 {
		sort.Strings(agentIDs)
	}

	return DiscoveryFilters{
		AgentIDs:            agentIDs,
		ReasonerPattern:     optionalString(c.Query("reasoner")),
		SkillPattern:        optionalString(c.Query("skill")),
		Tags:                parseCSV(c.Query("tags")),
		IncludeInputSchema:  includeInputSchema,
		IncludeOutputSchema: includeOutputSchema,
		IncludeDescriptions: includeDescriptions,
		IncludeExamples:     includeExamples,
		Format:              format,
		HealthStatus:        healthStatus,
		Limit:               limit,
		Offset:              offset,
	}, nil
}

func collectAgentIDs(c *gin.Context) []string {
	var ids []string
	single := c.Query("agent")
	if single == "" {
		single = c.Query("node_id")
	}
	if single != "" {
		ids = append(ids, strings.TrimSpace(single))
	}
	if multi := c.Query("agent_ids"); multi != "" {
		ids = append(ids, parseCSV(multi)...)
	}
	if multi := c.Query("node_ids"); multi != "" {
		ids = append(ids, parseCSV(multi)...)
	}
	return ids
}

func parseCSV(value string) []string {
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	results := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			results = append(results, p)
		}
	}
	return results
}

func parseBool(value string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1", "yes", "y":
		return true, nil
	case "false", "0", "no", "n":
		return false, nil
	default:
		return false, fmt.Errorf("invalid boolean")
	}
}

func parseInt(value string, min, max int) (int, error) {
	var parsed int
	if _, err := fmt.Sscanf(value, "%d", &parsed); err != nil {
		return 0, err
	}
	if parsed < min || parsed > max {
		return 0, fmt.Errorf("out of bounds")
	}
	return parsed, nil
}

func optionalString(value string) *string {
	v := strings.TrimSpace(value)
	if v == "" {
		return nil
	}
	return &v
}

func buildDiscoveryResponse(agents []*types.AgentNode, filters DiscoveryFilters) DiscoveryResponse {
	allowedAgents := make(map[string]struct{})
	for _, id := range filters.AgentIDs {
		allowedAgents[id] = struct{}{}
	}

	var (
		matchedCapabilities []AgentCapability
		totalReasoners      int
		totalSkills         int
	)

	for _, agent := range agents {
		if len(allowedAgents) > 0 {
			if _, ok := allowedAgents[agent.ID]; !ok {
				continue
			}
		}

		if filters.HealthStatus != nil && agent.HealthStatus != *filters.HealthStatus {
			continue
		}

		capability := AgentCapability{
			AgentID:        agent.ID,
			GroupID:        agent.GroupID,
			BaseURL:        agent.BaseURL,
			Version:        agent.Version,
			HealthStatus:   string(agent.HealthStatus),
			DeploymentType: agent.DeploymentType,
			LastHeartbeat:  agent.LastHeartbeat,
		}

		for _, reasoner := range agent.Reasoners {
			if filters.ReasonerPattern != nil && !matchesPattern(reasoner.ID, *filters.ReasonerPattern) {
				continue
			}
			if len(filters.Tags) > 0 && !matchesTags(reasoner.Tags, filters.Tags) {
				continue
			}

			reasonerCap := ReasonerCapability{
				ID:               reasoner.ID,
				Tags:             reasoner.Tags,
				InvocationTarget: fmt.Sprintf("%s:%s", agent.ID, reasoner.ID),
			}

			if filters.IncludeInputSchema {
				reasonerCap.InputSchema = decodeSchema(reasoner.InputSchema)
			}
			if filters.IncludeOutputSchema {
				reasonerCap.OutputSchema = decodeSchema(reasoner.OutputSchema)
			}
			if filters.IncludeDescriptions {
				reasonerCap.Description = extractDescription(agent.Metadata, reasoner.ID)
			}
			if filters.IncludeExamples {
				reasonerCap.Examples = extractExamples(agent.Metadata, reasoner.ID)
			}

			capability.Reasoners = append(capability.Reasoners, reasonerCap)
		}

		for _, skill := range agent.Skills {
			if filters.SkillPattern != nil && !matchesPattern(skill.ID, *filters.SkillPattern) {
				continue
			}
			if len(filters.Tags) > 0 && !matchesTags(skill.Tags, filters.Tags) {
				continue
			}

			skillCap := SkillCapability{
				ID:               skill.ID,
				Tags:             skill.Tags,
				InvocationTarget: fmt.Sprintf("%s:skill:%s", agent.ID, skill.ID),
			}

			if filters.IncludeInputSchema {
				skillCap.InputSchema = decodeSchema(skill.InputSchema)
			}
			if filters.IncludeDescriptions {
				skillCap.Description = extractDescription(agent.Metadata, skill.ID)
			}

			capability.Skills = append(capability.Skills, skillCap)
		}

		if len(capability.Reasoners) > 0 || len(capability.Skills) > 0 {
			totalReasoners += len(capability.Reasoners)
			totalSkills += len(capability.Skills)
			matchedCapabilities = append(matchedCapabilities, capability)
		}
	}

	totalAgents := len(matchedCapabilities)
	start := filters.Offset
	if start > totalAgents {
		start = totalAgents
	}
	end := start + filters.Limit
	if end > totalAgents {
		end = totalAgents
	}

	paginated := matchedCapabilities[start:end]

	return DiscoveryResponse{
		DiscoveredAt:   time.Now().UTC(),
		TotalAgents:    totalAgents,
		TotalReasoners: totalReasoners,
		TotalSkills:    totalSkills,
		Pagination: DiscoveryPagination{
			Limit:   filters.Limit,
			Offset:  filters.Offset,
			HasMore: end < totalAgents,
		},
		Capabilities: paginated,
	}
}

func decodeSchema(raw json.RawMessage) map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	var schema map[string]interface{}
	if err := json.Unmarshal(raw, &schema); err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to decode schema; returning nil")
		return nil
	}
	return schema
}

func extractDescription(metadata types.AgentMetadata, id string) *string {
	if metadata.Custom == nil {
		return nil
	}
	if raw, ok := metadata.Custom["descriptions"]; ok {
		if m, ok := raw.(map[string]interface{}); ok {
			if desc, ok := m[id]; ok {
				if text, ok := desc.(string); ok && strings.TrimSpace(text) != "" {
					return &text
				}
			}
		}
	}
	return nil
}

func extractExamples(metadata types.AgentMetadata, id string) []map[string]interface{} {
	if metadata.Custom == nil {
		return nil
	}
	raw, ok := metadata.Custom["examples"]
	if !ok {
		return nil
	}

	examples, ok := raw.(map[string]interface{})
	if !ok {
		return nil
	}

	if entry, ok := examples[id]; ok {
		switch typed := entry.(type) {
		case []map[string]interface{}:
			return typed
		case []interface{}:
			results := make([]map[string]interface{}, 0, len(typed))
			for _, v := range typed {
				if m, ok := v.(map[string]interface{}); ok {
					results = append(results, m)
				}
			}
			return results
		}
	}
	return nil
}

func matchesPattern(value, pattern string) bool {
	if pattern == "" || pattern == "*" {
		return true
	}

	quoted := regexp.QuoteMeta(pattern)
	regex := "^" + strings.ReplaceAll(quoted, "\\*", ".*") + "$"
	matched, err := regexp.MatchString(regex, value)
	if err != nil {
		return false
	}
	return matched
}

func matchesTags(tags, patterns []string) bool {
	if len(patterns) == 0 {
		return true
	}
	for _, tag := range tags {
		for _, pattern := range patterns {
			if matchesPattern(tag, pattern) {
				return true
			}
		}
	}
	return false
}

func formatJSONResponse(response DiscoveryResponse) interface{} {
	return response
}

func formatXMLResponse(response DiscoveryResponse) (string, error) {
	type xmlReasoner struct {
		ID           string   `xml:"id,attr"`
		Target       string   `xml:"target,attr"`
		Description  *string  `xml:"description,omitempty"`
		Tags         []string `xml:"tags>tag,omitempty"`
		InputSchema  string   `xml:"input_schema,omitempty"`
		OutputSchema string   `xml:"output_schema,omitempty"`
	}

	type xmlSkill struct {
		ID          string   `xml:"id,attr"`
		Target      string   `xml:"target,attr"`
		Description *string  `xml:"description,omitempty"`
		Tags        []string `xml:"tags>tag,omitempty"`
		InputSchema string   `xml:"input_schema,omitempty"`
	}

	type xmlAgent struct {
		ID             string        `xml:"id,attr"`
		BaseURL        string        `xml:"base_url,attr"`
		Version        string        `xml:"version,attr"`
		HealthStatus   string        `xml:"health_status,attr"`
		DeploymentType string        `xml:"deployment_type,attr"`
		LastHeartbeat  string        `xml:"last_heartbeat,attr"`
		Reasoners      []xmlReasoner `xml:"reasoners>reasoner,omitempty"`
		Skills         []xmlSkill    `xml:"skills>skill,omitempty"`
	}

	type xmlDiscovery struct {
		XMLName      xml.Name `xml:"discovery"`
		DiscoveredAt string   `xml:"discovered_at,attr"`
		Summary      struct {
			TotalAgents    int `xml:"total_agents,attr"`
			TotalReasoners int `xml:"total_reasoners,attr"`
			TotalSkills    int `xml:"total_skills,attr"`
		} `xml:"summary"`
		Agents []xmlAgent `xml:"capabilities>agent"`
	}

	payload := xmlDiscovery{
		DiscoveredAt: response.DiscoveredAt.Format(time.RFC3339),
		Agents:       make([]xmlAgent, 0, len(response.Capabilities)),
	}
	payload.Summary.TotalAgents = response.TotalAgents
	payload.Summary.TotalReasoners = response.TotalReasoners
	payload.Summary.TotalSkills = response.TotalSkills

	for _, cap := range response.Capabilities {
		agent := xmlAgent{
			ID:             cap.AgentID,
			BaseURL:        cap.BaseURL,
			Version:        cap.Version,
			HealthStatus:   cap.HealthStatus,
			DeploymentType: cap.DeploymentType,
			LastHeartbeat:  cap.LastHeartbeat.Format(time.RFC3339),
		}

		for _, r := range cap.Reasoners {
			agent.Reasoners = append(agent.Reasoners, xmlReasoner{
				ID:           r.ID,
				Target:       r.InvocationTarget,
				Description:  r.Description,
				Tags:         r.Tags,
				InputSchema:  encodeSchema(r.InputSchema),
				OutputSchema: encodeSchema(r.OutputSchema),
			})
		}
		for _, s := range cap.Skills {
			agent.Skills = append(agent.Skills, xmlSkill{
				ID:          s.ID,
				Target:      s.InvocationTarget,
				Description: s.Description,
				Tags:        s.Tags,
				InputSchema: encodeSchema(s.InputSchema),
			})
		}
		payload.Agents = append(payload.Agents, agent)
	}

	output, err := xml.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", err
	}
	return xml.Header + string(output), nil
}

func formatCompactResponse(response DiscoveryResponse) CompactDiscoveryResponse {
	result := CompactDiscoveryResponse{
		DiscoveredAt: response.DiscoveredAt,
	}
	for _, cap := range response.Capabilities {
		for _, r := range cap.Reasoners {
			result.Reasoners = append(result.Reasoners, CompactCapability{
				ID:      r.ID,
				AgentID: cap.AgentID,
				Target:  r.InvocationTarget,
				Tags:    r.Tags,
			})
		}
		for _, s := range cap.Skills {
			result.Skills = append(result.Skills, CompactCapability{
				ID:      s.ID,
				AgentID: cap.AgentID,
				Target:  s.InvocationTarget,
				Tags:    s.Tags,
			})
		}
	}
	return result
}

func encodeSchema(schema map[string]interface{}) string {
	if len(schema) == 0 {
		return ""
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return ""
	}
	return string(b)
}

func dedupeStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

func normalizeDiscoveryFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "xml":
		return "xml"
	case "compact":
		return "compact"
	default:
		return "json"
	}
}

func recordDiscoveryMetrics(format, status string, duration time.Duration) {
	normalized := normalizeDiscoveryFormat(format)
	discoveryRequestCounter.WithLabelValues(normalized, status).Inc()
	if status == "success" {
		discoveryRequestDuration.WithLabelValues(normalized).Observe(duration.Seconds())
	}
}

func trackFilterUsage(filters DiscoveryFilters) {
	if len(filters.AgentIDs) > 0 {
		discoveryFilterUsage.WithLabelValues("agent").Inc()
	}
	if filters.ReasonerPattern != nil && strings.TrimSpace(*filters.ReasonerPattern) != "" {
		discoveryFilterUsage.WithLabelValues("reasoner").Inc()
	}
	if filters.SkillPattern != nil && strings.TrimSpace(*filters.SkillPattern) != "" {
		discoveryFilterUsage.WithLabelValues("skill").Inc()
	}
	if len(filters.Tags) > 0 {
		discoveryFilterUsage.WithLabelValues("tag").Inc()
	}
}

func logDiscoverySuccess(c *gin.Context, filters DiscoveryFilters, response DiscoveryResponse, cacheHit bool, duration time.Duration) {
	event := logger.Logger.Info().
		Str("path", c.FullPath()).
		Dur("duration", duration).
		Str("format", normalizeDiscoveryFormat(filters.Format)).
		Int("agents", response.TotalAgents).
		Int("reasoners", response.TotalReasoners).
		Int("skills", response.TotalSkills).
		Bool("cache_hit", cacheHit).
		Interface("filters", gin.H{
			"agent_ids": filters.AgentIDs,
			"reasoner":  derefOrEmpty(filters.ReasonerPattern),
			"skill":     derefOrEmpty(filters.SkillPattern),
			"tags":      filters.Tags,
			"health":    derefHealth(filters.HealthStatus),
			"limit":     filters.Limit,
			"offset":    filters.Offset,
		})
	if requestID := c.GetString("request_id"); requestID != "" {
		event = event.Str("request_id", requestID)
	}
	event.Msg("discovery request completed")
}

func logDiscoveryError(c *gin.Context, format string, duration time.Duration, err error) {
	event := logger.Logger.Warn().
		Err(err).
		Str("path", c.FullPath()).
		Dur("duration", duration).
		Str("format", normalizeDiscoveryFormat(format))
	if requestID := c.GetString("request_id"); requestID != "" {
		event = event.Str("request_id", requestID)
	}
	event.Msg("discovery request failed")
}

func derefOrEmpty(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func derefHealth(status *types.HealthStatus) string {
	if status == nil {
		return ""
	}
	return string(*status)
}
