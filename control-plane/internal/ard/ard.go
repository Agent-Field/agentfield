package ard

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

const StateConfigKey = "ard.state"

var externalSearchHTTPClient = &http.Client{
	Timeout: 10 * time.Second,
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) == 0 {
			return nil
		}
		if len(via) >= 3 {
			return http.ErrUseLastResponse
		}
		origin := via[0].URL
		if req.URL.Scheme != origin.Scheme || req.URL.Host != origin.Host {
			return http.ErrUseLastResponse
		}
		return nil
	},
}

type Store interface {
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	GetConfig(ctx context.Context, key string) (*storage.ConfigEntry, error)
	SetConfig(ctx context.Context, key string, value string, updatedBy string) error
}

type StateReader interface {
	GetConfig(ctx context.Context, key string) (*storage.ConfigEntry, error)
}

type State struct {
	Settings      RuntimeSettings            `json:"settings"`
	Publications  map[string]Publication     `json:"publications"`
	Imports       []ExternalEntry            `json:"imports"`
	Bindings      map[string]ExternalBinding `json:"bindings"`
	Registries    []RegistryRecord           `json:"registries"`
	UpdatedAt     time.Time                  `json:"updated_at"`
	LastUpdatedBy string                     `json:"last_updated_by,omitempty"`
}

type RuntimeSettings struct {
	Enabled          *bool  `json:"enabled,omitempty"`
	PublishEnabled   *bool  `json:"publish_enabled,omitempty"`
	RegistryPublic   *bool  `json:"registry_public,omitempty"`
	PublicBaseURL    string `json:"public_base_url,omitempty"`
	PublisherDomain  string `json:"publisher_domain,omitempty"`
	DisplayName      string `json:"display_name,omitempty"`
	DocumentationURL string `json:"documentation_url,omitempty"`
	LogoURL          string `json:"logo_url,omitempty"`
}

type Publication struct {
	TargetKind            string    `json:"target_kind"`
	NodeID                string    `json:"node_id"`
	TargetID              string    `json:"target_id"`
	Published             bool      `json:"published"`
	DisplayName           string    `json:"display_name"`
	Description           string    `json:"description"`
	Tags                  []string  `json:"tags"`
	Capabilities          []string  `json:"capabilities"`
	RepresentativeQueries []string  `json:"representative_queries"`
	ArtifactType          string    `json:"artifact_type"`
	ArtifactURLOverride   string    `json:"artifact_url_override,omitempty"`
	ValidationStatus      string    `json:"validation_status"`
	ValidationErrors      []string  `json:"validation_errors,omitempty"`
	LastValidatedAt       time.Time `json:"last_validated_at,omitempty"`
	UpdatedAt             time.Time `json:"updated_at"`
}

type ExternalEntry struct {
	ID                    string          `json:"id"`
	SourceRegistry        string          `json:"source_registry,omitempty"`
	Identifier            string          `json:"identifier"`
	Type                  string          `json:"type"`
	DisplayName           string          `json:"display_name"`
	Description           string          `json:"description,omitempty"`
	URL                   string          `json:"url,omitempty"`
	Data                  json.RawMessage `json:"data,omitempty"`
	Publisher             string          `json:"publisher,omitempty"`
	TrustSummary          string          `json:"trust_summary,omitempty"`
	RepresentativeQueries []string        `json:"representative_queries,omitempty"`
	ImportedAt            time.Time       `json:"imported_at"`
}

type ExternalBinding struct {
	ExternalEntryID   string    `json:"external_entry_id"`
	Callable          bool      `json:"callable"`
	LocalTarget       string    `json:"local_target"`
	Adapter           string    `json:"adapter"`
	AuthRef           string    `json:"auth_ref,omitempty"`
	TimeoutMS         int       `json:"timeout_ms"`
	AllowedOperations []string  `json:"allowed_operations,omitempty"`
	Policy            string    `json:"policy,omitempty"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type RegistryRecord struct {
	URL             string    `json:"url"`
	Name            string    `json:"name,omitempty"`
	SubmissionState string    `json:"submission_state,omitempty"`
	LastCheckedAt   time.Time `json:"last_checked_at,omitempty"`
}

type CatalogManifest struct {
	SpecVersion string         `json:"specVersion"`
	Host        HostInfo       `json:"host"`
	Entries     []CatalogEntry `json:"entries"`
}

type HostInfo struct {
	DisplayName      string `json:"displayName"`
	Identifier       string `json:"identifier,omitempty"`
	DocumentationURL string `json:"documentationUrl,omitempty"`
	LogoURL          string `json:"logoUrl,omitempty"`
}

type CatalogEntry struct {
	Identifier            string          `json:"identifier"`
	DisplayName           string          `json:"displayName"`
	Description           string          `json:"description,omitempty"`
	Type                  string          `json:"type"`
	URL                   string          `json:"url,omitempty"`
	Data                  json.RawMessage `json:"data,omitempty"`
	Tags                  []string        `json:"tags,omitempty"`
	Capabilities          []string        `json:"capabilities,omitempty"`
	RepresentativeQueries []string        `json:"representativeQueries,omitempty"`
	Version               string          `json:"version,omitempty"`
	UpdatedAt             string          `json:"updatedAt,omitempty"`
	TrustManifest         *TrustManifest  `json:"trustManifest,omitempty"`
}

type TrustManifest struct {
	Identity     string `json:"identity"`
	IdentityType string `json:"identityType,omitempty"`
}

type AgentFieldMeta struct {
	TargetKind       string `json:"targetKind"`
	NodeID           string `json:"nodeId"`
	TargetID         string `json:"targetId"`
	InvocationTarget string `json:"invocationTarget"`
	HealthStatus     string `json:"healthStatus,omitempty"`
	Version          string `json:"version,omitempty"`
}

type Dashboard struct {
	Config       EffectiveConfig      `json:"config"`
	State        State                `json:"state"`
	Summary      Summary              `json:"summary"`
	Publications []PublicationView    `json:"publications"`
	Catalog      CatalogManifest      `json:"catalog"`
	Imports      []ExternalImportView `json:"imports"`
	Registries   []RegistryRecord     `json:"registries"`
}

type EffectiveConfig struct {
	Enabled                   bool            `json:"enabled"`
	PublishEnabled            bool            `json:"publish_enabled"`
	RegistryEnabled           bool            `json:"registry_enabled"`
	RegistryPublic            bool            `json:"registry_public"`
	ExternalSearchEnabled     bool            `json:"external_search_enabled"`
	ExternalInvocationEnabled bool            `json:"external_invocation_enabled"`
	PublicBaseURL             string          `json:"public_base_url"`
	PublisherDomain           string          `json:"publisher_domain"`
	DisplayName               string          `json:"display_name"`
	DocumentationURL          string          `json:"documentation_url"`
	LogoURL                   string          `json:"logo_url"`
	Identifier                string          `json:"identifier,omitempty"`
	DefaultType               string          `json:"default_type"`
	AllowedRegistries         []string        `json:"allowed_registries"`
	IncludeHealthStatuses     []string        `json:"include_health_statuses"`
	Locked                    map[string]bool `json:"locked"`
}

type Summary struct {
	ARDEnabled                bool   `json:"ard_enabled"`
	CatalogPublished          bool   `json:"catalog_published"`
	PublicURLReachable        bool   `json:"public_url_reachable"`
	DIDAvailable              bool   `json:"did_available"`
	PublishedReasoners        int    `json:"published_reasoners"`
	PublishedSkills           int    `json:"published_skills"`
	ImportedResources         int    `json:"imported_resources"`
	CallableExternalResources int    `json:"callable_external_resources"`
	CatalogURL                string `json:"catalog_url"`
}

type PublicationView struct {
	Publication
	Key         string          `json:"key"`
	Status      string          `json:"status"`
	Entry       CatalogEntry    `json:"entry,omitempty"`
	AgentField  AgentFieldMeta  `json:"agentfield"`
	ArtifactURL string          `json:"artifact_url,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type ExternalImportView struct {
	Entry   ExternalEntry    `json:"entry"`
	Binding *ExternalBinding `json:"binding,omitempty"`
	Status  string           `json:"status"`
}

type SearchRequest struct {
	Query      QueryModel `json:"query"`
	Federation string     `json:"federation,omitempty"`
	PageSize   int        `json:"pageSize,omitempty"`
	PageToken  string     `json:"pageToken,omitempty"`
	Registries []string   `json:"registries,omitempty"`
}

type SearchResponse struct {
	Results   []SearchResult `json:"results"`
	Referrals []CatalogEntry `json:"referrals,omitempty"`
	PageToken string         `json:"pageToken,omitempty"`
	Sources   []SearchSource `json:"sources,omitempty"`
}

type QueryModel struct {
	Text   string              `json:"text,omitempty"`
	Filter map[string][]string `json:"filter,omitempty"`
}

func (q *QueryModel) UnmarshalJSON(data []byte) error {
	var raw struct {
		Text   string                     `json:"text,omitempty"`
		Filter map[string]json.RawMessage `json:"filter,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	q.Text = raw.Text
	q.Filter = nil
	if len(raw.Filter) == 0 {
		return nil
	}
	q.Filter = make(map[string][]string, len(raw.Filter))
	for field, value := range raw.Filter {
		var values []any
		if err := json.Unmarshal(value, &values); err == nil {
			for _, item := range values {
				if filterValue := filterValueString(item); filterValue != "" {
					q.Filter[field] = append(q.Filter[field], filterValue)
				}
			}
			continue
		}
		var single any
		if err := json.Unmarshal(value, &single); err != nil {
			return fmt.Errorf("invalid filter %q: %w", field, err)
		}
		if filterValue := filterValueString(single); filterValue != "" {
			q.Filter[field] = []string{filterValue}
		}
	}
	for field, values := range q.Filter {
		if len(values) == 0 {
			delete(q.Filter, field)
		}
	}
	if len(q.Filter) == 0 {
		q.Filter = nil
	}
	return nil
}

type SearchSource struct {
	URL    string `json:"url"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

type SearchResult struct {
	CatalogEntry
	Score  float64 `json:"score"`
	Source string  `json:"source"`
}

type ExploreRequest struct {
	Query      QueryModel        `json:"query,omitempty"`
	ResultType ExploreResultType `json:"resultType"`
}

type ExploreResultType struct {
	Facets []ExploreFacetRequest `json:"facets"`
}

type ExploreFacetRequest struct {
	Field    string `json:"field"`
	Limit    int    `json:"limit,omitempty"`
	MinCount int    `json:"minCount,omitempty"`
}

type ExploreResponse struct {
	ResultType string                        `json:"resultType"`
	Facets     map[string]ExploreFacetResult `json:"facets"`
}

type ExploreFacetResult struct {
	Buckets    []ExploreFacetBucket `json:"buckets"`
	OtherCount int                  `json:"otherCount,omitempty"`
}

type ExploreFacetBucket struct {
	Value string `json:"value"`
	Count int    `json:"count"`
}

type AgentListResponse struct {
	Items     []CatalogEntry `json:"items"`
	Total     int            `json:"total,omitempty"`
	PageToken string         `json:"pageToken,omitempty"`
}

func LoadState(ctx context.Context, st Store) (State, error) {
	return LoadStateReadOnly(ctx, st)
}

func LoadStateReadOnly(ctx context.Context, st StateReader) (State, error) {
	state := defaultState()
	entry, err := st.GetConfig(ctx, StateConfigKey)
	if err != nil {
		return state, err
	}
	if entry == nil || strings.TrimSpace(entry.Value) == "" {
		return state, nil
	}
	if err := json.Unmarshal([]byte(entry.Value), &state); err != nil {
		return defaultState(), err
	}
	normalizeState(&state)
	return state, nil
}

func SaveState(ctx context.Context, st Store, state State, updatedBy string) error {
	normalizeState(&state)
	state.UpdatedAt = time.Now().UTC()
	state.LastUpdatedBy = updatedBy
	payload, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return st.SetConfig(ctx, StateConfigKey, string(payload), updatedBy)
}

func BuildDashboard(ctx context.Context, st Store, cfg config.ARDConfig, didAvailable bool) (Dashboard, error) {
	state, err := LoadState(ctx, st)
	if err != nil {
		return Dashboard{}, err
	}
	effective := Effective(cfg, state)
	catalog, publications, err := BuildCatalog(ctx, st, effective, state, didAvailable)
	if err != nil {
		return Dashboard{}, err
	}
	imports := make([]ExternalImportView, 0, len(state.Imports))
	for _, entry := range state.Imports {
		binding, ok := state.Bindings[entry.ID]
		status := "imported"
		if ok && binding.Callable {
			status = "callable"
		}
		view := ExternalImportView{Entry: entry, Status: status}
		if ok {
			b := binding
			view.Binding = &b
		}
		imports = append(imports, view)
	}
	summary := Summary{
		ARDEnabled:         effective.Enabled,
		CatalogPublished:   effective.Enabled && effective.PublishEnabled,
		PublicURLReachable: effective.PublicBaseURL != "",
		DIDAvailable:       didAvailable,
		ImportedResources:  len(state.Imports),
		CatalogURL:         strings.TrimRight(effective.PublicBaseURL, "/") + "/.well-known/ai-catalog.json",
	}
	if effective.PublicBaseURL == "" {
		summary.CatalogURL = "/.well-known/ai-catalog.json"
	}
	for _, pub := range publications {
		if isCatalogVisible(pub, effective) {
			if pub.TargetKind == "skill" {
				summary.PublishedSkills++
			} else {
				summary.PublishedReasoners++
			}
		}
	}
	for _, binding := range state.Bindings {
		if binding.Callable {
			summary.CallableExternalResources++
		}
	}
	return Dashboard{
		Config:       effective,
		State:        state,
		Summary:      summary,
		Publications: publications,
		Catalog:      catalog,
		Imports:      imports,
		Registries:   state.Registries,
	}, nil
}

func Effective(cfg config.ARDConfig, state State) EffectiveConfig {
	enabled := cfg.Enabled
	if cfg.Enabled && state.Settings.Enabled != nil {
		enabled = *state.Settings.Enabled
	}
	publishEnabled := cfg.Publish.Enabled
	if cfg.Publish.Enabled && state.Settings.PublishEnabled != nil {
		publishEnabled = *state.Settings.PublishEnabled
	}
	registryPublic := cfg.Registry.Public
	if cfg.Registry.Public && state.Settings.RegistryPublic != nil {
		registryPublic = *state.Settings.RegistryPublic
	}
	publicBaseURL := strings.TrimRight(firstNonEmpty(cfg.PublicBaseURL, state.Settings.PublicBaseURL), "/")
	publisherDomain := firstNonEmpty(cfg.PublisherDomain, state.Settings.PublisherDomain)
	displayName := firstNonEmpty(cfg.Host.DisplayName, state.Settings.DisplayName, "AgentField Control Plane")
	return EffectiveConfig{
		Enabled:                   enabled,
		PublishEnabled:            publishEnabled,
		RegistryEnabled:           cfg.Registry.Enabled,
		RegistryPublic:            registryPublic,
		ExternalSearchEnabled:     cfg.External.SearchEnabled,
		ExternalInvocationEnabled: cfg.External.InvocationEnabled,
		PublicBaseURL:             publicBaseURL,
		PublisherDomain:           publisherDomain,
		DisplayName:               displayName,
		DocumentationURL:          firstNonEmpty(cfg.Host.DocumentationURL, state.Settings.DocumentationURL),
		LogoURL:                   firstNonEmpty(cfg.Host.LogoURL, state.Settings.LogoURL),
		Identifier:                cfg.Host.Identifier,
		DefaultType:               firstNonEmpty(cfg.Publish.DefaultType, "application/openapi+json"),
		AllowedRegistries:         append([]string{}, cfg.External.AllowedRegistries...),
		IncludeHealthStatuses:     append([]string{}, cfg.Publish.IncludeHealthStatuses...),
		Locked: map[string]bool{
			"enabled":          !cfg.Enabled,
			"publish_enabled":  !cfg.Publish.Enabled,
			"registry_public":  !cfg.Registry.Public,
			"public_base_url":  cfg.PublicBaseURL != "",
			"publisher_domain": cfg.PublisherDomain != "",
			"display_name":     cfg.Host.DisplayName != "",
		},
	}
}

func BuildCatalog(ctx context.Context, st Store, cfg EffectiveConfig, state State, didAvailable bool) (CatalogManifest, []PublicationView, error) {
	catalog := CatalogManifest{
		SpecVersion: "1.0",
		Host: HostInfo{
			DisplayName:      cfg.DisplayName,
			Identifier:       cfg.Identifier,
			DocumentationURL: cfg.DocumentationURL,
			LogoURL:          cfg.LogoURL,
		},
		Entries: []CatalogEntry{},
	}
	agents, err := st.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return catalog, nil, err
	}
	publications := make([]PublicationView, 0, len(state.Publications))
	for _, agent := range agents {
		for _, reasoner := range agent.Reasoners {
			key := TargetKey("reasoner", agent.ID, reasoner.ID)
			pub, ok := state.Publications[key]
			if !ok {
				pub = defaultPublication("reasoner", agent.ID, reasoner.ID, reasoner.Tags, "")
				pub.ArtifactType = firstNonEmpty(cfg.DefaultType, pub.ArtifactType)
			}
			view := buildPublicationView(cfg, agent, "reasoner", reasoner.ID, reasoner.InputSchema, reasoner.OutputSchema, pub, didAvailable)
			publications = append(publications, view)
			if isCatalogVisible(view, cfg) {
				catalog.Entries = append(catalog.Entries, view.Entry)
			}
		}
		for _, skill := range agent.Skills {
			key := TargetKey("skill", agent.ID, skill.ID)
			pub, ok := state.Publications[key]
			if !ok {
				pub = defaultPublication("skill", agent.ID, skill.ID, skill.Tags, "")
				pub.ArtifactType = firstNonEmpty(cfg.DefaultType, pub.ArtifactType)
			}
			view := buildPublicationView(cfg, agent, "skill", skill.ID, skill.InputSchema, nil, pub, didAvailable)
			publications = append(publications, view)
			if isCatalogVisible(view, cfg) {
				catalog.Entries = append(catalog.Entries, view.Entry)
			}
		}
	}
	sort.Slice(publications, func(i, j int) bool { return publications[i].Key < publications[j].Key })
	sort.Slice(catalog.Entries, func(i, j int) bool { return catalog.Entries[i].Identifier < catalog.Entries[j].Identifier })
	return catalog, publications, nil
}

func ValidatePublication(pub Publication, cfg EffectiveConfig) []string {
	var errs []string
	if !cfg.Enabled {
		errs = append(errs, "ARD is disabled")
	}
	if !cfg.PublishEnabled {
		errs = append(errs, "catalog publishing is disabled")
	}
	if cfg.PublicBaseURL == "" {
		errs = append(errs, "public base URL is required")
	}
	if cfg.PublisherDomain == "" {
		errs = append(errs, "publisher domain is required")
	}
	if strings.TrimSpace(pub.DisplayName) == "" {
		errs = append(errs, "display name is required")
	}
	if strings.TrimSpace(pub.Description) == "" {
		errs = append(errs, "description is required")
	}
	if len(nonEmpty(pub.RepresentativeQueries)) < 2 {
		errs = append(errs, "at least two representative queries are required")
	}
	if len(nonEmpty(pub.RepresentativeQueries)) > 5 {
		errs = append(errs, "at most five representative queries are allowed")
	}
	if strings.TrimSpace(pub.ArtifactType) == "" {
		errs = append(errs, "artifact type is required")
	}
	if strings.TrimSpace(pub.ArtifactType) != "" &&
		strings.TrimSpace(pub.ArtifactType) != "application/openapi+json" &&
		strings.TrimSpace(pub.ArtifactURLOverride) == "" {
		errs = append(errs, "artifact URL override is required for non-OpenAPI artifact types")
	}
	return errs
}

func TargetKey(kind, nodeID, targetID string) string {
	if kind == "skill" {
		return nodeID + ":skill:" + targetID
	}
	return nodeID + "." + targetID
}

func ImportID(identifier string) string {
	sum := sha256.Sum256([]byte(identifier))
	return "ext_" + hex.EncodeToString(sum[:])[:16]
}

func BuildArtifact(ctx context.Context, st Store, cfg EffectiveConfig, state State, entryID string, didAvailable bool) (map[string]any, bool, error) {
	_, publications, err := BuildCatalog(ctx, st, cfg, state, didAvailable)
	if err != nil {
		return nil, false, err
	}
	for _, pub := range publications {
		if pub.Key == entryID && isCatalogVisible(pub, cfg) {
			target := pub.AgentField.InvocationTarget
			return map[string]any{
				"openapi": "3.1.0",
				"info": map[string]any{
					"title":       pub.DisplayName,
					"description": pub.Description,
					"version":     pub.AgentField.Version,
				},
				"paths": map[string]any{
					"/api/v1/execute/" + target: map[string]any{
						"post": map[string]any{
							"summary":     "Execute " + pub.DisplayName,
							"description": "AgentField callable projected for ARD discovery.",
							"requestBody": map[string]any{
								"required": true,
								"content": map[string]any{
									"application/json": map[string]any{
										"schema": map[string]any{
											"type": "object",
											"properties": map[string]any{
												"input": schemaOrObject(pub.InputSchema),
											},
										},
									},
								},
							},
							"responses": map[string]any{"200": map[string]any{"description": "Execution result"}},
						},
					},
				},
				"x-agentfield": pub.AgentField,
			}, true, nil
		}
	}
	return nil, false, nil
}

func LocalSearch(entries []CatalogEntry, req SearchRequest, source string) []SearchResult {
	query := strings.ToLower(strings.TrimSpace(req.Query.Text))
	var results []SearchResult
	for _, entry := range entries {
		if !matchesFilter(entry, req.Query.Filter) {
			continue
		}
		score := scoreEntry(entry, query)
		if query == "" || score > 0 {
			results = append(results, SearchResult{CatalogEntry: entry, Score: float64(score), Source: source})
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if req.PageSize > 0 && len(results) > req.PageSize {
		results = results[:req.PageSize]
	}
	return results
}

func SearchExternal(ctx context.Context, registries []string, req SearchRequest) SearchResponse {
	allowed := make(map[string]string, len(registries))
	for _, registry := range registries {
		if normalized, ok := normalizeRegistryURL(registry); ok {
			allowed[normalized] = normalized
		}
	}
	targets := registries
	if len(req.Registries) > 0 {
		targets = req.Registries
	}
	resp := SearchResponse{Results: []SearchResult{}, Sources: []SearchSource{}}
	for _, rawURL := range targets {
		requested, valid := normalizeRegistryURL(rawURL)
		if !valid {
			resp.Sources = append(resp.Sources, SearchSource{URL: rawURL, Status: "error", Error: "invalid registry URL"})
			continue
		}
		base, ok := allowed[requested]
		if !ok {
			resp.Sources = append(resp.Sources, SearchSource{URL: rawURL, Status: "blocked", Error: "registry is not allowlisted"})
			continue
		}
		outboundReq := req
		outboundReq.Federation = "none"
		outboundReq.Registries = nil
		body, _ := json.Marshal(outboundReq)
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/search", bytes.NewReader(body))
		if err != nil {
			resp.Sources = append(resp.Sources, SearchSource{URL: base, Status: "error", Error: err.Error()})
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpResp, err := externalSearchHTTPClient.Do(httpReq)
		if err != nil {
			resp.Sources = append(resp.Sources, SearchSource{URL: base, Status: "error", Error: err.Error()})
			continue
		}
		func() {
			defer httpResp.Body.Close()
			if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
				resp.Sources = append(resp.Sources, SearchSource{URL: base, Status: "error", Error: httpResp.Status})
				return
			}
			var payload struct {
				Results []SearchResult `json:"results"`
			}
			if err := json.NewDecoder(httpResp.Body).Decode(&payload); err != nil {
				resp.Sources = append(resp.Sources, SearchSource{URL: base, Status: "error", Error: err.Error()})
				return
			}
			for _, result := range payload.Results {
				result.Source = base
				resp.Results = append(resp.Results, result)
			}
			resp.Sources = append(resp.Sources, SearchSource{URL: base, Status: "ok"})
		}()
	}
	sort.Slice(resp.Results, func(i, j int) bool { return resp.Results[i].Score > resp.Results[j].Score })
	if req.PageSize > 0 && len(resp.Results) > req.PageSize {
		resp.Results = resp.Results[:req.PageSize]
	}
	return resp
}

func normalizeRegistryURL(raw string) (string, bool) {
	trimmed := strings.TrimRight(strings.TrimSpace(raw), "/")
	if trimmed == "" {
		return "", false
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" {
		return "", false
	}
	if parsed.Scheme != "https" && parsed.Scheme != "http" {
		return "", false
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", false
	}
	return parsed.String(), true
}

func Explore(entries []CatalogEntry, req ExploreRequest) ExploreResponse {
	matched := make([]CatalogEntry, 0, len(entries))
	query := strings.ToLower(strings.TrimSpace(req.Query.Text))
	for _, entry := range entries {
		if !matchesFilter(entry, req.Query.Filter) {
			continue
		}
		if query != "" && scoreEntry(entry, query) == 0 {
			continue
		}
		matched = append(matched, entry)
	}
	facetCounts := map[string]map[string]int{
		"type":         {},
		"tags":         {},
		"capabilities": {},
		"publisher":    {},
	}
	for _, entry := range matched {
		facetCounts["type"][entry.Type]++
		if publisher := publisherFromURN(entry.Identifier); publisher != "" {
			facetCounts["publisher"][publisher]++
		}
		for _, tag := range entry.Tags {
			facetCounts["tags"][tag]++
		}
		for _, capability := range entry.Capabilities {
			facetCounts["capabilities"][capability]++
		}
	}
	facets := requestedFacets(req)
	resp := ExploreResponse{
		ResultType: "facets",
		Facets:     make(map[string]ExploreFacetResult, len(facets)),
	}
	for _, facet := range facets {
		limit := facet.Limit
		if limit <= 0 {
			limit = 20
		}
		minCount := facet.MinCount
		if minCount <= 0 {
			minCount = 1
		}
		resp.Facets[facet.Field] = ExploreFacetResult{Buckets: facetBuckets(facetCounts[facet.Field], limit, minCount)}
	}
	return resp
}

func defaultState() State {
	return State{
		Publications: make(map[string]Publication),
		Bindings:     make(map[string]ExternalBinding),
		Imports:      []ExternalEntry{},
		Registries:   []RegistryRecord{},
	}
}

func normalizeState(state *State) {
	if state.Publications == nil {
		state.Publications = make(map[string]Publication)
	}
	if state.Bindings == nil {
		state.Bindings = make(map[string]ExternalBinding)
	}
	if state.Imports == nil {
		state.Imports = []ExternalEntry{}
	}
	if state.Registries == nil {
		state.Registries = []RegistryRecord{}
	}
}

func buildPublicationView(cfg EffectiveConfig, agent *types.AgentNode, kind, targetID string, inputSchema, outputSchema json.RawMessage, pub Publication, didAvailable bool) PublicationView {
	if pub.TargetKind == "" {
		pub.TargetKind = kind
	}
	if pub.NodeID == "" {
		pub.NodeID = agent.ID
	}
	if pub.TargetID == "" {
		pub.TargetID = targetID
	}
	if pub.ArtifactType == "" {
		pub.ArtifactType = firstNonEmpty(cfg.DefaultType, "application/openapi+json")
	}
	pub.Tags = nonEmpty(pub.Tags)
	pub.Capabilities = nonEmpty(pub.Capabilities)
	pub.RepresentativeQueries = nonEmpty(pub.RepresentativeQueries)
	if pub.DisplayName == "" {
		pub.DisplayName = targetID
	}
	if pub.Description == "" {
		pub.Description = fmt.Sprintf("%s %s from AgentField node %s", strings.Title(kind), targetID, agent.ID)
	}
	pub.ValidationErrors = ValidatePublication(pub, cfg)
	pub.ValidationStatus = "valid"
	if len(pub.ValidationErrors) > 0 {
		pub.ValidationStatus = "invalid"
	}
	pub.LastValidatedAt = time.Now().UTC()
	key := TargetKey(kind, agent.ID, targetID)
	artifactURL := pub.ArtifactURLOverride
	if artifactURL == "" && cfg.PublicBaseURL != "" {
		artifactURL = strings.TrimRight(cfg.PublicBaseURL, "/") + "/api/v1/ard/artifacts/" + url.PathEscape(key)
	}
	entry := CatalogEntry{
		Identifier:            fmt.Sprintf("urn:ai:%s:agentfield:%s:%s:%s", sanitizeURNPart(cfg.PublisherDomain), sanitizeURNPart(agent.ID), kind, sanitizeURNPart(targetID)),
		DisplayName:           pub.DisplayName,
		Description:           pub.Description,
		Type:                  pub.ArtifactType,
		URL:                   artifactURL,
		Tags:                  pub.Tags,
		Capabilities:          pub.Capabilities,
		RepresentativeQueries: pub.RepresentativeQueries,
		Version:               agent.Version,
	}
	if trust := trustManifest(cfg, didAvailable); trust != nil {
		entry.TrustManifest = trust
	}
	agentField := AgentFieldMeta{
		TargetKind:       kind,
		NodeID:           agent.ID,
		TargetID:         targetID,
		InvocationTarget: invocationTarget(kind, agent.ID, targetID),
		HealthStatus:     string(agent.HealthStatus),
		Version:          agent.Version,
	}
	return PublicationView{
		Publication: pub,
		Key:         key,
		Status:      publicationStatus(pub),
		Entry:       entry,
		AgentField:  agentField,
		ArtifactURL: artifactURL,
		InputSchema: inputSchema,
	}
}

func defaultPublication(kind, nodeID, targetID string, tags []string, description string) Publication {
	return Publication{
		TargetKind:            kind,
		NodeID:                nodeID,
		TargetID:              targetID,
		DisplayName:           targetID,
		Description:           description,
		Tags:                  append([]string{}, tags...),
		Capabilities:          []string{},
		RepresentativeQueries: []string{},
		ArtifactType:          "application/openapi+json",
		ValidationStatus:      "private",
	}
}

func filterValueString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case bool:
		return fmt.Sprint(typed)
	case float64:
		return fmt.Sprint(typed)
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}

func requestedFacets(req ExploreRequest) []ExploreFacetRequest {
	if len(req.ResultType.Facets) == 0 {
		return []ExploreFacetRequest{
			{Field: "type", Limit: 20, MinCount: 1},
			{Field: "tags", Limit: 20, MinCount: 1},
			{Field: "capabilities", Limit: 20, MinCount: 1},
			{Field: "publisher", Limit: 20, MinCount: 1},
		}
	}
	facets := make([]ExploreFacetRequest, 0, len(req.ResultType.Facets))
	seen := map[string]struct{}{}
	for _, facet := range req.ResultType.Facets {
		field := strings.TrimSpace(facet.Field)
		if field == "" {
			continue
		}
		if _, ok := seen[field]; ok {
			continue
		}
		seen[field] = struct{}{}
		facet.Field = field
		facets = append(facets, facet)
	}
	return facets
}

func publicationStatus(pub Publication) string {
	if !pub.Published {
		return "private"
	}
	if pub.ValidationStatus == "valid" {
		return "published"
	}
	return "needs_setup"
}

func isCatalogVisible(pub PublicationView, cfg EffectiveConfig) bool {
	if !pub.Published || pub.ValidationStatus != "valid" {
		return false
	}
	if len(cfg.IncludeHealthStatuses) == 0 {
		return true
	}
	status := strings.ToLower(strings.TrimSpace(pub.AgentField.HealthStatus))
	if status == "" {
		status = "unknown"
	}
	for _, allowed := range cfg.IncludeHealthStatuses {
		if strings.ToLower(strings.TrimSpace(allowed)) == status {
			return true
		}
	}
	return false
}

func invocationTarget(kind, nodeID, targetID string) string {
	if kind == "skill" {
		return nodeID + ":skill:" + targetID
	}
	return nodeID + "." + targetID
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func trustManifest(cfg EffectiveConfig, didAvailable bool) *TrustManifest {
	if !didAvailable {
		return nil
	}
	identity := strings.TrimSpace(cfg.Identifier)
	identityType := "other"
	if strings.HasPrefix(identity, "did:") {
		identityType = "did"
	}
	if identity == "" && cfg.PublisherDomain != "" {
		identity = "https://" + cfg.PublisherDomain
		identityType = "https"
	}
	if identity == "" {
		return nil
	}
	return &TrustManifest{Identity: identity, IdentityType: identityType}
}

func nonEmpty(values []string) []string {
	out := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	return out
}

var urnPartPattern = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)

func sanitizeURNPart(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = urnPartPattern.ReplaceAllString(value, "-")
	value = strings.Trim(value, "-")
	if value == "" {
		return "unknown"
	}
	return value
}

func scoreEntry(entry CatalogEntry, query string) int {
	if query == "" {
		return 100
	}
	haystack := strings.ToLower(strings.Join([]string{
		entry.DisplayName,
		entry.Description,
		entry.Identifier,
		strings.Join(entry.Tags, " "),
		strings.Join(entry.Capabilities, " "),
		strings.Join(entry.RepresentativeQueries, " "),
	}, " "))
	score := 0
	for _, term := range strings.Fields(query) {
		if strings.Contains(strings.ToLower(entry.DisplayName), term) {
			score += 40
		}
		if strings.Contains(haystack, term) {
			score += 15
		}
	}
	if score > 100 {
		return 100
	}
	return score
}

func matchesFilter(entry CatalogEntry, filters map[string][]string) bool {
	for key, values := range filters {
		if len(values) == 0 {
			continue
		}
		if !entryMatchesAny(entry, key, values) {
			return false
		}
	}
	return true
}

func entryMatchesAny(entry CatalogEntry, key string, values []string) bool {
	for _, raw := range values {
		value := strings.ToLower(strings.TrimSpace(raw))
		if value == "" {
			continue
		}
		switch key {
		case "type":
			if strings.EqualFold(entry.Type, value) {
				return true
			}
		case "displayName":
			if strings.Contains(strings.ToLower(entry.DisplayName), value) {
				return true
			}
		case "publisher", "publisherId":
			if strings.EqualFold(publisherFromURN(entry.Identifier), value) {
				return true
			}
		case "tags":
			if containsFold(entry.Tags, value) {
				return true
			}
		case "capabilities":
			if containsFold(entry.Capabilities, value) {
				return true
			}
		default:
			return true
		}
	}
	return false
}

func containsFold(values []string, target string) bool {
	for _, value := range values {
		if strings.EqualFold(value, target) {
			return true
		}
	}
	return false
}

func publisherFromURN(identifier string) string {
	parts := strings.Split(identifier, ":")
	if len(parts) < 3 || parts[0] != "urn" || parts[1] != "ai" {
		return ""
	}
	return parts[2]
}

func facetBuckets(counts map[string]int, limit int, minCount int) []ExploreFacetBucket {
	if limit <= 0 {
		limit = 20
	}
	if minCount <= 0 {
		minCount = 1
	}
	buckets := make([]ExploreFacetBucket, 0, len(counts))
	for value, count := range counts {
		if count < minCount {
			continue
		}
		buckets = append(buckets, ExploreFacetBucket{Value: value, Count: count})
	}
	sort.Slice(buckets, func(i, j int) bool {
		if buckets[i].Count == buckets[j].Count {
			return buckets[i].Value < buckets[j].Value
		}
		return buckets[i].Count > buckets[j].Count
	})
	if len(buckets) > limit {
		return buckets[:limit]
	}
	return buckets
}

func schemaOrObject(schema json.RawMessage) any {
	if len(schema) > 0 && json.Valid(schema) {
		var out any
		if err := json.Unmarshal(schema, &out); err == nil {
			return out
		}
	}
	return map[string]any{"type": "object"}
}
