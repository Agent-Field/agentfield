package ard

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type testStore struct {
	agents  []*types.AgentNode
	config  map[string]string
	getErr  error
	setErr  error
	listErr error
}

func (s *testStore) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.agents, nil
}

func (s *testStore) GetConfig(_ context.Context, key string) (*storage.ConfigEntry, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	if s.config == nil {
		return nil, nil
	}
	value, ok := s.config[key]
	if !ok {
		return nil, nil
	}
	return &storage.ConfigEntry{Key: key, Value: value}, nil
}

func (s *testStore) SetConfig(_ context.Context, key string, value string, _ string) error {
	if s.setErr != nil {
		return s.setErr
	}
	if s.config == nil {
		s.config = map[string]string{}
	}
	s.config[key] = value
	return nil
}

func TestBuildCatalogProducesSchemaCleanManifest(t *testing.T) {
	ctx := context.Background()
	store := &testStore{
		agents: []*types.AgentNode{{
			ID:            "node-1",
			Version:       "1.2.3",
			HealthStatus:  types.HealthStatusActive,
			LastHeartbeat: time.Now().UTC(),
			Reasoners: []types.ReasonerDefinition{{
				ID:          "review_contract",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
				Tags:        []string{"legal"},
			}},
		}},
	}
	state := defaultState()
	state.Publications[TargetKey("reasoner", "node-1", "review_contract")] = Publication{
		TargetKind:            "reasoner",
		NodeID:                "node-1",
		TargetID:              "review_contract",
		Published:             true,
		DisplayName:           "Review Contract",
		Description:           "Review contract language for risk and missing clauses.",
		Tags:                  []string{"legal"},
		Capabilities:          []string{"ContractReview"},
		RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
		ArtifactType:          "application/openapi+json",
	}
	cfg := Effective(config.ARDConfig{
		Enabled:         true,
		PublicBaseURL:   "https://cp.example.com",
		PublisherDomain: "example.com",
		Host: config.ARDHostConfig{
			DisplayName:      "Example Control Plane",
			Identifier:       "did:web:example.com",
			DocumentationURL: "https://docs.example.com",
			LogoURL:          "https://docs.example.com/logo.png",
		},
		Publish: config.ARDPublishConfig{
			Enabled:               true,
			IncludeHealthStatuses: []string{"active"},
		},
	}, state)

	catalog, publications, err := BuildCatalog(ctx, store, cfg, state, true)
	if err != nil {
		t.Fatalf("BuildCatalog returned error: %v", err)
	}
	if len(catalog.Entries) != 1 {
		t.Fatalf("expected one published entry, got %d", len(catalog.Entries))
	}
	entry := catalog.Entries[0]
	if entry.URL == "" || len(entry.Data) != 0 {
		t.Fatalf("expected URL-only catalog entry, got url=%q data=%s", entry.URL, string(entry.Data))
	}

	raw, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("manifest did not marshal: %v", err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatalf("manifest did not unmarshal: %v", err)
	}
	for _, forbidden := range []string{"generatedAt"} {
		if _, ok := manifest[forbidden]; ok {
			t.Fatalf("manifest includes non-schema field %q: %s", forbidden, string(raw))
		}
	}
	host := manifest["host"].(map[string]any)
	for _, forbidden := range []string{"publisherDomain"} {
		if _, ok := host[forbidden]; ok {
			t.Fatalf("host includes non-schema field %q: %s", forbidden, string(raw))
		}
	}
	publicEntry := manifest["entries"].([]any)[0].(map[string]any)
	for _, forbidden := range []string{"publisher", "agentfield"} {
		if _, ok := publicEntry[forbidden]; ok {
			t.Fatalf("entry includes non-schema field %q: %s", forbidden, string(raw))
		}
	}
	if _, ok := publicEntry["url"]; !ok {
		t.Fatalf("entry missing url: %s", string(raw))
	}
	if _, ok := publicEntry["data"]; ok {
		t.Fatalf("entry includes both url and data: %s", string(raw))
	}
	if len(publications) != 1 || publications[0].AgentField.InvocationTarget != "node-1.review_contract" {
		t.Fatalf("expected internal invocation metadata to stay on publication view, got %#v", publications)
	}
}

func TestLocalSearchAndExploreUseRegistryProtocolShape(t *testing.T) {
	entries := []CatalogEntry{{
		Identifier:            "urn:ai:example.com:agentfield:node-1:reasoner:review_contract",
		DisplayName:           "Review Contract",
		Description:           "Review contract language.",
		Type:                  "application/openapi+json",
		URL:                   "https://cp.example.com/api/v1/ard/artifacts/node-1.review_contract",
		Tags:                  []string{"legal"},
		Capabilities:          []string{"ContractReview"},
		RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
	}}
	req := SearchRequest{
		Query:    QueryModel{Text: "review contract", Filter: map[string][]string{"type": {"application/openapi+json"}}},
		PageSize: 5,
	}
	results := LocalSearch(entries, req, "https://cp.example.com/api/v1/ard")
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	resultJSON, err := json.Marshal(results[0])
	if err != nil {
		t.Fatalf("result did not marshal: %v", err)
	}
	var result map[string]any
	if err := json.Unmarshal(resultJSON, &result); err != nil {
		t.Fatalf("result did not unmarshal: %v", err)
	}
	for _, key := range []string{"identifier", "displayName", "type", "url", "score", "source"} {
		if _, ok := result[key]; !ok {
			t.Fatalf("search result missing %q: %s", key, string(resultJSON))
		}
	}
	if _, ok := result["entry"]; ok {
		t.Fatalf("search result should not wrap catalog entry in entry key: %s", string(resultJSON))
	}

	explore := Explore(entries, ExploreRequest{
		Query: QueryModel{Text: "contract"},
		ResultType: ExploreResultType{Facets: []ExploreFacetRequest{
			{Field: "type", Limit: 5, MinCount: 1},
		}},
	})
	buckets := explore.Facets["type"].Buckets
	if explore.ResultType != "facets" || len(buckets) != 1 || len(explore.Facets) != 1 {
		t.Fatalf("unexpected explore response: %#v", explore)
	}
	if buckets[0].Value != "application/openapi+json" || buckets[0].Count != 1 {
		t.Fatalf("unexpected type bucket: %#v", buckets[0])
	}
}

func TestValidatePublicationRepresentativeQueries(t *testing.T) {
	cfg := Effective(config.ARDConfig{
		Enabled:         true,
		PublicBaseURL:   "https://cp.example.com",
		PublisherDomain: "example.com",
		Publish:         config.ARDPublishConfig{Enabled: true},
	}, defaultState())
	pub := Publication{
		TargetKind:            "reasoner",
		NodeID:                "node-1",
		TargetID:              "review_contract",
		Published:             true,
		DisplayName:           "Review Contract",
		Description:           "Review contract language.",
		RepresentativeQueries: []string{"review this MSA"},
		ArtifactType:          "application/openapi+json",
	}
	errs := ValidatePublication(pub, cfg)
	if len(errs) == 0 {
		t.Fatal("expected representative query validation error")
	}
	pub.RepresentativeQueries = []string{"review this MSA", "find risky indemnity clauses"}
	pub.ArtifactType = "application/a2a-agent-card+json"
	errs = ValidatePublication(pub, cfg)
	var hasArtifactURLError bool
	for _, err := range errs {
		if err == "artifact URL override is required for non-OpenAPI artifact types" {
			hasArtifactURLError = true
		}
	}
	if !hasArtifactURLError {
		t.Fatalf("expected non-OpenAPI generated artifact validation error, got %#v", errs)
	}
	pub.ArtifactURLOverride = "https://vendor.example/agent-card.json"
	if errs := ValidatePublication(pub, cfg); len(errs) != 0 {
		t.Fatalf("expected explicit non-OpenAPI artifact URL to validate, got %#v", errs)
	}
}

func TestBuildArtifactHonorsCatalogVisibility(t *testing.T) {
	ctx := context.Background()
	store := &testStore{
		agents: []*types.AgentNode{{
			ID:           "node-1",
			Version:      "1.2.3",
			HealthStatus: types.HealthStatusInactive,
			Reasoners: []types.ReasonerDefinition{{
				ID: "review_contract",
			}},
		}},
	}
	state := defaultState()
	state.Publications[TargetKey("reasoner", "node-1", "review_contract")] = Publication{
		TargetKind:            "reasoner",
		NodeID:                "node-1",
		TargetID:              "review_contract",
		Published:             true,
		DisplayName:           "Review Contract",
		Description:           "Review contract language.",
		RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
		ArtifactType:          "application/openapi+json",
	}
	cfg := Effective(config.ARDConfig{
		Enabled:         true,
		PublicBaseURL:   "https://cp.example.com",
		PublisherDomain: "example.com",
		Publish: config.ARDPublishConfig{
			Enabled:               true,
			IncludeHealthStatuses: []string{"active"},
		},
	}, state)

	artifact, ok, err := BuildArtifact(ctx, store, cfg, state, "node-1.review_contract", false)
	if err != nil {
		t.Fatalf("BuildArtifact returned error: %v", err)
	}
	if ok || artifact != nil {
		t.Fatalf("expected inactive publication artifact to be hidden, got ok=%v artifact=%#v", ok, artifact)
	}
}

func TestQueryModelAcceptsScalarFilterValues(t *testing.T) {
	var req SearchRequest
	if err := json.Unmarshal([]byte(`{"query":{"text":"review","filter":{"type":"application/openapi+json","tags":["legal","contracts"],"active":true}}}`), &req); err != nil {
		t.Fatalf("expected scalar filters to decode: %v", err)
	}
	if got := req.Query.Filter["type"]; len(got) != 1 || got[0] != "application/openapi+json" {
		t.Fatalf("unexpected type filter: %#v", got)
	}
	if got := req.Query.Filter["tags"]; len(got) != 2 || got[0] != "legal" || got[1] != "contracts" {
		t.Fatalf("unexpected tags filter: %#v", got)
	}
	if got := req.Query.Filter["active"]; len(got) != 1 || got[0] != "true" {
		t.Fatalf("unexpected boolean filter: %#v", got)
	}
}

func TestSearchExternalBlocksCrossHostRedirects(t *testing.T) {
	ctx := context.Background()
	redirectHit := false
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirectHit = true
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer redirectTarget.Close()

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, redirectTarget.URL+"/search", http.StatusFound)
	}))
	defer registry.Close()

	resp := SearchExternal(ctx, []string{registry.URL}, SearchRequest{
		Query:    QueryModel{Text: "review"},
		PageSize: 5,
	})

	if redirectHit {
		t.Fatal("cross-host redirect target was called")
	}
	if len(resp.Sources) != 1 || resp.Sources[0].Status != "error" {
		t.Fatalf("expected redirect to be reported as source error, got %#v", resp.Sources)
	}
	if len(resp.Results) != 0 {
		t.Fatalf("expected no results from blocked redirect, got %#v", resp.Results)
	}
}

func TestStateEffectiveDashboardAndArtifactFlow(t *testing.T) {
	ctx := context.Background()
	enabled := true
	publish := true
	registryPublic := true
	store := &testStore{
		agents: []*types.AgentNode{{
			ID:            "Node One",
			Version:       "2.0.0",
			HealthStatus:  types.HealthStatusUnknown,
			LastHeartbeat: time.Now().UTC(),
			Reasoners: []types.ReasonerDefinition{{
				ID:          "Review Contract",
				Tags:        []string{" legal ", "legal", ""},
				InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
			}},
			Skills: []types.SkillDefinition{{
				ID:          "summarize",
				Tags:        []string{"text"},
				InputSchema: json.RawMessage(`not-json`),
			}},
		}},
		config: map[string]string{},
	}
	state := State{
		Settings: RuntimeSettings{
			Enabled:          &enabled,
			PublishEnabled:   &publish,
			RegistryPublic:   &registryPublic,
			PublicBaseURL:    " https://runtime.example.com/ ",
			PublisherDomain:  "Runtime Example",
			DisplayName:      "Runtime CP",
			DocumentationURL: "https://docs.runtime.example.com",
			LogoURL:          "https://docs.runtime.example.com/logo.png",
		},
		Publications: map[string]Publication{
			TargetKey("reasoner", "Node One", "Review Contract"): {
				TargetKind:            "reasoner",
				NodeID:                "Node One",
				TargetID:              "Review Contract",
				Published:             true,
				DisplayName:           "Review Contract",
				Description:           "Review contract language.",
				Tags:                  []string{"legal", "risk"},
				Capabilities:          []string{"ContractReview"},
				RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
				ArtifactType:          "application/openapi+json",
			},
			TargetKey("skill", "Node One", "summarize"): {
				TargetKind:            "skill",
				NodeID:                "Node One",
				TargetID:              "summarize",
				Published:             true,
				DisplayName:           "Summarize",
				Description:           "Summarize input text.",
				RepresentativeQueries: []string{"summarize this note", "make this shorter"},
				ArtifactType:          "application/openapi+json",
				ArtifactURLOverride:   "https://override.example.com/openapi.json",
			},
		},
		Imports: []ExternalEntry{{
			ID:          "ext_1",
			Identifier:  "urn:ai:vendor.example:agent:review",
			Type:        "application/a2a-agent-card+json",
			DisplayName: "Vendor Review",
			ImportedAt:  time.Now().UTC(),
		}},
		Bindings: map[string]ExternalBinding{
			"ext_1": {ExternalEntryID: "ext_1", Callable: true, Adapter: "a2a", TimeoutMS: 1000},
		},
		Registries: []RegistryRecord{{URL: "https://registry.example.com/api/v1/ard", Name: "Registry"}},
	}
	if err := SaveState(ctx, store, state, "tester"); err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	loaded, err := LoadState(ctx, store)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if loaded.LastUpdatedBy != "tester" || loaded.UpdatedAt.IsZero() {
		t.Fatalf("state audit fields were not saved: %#v", loaded)
	}

	cfg := config.ARDConfig{
		Enabled:         true,
		PublicBaseURL:   "https://config.example.com/",
		PublisherDomain: "config.example.com",
		Host: config.ARDHostConfig{
			DisplayName: "Config CP",
			Identifier:  "https://identity.example.com",
		},
		Publish: config.ARDPublishConfig{
			Enabled:               true,
			IncludeHealthStatuses: []string{"unknown"},
		},
		Registry: config.ARDRegistryConfig{Enabled: true, Public: true},
		External: config.ARDExternalConfig{
			SearchEnabled:     true,
			InvocationEnabled: true,
			AllowedRegistries: []string{"https://registry.example.com/api/v1/ard"},
		},
	}
	effective := Effective(cfg, loaded)
	if !effective.Enabled || !effective.PublishEnabled || effective.PublicBaseURL != "https://config.example.com" {
		t.Fatalf("unexpected effective config: %#v", effective)
	}
	if !effective.Locked["public_base_url"] || !effective.Locked["publisher_domain"] || !effective.Locked["display_name"] {
		t.Fatalf("expected config-backed fields to be locked: %#v", effective.Locked)
	}

	dashboard, err := BuildDashboard(ctx, store, cfg, true)
	if err != nil {
		t.Fatalf("BuildDashboard: %v", err)
	}
	if !dashboard.Summary.ARDEnabled || dashboard.Summary.PublishedReasoners != 1 || dashboard.Summary.PublishedSkills != 1 {
		t.Fatalf("unexpected dashboard summary: %#v", dashboard.Summary)
	}
	if dashboard.Summary.CallableExternalResources != 1 || len(dashboard.Imports) != 1 || dashboard.Imports[0].Status != "callable" {
		t.Fatalf("unexpected import summary: %#v", dashboard.Imports)
	}
	if len(dashboard.Catalog.Entries) != 2 {
		t.Fatalf("expected reasoner and skill entries, got %#v", dashboard.Catalog.Entries)
	}
	if dashboard.Catalog.Entries[0].TrustManifest == nil || dashboard.Catalog.Entries[0].TrustManifest.IdentityType != "other" {
		t.Fatalf("expected https trust manifest, got %#v", dashboard.Catalog.Entries[0].TrustManifest)
	}

	artifact, ok, err := BuildArtifact(ctx, store, dashboard.Config, loaded, TargetKey("reasoner", "Node One", "Review Contract"), true)
	if err != nil || !ok {
		t.Fatalf("BuildArtifact ok=%v err=%v artifact=%#v", ok, err, artifact)
	}
	paths := artifact["paths"].(map[string]any)
	if _, ok := paths["/api/v1/execute/Node One.Review Contract"]; !ok {
		t.Fatalf("artifact missing execution path: %#v", paths)
	}
	if got := ImportID("urn:ai:vendor.example:agent:review"); got == "" || got == ImportID("different") {
		t.Fatalf("ImportID should be deterministic and identifier-specific")
	}
}

func TestARDSearchExploreAndExternalBranches(t *testing.T) {
	entries := []CatalogEntry{
		{
			Identifier:            "urn:ai:alpha.example:agentfield:node:reasoner:review",
			DisplayName:           "Contract Review",
			Description:           "Review legal contracts.",
			Type:                  "application/openapi+json",
			Tags:                  []string{"legal", "contracts"},
			Capabilities:          []string{"ContractReview"},
			RepresentativeQueries: []string{"review contract", "find clause risk"},
		},
		{
			Identifier:            "urn:ai:beta.example:agentfield:node:skill:summarize",
			DisplayName:           "Summarize",
			Description:           "Summarize notes.",
			Type:                  "application/a2a-agent-card+json",
			Tags:                  []string{"text"},
			Capabilities:          []string{"Summarization"},
			RepresentativeQueries: []string{"summarize this", "make it shorter"},
		},
	}
	results := LocalSearch(entries, SearchRequest{
		Query: QueryModel{
			Text: "contract review",
			Filter: map[string][]string{
				"type":         {"application/openapi+json"},
				"displayName":  {"contract"},
				"publisher":    {"alpha.example"},
				"tags":         {"legal"},
				"capabilities": {"ContractReview"},
			},
		},
		PageSize: 1,
	}, "local")
	if len(results) != 1 || results[0].DisplayName != "Contract Review" || results[0].Score <= 0 {
		t.Fatalf("unexpected local search results: %#v", results)
	}
	if got := LocalSearch(entries, SearchRequest{Query: QueryModel{Filter: map[string][]string{"unknown": {"anything"}}}}, "local"); len(got) != 2 {
		t.Fatalf("unknown filters should not remove entries, got %#v", got)
	}

	explore := Explore(entries, ExploreRequest{
		Query: QueryModel{Filter: map[string][]string{"tags": {"legal"}}},
		ResultType: ExploreResultType{Facets: []ExploreFacetRequest{
			{Field: "tags", Limit: 1},
			{Field: "tags", Limit: 3},
			{Field: "publisher", MinCount: 1},
			{Field: "capabilities", Limit: 10},
			{Field: ""},
		}},
	})
	if len(explore.Facets) != 3 || explore.Facets["tags"].Buckets[0].Value != "contracts" {
		t.Fatalf("unexpected explore facets: %#v", explore.Facets)
	}
	if defaults := Explore(entries, ExploreRequest{}); len(defaults.Facets) != 4 {
		t.Fatalf("default explore should include four facets, got %#v", defaults.Facets)
	}

	var okRegistryRequest SearchRequest
	okRegistry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search" {
			t.Fatalf("unexpected search path %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&okRegistryRequest); err != nil {
			t.Fatalf("decode forwarded request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"identifier":"urn:ai:vendor.example:agent:review","displayName":"Vendor Review","type":"application/openapi+json","url":"https://vendor.example/openapi.json","score":0.92}]}`))
	}))
	defer okRegistry.Close()
	badRegistry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusTeapot)
	}))
	defer badRegistry.Close()
	malformedRegistry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{`))
	}))
	defer malformedRegistry.Close()

	external := SearchExternal(context.Background(), []string{okRegistry.URL, badRegistry.URL, malformedRegistry.URL}, SearchRequest{
		Query:      QueryModel{Text: "review"},
		Registries: []string{okRegistry.URL, badRegistry.URL, malformedRegistry.URL, "https://blocked.example.com/api/v1/ard"},
		PageSize:   1,
	})
	if len(external.Results) != 1 || external.Results[0].Source != okRegistry.URL {
		t.Fatalf("expected one result from ok registry, got %#v", external)
	}
	if external.Results[0].Score != 0.92 {
		t.Fatalf("expected fractional external score to decode, got %#v", external.Results[0].Score)
	}
	statuses := map[string]string{}
	for _, source := range external.Sources {
		statuses[source.URL] = source.Status
	}
	if statuses[okRegistry.URL] != "ok" || statuses[badRegistry.URL] != "error" || statuses[malformedRegistry.URL] != "error" || statuses["https://blocked.example.com/api/v1/ard"] != "blocked" {
		t.Fatalf("unexpected source statuses: %#v", external.Sources)
	}
	if okRegistryRequest.Federation != "none" || len(okRegistryRequest.Registries) != 0 {
		t.Fatalf("forwarded request should suppress federation and caller registry list, got %#v", okRegistryRequest)
	}
}

func TestARDStateAndErrorBranches(t *testing.T) {
	ctx := context.Background()
	if _, err := LoadState(ctx, &testStore{getErr: errors.New("get failed")}); err == nil {
		t.Fatal("expected LoadState to surface storage errors")
	}
	empty, err := LoadState(ctx, &testStore{config: map[string]string{StateConfigKey: " "}})
	if err != nil || len(empty.Publications) != 0 || len(empty.Imports) != 0 {
		t.Fatalf("expected empty state defaults, got state=%#v err=%v", empty, err)
	}
	if _, err := LoadState(ctx, &testStore{config: map[string]string{StateConfigKey: "{"}}); err == nil {
		t.Fatal("expected malformed state JSON to fail")
	}
	if err := SaveState(ctx, &testStore{setErr: errors.New("set failed")}, State{}, "tester"); err == nil {
		t.Fatal("expected SaveState to surface storage errors")
	}
	if _, err := BuildDashboard(ctx, &testStore{getErr: errors.New("get failed")}, config.ARDConfig{}, false); err == nil {
		t.Fatal("expected dashboard load error")
	}
	if _, err := BuildDashboard(ctx, &testStore{listErr: errors.New("list failed")}, config.ARDConfig{}, false); err == nil {
		t.Fatal("expected dashboard catalog error")
	}
	dashboard, err := BuildDashboard(ctx, &testStore{}, config.ARDConfig{Enabled: true, Publish: config.ARDPublishConfig{Enabled: true}}, false)
	if err != nil {
		t.Fatalf("BuildDashboard default URL: %v", err)
	}
	if dashboard.Summary.CatalogURL != "/.well-known/ai-catalog.json" {
		t.Fatalf("expected relative catalog URL, got %q", dashboard.Summary.CatalogURL)
	}
}

func TestARDDefaultPublicationsValidationAndSearchBranches(t *testing.T) {
	ctx := context.Background()
	store := &testStore{
		agents: []*types.AgentNode{{
			ID:           "node-1",
			Version:      "1.0.0",
			HealthStatus: types.HealthStatusActive,
			Reasoners: []types.ReasonerDefinition{{
				ID:   "review",
				Tags: []string{"legal"},
			}},
			Skills: []types.SkillDefinition{{
				ID:   "summarize",
				Tags: []string{"text"},
			}},
		}},
	}
	cfg := Effective(config.ARDConfig{
		Enabled:         true,
		PublicBaseURL:   "https://cp.example.com",
		PublisherDomain: "example.com",
		Publish: config.ARDPublishConfig{
			Enabled:               true,
			DefaultType:           "application/mcp-server+json",
			IncludeHealthStatuses: []string{"active"},
		},
	}, State{})
	catalog, publications, err := BuildCatalog(ctx, store, cfg, State{}, false)
	if err != nil {
		t.Fatalf("BuildCatalog: %v", err)
	}
	if len(publications) != 2 || len(catalog.Entries) != 0 {
		t.Fatalf("expected private default publications only, catalog=%#v publications=%#v", catalog.Entries, publications)
	}
	if publications[0].TargetKind == "" || publications[0].NodeID == "" || publications[0].ArtifactType == "" {
		t.Fatalf("default publication fields were not filled: %#v", publications[0])
	}
	if publications[0].ArtifactType != "application/mcp-server+json" || publications[1].ArtifactType != "application/mcp-server+json" {
		t.Fatalf("default publication artifact type did not honor config: %#v", publications)
	}
	disabledCfg := Effective(config.ARDConfig{}, State{})
	errs := ValidatePublication(Publication{Published: true, RepresentativeQueries: []string{"one", "two", "three", "four", "five", "six"}}, disabledCfg)
	for _, want := range []string{
		"ARD is disabled",
		"catalog publishing is disabled",
		"public base URL is required",
		"publisher domain is required",
		"display name is required",
		"description is required",
		"at most five representative queries are allowed",
		"artifact type is required",
	} {
		found := false
		for _, got := range errs {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing validation error %q in %#v", want, errs)
		}
	}
	if _, _, err := BuildArtifact(ctx, &testStore{listErr: errors.New("list failed")}, cfg, State{}, "missing", false); err == nil {
		t.Fatal("expected BuildArtifact to surface catalog errors")
	}

	entries := []CatalogEntry{
		{Identifier: "urn:ai:alpha.example:agent:a", DisplayName: "Alpha Review", Description: "Review", Type: "application/openapi+json", Tags: []string{"legal"}},
		{Identifier: "urn:ai:beta.example:agent:b", DisplayName: "Beta Review", Description: "Review", Type: "application/openapi+json", Tags: []string{"finance"}},
	}
	limited := LocalSearch(entries, SearchRequest{Query: QueryModel{Text: "review"}, PageSize: 1}, "local")
	if len(limited) != 1 {
		t.Fatalf("expected local search page size limit, got %#v", limited)
	}
	explore := Explore(entries, ExploreRequest{Query: QueryModel{Text: "missing"}, ResultType: ExploreResultType{Facets: []ExploreFacetRequest{{Field: "tags"}}}})
	if len(explore.Facets["tags"].Buckets) != 0 {
		t.Fatalf("expected query miss to remove facet buckets: %#v", explore.Facets)
	}
}

func TestQueryModelRejectsInvalidFilters(t *testing.T) {
	for _, raw := range []string{
		`{"query":{"filter":{"bad":{`,
		`{"query":{"filter":{"empty":["",null]}}}`,
	} {
		var req SearchRequest
		err := json.Unmarshal([]byte(raw), &req)
		if raw == `{"query":{"filter":{"bad":{` {
			if err == nil {
				t.Fatal("expected invalid filter JSON to fail")
			}
			continue
		}
		if err != nil {
			t.Fatalf("empty filters should decode and normalize away: %v", err)
		}
		if req.Query.Filter != nil {
			t.Fatalf("expected empty filters to normalize to nil, got %#v", req.Query.Filter)
		}
	}
}

func TestSearchExternalNetworkAndLimitBranches(t *testing.T) {
	okRegistry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"identifier":"urn:ai:one","displayName":"One","type":"application/openapi+json","score":10},{"identifier":"urn:ai:two","displayName":"Two","type":"application/openapi+json","score":9}]}`))
	}))
	defer okRegistry.Close()

	resp := SearchExternal(context.Background(), []string{"://bad-url", okRegistry.URL}, SearchRequest{
		Query:      QueryModel{Text: "review"},
		Registries: []string{"://bad-url", "http://127.0.0.1:1", okRegistry.URL},
		PageSize:   1,
	})
	if len(resp.Results) != 1 {
		t.Fatalf("expected external results to be page-size limited, got %#v", resp.Results)
	}
	statuses := map[string]string{}
	for _, source := range resp.Sources {
		statuses[source.URL] = source.Status
	}
	if statuses["://bad-url"] != "error" || statuses["http://127.0.0.1:1"] != "blocked" || statuses[okRegistry.URL] != "ok" {
		t.Fatalf("unexpected source statuses: %#v", resp.Sources)
	}
}
