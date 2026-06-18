package handlers

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/ard"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/gin-gonic/gin"
)

type ARDHandler struct {
	Store        storage.StorageProvider
	ReadConfig   func() config.ARDConfig
	DIDAvailable func() bool
}

func NewARDHandler(store storage.StorageProvider, readConfig func() config.ARDConfig, didAvailable func() bool) *ARDHandler {
	return &ARDHandler{
		Store:        store,
		ReadConfig:   readConfig,
		DIDAvailable: didAvailable,
	}
}

func (h *ARDHandler) GetDashboard(c *gin.Context) {
	dashboard, err := ard.BuildDashboard(c.Request.Context(), h.Store, h.config(), h.didAvailable())
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	c.JSON(http.StatusOK, dashboard)
}

func (h *ARDHandler) UpdateSettings(c *gin.Context) {
	var settings ard.RuntimeSettings
	if err := c.ShouldBindJSON(&settings); err != nil {
		RespondBadRequest(c, "invalid ARD settings payload")
		return
	}
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	state.Settings = settings
	if err := ard.SaveState(c.Request.Context(), h.Store, state, requestActor(c)); err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	h.GetDashboard(c)
}

func (h *ARDHandler) SavePublication(c *gin.Context) {
	var pub ard.Publication
	if err := c.ShouldBindJSON(&pub); err != nil {
		RespondBadRequest(c, "invalid ARD publication payload")
		return
	}
	pub.TargetKind = normalizeTargetKind(pub.TargetKind)
	if pub.TargetKind == "" || strings.TrimSpace(pub.NodeID) == "" || strings.TrimSpace(pub.TargetID) == "" {
		RespondBadRequest(c, "target_kind, node_id, and target_id are required")
		return
	}
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	effective := ard.Effective(h.config(), state)
	pub.ValidationErrors = ard.ValidatePublication(pub, effective)
	pub.ValidationStatus = "valid"
	if len(pub.ValidationErrors) > 0 {
		pub.ValidationStatus = "invalid"
	}
	pub.LastValidatedAt = time.Now().UTC()
	pub.UpdatedAt = pub.LastValidatedAt
	key := ard.TargetKey(pub.TargetKind, pub.NodeID, pub.TargetID)
	state.Publications[key] = pub
	if err := ard.SaveState(c.Request.Context(), h.Store, state, requestActor(c)); err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	h.GetDashboard(c)
}

func (h *ARDHandler) SearchExternal(c *gin.Context) {
	cfg := h.config()
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	effective := ard.Effective(cfg, state)
	if !effective.ExternalSearchEnabled {
		RespondError(c, http.StatusForbidden, "external ARD search is disabled by config")
		return
	}
	var req ard.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "invalid ARD search payload")
		return
	}
	if req.PageSize <= 0 {
		req.PageSize = cfg.External.DefaultSearchLimit
	}
	c.JSON(http.StatusOK, ard.SearchExternal(c.Request.Context(), effective.AllowedRegistries, req))
}

func (h *ARDHandler) ImportExternal(c *gin.Context) {
	var req struct {
		SourceRegistry string           `json:"source_registry"`
		Entry          ard.CatalogEntry `json:"entry"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "invalid ARD import payload")
		return
	}
	if strings.TrimSpace(req.Entry.Identifier) == "" {
		RespondBadRequest(c, "entry.identifier is required")
		return
	}
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	entry := ard.ExternalEntry{
		ID:                    ard.ImportID(req.Entry.Identifier),
		SourceRegistry:        req.SourceRegistry,
		Identifier:            req.Entry.Identifier,
		Type:                  req.Entry.Type,
		DisplayName:           req.Entry.DisplayName,
		Description:           req.Entry.Description,
		URL:                   req.Entry.URL,
		Data:                  req.Entry.Data,
		Publisher:             publisherFromEntry(req.Entry),
		RepresentativeQueries: req.Entry.RepresentativeQueries,
		ImportedAt:            time.Now().UTC(),
	}
	if req.Entry.TrustManifest != nil {
		entry.TrustSummary = req.Entry.TrustManifest.Identity
	}
	upsertImport(&state, entry)
	if err := ard.SaveState(c.Request.Context(), h.Store, state, requestActor(c)); err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	h.GetDashboard(c)
}

func (h *ARDHandler) SaveExternalBinding(c *gin.Context) {
	entryID := strings.TrimSpace(c.Param("entryID"))
	var binding ard.ExternalBinding
	if err := c.ShouldBindJSON(&binding); err != nil {
		RespondBadRequest(c, "invalid ARD binding payload")
		return
	}
	if binding.ExternalEntryID == "" {
		binding.ExternalEntryID = entryID
	}
	if binding.ExternalEntryID != entryID {
		RespondBadRequest(c, "binding external_entry_id must match the URL entry id")
		return
	}
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	effective := ard.Effective(h.config(), state)
	if binding.Callable && !effective.ExternalInvocationEnabled {
		RespondError(c, http.StatusForbidden, "external ARD invocation is disabled by config")
		return
	}
	if !hasImport(state, entryID) {
		RespondNotFound(c, "imported ARD entry not found")
		return
	}
	if binding.TimeoutMS <= 0 {
		binding.TimeoutMS = 30000
	}
	if binding.Adapter == "" {
		binding.Adapter = "openapi"
	}
	binding.LocalTarget = strings.TrimSpace(binding.LocalTarget)
	binding.Adapter = strings.TrimSpace(binding.Adapter)
	binding.AuthRef = strings.TrimSpace(binding.AuthRef)
	binding.AllowedOperations = nonEmptyStrings(binding.AllowedOperations)
	if binding.Callable && !strings.HasPrefix(binding.LocalTarget, "external.") {
		RespondBadRequest(c, "callable ARD bindings require a local_target that starts with external.")
		return
	}
	if binding.Callable {
		for existingEntryID, existing := range state.Bindings {
			if existingEntryID == entryID || existing.ExternalEntryID == entryID {
				continue
			}
			if existing.Callable && strings.TrimSpace(existing.LocalTarget) == binding.LocalTarget {
				RespondBadRequest(c, "callable ARD binding local_target is already in use.")
				return
			}
		}
	}
	binding.UpdatedAt = time.Now().UTC()
	state.Bindings[entryID] = binding
	if err := ard.SaveState(c.Request.Context(), h.Store, state, requestActor(c)); err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	h.GetDashboard(c)
}

func (h *ARDHandler) SaveRegistries(c *gin.Context) {
	var req struct {
		Registries []ard.RegistryRecord `json:"registries"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "invalid ARD registry payload")
		return
	}
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	state.Registries = req.Registries
	if err := ard.SaveState(c.Request.Context(), h.Store, state, requestActor(c)); err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	h.GetDashboard(c)
}

func (h *ARDHandler) GetPublicCatalog(c *gin.Context) {
	catalog, _, ok := h.publicCatalog(c)
	if !ok {
		return
	}
	writePublicJSON(c, catalog)
}

func (h *ARDHandler) GetArtifact(c *gin.Context) {
	cfg := h.config()
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	effective := ard.Effective(cfg, state)
	if !effective.Enabled || !effective.PublishEnabled {
		RespondNotFound(c, "ARD catalog is not published")
		return
	}
	artifact, ok, err := ard.BuildArtifact(c.Request.Context(), h.Store, effective, state, c.Param("entryID"), h.didAvailable())
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	if !ok {
		RespondNotFound(c, "ARD artifact not found")
		return
	}
	writePublicJSON(c, artifact)
}

func (h *ARDHandler) SearchRegistry(c *gin.Context) {
	catalog, ok := h.publicRegistryCatalog(c)
	if !ok {
		return
	}
	var req ard.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "invalid ARD search payload")
		return
	}
	if strings.TrimSpace(req.Query.Text) == "" {
		RespondError(c, http.StatusBadRequest, "query.text is required")
		return
	}
	if req.PageSize <= 0 {
		req.PageSize = 10
	}
	if req.PageSize > 100 {
		req.PageSize = 100
	}
	source := h.registrySource(c)
	c.JSON(http.StatusOK, ard.SearchResponse{
		Results: ard.LocalSearch(catalog.Entries, req, source),
	})
}

func (h *ARDHandler) ListRegistryAgents(c *gin.Context) {
	catalog, ok := h.publicRegistryCatalog(c)
	if !ok {
		return
	}
	items, nextToken, ok := paginateCatalogEntries(c, catalog.Entries)
	if !ok {
		return
	}
	c.JSON(http.StatusOK, ard.AgentListResponse{Items: items, Total: len(catalog.Entries), PageToken: nextToken})
}

func (h *ARDHandler) ExploreRegistry(c *gin.Context) {
	catalog, ok := h.publicRegistryCatalog(c)
	if !ok {
		return
	}
	var req ard.ExploreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "invalid ARD explore payload")
		return
	}
	if len(req.ResultType.Facets) == 0 {
		RespondBadRequest(c, "resultType.facets is required")
		return
	}
	c.JSON(http.StatusOK, ard.Explore(catalog.Entries, req))
}

func (h *ARDHandler) publicCatalog(c *gin.Context) (ard.CatalogManifest, ard.EffectiveConfig, bool) {
	cfg := h.config()
	state, err := ard.LoadState(c.Request.Context(), h.Store)
	if err != nil {
		RespondInternalError(c, err.Error())
		return ard.CatalogManifest{}, ard.EffectiveConfig{}, false
	}
	effective := ard.Effective(cfg, state)
	if !effective.Enabled || !effective.PublishEnabled {
		RespondNotFound(c, "ARD catalog is not published")
		return ard.CatalogManifest{}, ard.EffectiveConfig{}, false
	}
	catalog, _, err := ard.BuildCatalog(c.Request.Context(), h.Store, effective, state, h.didAvailable())
	if err != nil {
		RespondInternalError(c, err.Error())
		return ard.CatalogManifest{}, ard.EffectiveConfig{}, false
	}
	return catalog, effective, true
}

func (h *ARDHandler) publicRegistryCatalog(c *gin.Context) (ard.CatalogManifest, bool) {
	catalog, effective, ok := h.publicCatalog(c)
	if !ok {
		return ard.CatalogManifest{}, false
	}
	if !effective.RegistryEnabled || !effective.RegistryPublic {
		RespondNotFound(c, "ARD registry is not public")
		return ard.CatalogManifest{}, false
	}
	return catalog, true
}

func (h *ARDHandler) config() config.ARDConfig {
	if h.ReadConfig == nil {
		return config.ARDConfig{}
	}
	return h.ReadConfig()
}

func (h *ARDHandler) didAvailable() bool {
	return h.DIDAvailable != nil && h.DIDAvailable()
}

func writePublicJSON(c *gin.Context, payload any) {
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Cache-Control", "public, max-age=60")
	c.JSON(http.StatusOK, payload)
}

func (h *ARDHandler) registrySource(c *gin.Context) string {
	if base := strings.TrimRight(h.config().PublicBaseURL, "/"); base != "" {
		return base + "/api/v1/ard"
	}
	scheme := "http"
	if c.Request.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + c.Request.Host + "/api/v1/ard"
}

func paginateCatalogEntries(c *gin.Context, entries []ard.CatalogEntry) ([]ard.CatalogEntry, string, bool) {
	pageSize := len(entries)
	if raw := strings.TrimSpace(c.Query("pageSize")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			RespondBadRequest(c, "pageSize must be a non-negative integer")
			return nil, "", false
		}
		if parsed > 0 {
			pageSize = parsed
		}
	}
	if pageSize > 100 {
		pageSize = 100
	}
	offset := 0
	if raw := strings.TrimSpace(c.Query("pageToken")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			RespondBadRequest(c, "pageToken must be a non-negative integer offset")
			return nil, "", false
		}
		offset = parsed
	}
	if offset >= len(entries) {
		return []ard.CatalogEntry{}, "", true
	}
	end := offset + pageSize
	if end > len(entries) {
		end = len(entries)
	}
	nextToken := ""
	if end < len(entries) {
		nextToken = strconv.Itoa(end)
	}
	return entries[offset:end], nextToken, true
}

func publisherFromEntry(entry ard.CatalogEntry) string {
	parts := strings.Split(entry.Identifier, ":")
	if len(parts) >= 3 && parts[0] == "urn" && parts[1] == "ai" {
		return parts[2]
	}
	return ""
}

func requestActor(c *gin.Context) string {
	if actor := strings.TrimSpace(c.GetHeader("X-Caller-Agent-ID")); actor != "" {
		return actor
	}
	if actor := strings.TrimSpace(c.GetHeader("X-Agent-Node-ID")); actor != "" {
		return actor
	}
	return "ui"
}

func normalizeTargetKind(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "skill":
		return "skill"
	case "reasoner":
		return "reasoner"
	default:
		return ""
	}
}

func upsertImport(state *ard.State, entry ard.ExternalEntry) {
	for i, existing := range state.Imports {
		if existing.ID == entry.ID {
			if entry.ImportedAt.IsZero() {
				entry.ImportedAt = existing.ImportedAt
			}
			state.Imports[i] = entry
			return
		}
	}
	state.Imports = append(state.Imports, entry)
}

func hasImport(state ard.State, id string) bool {
	for _, entry := range state.Imports {
		if entry.ID == id {
			return true
		}
	}
	return false
}

func nonEmptyStrings(values []string) []string {
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
