package ui

import (
	"encoding/json"
	"net/http"
	"regexp"
	"sort"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

const (
	maxAgentSecretValueBytes = 32 * 1024
	globalSecretScope        = "global"
)

var agentSecretKeyPattern = regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)

// AgentSecretsHandler manages the encrypted secret store used when agents start.
type AgentSecretsHandler struct {
	storage        storage.StorageProvider
	agentfieldHome string
}

// NewAgentSecretsHandler creates an AgentSecretsHandler.
func NewAgentSecretsHandler(storage storage.StorageProvider, agentfieldHome string) *AgentSecretsHandler {
	return &AgentSecretsHandler{storage: storage, agentfieldHome: agentfieldHome}
}

type agentSecretStatus struct {
	Key   string `json:"key"`
	IsSet bool   `json:"is_set"`
	// Scope reports where the stored value lives ("node" or "global");
	// empty when the key is not set anywhere.
	Scope            string `json:"scope,omitempty"`
	DeclaredScope    string `json:"declared_scope,omitempty"`
	Description      string `json:"description,omitempty"`
	Secret           bool   `json:"secret,omitempty"`
	Default          string `json:"default,omitempty"`
	Requirement      string `json:"requirement,omitempty"`
	Group            string `json:"group,omitempty"`
	GroupDescription string `json:"group_description,omitempty"`
}

type declaredAgentEnvironmentVar struct {
	Name             string `json:"name"`
	Description      string `json:"description"`
	Type             string `json:"type"`
	Scope            string `json:"scope"`
	Default          string `json:"default"`
	Requirement      string `json:"-"`
	Group            string `json:"-"`
	GroupDescription string `json:"-"`
}

type setAgentSecretRequest struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	// Scope optionally forces "node" or "global". When empty the manifest's
	// declared scope for the key wins, defaulting to global — the same rule
	// `af secrets set` and the desktop app follow.
	Scope string `json:"scope"`
}

// ListAgentSecretsHandler lists secret names and whether each resolves for
// this agent. Resolution mirrors the runner (EnvResolver): node scope first,
// then global. Undeclared node-scoped keys are included because the runner
// injects them; undeclared global keys are not injected, so they are omitted.
func (h *AgentSecretsHandler) ListAgentSecretsHandler(c *gin.Context) {
	agentPackage, ok := h.resolveAgentPackage(c)
	if !ok {
		return
	}

	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	nodeKeys, err := store.List(agentSecretScope(agentPackage))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list secrets"})
		return
	}
	globalKeys, err := store.List(globalSecretScope)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list secrets"})
		return
	}

	inNode := make(map[string]bool, len(nodeKeys))
	for _, key := range nodeKeys {
		inNode[key] = true
	}
	inGlobal := make(map[string]bool, len(globalKeys))
	for _, key := range globalKeys {
		inGlobal[key] = true
	}

	includeEnvironment := c.Query("include") == "env"
	declared := declaredAgentSecrets(agentPackage.ConfigurationSchema)
	listed := make(map[string]struct{})
	for key := range declared {
		listed[key] = struct{}{}
	}
	var environment map[string]declaredAgentEnvironmentVar
	if includeEnvironment {
		environment = declaredAgentEnvironment(agentPackage.ConfigurationSchema)
		for key := range environment {
			listed[key] = struct{}{}
		}
	}
	for _, key := range nodeKeys {
		listed[key] = struct{}{}
	}

	keys := make([]string, 0, len(listed))
	for key := range listed {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	secrets := make([]agentSecretStatus, 0, len(keys))
	for _, key := range keys {
		status := agentSecretStatus{Key: key}
		switch {
		case inNode[key]:
			status.IsSet = true
			status.Scope = "node"
		case inGlobal[key]:
			status.IsSet = true
			status.Scope = globalSecretScope
		}
		if includeEnvironment {
			if variable, ok := environment[key]; ok {
				status.DeclaredScope = variable.Scope
				status.Description = variable.Description
				status.Secret = variable.Type == "secret"
				status.Default = variable.Default
				status.Requirement = variable.Requirement
				status.Group = variable.Group
				status.GroupDescription = variable.GroupDescription
			}
		}
		secrets = append(secrets, status)
	}
	c.JSON(http.StatusOK, gin.H{"secrets": secrets})
}

// SetAgentSecretHandler stores one secret in the scope selected by the
// request, the manifest declaration, or the global default, in that order.
func (h *AgentSecretsHandler) SetAgentSecretHandler(c *gin.Context) {
	agentPackage, ok := h.resolveAgentPackage(c)
	if !ok {
		return
	}

	var req setAgentSecretRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}
	if !agentSecretKeyPattern.MatchString(req.Key) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid secret key"})
		return
	}
	if req.Value == "" || len([]byte(req.Value)) > maxAgentSecretValueBytes {
		c.JSON(http.StatusBadRequest, gin.H{"error": "secret value must be non-empty and at most 32KiB"})
		return
	}
	scope, ok := h.selectScope(c, agentPackage, req.Key, req.Scope)
	if !ok {
		return
	}

	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	if err := store.Set(scope, req.Key, req.Value); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to store secret"})
		return
	}
	c.Status(http.StatusNoContent)
}

// DeleteAgentSecretHandler deletes one secret. The scope defaults exactly as
// SetAgentSecretHandler's does and can be forced with ?scope=node|global.
func (h *AgentSecretsHandler) DeleteAgentSecretHandler(c *gin.Context) {
	agentPackage, ok := h.resolveAgentPackage(c)
	if !ok {
		return
	}
	key := c.Param("key")
	scope, ok := h.selectScope(c, agentPackage, key, c.Query("scope"))
	if !ok {
		return
	}

	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	if err := store.Delete(scope, key); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to delete secret"})
		return
	}
	c.Status(http.StatusNoContent)
}

// ListAllSecretsHandler lists every stored secret reference (key + scope)
// across the global scope and all node scopes. Values are never returned.
// This backs store-wide management UIs, mirroring `af secrets ls`.
func (h *AgentSecretsHandler) ListAllSecretsHandler(c *gin.Context) {
	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	refs, err := store.ListAll()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list secrets"})
		return
	}
	secrets := make([]gin.H, 0, len(refs))
	for _, ref := range refs {
		secrets = append(secrets, gin.H{"key": ref.Key, "scope": ref.Scope})
	}
	c.JSON(http.StatusOK, gin.H{"secrets": secrets})
}

// selectScope maps a requested scope ("", "node", "global") to the concrete
// store scope for this agent, falling back to the manifest's declaration and
// then to global. Responds 400 and returns ok=false on anything else.
func (h *AgentSecretsHandler) selectScope(c *gin.Context, agentPackage *types.AgentPackage, key, requested string) (string, bool) {
	switch requested {
	case "node":
		return agentSecretScope(agentPackage), true
	case globalSecretScope:
		return globalSecretScope, true
	case "":
		if declaredAgentSecrets(agentPackage.ConfigurationSchema)[key] == "node" {
			return agentSecretScope(agentPackage), true
		}
		return globalSecretScope, true
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "scope must be \"node\" or \"global\""})
		return "", false
	}
}

func (h *AgentSecretsHandler) resolveAgentPackage(c *gin.Context) (*types.AgentPackage, bool) {
	agentPackage, err := h.storage.GetAgentPackage(c.Request.Context(), c.Param("agentId"))
	if err != nil || agentPackage == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent package not found"})
		return nil, false
	}
	return agentPackage, true
}

func agentSecretScope(agentPackage *types.AgentPackage) string {
	if agentPackage.Name != "" {
		return agentPackage.Name
	}
	return agentPackage.ID
}

// declaredAgentSecrets returns the manifest-declared environment keys mapped
// to their declared scope ("node" or "global"; global is the default, the
// same rule packages.UserEnvironmentVar.SecretScope applies).
func declaredAgentSecrets(schema json.RawMessage) map[string]string {
	type declaredVar struct {
		Name  string `json:"name"`
		Scope string `json:"scope"`
	}
	var manifest struct {
		UserEnvironment struct {
			Required     []declaredVar `json:"required"`
			RequireOneOf []struct {
				Options []declaredVar `json:"options"`
			} `json:"require_one_of"`
		} `json:"user_environment"`
	}
	if json.Unmarshal(schema, &manifest) != nil {
		return nil
	}

	declared := make(map[string]string)
	record := func(v declaredVar) {
		if v.Name == "" {
			return
		}
		scope := globalSecretScope
		if v.Scope == "node" {
			scope = "node"
		}
		declared[v.Name] = scope
	}
	for _, variable := range manifest.UserEnvironment.Required {
		record(variable)
	}
	for _, group := range manifest.UserEnvironment.RequireOneOf {
		for _, variable := range group.Options {
			record(variable)
		}
	}
	return declared
}

func declaredAgentEnvironment(schema json.RawMessage) map[string]declaredAgentEnvironmentVar {
	var manifest struct {
		UserEnvironment struct {
			Required     []declaredAgentEnvironmentVar `json:"required"`
			Optional     []declaredAgentEnvironmentVar `json:"optional"`
			RequireOneOf []struct {
				ID          string                        `json:"id"`
				Description string                        `json:"description"`
				Options     []declaredAgentEnvironmentVar `json:"options"`
			} `json:"require_one_of"`
		} `json:"user_environment"`
	}
	if json.Unmarshal(schema, &manifest) != nil {
		return nil
	}

	declared := make(map[string]declaredAgentEnvironmentVar)
	record := func(v declaredAgentEnvironmentVar, requirement, group, groupDescription string) {
		if v.Name == "" {
			return
		}
		if v.Scope != "node" {
			v.Scope = globalSecretScope
		}
		v.Requirement = requirement
		v.Group = group
		v.GroupDescription = groupDescription
		declared[v.Name] = v
	}
	for _, variable := range manifest.UserEnvironment.Required {
		record(variable, "required", "", "")
	}
	for _, group := range manifest.UserEnvironment.RequireOneOf {
		for _, variable := range group.Options {
			record(variable, "one_of", group.ID, group.Description)
		}
	}
	for _, variable := range manifest.UserEnvironment.Optional {
		record(variable, "optional", "", "")
	}
	return declared
}
