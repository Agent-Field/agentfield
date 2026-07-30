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

const maxAgentSecretValueBytes = 32 * 1024

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
}

type setAgentSecretRequest struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ListAgentSecretsHandler lists secret names and whether each is set.
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
	storedKeys, err := store.List(agentSecretScope(agentPackage))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list secrets"})
		return
	}

	statuses := make(map[string]bool, len(storedKeys))
	for _, key := range declaredAgentSecretKeys(agentPackage.ConfigurationSchema) {
		statuses[key] = false
	}
	for _, key := range storedKeys {
		statuses[key] = true
	}

	keys := make([]string, 0, len(statuses))
	for key := range statuses {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	secrets := make([]agentSecretStatus, 0, len(keys))
	for _, key := range keys {
		secrets = append(secrets, agentSecretStatus{Key: key, IsSet: statuses[key]})
	}
	c.JSON(http.StatusOK, gin.H{"secrets": secrets})
}

// SetAgentSecretHandler stores one node-scoped secret.
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

	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	if err := store.Set(agentSecretScope(agentPackage), req.Key, req.Value); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to store secret"})
		return
	}
	c.Status(http.StatusNoContent)
}

// DeleteAgentSecretHandler deletes one node-scoped secret.
func (h *AgentSecretsHandler) DeleteAgentSecretHandler(c *gin.Context) {
	agentPackage, ok := h.resolveAgentPackage(c)
	if !ok {
		return
	}

	store, err := packages.NewSecretStore(h.agentfieldHome)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open secret store"})
		return
	}
	if err := store.Delete(agentSecretScope(agentPackage), c.Param("key")); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to delete secret"})
		return
	}
	c.Status(http.StatusNoContent)
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

func declaredAgentSecretKeys(schema json.RawMessage) []string {
	var manifest struct {
		UserEnvironment struct {
			Required []struct {
				Name string `json:"name"`
			} `json:"required"`
			RequireOneOf []struct {
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"require_one_of"`
		} `json:"user_environment"`
	}
	if json.Unmarshal(schema, &manifest) != nil {
		return nil
	}

	seen := make(map[string]struct{})
	for _, variable := range manifest.UserEnvironment.Required {
		if variable.Name != "" {
			seen[variable.Name] = struct{}{}
		}
	}
	for _, group := range manifest.UserEnvironment.RequireOneOf {
		for _, variable := range group.Options {
			if variable.Name != "" {
				seen[variable.Name] = struct{}{}
			}
		}
	}
	keys := make([]string, 0, len(seen))
	for key := range seen {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
