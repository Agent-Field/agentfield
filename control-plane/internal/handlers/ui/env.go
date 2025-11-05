package ui

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/storage"

	"github.com/gin-gonic/gin"
)

// EnvHandler provides handlers for .env file management operations.
type EnvHandler struct {
	storage        storage.StorageProvider
	agentService   interfaces.AgentService
	agentfieldHome string
}

// DELETE /api/ui/v1/agents/:agentId/env/:key
func (h *EnvHandler) DeleteEnvVarHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agentId is required"})
		return
	}

	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "packageId query parameter is required"})
		return
	}

	key := c.Param("key")
	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "key is required"})
		return
	}

	agentPackage, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "package not found"})
		return
	}

	envPath := filepath.Join(agentPackage.InstallPath, ".env")
	backupPath := envPath + ".backup"

	// Read existing .env variables
	existingVars := make(map[string]string)
	if data, err := os.ReadFile(envPath); err == nil {
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				k := strings.TrimSpace(parts[0])
				value := strings.TrimSpace(parts[1])
				// Remove quotes if present
				if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
					(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
					value = value[1 : len(value)-1]
				}
				existingVars[k] = value
			}
		}
	}

	// Remove the specified key
	if _, exists := existingVars[key]; !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "variable not found"})
		return
	}
	delete(existingVars, key)

	// Validate resulting variables
	configVars := make(map[string]interface{}, len(existingVars))
	for k, v := range existingVars {
		configVars[k] = v
	}
	validationResult, err := h.storage.ValidateAgentConfiguration(ctx, agentID, packageID, configVars)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to validate configuration"})
		return
	}
	if !validationResult.Valid {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":             "configuration validation failed",
			"validation_errors": validationResult.Errors,
		})
		return
	}

	// Backup existing .env if it exists
	if _, err := os.Stat(envPath); err == nil {
		_ = os.Rename(envPath, backupPath)
	}

	// Write updated .env file
	var lines []string
	for k, v := range existingVars {
		// Quote value if it contains spaces or special chars
		if strings.ContainsAny(v, " #") {
			v = `"` + v + `"`
		}
		lines = append(lines, k+"="+v)
	}
	content := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(envPath, []byte(content), 0600); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to write .env file"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":    "variable deleted from .env file",
		"agent_id":   agentID,
		"package_id": packageID,
		"key":        key,
	})
}

// PATCH /api/ui/v1/agents/:agentId/env
func (h *EnvHandler) PatchEnvHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agentId is required"})
		return
	}

	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "packageId query parameter is required"})
		return
	}

	var req SetEnvRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}

	agentPackage, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "package not found"})
		return
	}

	envPath := filepath.Join(agentPackage.InstallPath, ".env")
	backupPath := envPath + ".backup"

	// Read existing .env variables
	existingVars := make(map[string]string)
	if data, err := os.ReadFile(envPath); err == nil {
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				value := strings.TrimSpace(parts[1])
				// Remove quotes if present
				if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
					(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
					value = value[1 : len(value)-1]
				}
				existingVars[key] = value
			}
		}
	}

	// Merge new variables into existing
	for k, v := range req.Variables {
		existingVars[k] = v
	}

	// Validate merged variables
	configVars := make(map[string]interface{}, len(existingVars))
	for k, v := range existingVars {
		configVars[k] = v
	}
	validationResult, err := h.storage.ValidateAgentConfiguration(ctx, agentID, packageID, configVars)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to validate configuration"})
		return
	}
	if !validationResult.Valid {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":             "configuration validation failed",
			"validation_errors": validationResult.Errors,
		})
		return
	}

	// Backup existing .env if it exists
	if _, err := os.Stat(envPath); err == nil {
		_ = os.Rename(envPath, backupPath)
	}

	// Write merged .env file
	var lines []string
	for k, v := range existingVars {
		// Quote value if it contains spaces or special chars
		if strings.ContainsAny(v, " #") {
			v = `"` + v + `"`
		}
		lines = append(lines, k+"="+v)
	}
	content := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(envPath, []byte(content), 0600); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to write .env file"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":    ".env file patched successfully",
		"agent_id":   agentID,
		"package_id": packageID,
	})
}

// EnvResponse represents the response for GET .env
type EnvResponse struct {
	AgentID      string            `json:"agent_id"`
	PackageID    string            `json:"package_id"`
	Variables    map[string]string `json:"variables"`
	MaskedKeys   []string          `json:"masked_keys"`
	FileExists   bool              `json:"file_exists"`
	LastModified *time.Time        `json:"last_modified,omitempty"`
}

// NewEnvHandler creates a new EnvHandler.
func NewEnvHandler(storage storage.StorageProvider, agentService interfaces.AgentService, agentfieldHome string) *EnvHandler {
	return &EnvHandler{
		storage:        storage,
		agentService:   agentService,
		agentfieldHome: agentfieldHome,
	}
}

// GET /api/ui/v1/agents/:agentId/env
func (h *EnvHandler) GetEnvHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agentId is required"})
		return
	}

	// Get packageId from query parameter
	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "packageId query parameter is required"})
		return
	}

	// Get the agent package to resolve install path and schema
	agentPackage, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "package not found"})
		return
	}

	envPath := filepath.Join(agentPackage.InstallPath, ".env")
	vars := make(map[string]string)
	fileExists := false
	var lastModified *time.Time

	// Read .env file if it exists
	if stat, err := os.Stat(envPath); err == nil {
		fileExists = true
		modTime := stat.ModTime()
		lastModified = &modTime

		data, err := os.ReadFile(envPath)
		if err == nil {
			lines := strings.Split(string(data), "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				parts := strings.SplitN(line, "=", 2)
				if len(parts) == 2 {
					key := strings.TrimSpace(parts[0])
					value := strings.TrimSpace(parts[1])
					// Remove quotes if present
					if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
						(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
						value = value[1 : len(value)-1]
					}
					vars[key] = value
				}
			}
		}
	}

	// Parse the configuration schema to determine secret fields
	var schema map[string]interface{}
	maskedKeys := []string{}
	if err := json.Unmarshal(agentPackage.ConfigurationSchema, &schema); err == nil {
		// Try to find secret fields in schema
		if userEnv, ok := schema["user_environment"].(map[string]interface{}); ok {
			for _, section := range []string{"required", "optional"} {
				if arr, ok := userEnv[section].([]interface{}); ok {
					for _, item := range arr {
						if field, ok := item.(map[string]interface{}); ok {
							if t, ok := field["type"].(string); ok && t == "secret" {
								if name, ok := field["name"].(string); ok {
									maskedKeys = append(maskedKeys, name)
									// Mask value if present
									if v, exists := vars[name]; exists && len(v) > 6 {
										vars[name] = v[:3] + "..." + v[len(v)-2:]
									} else if exists {
										vars[name] = "***"
									}
								}
							}
						}
					}
				}
			}
		}
	}

	resp := EnvResponse{
		AgentID:      agentID,
		PackageID:    packageID,
		Variables:    vars,
		MaskedKeys:   maskedKeys,
		FileExists:   fileExists,
		LastModified: lastModified,
	}
	c.JSON(http.StatusOK, resp)
}

// PUT /api/ui/v1/agents/:agentId/env
type SetEnvRequest struct {
	Variables map[string]string `json:"variables" binding:"required"`
}

func (h *EnvHandler) PutEnvHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agentId is required"})
		return
	}

	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "packageId query parameter is required"})
		return
	}

	var req SetEnvRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}

	agentPackage, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "package not found"})
		return
	}

	// Validate variables using existing validation logic
	// Convert map[string]string to map[string]interface{}
	configVars := make(map[string]interface{}, len(req.Variables))
	for k, v := range req.Variables {
		configVars[k] = v
	}
	validationResult, err := h.storage.ValidateAgentConfiguration(ctx, agentID, packageID, configVars)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to validate configuration"})
		return
	}
	if !validationResult.Valid {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":             "configuration validation failed",
			"validation_errors": validationResult.Errors,
		})
		return
	}

	envPath := filepath.Join(agentPackage.InstallPath, ".env")
	backupPath := envPath + ".backup"

	// Backup existing .env if it exists
	if _, err := os.Stat(envPath); err == nil {
		_ = os.Rename(envPath, backupPath)
	}

	// Write new .env file
	var lines []string
	for k, v := range req.Variables {
		// Quote value if it contains spaces or special chars
		if strings.ContainsAny(v, " #") {
			v = `"` + v + `"`
		}
		lines = append(lines, k+"="+v)
	}
	content := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(envPath, []byte(content), 0600); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to write .env file"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":    ".env file updated successfully",
		"agent_id":   agentID,
		"package_id": packageID,
	})
}
