package ui

import (
	"context"
	"net/http"
	"os"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

const agentfieldYAMLConfigKey = "agentfield.yaml"

// NodeLogSettingsHandler reads/updates node log proxy limits (DB + runtime config).
type NodeLogSettingsHandler struct {
	Storage     storage.StorageProvider
	ReadConfig  func(func(*config.Config))
	WriteConfig func(func(*config.Config))
}

type nodeLogProxyJSON struct {
	ConnectTimeout      string `json:"connect_timeout"`
	StreamIdleTimeout   string `json:"stream_idle_timeout"`
	MaxStreamDuration   string `json:"max_stream_duration"`
	MaxTailLines        int    `json:"max_tail_lines"`
}

func envLocksNodeLogProxy() map[string]bool {
	return map[string]bool{
		"connect_timeout":       os.Getenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT") != "",
		"stream_idle_timeout":   os.Getenv("AGENTFIELD_NODE_LOG_PROXY_STREAM_IDLE_TIMEOUT") != "",
		"max_stream_duration":   os.Getenv("AGENTFIELD_NODE_LOG_PROXY_MAX_DURATION") != "",
		"max_tail_lines":        os.Getenv("AGENTFIELD_NODE_LOG_MAX_TAIL_LINES") != "",
	}
}

// GetNodeLogProxySettingsHandler GET /api/ui/v1/settings/node-log-proxy
func (h *NodeLogSettingsHandler) GetNodeLogProxySettingsHandler(c *gin.Context) {
	if h.ReadConfig == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "not_configured"})
		return
	}
	var eff config.NodeLogProxyConfig
	h.ReadConfig(func(cfg *config.Config) {
		eff = config.EffectiveNodeLogProxy(cfg.AgentField.NodeLogProxy)
	})
	c.JSON(http.StatusOK, gin.H{
		"effective": gin.H{
			"connect_timeout":      eff.ConnectTimeout.String(),
			"stream_idle_timeout":  eff.StreamIdleTimeout.String(),
			"max_stream_duration":  eff.MaxStreamDuration.String(),
			"max_tail_lines":       eff.MaxTailLines,
		},
		"env_locks": envLocksNodeLogProxy(),
	})
}

// PutNodeLogProxySettingsHandler PUT /api/ui/v1/settings/node-log-proxy
func (h *NodeLogSettingsHandler) PutNodeLogProxySettingsHandler(c *gin.Context) {
	if h.Storage == nil || h.WriteConfig == nil || h.ReadConfig == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "not_configured"})
		return
	}
	locks := envLocksNodeLogProxy()
	for k, v := range locks {
		if v {
			c.JSON(http.StatusConflict, gin.H{
				"error":   "env_locked",
				"message": "Clear environment override for " + k + " before editing from UI",
				"locks":   locks,
			})
			return
		}
	}

	var body nodeLogProxyJSON
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_json", "message": err.Error()})
		return
	}

	next := config.NodeLogProxyConfig{}
	if body.ConnectTimeout != "" {
		d, err := time.ParseDuration(body.ConnectTimeout)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_connect_timeout"})
			return
		}
		next.ConnectTimeout = d
	}
	if body.StreamIdleTimeout != "" {
		d, err := time.ParseDuration(body.StreamIdleTimeout)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_stream_idle_timeout"})
			return
		}
		next.StreamIdleTimeout = d
	}
	if body.MaxStreamDuration != "" {
		d, err := time.ParseDuration(body.MaxStreamDuration)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_max_stream_duration"})
			return
		}
		next.MaxStreamDuration = d
	}
	if body.MaxTailLines > 0 {
		next.MaxTailLines = body.MaxTailLines
	}
	if next.ConnectTimeout == 0 && next.StreamIdleTimeout == 0 && next.MaxStreamDuration == 0 && next.MaxTailLines == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no_fields", "message": "Provide at least one field to update"})
		return
	}

	ctx := c.Request.Context()
	if err := persistNodeLogProxyOverlay(ctx, h.Storage, next); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "persist_failed", "message": err.Error()})
		return
	}

	h.WriteConfig(func(cfg *config.Config) {
		if next.ConnectTimeout > 0 {
			cfg.AgentField.NodeLogProxy.ConnectTimeout = next.ConnectTimeout
		}
		if next.StreamIdleTimeout > 0 {
			cfg.AgentField.NodeLogProxy.StreamIdleTimeout = next.StreamIdleTimeout
		}
		if next.MaxStreamDuration > 0 {
			cfg.AgentField.NodeLogProxy.MaxStreamDuration = next.MaxStreamDuration
		}
		if next.MaxTailLines > 0 {
			cfg.AgentField.NodeLogProxy.MaxTailLines = next.MaxTailLines
		}
	})

	var eff config.NodeLogProxyConfig
	h.ReadConfig(func(cfg *config.Config) {
		eff = config.EffectiveNodeLogProxy(cfg.AgentField.NodeLogProxy)
	})
	c.JSON(http.StatusOK, gin.H{
		"effective": gin.H{
			"connect_timeout":     eff.ConnectTimeout.String(),
			"stream_idle_timeout": eff.StreamIdleTimeout.String(),
			"max_stream_duration": eff.MaxStreamDuration.String(),
			"max_tail_lines":      eff.MaxTailLines,
		},
	})
}

func persistNodeLogProxyOverlay(ctx context.Context, st storage.StorageProvider, patch config.NodeLogProxyConfig) error {
	var root map[string]interface{}
	entry, err := st.GetConfig(ctx, agentfieldYAMLConfigKey)
	if err != nil {
		return err
	}
	if entry != nil && entry.Value != "" {
		if err := yaml.Unmarshal([]byte(entry.Value), &root); err != nil {
			return err
		}
	}
	if root == nil {
		root = make(map[string]interface{})
	}
	af, _ := root["agentfield"].(map[string]interface{})
	if af == nil {
		af = make(map[string]interface{})
		root["agentfield"] = af
	}
	nlp, _ := af["node_log_proxy"].(map[string]interface{})
	if nlp == nil {
		nlp = make(map[string]interface{})
		af["node_log_proxy"] = nlp
	}
	if patch.ConnectTimeout > 0 {
		nlp["connect_timeout"] = patch.ConnectTimeout.String()
	}
	if patch.StreamIdleTimeout > 0 {
		nlp["stream_idle_timeout"] = patch.StreamIdleTimeout.String()
	}
	if patch.MaxStreamDuration > 0 {
		nlp["max_stream_duration"] = patch.MaxStreamDuration.String()
	}
	if patch.MaxTailLines > 0 {
		nlp["max_tail_lines"] = patch.MaxTailLines
	}
	out, err := yaml.Marshal(root)
	if err != nil {
		return err
	}
	return st.SetConfig(ctx, agentfieldYAMLConfigKey, string(out), "ui")
}
