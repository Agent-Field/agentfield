package config

import (
	"os"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/databricks"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Config struct {
	NodeID            string
	Version           string
	AgentFieldURL     string
	ListenAddress     string
	PublicURL         string
	Token             string
	PromptFile        string
	DefaultPromptFile string
	Databricks        databricks.Config
}

func Load() Config {
	return Config{
		NodeID:            env("AGENTFIELD_NODE_ID", "databricks"),
		Version:           env("DATABRICKS_NODE_VERSION", "0.1.0"),
		AgentFieldURL:     env("AGENTFIELD_SERVER", env("AGENTFIELD_URL", "http://localhost:8080")),
		ListenAddress:     env("DATABRICKS_NODE_LISTEN", ":8013"),
		PublicURL:         os.Getenv("DATABRICKS_NODE_PUBLIC_URL"),
		Token:             os.Getenv("AGENTFIELD_TOKEN"),
		PromptFile:        os.Getenv("DATABRICKS_PROMPTS_FILE"),
		DefaultPromptFile: env("DATABRICKS_DEFAULT_PROMPTS_FILE", "../prompts/default-prompts.yaml"),
		Databricks: databricks.Config{
			WorkspaceURL:   firstNonBlank(os.Getenv("DATABRICKS_HOST"), os.Getenv("DATABRICKS_WORKSPACE_URL")),
			Token:          firstNonBlank(os.Getenv("DATABRICKS_TOKEN"), os.Getenv("DATABRICKS_PAT")),
			WarehouseID:    os.Getenv("DATABRICKS_WAREHOUSE_ID"),
			Catalog:        os.Getenv("DATABRICKS_CATALOG"),
			Schema:         os.Getenv("DATABRICKS_SCHEMA"),
			AIEndpoint:     os.Getenv("DATABRICKS_AI_ENDPOINT"),
			TimeoutSeconds: envInt("DATABRICKS_TIMEOUT_SECONDS", 30),
			MaxRows:        envInt("DATABRICKS_MAX_ROWS", 100),
		},
	}
}

func (c Config) AgentConfig() afagent.Config {
	return afagent.Config{
		NodeID:        c.NodeID,
		Version:       c.Version,
		AgentFieldURL: c.AgentFieldURL,
		ListenAddress: c.ListenAddress,
		PublicURL:     c.PublicURL,
		Token:         c.Token,
		Tags:          []string{"databricks", "data", "warehouse", "lakehouse", "ai-functions", "model-serving"},
	}
}

func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
