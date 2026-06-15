package config

import (
	"os"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/snowflake"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Config struct {
	NodeID             string
	Version            string
	AgentFieldURL      string
	ListenAddress      string
	PublicURL          string
	Token              string
	PromptFile         string
	DefaultPromptFile  string
	DefaultCortexModel string
	Snowflake          snowflake.Config
}

func Load() Config {
	return Config{
		NodeID:             env("AGENTFIELD_NODE_ID", "snowflake"),
		Version:            env("SNOWFLAKE_NODE_VERSION", "0.1.0"),
		AgentFieldURL:      env("AGENTFIELD_SERVER", env("AGENTFIELD_URL", "http://localhost:8080")),
		ListenAddress:      env("SNOWFLAKE_NODE_LISTEN", ":8012"),
		PublicURL:          os.Getenv("SNOWFLAKE_NODE_PUBLIC_URL"),
		Token:              os.Getenv("AGENTFIELD_TOKEN"),
		PromptFile:         os.Getenv("SNOWFLAKE_PROMPTS_FILE"),
		DefaultPromptFile:  env("SNOWFLAKE_DEFAULT_PROMPTS_FILE", "../prompts/default-prompts.yaml"),
		DefaultCortexModel: os.Getenv("SNOWFLAKE_CORTEX_MODEL"),
		Snowflake: snowflake.Config{
			AccountURL:     os.Getenv("SNOWFLAKE_ACCOUNT_URL"),
			Token:          firstNonBlank(os.Getenv("SNOWFLAKE_PAT"), os.Getenv("SNOWFLAKE_TOKEN")),
			Database:       os.Getenv("SNOWFLAKE_DATABASE"),
			Schema:         os.Getenv("SNOWFLAKE_SCHEMA"),
			Warehouse:      os.Getenv("SNOWFLAKE_WAREHOUSE"),
			Role:           os.Getenv("SNOWFLAKE_ROLE"),
			TimeoutSeconds: envInt("SNOWFLAKE_TIMEOUT_SECONDS", 30),
			MaxRows:        envInt("SNOWFLAKE_MAX_ROWS", 100),
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
		Tags:          []string{"snowflake", "data", "warehouse", "cortex"},
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
