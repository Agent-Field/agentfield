package config

import (
	"os"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/sentry"
	"github.com/Agent-Field/agentfield/sdk/go/inputs"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Config struct {
	NodeID        string
	Version       string
	AgentFieldURL string
	ListenAddress string
	PublicURL     string
	Token         string
	Sentry        sentry.Config
}

func Load() Config {
	return Config{
		NodeID:        env("AGENTFIELD_NODE_ID", "sentry"),
		Version:       env("SENTRY_NODE_VERSION", "0.1.0"),
		AgentFieldURL: env("AGENTFIELD_SERVER", env("AGENTFIELD_URL", "http://localhost:8080")),
		ListenAddress: env("SENTRY_NODE_LISTEN", ":8014"),
		PublicURL:     os.Getenv("SENTRY_NODE_PUBLIC_URL"),
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		Sentry: sentry.Config{
			BaseURL:        env("SENTRY_BASE_URL", "https://sentry.io"),
			Organization:   os.Getenv("SENTRY_ORG"),
			Token:          inputs.FirstNonBlank(os.Getenv("SENTRY_AUTH_TOKEN"), os.Getenv("SENTRY_TOKEN")),
			TimeoutSeconds: envInt("SENTRY_TIMEOUT_SECONDS", 20),
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
		Tags:          []string{"sentry", "observability", "errors"},
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
