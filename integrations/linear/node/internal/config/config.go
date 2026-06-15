package config

import (
	"os"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/linear"
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
	Linear        linear.Config
}

func Load() Config {
	return Config{
		NodeID:        env("AGENTFIELD_NODE_ID", "linear"),
		Version:       env("LINEAR_NODE_VERSION", "0.1.0"),
		AgentFieldURL: env("AGENTFIELD_SERVER", env("AGENTFIELD_URL", "http://localhost:8080")),
		ListenAddress: env("LINEAR_NODE_LISTEN", ":8013"),
		PublicURL:     os.Getenv("LINEAR_NODE_PUBLIC_URL"),
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		Linear: linear.Config{
			APIURL:         env("LINEAR_API_URL", "https://api.linear.app/graphql"),
			Token:          inputs.FirstNonBlank(os.Getenv("LINEAR_API_KEY"), os.Getenv("LINEAR_TOKEN")),
			TimeoutSeconds: envInt("LINEAR_TIMEOUT_SECONDS", 20),
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
		Tags:          []string{"linear", "issue-tracking", "project-management"},
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
