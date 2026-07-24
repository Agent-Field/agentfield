package main

import (
	"context"
	"log"
	"os"

	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/capabilities"
	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/sentry"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	cfg := config.Load()
	logger := log.New(os.Stdout, "[sentry-node] ", log.LstdFlags)

	client, err := sentry.NewClient(cfg.Sentry)
	if err != nil {
		logger.Fatalf("initialize sentry client: %v", err)
	}
	agentCfg := cfg.AgentConfig()
	agentCfg.Logger = logger
	agent, err := afagent.New(agentCfg)
	if err != nil {
		logger.Fatalf("initialize agent: %v", err)
	}
	capabilities.Register(agent, capabilities.Runtime{Config: cfg, Sentry: client})
	if err := agent.Run(context.Background()); err != nil {
		logger.Fatal(err)
	}
}
