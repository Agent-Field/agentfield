package main

import (
	"context"
	"log"
	"os"

	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/capabilities"
	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/linear"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	cfg := config.Load()
	logger := log.New(os.Stdout, "[linear-node] ", log.LstdFlags)

	client, err := linear.NewClient(cfg.Linear)
	if err != nil {
		logger.Fatalf("initialize linear client: %v", err)
	}
	agentCfg := cfg.AgentConfig()
	agentCfg.Logger = logger
	agent, err := afagent.New(agentCfg)
	if err != nil {
		logger.Fatalf("initialize agent: %v", err)
	}
	capabilities.Register(agent, capabilities.Runtime{Config: cfg, Linear: client})
	if err := agent.Run(context.Background()); err != nil {
		logger.Fatal(err)
	}
}
