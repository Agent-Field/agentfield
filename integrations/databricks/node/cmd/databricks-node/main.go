package main

import (
	"context"
	"log"
	"os"

	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/capabilities"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/databricks"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/prompts"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	cfg := config.Load()
	logger := log.New(os.Stdout, "[databricks-node] ", log.LstdFlags)

	promptStore, err := prompts.Load(cfg.DefaultPromptFile, cfg.PromptFile)
	if err != nil {
		logger.Fatalf("load prompts: %v", err)
	}
	dbClient, err := databricks.NewClient(cfg.Databricks)
	if err != nil {
		logger.Fatalf("initialize databricks client: %v", err)
	}
	agentCfg := cfg.AgentConfig()
	agentCfg.Logger = logger
	a, err := afagent.New(agentCfg)
	if err != nil {
		logger.Fatalf("initialize agent: %v", err)
	}
	capabilities.Register(a, capabilities.Runtime{
		Config:     cfg,
		Databricks: dbClient,
		Prompts:    promptStore,
	})
	if err := a.Run(context.Background()); err != nil {
		logger.Fatal(err)
	}
}
