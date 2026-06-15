package main

import (
	"context"
	"log"
	"os"

	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/capabilities"
	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/prompts"
	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/snowflake"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	cfg := config.Load()
	logger := log.New(os.Stdout, "[snowflake-node] ", log.LstdFlags)

	promptStore, err := prompts.Load(cfg.DefaultPromptFile, cfg.PromptFile)
	if err != nil {
		logger.Fatalf("load prompts: %v", err)
	}
	sfClient, err := snowflake.NewClient(cfg.Snowflake)
	if err != nil {
		logger.Fatalf("initialize snowflake client: %v", err)
	}
	agentCfg := cfg.AgentConfig()
	agentCfg.Logger = logger
	a, err := afagent.New(agentCfg)
	if err != nil {
		logger.Fatalf("initialize agent: %v", err)
	}
	capabilities.Register(a, capabilities.Runtime{
		Config:    cfg,
		Snowflake: sfClient,
		Prompts:   promptStore,
	})
	if err := a.Run(context.Background()); err != nil {
		logger.Fatal(err)
	}
}
