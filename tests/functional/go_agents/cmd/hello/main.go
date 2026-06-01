package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	nodeID := strings.TrimSpace(os.Getenv("AGENT_NODE_ID"))
	if nodeID == "" {
		nodeID = "my-agent"
	}

	agentFieldURL := strings.TrimSpace(os.Getenv("AGENTFIELD_URL"))
	listenAddr := strings.TrimSpace(os.Getenv("AGENT_LISTEN_ADDR"))
	if listenAddr == "" {
		listenAddr = ":8001"
	}
	publicURL := strings.TrimSpace(os.Getenv("AGENT_PUBLIC_URL"))
	if publicURL == "" {
		publicURL = "http://localhost" + listenAddr
	}

	cfg := agent.Config{
		NodeID:        nodeID,
		Version:       "1.0.0",
		AgentFieldURL: agentFieldURL,
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		ListenAddress: listenAddr,
		PublicURL:     publicURL,
		CLIConfig: &agent.CLIConfig{
			AppName:        "go-agent-hello",
			AppDescription: "Functional test agent for Go SDK CLI + control plane flows",
			HelpPreamble:   "Pass --set message=YourName to customize the greeting.",
			EnvironmentVars: []string{
				"AGENTFIELD_URL (optional) Control plane URL for server mode",
				"AGENTFIELD_TOKEN (optional) Bearer token",
				"AGENT_NODE_ID (optional) Override node id (default: my-agent)",
			},
		},
	}

	ag, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	addEmojiLocal := func(message string) map[string]any {
		trimmed := strings.TrimSpace(message)
		if trimmed == "" {
			trimmed = "Hello!"
		}
		return map[string]any{
			"text":  trimmed,
			"emoji": "👋",
		}
	}

	ag.RegisterReasoner("add_emoji", func(ctx context.Context, input map[string]any) (any, error) {
		msg := fmt.Sprintf("%v", input["message"])
		return addEmojiLocal(msg), nil
	},
		agent.WithDescription("Adds a friendly emoji to a message"),
	)

	ag.RegisterReasoner("say_hello", func(ctx context.Context, input map[string]any) (any, error) {
		name := strings.TrimSpace(fmt.Sprintf("%v", input["name"]))
		if name == "" || name == "<nil>" {
			name = "World"
		}
		greeting := fmt.Sprintf("Hello, %s!", name)

		decorated := addEmojiLocal(greeting)
		if res, callErr := ag.CallLocal(ctx, "add_emoji", map[string]any{"message": greeting}); callErr == nil {
			if typed, ok := res.(map[string]any); ok {
				decorated = typed
			} else {
				log.Printf("warn: unexpected add_emoji result type: %T", res)
			}
		} else {
			log.Printf("warn: local call to add_emoji failed: %v", callErr)
		}

		return map[string]any{
			"greeting": fmt.Sprintf("%s %s", decorated["text"], decorated["emoji"]),
			"name":     name,
		}, nil
	},
		agent.WithCLI(),
		agent.WithDescription("Greets a user, enriching the message via add_emoji"),
	)

	ag.RegisterReasoner("demo_echo", func(ctx context.Context, input map[string]any) (any, error) {
		message := strings.TrimSpace(fmt.Sprintf("%v", input["message"]))
		if message == "" || message == "<nil>" {
			message = "Agentfield"
		}

		res, callErr := ag.CallLocal(ctx, "say_hello", map[string]any{"name": message})
		if callErr == nil {
			return res, nil
		}
		log.Printf("warn: local call to say_hello failed: %v", callErr)

		return ag.Execute(ctx, "say_hello", map[string]any{"name": message})
	},
		agent.WithCLI(),
		agent.WithDefaultCLI(),
		agent.WithDescription("Echo entry point chaining say_hello -> add_emoji"),
		agent.WithCLIFormatter(func(ctx context.Context, result any, err error) {
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				return
			}
			if resMap, ok := result.(map[string]any); ok {
				fmt.Printf("%s (%s)\n", resMap["greeting"], resMap["name"])
				return
			}
			fmt.Println(result)
		}),
	)

	if err := ag.Run(context.Background()); err != nil {
		if cliErr, ok := err.(*agent.CLIError); ok {
			os.Exit(cliErr.ExitCode())
		}
		log.Fatal(err)
	}
}
