package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

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
		AgentFieldURL: agentFieldURL, // optional for CLI-only
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		InternalToken: strings.TrimSpace(os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN")),
		ListenAddress: listenAddr,
		PublicURL:     publicURL,
		CLIConfig: &agent.CLIConfig{
			AppName:        "go-agent-hello",
			AppDescription: "Go SDK hello-world with CLI + control plane",
			HelpPreamble:   "Pass --set message=YourName to customize the greeting.",
			EnvironmentVars: []string{
				"AGENTFIELD_URL (optional) Control plane URL for server mode",
				"AGENTFIELD_TOKEN (optional) Bearer token",
				"AGENT_NODE_ID (optional) Override node id (default: my-agent)",
			},
		},
	}

	hello, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	hasControlPlane := strings.TrimSpace(cfg.AgentFieldURL) != ""

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

	hello.RegisterReasoner("add_emoji", func(ctx context.Context, input map[string]any) (any, error) {
		msg := fmt.Sprintf("%v", input["message"])
		return addEmojiLocal(msg), nil
	},
		agent.WithDescription("Adds a friendly emoji to a message"),
	)

	hello.RegisterReasoner("say_hello", func(ctx context.Context, input map[string]any) (any, error) {
		name := strings.TrimSpace(fmt.Sprintf("%v", input["name"]))
		if name == "" || name == "<nil>" {
			name = "World"
		}
		greeting := fmt.Sprintf("Hello, %s!", name)

		var decorated map[string]any
		if hasControlPlane {
			// Prefer control plane call so workflow edges are captured.
			res, callErr := hello.Call(ctx, "add_emoji", map[string]any{"message": greeting})
			if callErr == nil {
				decorated = res
			} else {
				log.Printf("warn: control plane call to add_emoji failed: %v", callErr)
			}
		}
		if decorated == nil {
			decorated = addEmojiLocal(greeting)
		}

		return map[string]any{
			"greeting": fmt.Sprintf("%s %s", decorated["text"], decorated["emoji"]),
			"name":     name,
		}, nil
	},
		agent.WithCLI(),
		agent.WithDescription("Greets a user, enriching the message via add_emoji"),
	)

	hello.RegisterReasoner("demo_echo", func(ctx context.Context, input map[string]any) (any, error) {
		message := strings.TrimSpace(fmt.Sprintf("%v", input["message"]))
		if message == "" || message == "<nil>" {
			message = "Agentfield"
		}

		if hasControlPlane {
			res, callErr := hello.Call(ctx, "say_hello", map[string]any{"name": message})
			if callErr == nil {
				return res, nil
			}
			log.Printf("warn: control plane call to say_hello failed: %v", callErr)
		}

		return hello.Execute(ctx, "say_hello", map[string]any{"name": message})
	},
		agent.WithCLI(),
		agent.WithDefaultCLI(),
		agent.WithDescription("Echo entry point that chains into say_hello -> add_emoji"),
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

	go func() {
		t := time.NewTicker(3 * time.Second)
		defer t.Stop()
		n := 0
		for range t.C {
			log.Printf("[%s] demo stdout-class log %d", nodeID, n)
			_, _ = fmt.Fprintf(os.Stderr, "[%s] demo stderr line %d\n", nodeID, n)
			n++
		}
	}()

	if err := hello.Run(context.Background()); err != nil {
		if cliErr, ok := err.(*agent.CLIError); ok {
			os.Exit(cliErr.ExitCode())
		}
		log.Fatal(err)
	}
}
