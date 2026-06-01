// Multi-Version Agent Example (Go)
//
// Demonstrates multi-version agent support using the composite primary key
// (id, version). All agents share the same NodeID but register with different
// versions, creating separate rows in the control plane.
//
// The execute endpoint transparently routes across versioned agents using
// weighted round-robin when no default (unversioned) agent exists.
//
// Usage:
//
//	# Start control plane first, then:
//	cd examples/go_agent_nodes
//	go run ./cmd/multi_version
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

const (
	agentID  = "mv-demo-go"
	basePort = 9300
)

type versionSpec struct {
	version string
	port    int
}

func cpURL() string {
	if v := strings.TrimSpace(os.Getenv("AGENTFIELD_URL")); v != "" {
		return v
	}
	return "http://localhost:8080"
}

func createAgent(spec versionSpec) (*agent.Agent, error) {
	listenAddr := fmt.Sprintf(":%d", spec.port)
	publicURL := fmt.Sprintf("http://localhost:%d", spec.port)

	cfg := agent.Config{
		NodeID:        agentID,
		Version:       spec.version,
		AgentFieldURL: cpURL(),
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		ListenAddress: listenAddr,
		PublicURL:     publicURL,
	}

	a, err := agent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent v%s: %w", spec.version, err)
	}

	// Echo reasoner present on every version
	ver := spec.version // capture for closure
	a.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
		msg := ""
		if v, ok := input["message"]; ok {
			msg = fmt.Sprintf("%v", v)
		}
		return map[string]any{
			"agent":   agentID,
			"version": ver,
			"echoed":  msg,
		}, nil
	}, agent.WithDescription("Echo back the input with version info"))

	// v2 has an extra capability
	if spec.version == "2.0.0" {
		a.RegisterReasoner("v2_feature", func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"agent":   agentID,
				"version": ver,
				"feature": "Only available in v2",
				"input":   input,
			}, nil
		}, agent.WithDescription("Feature only available in v2"))
	}

	return a, nil
}

func validateRegistration() {
	fmt.Println("\n--- Validating multi-version registration ---")
	cp := cpURL()
	client := &http.Client{Timeout: 30 * time.Second}

	// List all nodes and check that both versions are registered
	resp, err := client.Get(cp + "/api/v1/nodes?show_all=true")
	if err != nil {
		fmt.Printf("Failed to list nodes: %v\n", err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var nodesResp struct {
		Nodes []struct {
			ID      string `json:"id"`
			NodeID  string `json:"node_id"`
			Version string `json:"version"`
			BaseURL string `json:"base_url"`
		} `json:"nodes"`
		Agents []struct {
			ID      string `json:"id"`
			NodeID  string `json:"node_id"`
			Version string `json:"version"`
			BaseURL string `json:"base_url"`
		} `json:"agents"`
	}
	if err := json.Unmarshal(body, &nodesResp); err != nil {
		fmt.Printf("Failed to parse nodes response: %v\n", err)
		return
	}

	allNodes := nodesResp.Nodes
	if len(allNodes) == 0 {
		allNodes = nodesResp.Agents
	}

	var agentNodes []struct {
		ID      string
		Version string
		BaseURL string
	}
	for _, n := range allNodes {
		nid := n.ID
		if nid == "" {
			nid = n.NodeID
		}
		if nid == agentID {
			agentNodes = append(agentNodes, struct {
				ID      string
				Version string
				BaseURL string
			}{nid, n.Version, n.BaseURL})
		}
	}

	fmt.Printf("\n[Nodes] Found %d versions of %q:\n", len(agentNodes), agentID)
	for _, n := range agentNodes {
		fmt.Printf("  - id=%s, version=%s, base_url=%s\n", n.ID, n.Version, n.BaseURL)
	}

	// Execute against the shared ID - the CP will route via round-robin
	fmt.Printf("\n[Execute] Sending requests to %s.echo:\n", agentID)
	for i := 0; i < 4; i++ {
		payload := fmt.Sprintf(`{"input":{"message":"request-%d"}}`, i)
		req, _ := http.NewRequest("POST",
			fmt.Sprintf("%s/api/v1/execute/%s.echo", cp, agentID),
			strings.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("  Request %d: failed - %v\n", i, err)
			continue
		}

		var result map[string]any
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		routedVersion := resp.Header.Get("X-Routed-Version")
		if routedVersion == "" {
			if payload, ok := result["result"].(map[string]any); ok {
				if payloadVersion, ok := payload["version"].(string); ok && payloadVersion != "" {
					routedVersion = payloadVersion
				}
			}
		}
		if routedVersion == "" {
			routedVersion = "(unknown)"
		}

		fmt.Printf("  Request %d: routed to version=%s, result=%v\n", i, routedVersion, result["result"])
	}

	fmt.Println("\n--- Validation complete ---")
}

func main() {
	versions := []versionSpec{
		{version: "1.0.0", port: basePort},
		{version: "2.0.0", port: basePort + 1},
	}

	fmt.Println("Multi-version agent example (Go)")
	fmt.Printf("  Control plane: %s\n", cpURL())
	fmt.Printf("  Agent ID:      %s\n", agentID)
	parts := make([]string, len(versions))
	for i, v := range versions {
		parts[i] = fmt.Sprintf("%s@:%d", v.version, v.port)
	}
	fmt.Printf("  Versions:      %s\n\n", strings.Join(parts, ", "))

	// Create and start all agents
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for _, spec := range versions {
		a, err := createAgent(spec)
		if err != nil {
			log.Fatalf("Failed to create agent v%s: %v", spec.version, err)
		}

		wg.Add(1)
		go func(a *agent.Agent, spec versionSpec) {
			defer wg.Done()
			fmt.Printf("  Started %s v%s on port %d\n", agentID, spec.version, spec.port)
			if err := a.Run(ctx); err != nil && ctx.Err() == nil {
				log.Printf("Agent v%s exited with error: %v", spec.version, err)
			}
		}(a, spec)
	}

	// Give the CP a moment to process registrations (DID creation takes ~5s)
	fmt.Println("\n  Waiting for registrations to propagate...")
	time.Sleep(8 * time.Second)

	// Validate
	validateRegistration()

	// Keep running so heartbeats continue
	fmt.Println("\nAll agents running. Press Ctrl+C to stop.\n")
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	fmt.Println("Shutting down.")
	cancel()
	wg.Wait()
}
