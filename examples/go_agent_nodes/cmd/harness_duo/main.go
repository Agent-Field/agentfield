// Command harness_duo runs Pi and OMP concurrently inside one AgentField workflow.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

const defaultModel = "openrouter/minimax/minimax-m2.7"

type workerOutput struct {
	Summary  string   `json:"summary"`
	Evidence []string `json:"evidence"`
}

type branchResult struct {
	Provider     string       `json:"provider"`
	Model        string       `json:"model"`
	Output       workerOutput `json:"output"`
	DurationMS   int          `json:"duration_ms"`
	InputTokens  int          `json:"input_tokens"`
	OutputTokens int          `json:"output_tokens"`
	TotalTokens  int          `json:"total_tokens"`
	CostUSD      *float64     `json:"cost_usd,omitempty"`
	HarnessRunID string       `json:"harness_session_id,omitempty"`
}

type completedBranch struct {
	provider string
	result   map[string]any
	err      error
}

func main() {
	nodeID := envOr("AGENT_NODE_ID", "harness-duo-go")
	listenAddress := envOr("AGENT_LISTEN_ADDR", ":8017")
	publicURL := envOr("AGENT_PUBLIC_URL", "http://localhost"+listenAddress)

	duo, err := agent.New(agent.Config{
		NodeID:        nodeID,
		Version:       "1.0.0",
		AgentFieldURL: envOr("AGENTFIELD_URL", "http://localhost:8080"),
		Token:         os.Getenv("AGENTFIELD_TOKEN"),
		InternalToken: strings.TrimSpace(os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN")),
		ListenAddress: listenAddress,
		PublicURL:     publicURL,
	})
	if err != nil {
		log.Fatal(err)
	}

	registerWorker(duo, "pi_worker", harness.ProviderPi, "PI_BIN")
	// Both branches name their provider explicitly. Omitting Provider would
	// select the SDK default, aforge — not OMP.
	registerWorker(duo, "omp_worker", harness.ProviderOMP, "OMP_BIN")

	duo.RegisterReasoner("compare", func(ctx context.Context, input map[string]any) (any, error) {
		branchInput := map[string]any{
			"task":        inputString(input, "task", defaultTask()),
			"model":       inputString(input, "model", envOr("HARNESS_MODEL", defaultModel)),
			"project_dir": inputString(input, "project_dir", projectDir()),
		}

		completed := make(chan completedBranch, 2)
		for _, provider := range []string{"pi", "omp"} {
			provider := provider
			go func() {
				result, callErr := duo.Call(ctx, provider+"_worker", branchInput)
				completed <- completedBranch{provider: provider, result: result, err: callErr}
			}()
		}

		results := make(map[string]any, 2)
		for i := 0; i < 2; i++ {
			branch := <-completed
			if branch.err != nil {
				return nil, fmt.Errorf("%s harness failed: %w", branch.provider, branch.err)
			}
			results[branch.provider] = branch.result
		}

		return map[string]any{
			"model":    branchInput["model"],
			"branches": results,
		}, nil
	},
		agent.WithDescription("Fan out one task to Pi and OMP concurrently, then join their structured results"),
		agent.WithReasonerTags("entry", "harness-demo"),
	)

	if err := duo.Run(context.Background()); err != nil {
		if cliErr, ok := err.(*agent.CLIError); ok {
			os.Exit(cliErr.ExitCode())
		}
		log.Fatal(err)
	}
}

func registerWorker(duo *agent.Agent, reasoner, provider, binEnv string) {
	providerName := provider
	duo.RegisterReasoner(reasoner, func(ctx context.Context, input map[string]any) (any, error) {
		model := inputString(input, "model", envOr("HARNESS_MODEL", defaultModel))
		root := inputString(input, "project_dir", projectDir())

		var output workerOutput
		schema, err := harness.StructToJSONSchema(output)
		if err != nil {
			return nil, fmt.Errorf("build output schema: %w", err)
		}

		run, err := duo.Harness(ctx, inputString(input, "task", defaultTask()), schema, &output, harness.Options{
			Provider:         provider,
			Model:            model,
			PermissionMode:   "auto",
			ProjectDir:       root,
			BinPath:          strings.TrimSpace(os.Getenv(binEnv)),
			Tools:            []string{"Read", "Write", "Glob", "Grep"},
			SystemPrompt:     "Inspect the requested project carefully. Be concise, cite file paths as evidence, and follow the structured-output instructions exactly.",
			Timeout:          300,
			MaxRetries:       1,
			SchemaMaxRetries: 1,
		})
		if err != nil {
			return nil, err
		}
		if run.IsError {
			return nil, fmt.Errorf("%s harness: %s", providerName, run.ErrorMessage)
		}

		return branchResult{
			Provider:     providerName,
			Model:        model,
			Output:       output,
			DurationMS:   run.DurationMS,
			InputTokens:  run.InputTokens,
			OutputTokens: run.OutputTokens,
			TotalTokens:  run.TotalTokens,
			CostUSD:      run.CostUSD,
			HarnessRunID: run.SessionID,
		}, nil
	}, agent.WithDescription("Run the task with the "+providerName+" coding harness"))
}

func defaultTask() string {
	return "Read README.md and one directly relevant source file. Summarize what this project does in two sentences and provide both file paths as evidence. Do not modify project files."
}

func projectDir() string {
	if configured := strings.TrimSpace(os.Getenv("HARNESS_PROJECT_DIR")); configured != "" {
		return configured
	}
	root, err := os.Getwd()
	if err != nil {
		return "."
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return root
	}
	return absolute
}

func inputString(input map[string]any, key, fallback string) string {
	if value, ok := input[key].(string); ok && strings.TrimSpace(value) != "" {
		return strings.TrimSpace(value)
	}
	return fallback
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
