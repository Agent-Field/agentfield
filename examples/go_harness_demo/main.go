// Example: Go SDK harness package demonstration.
//
// Tests both opencode and claude-code providers with structured output.
// Requires ANTHROPIC_API_KEY and OPENROUTER_API_KEY in .env.
//
// Usage:
//
//	cd examples/go_harness_demo
//	go run .
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// ReviewResult is a simple structured output schema for testing.
type ReviewResult struct {
	Summary  string   `json:"summary"`
	Findings []string `json:"findings"`
	Severity string   `json:"severity"`
}

// loadEnv is a simplified .env loader for this demo. For production use,
// consider github.com/joho/godotenv which handles quoting, export prefix,
// multi-line values, and other edge cases.
func loadEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])
			if os.Getenv(key) == "" {
				os.Setenv(key, val)
			}
		}
	}
}

func main() {
	loadEnv(".env")

	fmt.Println("=== Go SDK Harness Demo ===")
	fmt.Println()

	// Create a temp directory for output files
	workDir, err := os.MkdirTemp("", "harness-demo-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(workDir)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary":  map[string]any{"type": "string", "description": "A one-sentence summary"},
			"findings": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "List of findings"},
			"severity": map[string]any{"type": "string", "enum": []string{"low", "medium", "high"}, "description": "Overall severity"},
		},
		"required": []string{"summary", "findings", "severity"},
	}

	prompt := `Analyze this small Go function for potential issues:

func divide(a, b int) int {
    return a / b
}

Identify any bugs, edge cases, or improvements needed.`

	ctx := context.Background()

	// --- Test 1: Claude Code harness ---
	fmt.Println("--- Test 1: Claude Code Provider ---")
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println("SKIP: ANTHROPIC_API_KEY not set")
	} else {
		runClaudeCode(ctx, prompt, schema, workDir)
	}

	fmt.Println()

	// --- Test 2: OpenCode harness ---
	fmt.Println("--- Test 2: OpenCode Provider ---")
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		fmt.Println("SKIP: OPENROUTER_API_KEY not set")
	} else {
		runOpenCode(ctx, prompt, schema, workDir)
	}

	fmt.Println()
	fmt.Println("=== Demo Complete ===")
}

func runClaudeCode(ctx context.Context, prompt string, schema map[string]any, workDir string) {
	runner := harness.NewRunner(harness.Options{
		Provider:         "claude-code",
		Model:            "sonnet",
		MaxTurns:         5,
		PermissionMode:   "auto",
		Timeout:          120,
		SchemaMaxRetries: 1,
		Env: map[string]string{
			"ANTHROPIC_API_KEY": os.Getenv("ANTHROPIC_API_KEY"),
		},
	})

	var dest ReviewResult
	claudeDir, _ := os.MkdirTemp(workDir, "claude-")

	start := time.Now()
	result, err := runner.Run(ctx, prompt, schema, &dest, harness.Options{
		Cwd: claudeDir,
	})
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	printResult("claude-code", result, &dest, elapsed)
}

func runOpenCode(ctx context.Context, prompt string, schema map[string]any, workDir string) {
	// OpenCode picks model from its config based on available API keys.
	// With OPENROUTER_API_KEY set, it defaults to Claude Sonnet via OpenRouter.
	// We unset ANTHROPIC_API_KEY so opencode doesn't prefer the Anthropic
	// provider (which may have a session-scoped key that doesn't work).
	runner := harness.NewRunner(harness.Options{
		Provider:         "opencode",
		Timeout:          180,
		SchemaMaxRetries: 1,
		Env: map[string]string{
			"OPENROUTER_API_KEY": os.Getenv("OPENROUTER_API_KEY"),
			"ANTHROPIC_API_KEY":  "", // unset to force OpenRouter
		},
	})

	var dest ReviewResult
	openDir, _ := os.MkdirTemp(workDir, "opencode-")

	start := time.Now()
	result, err := runner.Run(ctx, prompt, schema, &dest, harness.Options{
		Cwd: openDir,
	})
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	printResult("opencode", result, &dest, elapsed)
}

func printResult(provider string, result *harness.Result, dest *ReviewResult, elapsed time.Duration) {
	fmt.Printf("  Provider:    %s\n", provider)
	fmt.Printf("  Duration:    %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  IsError:     %v\n", result.IsError)
	fmt.Printf("  NumTurns:    %d\n", result.NumTurns)
	fmt.Printf("  SessionID:   %s\n", result.SessionID)

	if result.IsError {
		fmt.Printf("  Error:       %s\n", result.ErrorMessage)
		fmt.Printf("  FailureType: %s\n", result.FailureType)
		// Print raw output for debugging
		if result.Result != "" {
			raw := result.Result
			if len(raw) > 500 {
				raw = raw[:500] + "..."
			}
			fmt.Printf("  Raw output:  %s\n", raw)
		}
		return
	}

	fmt.Printf("  Parsed OK:   %v\n", dest != nil && dest.Summary != "")
	if dest != nil && dest.Summary != "" {
		fmt.Printf("  Summary:     %s\n", dest.Summary)
		fmt.Printf("  Severity:    %s\n", dest.Severity)
		fmt.Printf("  Findings:    %d items\n", len(dest.Findings))
		for i, f := range dest.Findings {
			fmt.Printf("    [%d] %s\n", i+1, f)
		}
	}

	// Also dump the raw parsed JSON for verification
	if result.Parsed != nil {
		b, _ := json.MarshalIndent(dest, "  ", "  ")
		fmt.Printf("  JSON:\n  %s\n", string(b))
	}
}
