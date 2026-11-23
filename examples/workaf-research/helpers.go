package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// harnessInto runs one coding-agent call in dir and fills dest from its
// structured answer.
func harnessInto[T any](ctx context.Context, app *agent.Agent, prompt, dir string, dest *T) (any, error) {
	schema, err := harness.StructToJSONSchema(*dest)
	if err != nil {
		return nil, err
	}
	res, err := app.Harness(ctx, prompt, schema, dest, harness.Options{Cwd: dir, ProjectDir: dir})
	if err != nil {
		return nil, err
	}
	if res.IsError {
		return nil, fmt.Errorf("%s", res.ErrorMessage)
	}
	return dest, nil
}

func render(v any) string {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil || string(raw) == "null" {
		return "(nothing)"
	}
	return string(raw)
}

func str(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

// secret reads KEY, or the file named by KEY_FILE.
func secret(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	if path := os.Getenv(key + "_FILE"); path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			log.Fatalf("workaf-research: %s_FILE: %v", key, err)
		}
		return strings.TrimSpace(string(raw))
	}
	return ""
}
