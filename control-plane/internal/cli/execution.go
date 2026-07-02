package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// NewExecutionCommand groups execution management subcommands.
func NewExecutionCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "execution",
		Short: "Manage workflow executions",
		Long:  "Cancel, pause, or resume workflow executions on the control plane.",
	}

	cmd.AddCommand(newCancelExecutionCommand())
	cmd.AddCommand(newCancelRunCommand())
	cmd.AddCommand(newPauseExecutionCommand())
	cmd.AddCommand(newResumeExecutionCommand())
	cmd.AddCommand(newRestartExecutionCommand())
	return cmd
}

type executionActionOptions struct {
	serverURL  string
	token      string
	timeout    time.Duration
	jsonOutput bool
	reason     string
	scope      string
	reuse      string
	fork       bool
	model      string
	input      string
}

func defaultExecutionActionOptions() executionActionOptions {
	return executionActionOptions{
		serverURL: os.Getenv("AGENTFIELD_SERVER"),
		token:     os.Getenv("AGENTFIELD_TOKEN"),
		timeout:   15 * time.Second,
	}
}

func newRestartExecutionCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()
	opts.scope = "workflow"
	opts.reuse = "succeeded-before"

	cmd := &cobra.Command{
		Use:   "restart <execution_id>",
		Short: "Restart a workflow from an execution point",
		Long:  "Start a new run from an existing execution. By default, restarts the containing workflow and reuses successful app.call outputs before that point.",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			body, err := buildRestartExecutionBody(opts)
			if err != nil {
				return err
			}
			_, err = runExecutionAction(executionActionConfig{
				actionName:  "restart",
				successVerb: "restarted",
				endpoint:    "/api/v1/executions/%s/restart",
				opts:        &opts,
				executionID: args[0],
				withBody:    true,
				body:        body,
			})
			return err
		},
	}

	cmd.Flags().StringVar(&opts.scope, "scope", opts.scope, "Restart scope: workflow or execution")
	cmd.Flags().StringVar(&opts.reuse, "reuse", opts.reuse, "Replay reuse mode: succeeded-before, all-succeeded, or none")
	cmd.Flags().BoolVar(&opts.fork, "fork", false, "Mark this restart as a fork with intentional changes")
	cmd.Flags().StringVar(&opts.model, "model", "", "Model override to send in restart context")
	cmd.Flags().StringVar(&opts.input, "input", "", "JSON input override or @path to a JSON file")
	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for restarting the execution")
	bindExecutionActionFlags(cmd, &opts)
	return cmd
}

func newCancelExecutionCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()

	cmd := &cobra.Command{
		Use:   "cancel <execution_id>",
		Short: "Cancel a workflow execution",
		Long:  "Cancel one execution by execution_id. This preserves the historical per-execution behavior; use `af execution cancel-run <run_id>` to stop every non-terminal execution in a run.",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			_, err := runExecutionAction(executionActionConfig{
				actionName:  "cancel",
				successVerb: "cancelled",
				endpoint:    "/api/v1/executions/%s/cancel",
				opts:        &opts,
				executionID: args[0],
				targetKind:  "execution",
				withReason:  true,
			})
			return err
		},
	}

	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for cancelling the execution")
	bindExecutionActionFlags(cmd, &opts)
	return cmd
}

func newCancelRunCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()

	cmd := &cobra.Command{
		Use:   "cancel-run <run_id>",
		Short: "Cancel every active execution in a run",
		Long:  "Cancel every non-terminal execution belonging to a run, bottom-up. This is the safe command when the intent is to stop a whole workflow with child executions.",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			_, err := runExecutionAction(executionActionConfig{
				actionName:  "cancel",
				successVerb: "cancelled",
				endpoint:    "/api/v1/workflows/%s/cancel-tree",
				opts:        &opts,
				executionID: args[0],
				targetKind:  "run",
				withReason:  true,
			})
			return err
		},
	}

	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for cancelling the run")
	bindExecutionActionFlags(cmd, &opts)
	return cmd
}

func newPauseExecutionCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()

	cmd := &cobra.Command{
		Use:   "pause <execution_id>",
		Short: "Pause a running workflow execution",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			_, err := runExecutionAction(executionActionConfig{
				actionName:  "pause",
				successVerb: "paused",
				endpoint:    "/api/v1/executions/%s/pause",
				opts:        &opts,
				executionID: args[0],
				targetKind:  "execution",
				withReason:  true,
			})
			return err
		},
	}

	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for pausing the execution")
	bindExecutionActionFlags(cmd, &opts)
	return cmd
}

func newResumeExecutionCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()

	cmd := &cobra.Command{
		Use:   "resume <execution_id>",
		Short: "Resume a paused workflow execution",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			_, err := runExecutionAction(executionActionConfig{
				actionName:  "resume",
				successVerb: "resumed",
				endpoint:    "/api/v1/executions/%s/resume",
				opts:        &opts,
				executionID: args[0],
				targetKind:  "execution",
			})
			return err
		},
	}

	bindExecutionActionFlags(cmd, &opts)
	return cmd
}

func bindExecutionActionFlags(cmd *cobra.Command, opts *executionActionOptions) {
	cmd.Flags().StringVar(&opts.serverURL, "server", opts.serverURL, "Control plane URL (default: http://localhost:8080 or $AGENTFIELD_SERVER)")
	cmd.Flags().StringVar(&opts.token, "token", opts.token, "Bearer token for the control plane (default: $AGENTFIELD_TOKEN)")
	cmd.Flags().DurationVar(&opts.timeout, "timeout", opts.timeout, "HTTP timeout for the execution request")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", false, "Print raw JSON response")
}

type executionActionConfig struct {
	actionName  string
	successVerb string
	endpoint    string
	opts        *executionActionOptions
	executionID string
	targetKind  string
	withReason  bool
	withBody    bool
	body        map[string]interface{}
}

func runExecutionAction(cfg executionActionConfig) (map[string]any, error) {
	server := strings.TrimSpace(cfg.opts.serverURL)
	if server == "" {
		server = "http://localhost:8080"
	}
	server = strings.TrimSuffix(server, "/")

	var bodyBytes []byte
	if cfg.withBody {
		payload := map[string]interface{}{}
		for key, value := range cfg.body {
			switch typed := value.(type) {
			case string:
				if strings.TrimSpace(typed) != "" {
					payload[key] = typed
				}
			case nil:
			default:
				payload[key] = value
			}
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("encode payload: %w", err)
		}
		bodyBytes = encoded
	} else if cfg.withReason {
		payload := map[string]string{}
		if strings.TrimSpace(cfg.opts.reason) != "" {
			payload["reason"] = cfg.opts.reason
		}

		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("encode payload: %w", err)
		}
		bodyBytes = encoded
	}

	url := server + fmt.Sprintf(cfg.endpoint, cfg.executionID)
	var bodyReader *bytes.Reader
	if bodyBytes != nil {
		bodyReader = bytes.NewReader(bodyBytes)
	} else {
		bodyReader = bytes.NewReader(nil)
	}

	client := &http.Client{Timeout: cfg.opts.timeout}
	req, err := http.NewRequest(http.MethodPost, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if cfg.withReason || cfg.withBody {
		req.Header.Set("Content-Type", "application/json")
	}
	if cfg.opts.token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.opts.token)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	parsed := map[string]any{}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if resp.StatusCode >= 300 {
		return nil, formatExecutionActionError(resp.StatusCode, cfg.actionName, cfg.targetKind, cfg.executionID, parsed)
	}

	if cfg.opts.jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(parsed); err != nil {
			return nil, fmt.Errorf("write json output: %w", err)
		}
		return parsed, nil
	}

	printExecutionActionHumanOutput(parsed, cfg.successVerb)
	return parsed, nil
}

func buildRestartExecutionBody(opts executionActionOptions) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"scope": opts.scope,
		"reuse": opts.reuse,
	}
	if strings.TrimSpace(opts.reason) != "" {
		body["reason"] = opts.reason
	}
	if opts.fork {
		body["fork"] = true
	}
	context := map[string]interface{}{}
	if strings.TrimSpace(opts.model) != "" {
		context["model"] = strings.TrimSpace(opts.model)
	}
	if len(context) > 0 {
		body["context"] = context
	}
	if strings.TrimSpace(opts.input) != "" {
		input, err := parseRestartInput(opts.input)
		if err != nil {
			return nil, err
		}
		body["input"] = input
	}
	return body, nil
}

func parseRestartInput(value string) (map[string]interface{}, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return nil, nil
	}
	raw := []byte(trimmed)
	if strings.HasPrefix(trimmed, "@") {
		path := strings.TrimSpace(strings.TrimPrefix(trimmed, "@"))
		if path == "" {
			return nil, fmt.Errorf("--input @path requires a file path")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read input file %q: %w", path, err)
		}
		raw = data
	}
	var input map[string]interface{}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("parse --input JSON: %w", err)
	}
	return input, nil
}

func printExecutionActionHumanOutput(parsed map[string]any, successVerb string) {
	executionID, _ := parsed["execution_id"].(string)
	previousStatus, _ := parsed["previous_status"].(string)
	runID, _ := parsed["run_id"].(string)
	newRunID, _ := parsed["run_id"].(string)
	sourceRunID, _ := parsed["source_run_id"].(string)
	sourceExecutionID, _ := parsed["source_execution_id"].(string)
	reuse, _ := parsed["replay_mode"].(string)

	if successVerb == "cancelled" && runID != "" && executionID == "" {
		fmt.Printf("Run %s cancelled", runID)
		cancelled, hasCancelled := numberAsInt(parsed["cancelled_count"])
		skipped, hasSkipped := numberAsInt(parsed["skipped_count"])
		errors, hasErrors := numberAsInt(parsed["error_count"])
		if hasCancelled || hasSkipped || hasErrors {
			fmt.Printf(": %d cancelled, %d skipped, %d errors", cancelled, skipped, errors)
		}
		fmt.Println()
		return
	}

	if successVerb == "restarted" && newRunID != "" {
		if sourceRunID != "" && sourceExecutionID != "" {
			fmt.Printf("Restarted run %s from %s\n", sourceRunID, sourceExecutionID)
		} else if executionID != "" {
			fmt.Printf("Execution %s restarted\n", executionID)
		}
		fmt.Printf("New run: %s\n", newRunID)
		if reuse != "" {
			fmt.Printf("Reuse: %s\n", reuse)
		}
		return
	}

	if executionID != "" && previousStatus != "" {
		fmt.Printf("Execution %s %s (was: %s)\n", executionID, successVerb, previousStatus)
	} else if executionID != "" {
		fmt.Printf("Execution %s %s\n", executionID, successVerb)
	} else {
		fmt.Printf("Execution %s\n", successVerb)
	}

	reason, _ := parsed["reason"].(string)
	if strings.TrimSpace(reason) != "" {
		fmt.Printf("Reason: %s\n", reason)
	}
	warning, _ := parsed["warning"].(string)
	if strings.TrimSpace(warning) != "" {
		fmt.Printf("Warning: %s\n", warning)
	}
	suggestedEndpoint, _ := parsed["suggested_endpoint"].(string)
	if strings.TrimSpace(suggestedEndpoint) != "" {
		fmt.Printf("Suggestion: POST %s\n", suggestedEndpoint)
	}
}

func numberAsInt(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		return int(typed), true
	case json.Number:
		asInt, err := typed.Int64()
		if err != nil {
			return 0, false
		}
		return int(asInt), true
	default:
		return 0, false
	}
}

func formatExecutionActionError(statusCode int, actionName, targetKind, executionID string, parsed map[string]any) error {
	if strings.TrimSpace(targetKind) == "" {
		targetKind = "execution"
	}
	message := ""
	if v, ok := parsed["message"].(string); ok {
		message = strings.TrimSpace(v)
	}
	if message == "" {
		if v, ok := parsed["error"].(string); ok {
			message = strings.TrimSpace(v)
		}
	}

	switch statusCode {
	case http.StatusNotFound:
		if message != "" {
			return fmt.Errorf("%s %s not found: %s", targetKind, executionID, message)
		}
		return fmt.Errorf("%s %s not found", targetKind, executionID)
	case http.StatusConflict:
		if message != "" {
			return fmt.Errorf("cannot %s %s %s: %s", actionName, targetKind, executionID, message)
		}
		return fmt.Errorf("cannot %s %s %s in its current state", actionName, targetKind, executionID)
	default:
		if message != "" {
			return fmt.Errorf("failed to %s %s %s (%d): %s", actionName, targetKind, executionID, statusCode, message)
		}
		return fmt.Errorf("failed to %s %s %s (%d)", actionName, targetKind, executionID, statusCode)
	}
}
