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
	cmd.AddCommand(newPauseExecutionCommand())
	cmd.AddCommand(newResumeExecutionCommand())
	return cmd
}

type executionActionOptions struct {
	serverURL  string
	token      string
	timeout    time.Duration
	jsonOutput bool
	reason     string
}

func defaultExecutionActionOptions() executionActionOptions {
	return executionActionOptions{
		serverURL: os.Getenv("AGENTFIELD_SERVER"),
		token:     os.Getenv("AGENTFIELD_TOKEN"),
		timeout:   15 * time.Second,
	}
}

func newCancelExecutionCommand() *cobra.Command {
	opts := defaultExecutionActionOptions()

	cmd := &cobra.Command{
		Use:   "cancel <execution_id>",
		Short: "Cancel a workflow execution",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			_, err := runExecutionAction(executionActionConfig{
				actionName:  "cancel",
				successVerb: "cancelled",
				endpoint:    "/api/v1/executions/%s/cancel",
				opts:        &opts,
				executionID: args[0],
				withReason:  true,
			})
			return err
		},
	}

	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for cancelling the execution")
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
	withReason  bool
}

func runExecutionAction(cfg executionActionConfig) (map[string]any, error) {
	server := strings.TrimSpace(cfg.opts.serverURL)
	if server == "" {
		server = "http://localhost:8080"
	}
	server = strings.TrimSuffix(server, "/")

	var bodyBytes []byte
	if cfg.withReason {
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
	if cfg.withReason {
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
		return nil, formatExecutionActionError(resp.StatusCode, cfg.actionName, cfg.executionID, parsed)
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

func printExecutionActionHumanOutput(parsed map[string]any, successVerb string) {
	executionID, _ := parsed["execution_id"].(string)
	previousStatus, _ := parsed["previous_status"].(string)

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
}

func formatExecutionActionError(statusCode int, actionName, executionID string, parsed map[string]any) error {
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
			return fmt.Errorf("execution %s not found: %s", executionID, message)
		}
		return fmt.Errorf("execution %s not found", executionID)
	case http.StatusConflict:
		if message != "" {
			return fmt.Errorf("cannot %s execution %s: %s", actionName, executionID, message)
		}
		return fmt.Errorf("cannot %s execution %s in its current state", actionName, executionID)
	default:
		if message != "" {
			return fmt.Errorf("failed to %s execution %s (%d): %s", actionName, executionID, statusCode, message)
		}
		return fmt.Errorf("failed to %s execution %s (%d)", actionName, executionID, statusCode)
	}
}
