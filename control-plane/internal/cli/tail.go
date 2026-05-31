package cli

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

type tailOptions struct {
	fromStep     int
	outputFormat string
	stdout       io.Writer
}

func NewTailCommand() *cobra.Command {
	opts := &tailOptions{}
	cmd := &cobra.Command{
		Use:   "tail <run_id>",
		Short: "Attach to a running execution stream",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.stdout = os.Stdout
			ctx, cancel := commandContext()
			defer cancel()
			format := autoOutputFormat(opts.outputFormat, isOutputTerminal())
			if format == "yaml" {
				return cliExitError{Code: 2, Err: fmt.Errorf("tail supports pretty or json output")}
			}
			err := streamExecutionEvents(ctx, args[0], opts.fromStep, format, opts.stdout)
			if err != nil {
				if ctx.Err() != nil {
					return nil
				}
				return err
			}
			return nil
		},
		SilenceUsage: true,
	}
	cmd.Flags().IntVar(&opts.fromStep, "from", 0, "Resume stream from a specific step")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "", "Output format: pretty, json")
	return cmd
}

func streamExecutionEvents(ctx context.Context, runID string, fromStep int, format string, out io.Writer) error {
	values := url.Values{}
	if fromStep > 0 {
		values.Set("from", strconv.Itoa(fromStep))
	}
	path := appendQuery("/api/v1/executions/"+url.PathEscape(strings.TrimSpace(runID))+"/events", values)
	resp, err := makeRequest(ctx, http.MethodGet, path, nil, "text/event-stream")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		body, readErr := readJSONResponse(resp, nil)
		if readErr != nil {
			return cliExitError{Code: 3, Err: readErr}
		}
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("tail failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	defer resp.Body.Close()

	seen := 0
	failed := false
	err = scanSSE(resp.Body, func(event map[string]interface{}) bool {
		seen++
		if format == "json" {
			_ = writeValue(out, event, "json")
		} else {
			renderProgressEvent(out, seen, event)
		}
		status, _ := event["status"].(string)
		if failedStatus(status) {
			failed = true
		}
		return !terminalStatus(status)
	})
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if failed {
		return cliExitError{Code: 1, Err: fmt.Errorf("execution failed")}
	}
	return nil
}

func renderProgressEvent(out io.Writer, step int, event map[string]interface{}) {
	status, _ := event["status"].(string)
	eventType, _ := event["type"].(string)
	executionID, _ := event["execution_id"].(string)
	if eventType == "" {
		eventType = "execution"
	}
	if status == "" {
		status = "updated"
	}
	fmt.Fprintf(out, "%d  %s  %s  %s\n", step, status, executionID, eventType)
}
