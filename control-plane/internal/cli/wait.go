package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

type waitOptions struct {
	// timeout is the total budget before giving up with exit code 2. Held as a
	// duration (not the raw --timeout seconds) so tests can inject sub-second
	// waits without sleeping a real second.
	timeout      time.Duration
	pollInterval time.Duration
	outputFormat string
	stdout       io.Writer
	stderr       io.Writer
	stdoutTTY    bool
}

// waitExecution is the slice of a run-overview execution record `af wait` needs
// to decide terminal state and surface the result. It mirrors the fields the
// `/api/v1/agentic/run/:run_id` overview (the same API `af agent run --id` uses)
// returns per execution.
type waitExecution struct {
	ExecutionID       string          `json:"execution_id"`
	ParentExecutionID *string         `json:"parent_execution_id"`
	Status            string          `json:"status"`
	Result            json.RawMessage `json:"result"`
	Error             *string         `json:"error"`
}

type runOverviewEnvelope struct {
	OK   bool `json:"ok"`
	Data struct {
		RunID      string          `json:"run_id"`
		Executions []waitExecution `json:"executions"`
	} `json:"data"`
}

type runOutcome struct {
	status string
	result interface{}
}

// NewWaitCommand builds `af wait <run_id>` — the blocking half of the async
// golden path (`af call --async` returns a run_id; `af wait` blocks on it).
func NewWaitCommand() *cobra.Command {
	opts := &waitOptions{}
	var timeoutSec int
	cmd := &cobra.Command{
		Use:   "wait <run_id>",
		Short: "Block until a run reaches a terminal state, then print its status and result",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.stdout = os.Stdout
			opts.stderr = os.Stderr
			opts.stdoutTTY = isOutputTerminal()
			opts.timeout = time.Duration(timeoutSec) * time.Second
			ctx, cancel := commandContext()
			defer cancel()
			return runWait(ctx, args[0], opts)
		},
		SilenceUsage: true,
	}
	cmd.Flags().IntVar(&timeoutSec, "timeout", 600, "Maximum seconds to wait before giving up (exit code 2)")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runWait(ctx context.Context, runID string, opts *waitOptions) error {
	if opts == nil {
		opts = &waitOptions{}
	}
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	if opts.stderr == nil {
		opts.stderr = os.Stderr
	}
	format := autoOutputFormat(opts.outputFormat, opts.stdoutTTY)
	if format != "pretty" && format != "json" && format != "yaml" {
		return cliExitError{Code: 2, Err: fmt.Errorf("output format must be pretty, json, or yaml")}
	}
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return cliExitError{Code: 2, Err: fmt.Errorf("run_id is required")}
	}

	timeout := opts.timeout
	if timeout <= 0 {
		timeout = 600 * time.Second
	}
	pollInterval := opts.pollInterval
	if pollInterval <= 0 {
		pollInterval = 2 * time.Second
	}

	deadline := time.Now().Add(timeout)
	for {
		outcome, ready, err := pollRunStatus(ctx, runID)
		if err != nil {
			return err
		}
		if ready {
			payload := map[string]interface{}{
				"run_id": runID,
				"status": outcome.status,
				"result": outcome.result,
			}
			if writeErr := writeValue(opts.stdout, payload, format); writeErr != nil {
				return cliExitError{Code: 2, Err: writeErr}
			}
			if failedStatus(outcome.status) {
				return cliExitError{Code: 1, Err: fmt.Errorf("run %s ended with status %s", runID, outcome.status)}
			}
			return nil
		}
		if !time.Now().Before(deadline) {
			fmt.Fprintf(opts.stderr, "timed out after %s waiting for run %s to finish\n", timeout, runID)
			return cliExitError{Code: 2, Err: fmt.Errorf("timed out waiting for run %s", runID)}
		}
		select {
		case <-ctx.Done():
			return cliExitError{Code: 2, Err: ctx.Err()}
		case <-time.After(pollInterval):
		}
	}
}

// pollRunStatus fetches the run overview once and reports whether the run has
// reached a terminal state. A run is terminal only when every one of its
// executions is terminal; a 404 (records not written yet after an async accept)
// is treated as "not ready" so a freshly-accepted run keeps polling.
func pollRunStatus(ctx context.Context, runID string) (runOutcome, bool, error) {
	resp, err := makeRequest(ctx, http.MethodGet, "/api/v1/agentic/run/"+url.PathEscape(runID), nil, "application/json")
	if err != nil {
		return runOutcome{}, false, cliExitError{Code: 3, Err: err}
	}
	var decoded runOverviewEnvelope
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return runOutcome{}, false, cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode == http.StatusNotFound {
		return runOutcome{}, false, nil
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return runOutcome{}, false, cliExitError{
			Code: httpExitCode(resp.StatusCode),
			Err:  fmt.Errorf("run status request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body))),
		}
	}

	execs := decoded.Data.Executions
	if len(execs) == 0 {
		return runOutcome{}, false, nil
	}
	for _, e := range execs {
		if !terminalStatus(e.Status) {
			return runOutcome{}, false, nil
		}
	}

	// A single failed execution makes the whole run failed.
	overall := "succeeded"
	for _, e := range execs {
		if failedStatus(e.Status) {
			overall = strings.ToLower(strings.TrimSpace(e.Status))
			break
		}
	}
	return runOutcome{status: overall, result: rootExecutionResult(execs)}, true, nil
}

// rootExecutionResult returns the result payload of the run's root execution
// (the one with no parent) — that is the reasoner the caller triggered. It
// falls back to the last execution when no explicit root is present.
func rootExecutionResult(execs []waitExecution) interface{} {
	chosen := &execs[len(execs)-1]
	for i := range execs {
		if execs[i].ParentExecutionID == nil {
			chosen = &execs[i]
			break
		}
	}
	if len(chosen.Result) == 0 {
		return nil
	}
	var decoded interface{}
	if err := json.Unmarshal(chosen.Result, &decoded); err != nil {
		return string(chosen.Result)
	}
	return decoded
}
