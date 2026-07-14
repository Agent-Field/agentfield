package cli

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

type psOptions struct {
	agent        string
	session      string
	outputFormat string
	stdout       io.Writer
	stdoutTTY    bool
}

type activeRunsResponse struct {
	Count int             `json:"count"`
	Runs  []activeRunItem `json:"runs"`
}

type activeRunItem struct {
	RunID            string         `json:"run_id"`
	RootExecutionID  string         `json:"root_execution_id,omitempty"`
	Target           string         `json:"target,omitempty"`
	AgentID          string         `json:"agent_id,omitempty"`
	ReasonerID       string         `json:"reasoner_id,omitempty"`
	RootStatus       string         `json:"root_status,omitempty"`
	ActiveExecutions int            `json:"active_executions"`
	TotalExecutions  int            `json:"total_executions"`
	StatusCounts     map[string]int `json:"status_counts"`
	SessionID        string         `json:"session_id,omitempty"`
	StartedAt        time.Time      `json:"started_at"`
	LatestActivity   time.Time      `json:"latest_activity"`
}

func NewPsCommand() *cobra.Command {
	opts := &psOptions{}
	cmd := &cobra.Command{
		Use:   "ps",
		Short: "List in-flight workflow runs",
		Long: `List every workflow run with at least one non-terminal execution.

A run whose LAST ACTIVITY is many minutes old while it still shows active
executions is likely wedged — inspect it with ` + "`af logs <agent>`" + ` and cancel the
whole run with POST /api/v1/workflows/<run_id>/cancel-tree if it is.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.stdout = os.Stdout
			opts.stdoutTTY = isOutputTerminal()
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			return runPs(ctx, opts)
		},
		SilenceUsage: true,
	}
	cmd.Flags().StringVar(&opts.agent, "agent", "", "Only runs touching this agent")
	cmd.Flags().StringVar(&opts.session, "session", "", "Only runs in this session")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runPs(ctx context.Context, opts *psOptions) error {
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	format := autoOutputFormat(opts.outputFormat, opts.stdoutTTY)
	if format != "pretty" && format != "json" && format != "yaml" {
		return cliExitError{Code: 2, Err: fmt.Errorf("output format must be pretty, json, or yaml")}
	}
	values := url.Values{}
	if strings.TrimSpace(opts.agent) != "" {
		values.Set("agent_id", strings.TrimSpace(opts.agent))
	}
	if strings.TrimSpace(opts.session) != "" {
		values.Set("session_id", strings.TrimSpace(opts.session))
	}
	resp, err := makeRequest(ctx, http.MethodGet, appendQuery("/api/v1/executions/active", values), nil, "application/json")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	var decoded activeRunsResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("ps failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	if format != "pretty" {
		return writeValue(opts.stdout, decoded, format)
	}
	renderActiveRuns(opts.stdout, decoded)
	return nil
}

func renderActiveRuns(out io.Writer, resp activeRunsResponse) {
	if resp.Count == 0 {
		fmt.Fprintln(out, "No runs in flight.")
		return
	}
	fmt.Fprintf(out, "%-34s %-28s %-9s %7s %7s  %-12s %s\n",
		"RUN", "TARGET", "STATUS", "ACTIVE", "TOTAL", "STARTED", "LAST ACTIVITY")
	for _, run := range resp.Runs {
		target := run.Target
		if target == "" {
			target = "-"
		}
		status := run.RootStatus
		if status == "" {
			status = "-"
		}
		fmt.Fprintf(out, "%-34s %-28s %-9s %7d %7d  %-12s %s\n",
			run.RunID,
			target,
			status,
			run.ActiveExecutions,
			run.TotalExecutions,
			relativeTimeValue(run.StartedAt),
			relativeTimeValue(run.LatestActivity),
		)
	}
	if resp.Count > len(resp.Runs) {
		fmt.Fprintf(out, "\n%d active runs total (showing %d)\n", resp.Count, len(resp.Runs))
	}
}

// relativeTimeValue is relativeTime for a concrete time.Time.
func relativeTimeValue(ts time.Time) string {
	if ts.IsZero() {
		return "-"
	}
	formatted := ts.UTC().Format(time.RFC3339)
	return relativeTime(&formatted)
}
