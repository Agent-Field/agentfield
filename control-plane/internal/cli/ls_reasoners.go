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

type lsOptions struct {
	all          bool
	node         string
	live         bool
	entrypoints  bool
	outputFormat string
	stdout       io.Writer
	stdoutTTY    bool
}

type reasonerListResponse struct {
	Reasoners []reasonerListItem `json:"reasoners"`
	Shown     int                `json:"shown"`
	Total     int                `json:"total"`
}

type reasonerListItem struct {
	Node        string   `json:"node"`
	Reasoner    string   `json:"reasoner"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	LastRunAt   *string  `json:"last_run_at,omitempty"`
	Status      string   `json:"status"`
}

func NewReasonerListCommand() *cobra.Command {
	opts := &lsOptions{}
	cmd := &cobra.Command{
		Use:   "ls [query]",
		Short: "List reasoners",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.stdout = os.Stdout
			opts.stdoutTTY = isOutputTerminal()
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			query := ""
			if len(args) > 0 {
				query = args[0]
			}
			return runReasonerList(ctx, query, opts)
		},
		SilenceUsage: true,
	}
	cmd.Flags().BoolVar(&opts.all, "all", false, "Show every reasoner")
	cmd.Flags().StringVar(&opts.node, "node", "", "Filter to a single node")
	cmd.Flags().BoolVar(&opts.live, "live", false, "Only show reasoners whose node is live")
	cmd.Flags().BoolVarP(&opts.entrypoints, "entrypoints", "e", false, "Only show reasoners tagged as entry points")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runReasonerList(ctx context.Context, query string, opts *lsOptions) error {
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	format := autoOutputFormat(opts.outputFormat, opts.stdoutTTY)
	if format != "pretty" && format != "json" && format != "yaml" {
		return cliExitError{Code: 2, Err: fmt.Errorf("output format must be pretty, json, or yaml")}
	}
	values := url.Values{}
	if strings.TrimSpace(query) != "" {
		values.Set("query", strings.TrimSpace(query))
	}
	if opts.all {
		values.Set("all", "true")
	}
	if strings.TrimSpace(opts.node) != "" {
		values.Set("node", strings.TrimSpace(opts.node))
	}
	if opts.live {
		values.Set("live", "true")
	}
	if opts.entrypoints {
		values.Set("entrypoints", "true")
	}
	resp, err := makeRequest(ctx, http.MethodGet, appendQuery("/api/v1/reasoners", values), nil, "application/json")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	var decoded reasonerListResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("list failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	if format != "pretty" {
		return writeValue(opts.stdout, decoded, format)
	}
	renderReasonerList(opts.stdout, decoded, !opts.all && query == "")
	return nil
}

func renderReasonerList(out io.Writer, resp reasonerListResponse, recentHeader bool) {
	if recentHeader {
		fmt.Fprintln(out, "RECENT")
	}
	for _, item := range resp.Reasoners {
		name := item.Node + "." + item.Reasoner
		fmt.Fprintf(out, "%-28s %-12s %-6s %s\n", name, relativeTime(item.LastRunAt), item.Status, describeReasoner(item))
	}
	if resp.Total > resp.Shown {
		fmt.Fprintf(out, "\n%d more recent  -  %d total  -  use `af ls --all`\n", resp.Total-resp.Shown, resp.Total)
	}
}

// describeReasoner renders the trailing description column: entry points are
// labeled so callers can spot a node's intended surface, and long descriptions
// are truncated to keep rows on one line.
func describeReasoner(item reasonerListItem) string {
	const maxLen = 80
	desc := strings.Join(strings.Fields(item.Description), " ")
	if len(desc) > maxLen {
		desc = desc[:maxLen-1] + "…"
	}
	entry := false
	for _, tag := range item.Tags {
		if strings.EqualFold(strings.TrimSpace(tag), "entrypoint") {
			entry = true
			break
		}
	}
	switch {
	case entry && desc != "":
		return "[entrypoint] " + desc
	case entry:
		return "[entrypoint]"
	default:
		return desc
	}
}

func relativeTime(value *string) string {
	if value == nil || strings.TrimSpace(*value) == "" {
		return "-"
	}
	ts, err := time.Parse(time.RFC3339, strings.TrimSpace(*value))
	if err != nil {
		return "-"
	}
	d := time.Since(ts)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "yesterday"
	default:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	}
}
