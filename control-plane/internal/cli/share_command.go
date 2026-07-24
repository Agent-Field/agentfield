package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/share"
	"github.com/spf13/cobra"
)

// shareOptions holds resolved flags for `af share`.
type shareOptions struct {
	output string
	title  string
	public bool
	demo   bool
	redact bool
}

// NewShareCommand builds `af share` — turn a workflow run into a self-contained,
// offline HTML artifact you can screenshot or hand to a teammate.
func NewShareCommand() *cobra.Command {
	opts := &shareOptions{}
	cmd := &cobra.Command{
		Use:   "share [workflow-id]",
		Short: "Export a workflow run as a self-contained shareable HTML file",
		Long: `Export a workflow run as a single, self-contained HTML file you can open
offline, screenshot, or hand to a teammate. Nothing leaves your machine.

The exported file inlines a versioned run bundle (DAG, timeline, per-node
input/output previews) and renders it with vanilla JS and inline CSS. No CDN,
no network, works from disk.

Examples:
  # Export a run to run-<id>.html in the current directory
  af share run-a1b2c3d4

  # Choose the output path
  af share run-a1b2c3d4 -o /tmp/audit.html

  # Strip all input/output previews from the artifact
  af share run-a1b2c3d4 --redact

  # Build from an embedded demo fixture (no control plane needed)
  af share --demo

  # Also publish a shareable permalink you can send or post (opt-in)
  af share run-a1b2c3d4 --public
  # => https://agentfield.ai/share/<token>

Server resolution matches other af commands: --server, then $AGENTFIELD_SERVER,
then http://localhost:8080. Hosted sharing (--public) posts the bundle JSON to
agentfield.ai and prints the permalink. Point AGENTFIELD_SHARE_URL at your own
share server to publish there instead.`,
		Args: func(cmd *cobra.Command, args []string) error {
			if opts.demo {
				return cobra.MaximumNArgs(1)(cmd, args)
			}
			return cobra.ExactArgs(1)(cmd, args)
		},
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			workflowID := ""
			if len(args) > 0 {
				workflowID = strings.TrimSpace(args[0])
			}
			return runShare(ctx, workflowID, opts)
		},
	}

	cmd.Flags().StringVarP(&opts.output, "output", "o", "", "Output HTML path (default: run-<workflow-id>.html in the current directory)")
	cmd.Flags().StringVar(&opts.title, "title", "", "Human title for the shared run (default: \"run of <agent>.<func>\" from the entry node)")
	cmd.Flags().BoolVar(&opts.public, "public", false, "Publish a shareable permalink (uploads to agentfield.ai; override with AGENTFIELD_SHARE_URL)")
	cmd.Flags().BoolVar(&opts.demo, "demo", false, "Render from an embedded demo fixture instead of the control plane")
	cmd.Flags().BoolVar(&opts.redact, "redact", false, "Replace every input/output preview with \"[redacted]\"")
	return cmd
}

func runShare(ctx context.Context, workflowID string, opts *shareOptions) error {
	var bundle *share.Bundle
	var err error

	if opts.demo {
		bundle = share.DemoBundle()
		if workflowID != "" {
			bundle.WorkflowID = workflowID
		}
		workflowID = bundle.WorkflowID
	} else {
		if workflowID == "" {
			return cliExitError{Code: 2, Err: fmt.Errorf("a workflow-id is required (or use --demo)")}
		}
		bundle, err = buildBundleFromServer(ctx, workflowID)
		if err != nil {
			return err
		}
	}

	// Resolve the human title: explicit --title wins, otherwise fall back to a
	// default derived from the entry node ("run of <agent>.<func>").
	if t := strings.TrimSpace(opts.title); t != "" {
		bundle.Title = t
	} else if strings.TrimSpace(bundle.Title) == "" {
		bundle.Title = deriveTitle(bundle)
	}

	if opts.redact {
		share.Redact(bundle)
	}

	html, err := share.RenderHTML(bundle)
	if err != nil {
		return cliExitError{Code: 1, Err: err}
	}

	outPath := opts.output
	if strings.TrimSpace(outPath) == "" {
		base := sanitizeFilename(workflowID)
		// Avoid a doubled "run-run-" prefix when the id is already a run id.
		if !strings.HasPrefix(base, "run-") {
			base = "run-" + base
		}
		outPath = base + ".html"
	}
	abs, err := filepath.Abs(outPath)
	if err != nil {
		abs = outPath
	}
	if err := os.WriteFile(abs, html, 0o644); err != nil {
		return cliExitError{Code: 1, Err: fmt.Errorf("write %s: %w", abs, err)}
	}
	fmt.Println(abs)

	if opts.public {
		if err := publishBundle(ctx, bundle); err != nil {
			return err
		}
	}
	return nil
}

// buildBundleFromServer fetches the run DAG and per-execution details from the
// control plane and maps them into the versioned share bundle.
func buildBundleFromServer(ctx context.Context, workflowID string) (*share.Bundle, error) {
	dag, err := fetchWorkflowDAG(ctx, workflowID)
	if err != nil {
		return nil, err
	}
	if len(dag.Timeline) == 0 {
		return nil, cliExitError{Code: 2, Err: fmt.Errorf("workflow %q has no executions", workflowID)}
	}

	nodes := make([]share.BundleNode, 0, len(dag.Timeline))
	edges := make([]share.BundleEdge, 0, len(dag.Timeline))
	var totalDuration int64

	// Fetch per-execution input/output previews concurrently. Cost and model
	// are not tracked on execution records, so they are left null.
	details := fetchExecutionDetails(ctx, dag.Timeline)

	for _, tn := range dag.Timeline {
		agent := tn.AgentNodeID
		fn := tn.ReasonerID
		node := share.BundleNode{
			ID:        tn.ExecutionID,
			Agent:     agent,
			Func:      fn,
			Status:    tn.Status,
			StartedAt: tn.StartedAt,
			Model:     nil, // not available from execution records
			CostUSD:   nil, // not available from execution records
		}
		if tn.CompletedAt != nil {
			node.EndedAt = *tn.CompletedAt
		}
		if tn.DurationMS != nil {
			node.DurationMS = tn.DurationMS
		}
		if d, ok := details[tn.ExecutionID]; ok {
			node.InputPreview = share.TruncatePreview(d.inputPreview)
			node.OutputPreview = share.TruncatePreview(d.outputPreview)
			if d.errMessage != "" {
				msg := d.errMessage
				node.Error = &msg
			}
		}
		nodes = append(nodes, node)

		if tn.ParentExecutionID != nil && *tn.ParentExecutionID != "" {
			edges = append(edges, share.BundleEdge{From: *tn.ParentExecutionID, To: tn.ExecutionID})
		}
	}

	// Wall-clock duration: span from earliest start to latest end across nodes.
	totalDuration = computeWallClock(nodes)

	return &share.Bundle{
		Version:     share.BundleVersion,
		WorkflowID:  workflowID,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Totals: share.BundleTotals{
			Agents:     len(nodes),
			DurationMS: totalDuration,
			CostUSD:    nil, // control plane does not track cost per execution
		},
		Nodes: nodes,
		Edges: edges,
	}, nil
}

// dagTimelineNode mirrors the subset of WorkflowDAGNode fields the share bundle
// needs. Declared locally so the CLI does not import the handlers package.
type dagTimelineNode struct {
	ExecutionID       string  `json:"execution_id"`
	AgentNodeID       string  `json:"agent_node_id"`
	ReasonerID        string  `json:"reasoner_id"`
	Status            string  `json:"status"`
	StartedAt         string  `json:"started_at"`
	CompletedAt       *string `json:"completed_at"`
	DurationMS        *int64  `json:"duration_ms"`
	ParentExecutionID *string `json:"parent_execution_id"`
}

type dagResponse struct {
	RootWorkflowID string            `json:"root_workflow_id"`
	WorkflowStatus string            `json:"workflow_status"`
	WorkflowName   string            `json:"workflow_name"`
	TotalNodes     int               `json:"total_nodes"`
	Timeline       []dagTimelineNode `json:"timeline"`
}

func fetchWorkflowDAG(ctx context.Context, workflowID string) (*dagResponse, error) {
	path := fmt.Sprintf("/api/ui/v1/workflows/%s/dag", workflowID)
	resp, err := makeRequest(ctx, http.MethodGet, path, nil, "application/json")
	if err != nil {
		return nil, cliExitError{Code: 3, Err: fmt.Errorf("fetch workflow DAG: %w", err)}
	}
	var decoded dagResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, cliExitError{Code: 2, Err: fmt.Errorf("workflow %q not found on the control plane", workflowID)}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("workflow DAG request failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	return &decoded, nil
}

// executionPreview holds the preview fields pulled from the execution details
// endpoint for a single execution.
type executionPreview struct {
	inputPreview  string
	outputPreview string
	errMessage    string
}

type executionDetailsResponse struct {
	InputData    json.RawMessage `json:"input_data"`
	OutputData   json.RawMessage `json:"output_data"`
	ErrorMessage *string         `json:"error_message"`
}

// fetchExecutionDetails fetches input/output previews for each execution in the
// timeline concurrently (bounded), returning a map keyed by execution id. A
// failure to fetch one execution degrades gracefully to an empty preview.
func fetchExecutionDetails(ctx context.Context, timeline []dagTimelineNode) map[string]executionPreview {
	out := make(map[string]executionPreview, len(timeline))
	var mu sync.Mutex

	sem := make(chan struct{}, 8)
	var wg sync.WaitGroup
	for _, tn := range timeline {
		execID := tn.ExecutionID
		if execID == "" {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			preview := fetchOneExecutionPreview(ctx, execID)
			mu.Lock()
			out[execID] = preview
			mu.Unlock()
		}()
	}
	wg.Wait()
	return out
}

func fetchOneExecutionPreview(ctx context.Context, execID string) executionPreview {
	path := fmt.Sprintf("/api/ui/v1/executions/%s/details", execID)
	resp, err := makeRequest(ctx, http.MethodGet, path, nil, "application/json")
	if err != nil {
		return executionPreview{}
	}
	var decoded executionDetailsResponse
	if _, err := readJSONResponse(resp, &decoded); err != nil {
		return executionPreview{}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return executionPreview{}
	}
	preview := executionPreview{
		inputPreview:  rawJSONPreview(decoded.InputData),
		outputPreview: rawJSONPreview(decoded.OutputData),
	}
	if decoded.ErrorMessage != nil {
		preview.errMessage = strings.TrimSpace(*decoded.ErrorMessage)
	}
	return preview
}

// rawJSONPreview renders raw JSON as a compact, human-readable preview string.
// Objects/arrays are pretty-printed; scalars are stringified.
func rawJSONPreview(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return ""
	}
	var v interface{}
	if err := json.Unmarshal(trimmed, &v); err != nil {
		return string(trimmed)
	}
	pretty, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return string(trimmed)
	}
	return string(pretty)
}

// computeWallClock returns the span in milliseconds from the earliest node
// start to the latest node end. Falls back to summing durations when timestamps
// are missing.
func computeWallClock(nodes []share.BundleNode) int64 {
	var minStart, maxEnd time.Time
	var haveBounds bool
	var sumDur int64

	for _, n := range nodes {
		if n.DurationMS != nil {
			sumDur += *n.DurationMS
		}
		start, err := time.Parse(time.RFC3339, n.StartedAt)
		if err != nil {
			continue
		}
		end := start
		if n.EndedAt != "" {
			if e, err := time.Parse(time.RFC3339, n.EndedAt); err == nil {
				end = e
			}
		} else if n.DurationMS != nil {
			end = start.Add(time.Duration(*n.DurationMS) * time.Millisecond)
		}
		if !haveBounds {
			minStart, maxEnd, haveBounds = start, end, true
			continue
		}
		if start.Before(minStart) {
			minStart = start
		}
		if end.After(maxEnd) {
			maxEnd = end
		}
	}
	if haveBounds {
		if span := maxEnd.Sub(minStart).Milliseconds(); span > 0 {
			return span
		}
	}
	return sumDur
}

// defaultShareHost is the hosted share server used by --public when the user
// hasn't pointed AGENTFIELD_SHARE_URL at a self-hosted one.
const defaultShareHost = "https://agentfield.ai"

// publishBundle posts the bundle JSON to a hosted share server and prints the
// returned permalink. Defaults to agentfield.ai; override with
// AGENTFIELD_SHARE_URL to publish to a self-hosted share server instead.
func publishBundle(ctx context.Context, bundle *share.Bundle) error {
	base := strings.TrimSpace(os.Getenv("AGENTFIELD_SHARE_URL"))
	if base == "" {
		base = defaultShareHost
	}
	base = strings.TrimRight(base, "/")
	url := base + "/api/v1/shares"

	body, err := json.Marshal(bundle)
	if err != nil {
		return cliExitError{Code: 1, Err: fmt.Errorf("encode bundle: %w", err)}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return cliExitError{Code: 1, Err: fmt.Errorf("build share request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "af-cli/share")
	if key := strings.TrimSpace(GetAPIKey()); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("publish to %s: %w", url, err)}
	}
	defer resp.Body.Close()

	var decoded struct {
		URL   string `json:"url"`
		ID    string `json:"id"`
		Error string `json:"error"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&decoded)

	if resp.StatusCode >= http.StatusBadRequest {
		msg := decoded.Error
		if msg == "" {
			msg = fmt.Sprintf("share server returned status %d", resp.StatusCode)
		}
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("hosted sharing failed: %s", msg)}
	}

	if decoded.URL != "" {
		fmt.Println(decoded.URL)
	} else if decoded.ID != "" {
		fmt.Printf("%s/s/%s\n", base, decoded.ID)
	} else {
		fmt.Println("published to hosted share server")
	}
	return nil
}

// deriveTitle builds a default human title from the run's entry node, e.g.
// "run of orchestrator.run_audit". The entry node is the DAG root: a node that
// is never the target of an edge. Falls back to the first node, then to a
// generic label when the bundle is empty.
func deriveTitle(b *share.Bundle) string {
	if b == nil || len(b.Nodes) == 0 {
		return "shared run"
	}
	hasParent := make(map[string]bool, len(b.Edges))
	for _, e := range b.Edges {
		hasParent[e.To] = true
	}
	entry := b.Nodes[0]
	for _, n := range b.Nodes {
		if !hasParent[n.ID] {
			entry = n
			break
		}
	}
	if entry.Agent == "" && entry.Func == "" {
		return "shared run"
	}
	return fmt.Sprintf("run of %s.%s", entry.Agent, entry.Func)
}

// sanitizeFilename makes a workflow id safe to use as a filename component.
func sanitizeFilename(id string) string {
	replacer := strings.NewReplacer("/", "-", "\\", "-", ":", "-", " ", "-", "..", "-")
	cleaned := replacer.Replace(id)
	cleaned = strings.Trim(cleaned, ".-")
	if cleaned == "" {
		return "run"
	}
	return cleaned
}
