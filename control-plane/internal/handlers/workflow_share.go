package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/share"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// GetWorkflowShareHandler returns a self-contained, offline HTML artifact for a
// single workflow run. It is the server-side twin of `af share`: same versioned
// bundle, same embedded template. Rather than round-tripping through its own
// HTTP API (as the CLI must, since the CLI may target a remote plane), the
// handler builds the bundle directly from the storage layer — the same data the
// DAG and execution-details UI handlers read — which avoids an internal HTTP
// dependency and the CLI's per-execution fan-out.
//
// Query params:
//
//	redact=1  — replace every input/output preview with "[redacted]".
func GetWorkflowShareHandler(store storage.StorageProvider, payloads services.PayloadStore) gin.HandlerFunc {
	b := &shareBundleBuilder{store: store, payloads: payloads}
	return b.handle
}

type shareBundleBuilder struct {
	store    storage.StorageProvider
	payloads services.PayloadStore
}

func (b *shareBundleBuilder) handle(c *gin.Context) {
	ctx := c.Request.Context()

	runID := strings.TrimSpace(c.Param("workflowId"))
	if runID == "" {
		runID = strings.TrimSpace(c.Param("workflow_id"))
	}
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "workflowId or workflow_id is required"})
		return
	}

	filter := types.ExecutionFilter{
		RunID:          &runID,
		SortBy:         "started_at",
		SortDescending: false,
	}
	executions, err := b.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to load workflow: %v", err)})
		return
	}
	if len(executions) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "workflow not found"})
		return
	}

	bundle := b.buildBundle(ctx, runID, executions)

	if isRedactRequest(c) {
		share.Redact(bundle)
	}

	html, err := share.RenderHTML(bundle)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to render share artifact: %v", err)})
		return
	}

	filename := shareFilename(runID)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
	c.Data(http.StatusOK, "text/html; charset=utf-8", html)
}

// buildBundle maps a run's execution records into the versioned share bundle.
// It mirrors the CLI's buildBundleFromServer, but reads execution records
// directly (payloads resolved from the payload store when stored by URI).
func (b *shareBundleBuilder) buildBundle(ctx context.Context, runID string, executions []*types.Execution) *share.Bundle {
	nodes := make([]share.BundleNode, 0, len(executions))
	edges := make([]share.BundleEdge, 0, len(executions))

	for _, exec := range executions {
		if exec == nil {
			continue
		}

		node := share.BundleNode{
			ID:        exec.ExecutionID,
			Agent:     exec.AgentNodeID,
			Func:      exec.ReasonerID,
			Status:    types.NormalizeExecutionStatus(exec.Status),
			StartedAt: formatShareTime(&exec.StartedAt),
			Model:     nil, // not tracked on execution records
			CostUSD:   nil, // control plane does not track per-execution cost
		}
		if exec.CompletedAt != nil {
			node.EndedAt = exec.CompletedAt.Format(time.RFC3339)
		}
		node.DurationMS = exec.DurationMS

		node.InputPreview = share.TruncatePreview(b.previewPayload(ctx, exec.InputPayload, exec.InputURI))
		node.OutputPreview = share.TruncatePreview(b.previewPayload(ctx, exec.ResultPayload, exec.ResultURI))
		if exec.ErrorMessage != nil {
			if msg := strings.TrimSpace(*exec.ErrorMessage); msg != "" {
				node.Error = &msg
			}
		}

		nodes = append(nodes, node)

		if exec.ParentExecutionID != nil && *exec.ParentExecutionID != "" {
			edges = append(edges, share.BundleEdge{From: *exec.ParentExecutionID, To: exec.ExecutionID})
		}
	}

	bundle := &share.Bundle{
		Version:     share.BundleVersion,
		WorkflowID:  runID,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Totals: share.BundleTotals{
			Agents:     len(nodes),
			DurationMS: shareWallClock(nodes),
			CostUSD:    nil,
		},
		Nodes: nodes,
		Edges: edges,
	}
	bundle.Title = deriveShareTitle(bundle)
	return bundle
}

// previewPayload renders a raw payload as a human-readable preview string,
// falling back to the payload store when the record only carries a URI. Any
// resolution failure degrades to an empty preview rather than failing the
// export.
func (b *shareBundleBuilder) previewPayload(ctx context.Context, raw json.RawMessage, uri *string) string {
	if preview := rawSharePreview(raw); preview != "" {
		return preview
	}
	if uri == nil || b.payloads == nil {
		return ""
	}
	trimmed := strings.TrimSpace(*uri)
	if trimmed == "" {
		return ""
	}
	reader, err := b.payloads.Open(ctx, trimmed)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("uri", trimmed).Msg("share: failed to open payload for preview")
		return ""
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("uri", trimmed).Msg("share: failed to read payload for preview")
		return ""
	}
	return rawSharePreview(data)
}

// rawSharePreview renders raw JSON as a compact, human-readable preview string.
// Objects/arrays are pretty-printed; scalars are stringified; empty/null yields
// an empty string.
func rawSharePreview(raw json.RawMessage) string {
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

// shareWallClock returns the span in milliseconds from the earliest node start
// to the latest node end. Falls back to summing durations when timestamps are
// missing. Mirrors the CLI's computeWallClock.
func shareWallClock(nodes []share.BundleNode) int64 {
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

// deriveShareTitle builds a default human title from the run's entry node, e.g.
// "run of orchestrator.run_audit". The entry node is the DAG root: a node that
// is never the target of an edge. Mirrors the CLI's deriveTitle.
func deriveShareTitle(b *share.Bundle) string {
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

// shareFilename produces the download filename: run-<sanitized id>.html, without
// doubling the "run-" prefix when the id is already a run id.
func shareFilename(runID string) string {
	replacer := strings.NewReplacer("/", "-", "\\", "-", ":", "-", " ", "-", "..", "-")
	cleaned := strings.Trim(replacer.Replace(runID), ".-")
	if cleaned == "" {
		cleaned = "run"
	}
	if !strings.HasPrefix(cleaned, "run-") {
		cleaned = "run-" + cleaned
	}
	return cleaned + ".html"
}

func formatShareTime(t *time.Time) string {
	if t == nil || t.IsZero() {
		return ""
	}
	return t.Format(time.RFC3339)
}

func isRedactRequest(c *gin.Context) bool {
	v := strings.TrimSpace(c.Query("redact"))
	return v == "1" || strings.EqualFold(v, "true")
}
