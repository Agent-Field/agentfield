package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/workspace"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ExecuteArtifacts is the optional artifacts block on an execute request. Only a
// workspace artifact is defined for the POC.
type ExecuteArtifacts struct {
	Workspace *WorkspaceArtifact `json:"workspace,omitempty"`
}

// WorkspaceArtifact attaches a sealed folder to an execution. The manifest is
// inlined for the POC; blobs live in the shared content store keyed by sha256.
type WorkspaceArtifact struct {
	ManifestID string              `json:"manifest_id"`
	Manifest   *workspace.Manifest `json:"manifest,omitempty"`
}

// workspaceDiffEnvelopeKey is the top-level result key the node SDK attaches to
// carry the workspace diff, sibling to the reasoner's output — the same
// convention used for the usage envelope.
const workspaceDiffEnvelopeKey = "workspace_diff"

// buildArtifactsEnvelope produces the artifacts object dispatched to the node.
// The node SDK consumes the inline manifest to materialize the workspace, so the
// envelope carries the manifest itself (the node re-derives the manifest_id).
func buildArtifactsEnvelope(ws *WorkspaceArtifact) map[string]interface{} {
	return map[string]interface{}{
		"workspace": map[string]interface{}{
			"manifest": ws.Manifest,
		},
	}
}

// workspaceCAS returns the shared local content store. The CLI seals directly
// into this same directory (~/.agentfield/cas) on the POC single-machine setup,
// so the control plane reads the caller's blobs without any copy step.
func workspaceCAS() *workspace.CAS {
	return workspace.NewCAS(workspace.DefaultCASDir())
}

// prepareWorkspaceOnNode transports the sealed workspace to the node before an
// execute dispatch: it asks the node which blobs it is missing, then uploads
// each missing blob from the shared CAS. It is a no-op when the execution
// carries no workspace.
func (c *executionController) prepareWorkspaceOnNode(ctx context.Context, plan *preparedExecution) error {
	if plan == nil || plan.workspace == nil || plan.workspace.Manifest == nil {
		return nil
	}
	base := workspaceNodeBaseURL(plan)
	if base == "" {
		return fmt.Errorf("agent %s has no base URL for workspace transport", plan.agent.ID)
	}
	cas := workspaceCAS()

	prepareBody, err := json.Marshal(map[string]interface{}{"manifest": plan.workspace.Manifest})
	if err != nil {
		return fmt.Errorf("encode prepare request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/v1/workspace/prepare", bytes.NewReader(prepareBody))
	if err != nil {
		return fmt.Errorf("build prepare request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyInternalAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("call node prepare: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		return fmt.Errorf("node prepare failed (%d): %s", resp.StatusCode, truncateForLog(body))
	}

	var prepared struct {
		Missing []string `json:"missing"`
	}
	if len(bytes.TrimSpace(body)) > 0 {
		if err := json.Unmarshal(body, &prepared); err != nil {
			return fmt.Errorf("decode prepare response: %w", err)
		}
	}

	for _, sha := range prepared.Missing {
		blob, err := cas.Get(sha)
		if err != nil {
			return fmt.Errorf("read blob %s from content store: %w", sha, err)
		}
		if err := c.putWorkspaceBlob(ctx, base, sha, blob); err != nil {
			return err
		}
	}

	logger.Logger.Debug().
		Str("execution_id", plan.exec.ExecutionID).
		Str("node", plan.target.NodeID).
		Int("uploaded_blobs", len(prepared.Missing)).
		Msg("workspace prepared on node")
	return nil
}

func (c *executionController) putWorkspaceBlob(ctx context.Context, base, sha string, blob []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, base+"/api/v1/workspace/blobs/"+sha, bytes.NewReader(blob))
	if err != nil {
		return fmt.Errorf("build blob upload for %s: %w", sha, err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	c.applyInternalAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("upload blob %s: %w", sha, err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		return fmt.Errorf("node rejected blob %s (%d): %s", sha, resp.StatusCode, truncateForLog(body))
	}
	return nil
}

// processWorkspaceResult extracts a workspace diff from the node's result
// envelope, fetches any changed blobs into the shared CAS, records a staged
// entry keyed by run_id, and returns the result with workspace_diff stripped so
// it never leaks into the stored reasoner output. It is a no-op when the
// execution carried no workspace or the result has no diff.
func (c *executionController) processWorkspaceResult(ctx context.Context, plan *preparedExecution, result []byte) []byte {
	if plan == nil || plan.workspace == nil {
		return result
	}
	diff, stripped, present := extractWorkspaceDiff(result)
	if !present {
		return result
	}
	if diff == nil {
		// The envelope carried a malformed workspace_diff; strip it but stage
		// nothing.
		logger.Logger.Warn().
			Str("execution_id", plan.exec.ExecutionID).
			Msg("workspace_diff present but could not be decoded; skipping staging")
		return stripped
	}

	base := workspaceNodeBaseURL(plan)
	cas := workspaceCAS()
	for _, sha := range diff.Blobs() {
		if cas.Has(sha) {
			continue
		}
		if err := c.fetchWorkspaceBlob(ctx, base, sha, cas); err != nil {
			logger.Logger.Error().
				Err(err).
				Str("execution_id", plan.exec.ExecutionID).
				Str("sha256", sha).
				Msg("failed to fetch workspace diff blob from node")
		}
	}

	if err := workspace.MergeStaged(plan.exec.RunID, func(rec *workspace.StagedRecord) {
		rec.ExecutionID = plan.exec.ExecutionID
		rec.Diff = diff
		rec.NodeBaseURL = base
		rec.NodeID = plan.target.NodeID
		// Echo the input manifest so `af apply` can detect conflicts even if the
		// CLI-side record write is unavailable.
		if rec.InputManifest == nil {
			rec.InputManifest = plan.workspace.Manifest
		}
		if rec.ManifestID == "" {
			rec.ManifestID = plan.workspace.ManifestID
		}
	}); err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", plan.exec.ExecutionID).
			Str("run_id", plan.exec.RunID).
			Msg("failed to write staged workspace record")
	}

	logger.Logger.Info().
		Str("execution_id", plan.exec.ExecutionID).
		Str("run_id", plan.exec.RunID).
		Int("changed", len(diff.Changed)).
		Int("deleted", len(diff.Deleted)).
		Msg("staged workspace diff")
	return stripped
}

// stageWorkspaceFromStatus stages a workspace diff that arrived via an async
// status callback (the default SDK path, where the reasoner self-acknowledges
// and reports its result asynchronously rather than on the synchronous
// response). It resolves the node base URL from the execution's node id, fetches
// any missing changed blobs into the content store, and records the staged entry
// keyed by run_id. Best-effort: failures are logged, never propagated.
func (c *executionController) stageWorkspaceFromStatus(ctx context.Context, exec *types.Execution, diff *workspace.Diff) {
	if exec == nil || diff == nil {
		return
	}

	base := c.nodeBaseURLForExecution(ctx, exec)
	cas := workspaceCAS()
	for _, sha := range diff.Blobs() {
		if cas.Has(sha) {
			continue
		}
		if base == "" {
			logger.Logger.Warn().
				Str("execution_id", exec.ExecutionID).
				Str("sha256", sha).
				Msg("cannot fetch workspace diff blob: node base URL unknown")
			continue
		}
		if err := c.fetchWorkspaceBlob(ctx, base, sha, cas); err != nil {
			logger.Logger.Error().
				Err(err).
				Str("execution_id", exec.ExecutionID).
				Str("sha256", sha).
				Msg("failed to fetch workspace diff blob from node")
		}
	}

	if err := workspace.MergeStaged(exec.RunID, func(rec *workspace.StagedRecord) {
		rec.ExecutionID = exec.ExecutionID
		rec.Diff = diff
		rec.NodeBaseURL = base
		rec.NodeID = exec.NodeID
	}); err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", exec.ExecutionID).
			Str("run_id", exec.RunID).
			Msg("failed to write staged workspace record from status callback")
		return
	}

	logger.Logger.Info().
		Str("execution_id", exec.ExecutionID).
		Str("run_id", exec.RunID).
		Int("changed", len(diff.Changed)).
		Int("deleted", len(diff.Deleted)).
		Msg("staged workspace diff from status callback")
}

// nodeBaseURLForExecution resolves the node's base URL for an execution so the
// control plane can fetch diff blobs. It tries the node id first, then the agent
// node id.
func (c *executionController) nodeBaseURLForExecution(ctx context.Context, exec *types.Execution) string {
	for _, id := range []string{exec.NodeID, exec.AgentNodeID} {
		if strings.TrimSpace(id) == "" {
			continue
		}
		if agent, err := c.store.GetAgent(ctx, id); err == nil && agent != nil {
			if base := strings.TrimRight(strings.TrimSpace(agent.BaseURL), "/"); base != "" {
				return base
			}
		}
	}
	return ""
}

func (c *executionController) fetchWorkspaceBlob(ctx context.Context, base, sha string, cas *workspace.CAS) error {
	if base == "" {
		return fmt.Errorf("no node base URL to fetch blob %s", sha)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/v1/workspace/blobs/"+sha, nil)
	if err != nil {
		return fmt.Errorf("build blob fetch for %s: %w", sha, err)
	}
	c.applyInternalAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("fetch blob %s: %w", sha, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("node returned %d for blob %s: %s", resp.StatusCode, sha, string(body))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read blob %s body: %w", sha, err)
	}
	return cas.Put(sha, data)
}

// applyInternalAuth forwards the internal service token to the node, matching
// how the reasoner dispatch authenticates.
func (c *executionController) applyInternalAuth(req *http.Request) {
	if c.internalToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.internalToken)
	}
}

// workspaceNodeBaseURL returns the node's base URL for workspace transport.
func workspaceNodeBaseURL(plan *preparedExecution) string {
	if plan == nil || plan.agent == nil {
		return ""
	}
	return strings.TrimRight(strings.TrimSpace(plan.agent.BaseURL), "/")
}

// extractWorkspaceDiff pulls the workspace_diff envelope key out of a result.
// present reports whether the key existed at all (so the caller can strip it);
// diff is non-nil only when it decoded cleanly.
func extractWorkspaceDiff(result []byte) (diff *workspace.Diff, stripped []byte, present bool) {
	if len(result) == 0 {
		return nil, result, false
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(result, &decoded); err != nil {
		// Not a JSON object — nothing to strip.
		return nil, result, false
	}
	raw, ok := decoded[workspaceDiffEnvelopeKey]
	if !ok {
		return nil, result, false
	}
	delete(decoded, workspaceDiffEnvelopeKey)

	strippedBytes, err := json.Marshal(decoded)
	if err != nil {
		strippedBytes = result
	}

	rawBytes, err := json.Marshal(raw)
	if err != nil {
		return nil, strippedBytes, true
	}
	var parsed workspace.Diff
	if err := json.Unmarshal(rawBytes, &parsed); err != nil {
		return nil, strippedBytes, true
	}
	return &parsed, strippedBytes, true
}
