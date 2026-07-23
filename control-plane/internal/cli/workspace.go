package cli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/workspace"
	"github.com/spf13/cobra"
)

// sealedWorkspace records what the CLI sealed for a call so it can persist the
// caller-side half of the staged record (the local original directory and the
// input manifest) once a run_id is known.
type sealedWorkspace struct {
	originalDir string
	manifestID  string
	manifest    *workspace.Manifest
}

// localWorkspaceCAS is the CLI's own content store. The CLI never assumes it
// shares a filesystem with the control plane: it seals into this local store,
// then pushes blobs to the control plane over HTTP. The control plane may be
// local or remote — that is a pure configuration detail (the control plane URL).
func localWorkspaceCAS() *workspace.CAS {
	return workspace.NewCAS(workspace.DefaultCASDir())
}

// sealWorkspaceForCall seals dir into the local content store and uploads any
// blobs the control plane is missing, returning the artifacts block to attach to
// the execute request. When dir is empty it is a no-op returning nil artifacts.
func sealWorkspaceForCall(ctx context.Context, dir string, stderr io.Writer) (map[string]interface{}, *sealedWorkspace, error) {
	if dir == "" {
		return nil, nil, nil
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, nil, cliExitError{Code: 2, Err: fmt.Errorf("resolve --dir: %w", err)}
	}
	info, err := os.Stat(absDir)
	if err != nil {
		return nil, nil, cliExitError{Code: 2, Err: fmt.Errorf("--dir %q: %w", dir, err)}
	}
	if !info.IsDir() {
		return nil, nil, cliExitError{Code: 2, Err: fmt.Errorf("--dir %q is not a directory", dir)}
	}

	cas := localWorkspaceCAS()
	sealStart := time.Now()
	manifest, err := workspace.Seal(absDir, cas)
	if err != nil {
		return nil, nil, cliExitError{Code: 3, Err: fmt.Errorf("seal workspace: %w", err)}
	}
	sealSeconds := time.Since(sealStart).Seconds()
	manifestID, err := workspace.ManifestID(manifest)
	if err != nil {
		return nil, nil, cliExitError{Code: 3, Err: fmt.Errorf("compute manifest id: %w", err)}
	}

	var sealedBytes int64
	for _, f := range manifest.Files {
		sealedBytes += f.Size
	}

	// Transport the sealed blobs to the control plane: ask which it is missing,
	// then upload just those in batched, compressed requests (falling back to
	// bounded-parallel PUTs against an older control plane).
	missing, err := cpPrepareWorkspace(ctx, manifest)
	if err != nil {
		return nil, nil, cliExitError{Code: 3, Err: fmt.Errorf("prepare workspace on control plane: %w", err)}
	}
	uploadStart := time.Now()
	stats, err := cpUploadBlobs(ctx, cas, missing)
	if err != nil {
		return nil, nil, cliExitError{Code: 3, Err: fmt.Errorf("upload workspace blobs: %w", err)}
	}
	uploadSeconds := time.Since(uploadStart).Seconds()

	if stderr != nil {
		fmt.Fprintf(stderr, "sealed %d files (%.1f MB) in %.1fs; uploaded %d blobs in %.1fs\n",
			len(manifest.Files), float64(sealedBytes)/(1<<20), sealSeconds, stats.Blobs, uploadSeconds)
	}

	artifacts := map[string]interface{}{
		"workspace": map[string]interface{}{
			"manifest_id": manifestID,
			"manifest":    manifest,
		},
	}
	return artifacts, &sealedWorkspace{originalDir: absDir, manifestID: manifestID, manifest: manifest}, nil
}

// stageWorkspaceCall records the caller-side staged record (original directory +
// input manifest) on the local machine so `af apply` can write changes back and
// detect conflicts. The control plane independently records the returned diff,
// which `af diff` / `af apply` fetch over HTTP.
func stageWorkspaceCall(stderr io.Writer, runID, executionID string, sealed *sealedWorkspace) error {
	if sealed == nil {
		return nil
	}
	if runID == "" {
		return cliExitError{Code: 3, Err: fmt.Errorf("server accepted workspace execution without run_id")}
	}
	err := workspace.MergeStaged(runID, func(rec *workspace.StagedRecord) {
		rec.OriginalDir = sealed.originalDir
		rec.ManifestID = sealed.manifestID
		rec.InputManifest = sealed.manifest
		if executionID != "" {
			rec.ExecutionID = executionID
		}
	})
	if err != nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("write staged record: %w", err)}
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "workspace staged for run %s (af diff %s / af apply %s)\n", runID, runID, runID)
	}
	return nil
}

// stagedDiffResponse mirrors the control plane's staged-diff fetch view.
type stagedDiffResponse struct {
	RunID       string          `json:"run_id"`
	ExecutionID string          `json:"execution_id"`
	ManifestID  string          `json:"manifest_id"`
	Diff        *workspace.Diff `json:"workspace_diff"`
}

// NewDiffCommand prints the staged workspace diff for a run.
func NewDiffCommand() *cobra.Command {
	var output string
	cmd := &cobra.Command{
		Use:   "diff <run_id>",
		Short: "Show the staged workspace changes for a run",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			return runDiff(ctx, args[0], output, os.Stdout, os.Stderr)
		},
		SilenceUsage: true,
	}
	cmd.Flags().StringVarP(&output, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runDiff(ctx context.Context, runID, output string, stdout, stderr io.Writer) error {
	format := autoOutputFormat(output, isOutputTerminal())
	staged, err := cpFetchStagedDiff(ctx, runID)
	if err != nil {
		return err
	}
	if staged.Diff == nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("no workspace diff staged yet for run %s (the execution may still be running)", runID)}
	}

	// The original directory is only known locally (it is a path on this
	// machine), so read it best-effort from the local staged record.
	originalDir := ""
	if local, _ := workspace.LoadStaged(runID); local != nil {
		originalDir = local.OriginalDir
	}

	changed := make([]map[string]interface{}, 0, len(staged.Diff.Changed))
	for _, c := range staged.Diff.Changed {
		changed = append(changed, map[string]interface{}{
			"path": c.Path,
			"size": c.Size,
			"mode": c.Mode,
		})
	}
	summary := map[string]interface{}{
		"run_id":       runID,
		"original_dir": originalDir,
		"changed":      changed,
		"deleted":      staged.Diff.Deleted,
	}

	if format == "json" || format == "yaml" {
		return writeValue(stdout, summary, format)
	}
	fmt.Fprintf(stdout, "Workspace diff for run %s\n", runID)
	if originalDir != "" {
		fmt.Fprintf(stdout, "  dir: %s\n", originalDir)
	}
	fmt.Fprintf(stdout, "  changed: %d\n", len(staged.Diff.Changed))
	for _, c := range staged.Diff.Changed {
		fmt.Fprintf(stdout, "    ~ %s (%d bytes)\n", c.Path, c.Size)
	}
	fmt.Fprintf(stdout, "  deleted: %d\n", len(staged.Diff.Deleted))
	for _, p := range staged.Diff.Deleted {
		fmt.Fprintf(stdout, "    - %s\n", p)
	}
	if len(staged.Diff.Changed) == 0 && len(staged.Diff.Deleted) == 0 {
		fmt.Fprintln(stdout, "  (no changes)")
	}
	return nil
}

// NewApplyCommand writes a run's staged changes back onto the original folder.
func NewApplyCommand() *cobra.Command {
	var force bool
	var output string
	cmd := &cobra.Command{
		Use:   "apply <run_id>",
		Short: "Apply a run's staged workspace changes onto the original folder",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			return runApply(ctx, args[0], force, output, os.Stdout, os.Stderr)
		},
		SilenceUsage: true,
	}
	cmd.Flags().BoolVar(&force, "force", false, "Overwrite files that were changed locally after sealing")
	cmd.Flags().StringVarP(&output, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runApply(ctx context.Context, runID string, force bool, output string, stdout, stderr io.Writer) error {
	format := autoOutputFormat(output, isOutputTerminal())

	// The original directory and input manifest are local-only state written by
	// `af call --dir`; apply must run on the machine that made the call.
	local, err := workspace.LoadStaged(runID)
	if err != nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("load local staged record: %w", err)}
	}
	if local == nil || local.OriginalDir == "" {
		return cliExitError{Code: 2, Err: fmt.Errorf("no local workspace record for run %s; apply must run on the machine that made the call", runID)}
	}

	staged, err := cpFetchStagedDiff(ctx, runID)
	if err != nil {
		return err
	}
	if staged.Diff == nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("no workspace diff staged yet for run %s (the execution may still be running)", runID)}
	}

	// Pull any changed blobs we do not already hold locally from the control plane.
	cas := localWorkspaceCAS()
	for _, sha := range staged.Diff.Blobs() {
		if cas.Has(sha) {
			continue
		}
		data, derr := cpDownloadBlob(ctx, sha)
		if derr != nil {
			return cliExitError{Code: 3, Err: fmt.Errorf("download blob %s: %w", sha, derr)}
		}
		if perr := cas.PutVerified(sha, data); perr != nil {
			return cliExitError{Code: 3, Err: fmt.Errorf("store blob %s: %w", sha, perr)}
		}
	}

	result, err := workspace.Apply(local.OriginalDir, local.InputManifest, staged.Diff, cas, force)
	if err != nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("apply workspace diff: %w", err)}
	}

	summary := map[string]interface{}{
		"run_id":       runID,
		"original_dir": local.OriginalDir,
		"written":      result.Written,
		"deleted":      result.Deleted,
		"conflicts":    result.Conflicts,
		"forced":       force,
	}

	if format == "json" || format == "yaml" {
		if werr := writeValue(stdout, summary, format); werr != nil {
			return werr
		}
	} else {
		fmt.Fprintf(stdout, "Applied run %s onto %s\n", runID, local.OriginalDir)
		fmt.Fprintf(stdout, "  written: %d\n", len(result.Written))
		for _, p := range result.Written {
			fmt.Fprintf(stdout, "    ~ %s\n", p)
		}
		fmt.Fprintf(stdout, "  deleted: %d\n", len(result.Deleted))
		for _, p := range result.Deleted {
			fmt.Fprintf(stdout, "    - %s\n", p)
		}
		if len(result.Conflicts) > 0 {
			fmt.Fprintf(stdout, "  conflicts (skipped): %d\n", len(result.Conflicts))
			for _, p := range result.Conflicts {
				fmt.Fprintf(stdout, "    ! %s\n", p)
			}
			fmt.Fprintln(stderr, "re-run with --force to overwrite conflicting files")
		}
	}

	// Signal unresolved conflicts with a non-zero exit so scripts can react,
	// while still having applied the non-conflicting changes above.
	if len(result.Conflicts) > 0 && !force {
		return cliExitError{Code: 1, Err: fmt.Errorf("%d conflict(s) skipped; re-run with --force to overwrite", len(result.Conflicts))}
	}
	return nil
}

// --- Control plane workspace HTTP client ---------------------------------

// cpPrepareWorkspace asks the control plane which of a manifest's blobs it is
// missing.
func cpPrepareWorkspace(ctx context.Context, manifest *workspace.Manifest) ([]string, error) {
	resp, err := makeRequest(ctx, http.MethodPost, "/api/v1/workspace/prepare",
		map[string]interface{}{"manifest": manifest}, "application/json")
	if err != nil {
		return nil, err
	}
	var decoded struct {
		Missing []string `json:"missing"`
	}
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, fmt.Errorf("prepare failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return decoded.Missing, nil
}

// cpUploadBlobs transports the given blobs from the local content store to the
// control plane using the shared batch uploader (with automatic parallel PUT
// fallback for older control planes). It is the single sender used by the
// CLI→control-plane hop, mirroring the control-plane→node hop.
func cpUploadBlobs(ctx context.Context, cas *workspace.CAS, shas []string) (workspace.UploadStats, error) {
	server := strings.TrimRight(GetServerURL(), "/")
	apiKey := strings.TrimSpace(GetAPIKey())
	return workspace.UploadBlobs(ctx, cas, shas, workspace.UploadOptions{
		BaseURL: server,
		Client:  &http.Client{Timeout: 5 * time.Minute},
		Decorate: func(req *http.Request) {
			if apiKey != "" {
				req.Header.Set("X-API-Key", apiKey)
			}
		},
	})
}

// cpUploadBlob PUTs a raw blob to the control plane.
func cpUploadBlob(ctx context.Context, sha string, data []byte) error {
	resp, err := cpBlobRequest(ctx, http.MethodPut, sha, bytes.NewReader(data), "application/octet-stream")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

// cpDownloadBlob GETs a raw blob from the control plane.
func cpDownloadBlob(ctx context.Context, sha string) ([]byte, error) {
	resp, err := cpBlobRequest(ctx, http.MethodGet, sha, nil, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return io.ReadAll(resp.Body)
}

// cpFetchStagedDiff retrieves the staged workspace diff for a run over HTTP.
func cpFetchStagedDiff(ctx context.Context, runID string) (*stagedDiffResponse, error) {
	resp, err := makeRequest(ctx, http.MethodGet,
		"/api/v1/workspace/staged/"+url.PathEscape(runID), nil, "application/json")
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	var decoded stagedDiffResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, cliExitError{Code: 2, Err: fmt.Errorf("no staged workspace record for run %s", runID)}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("staged fetch failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	return &decoded, nil
}

// cpBlobRequest issues a raw (non-JSON) blob request to the control plane's
// workspace endpoint, carrying the configured API key. Blobs can be large, so a
// generous timeout is used.
func cpBlobRequest(ctx context.Context, method, sha string, body io.Reader, contentType string) (*http.Response, error) {
	server := strings.TrimRight(GetServerURL(), "/")
	req, err := http.NewRequestWithContext(ctx, method, server+"/api/v1/workspace/blobs/"+url.PathEscape(sha), body)
	if err != nil {
		return nil, fmt.Errorf("build blob request: %w", err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if key := strings.TrimSpace(GetAPIKey()); key != "" {
		req.Header.Set("X-API-Key", key)
	}
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, err
		}
		return nil, controlPlaneUnreachableError(err)
	}
	return resp, nil
}
