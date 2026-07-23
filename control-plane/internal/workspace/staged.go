package workspace

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// StagedRecord is the on-disk record produced for a workspace-bearing run,
// stored at ~/.agentfield/staged/<run_id>.json. Two writers contribute to it on
// the shared-filesystem POC: the CLI records the caller's original directory and
// the input manifest at call time; the control plane records the returned diff
// and where its blobs can be fetched from. MergeStaged keeps both contributions
// in one file.
type StagedRecord struct {
	RunID         string    `json:"run_id"`
	ExecutionID   string    `json:"execution_id,omitempty"`
	OriginalDir   string    `json:"original_dir,omitempty"`
	ManifestID    string    `json:"manifest_id,omitempty"`
	InputManifest *Manifest `json:"input_manifest,omitempty"`
	Diff          *Diff     `json:"workspace_diff,omitempty"`
	// NodeBaseURL and NodeID identify where changed blobs were fetched from
	// (the node's workspace blob endpoint) — the "node blob source info".
	NodeBaseURL string `json:"node_base_url,omitempty"`
	NodeID      string `json:"node_id,omitempty"`
	CreatedAt   string `json:"created_at,omitempty"`
	UpdatedAt   string `json:"updated_at,omitempty"`
}

// stagedMu serializes read-modify-write cycles on staged records within a
// process. Cross-process ordering on the shared POC filesystem is naturally
// sequential (the control plane writes before returning; the CLI merges after).
var stagedMu sync.Mutex

// StagedDir returns the staged-records directory (~/.agentfield/staged).
func StagedDir() string {
	return filepath.Join(HomeDir(), "staged")
}

// StagedPath returns the on-disk path for a run's staged record.
func StagedPath(runID string) string {
	return filepath.Join(StagedDir(), runID+".json")
}

// LoadStaged reads a staged record. It returns (nil, nil) when none exists.
func LoadStaged(runID string) (*StagedRecord, error) {
	if runID == "" {
		return nil, fmt.Errorf("empty run id")
	}
	data, err := os.ReadFile(StagedPath(runID))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var rec StagedRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("decode staged record %s: %w", runID, err)
	}
	return &rec, nil
}

// SaveStaged writes a staged record atomically.
func SaveStaged(rec *StagedRecord) error {
	if rec == nil || rec.RunID == "" {
		return fmt.Errorf("staged record requires a run id")
	}
	dir := StagedDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create staged dir: %w", err)
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("encode staged record: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".tmp-"+rec.RunID+"-*")
	if err != nil {
		return fmt.Errorf("create staged temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("write staged temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("close staged temp: %w", err)
	}
	if err := os.Rename(tmpName, StagedPath(rec.RunID)); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("commit staged record: %w", err)
	}
	return nil
}

// MergeStaged applies mutate to the existing record for runID (or a fresh one)
// and persists the result. It is safe against concurrent in-process writers.
func MergeStaged(runID string, mutate func(*StagedRecord)) error {
	if runID == "" {
		return fmt.Errorf("empty run id")
	}
	stagedMu.Lock()
	defer stagedMu.Unlock()

	rec, err := LoadStaged(runID)
	if err != nil {
		return err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	if rec == nil {
		rec = &StagedRecord{RunID: runID, CreatedAt: now}
	}
	if rec.CreatedAt == "" {
		rec.CreatedAt = now
	}
	mutate(rec)
	rec.RunID = runID
	rec.UpdatedAt = now
	return SaveStaged(rec)
}
