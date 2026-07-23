package workspace

import (
	"path/filepath"
	"sort"
)

// WorkspacesDir returns the directory holding per-execution materialized
// workspaces (~/.agentfield/workspaces), honoring AGENTFIELD_HOME the same way
// the content store does.
func WorkspacesDir() string {
	return filepath.Join(HomeDir(), "workspaces")
}

// MissingBlobs returns the sha256 hexes referenced by the manifest that the CAS
// does not yet hold. The result is deduplicated and sorted ascending. It backs
// the node's `prepare` endpoint: the control plane uploads exactly these blobs
// before dispatching a workspace-bearing execution.
func MissingBlobs(m *Manifest, cas *CAS) []string {
	if m == nil || cas == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(m.Files))
	var missing []string
	for _, f := range m.Files {
		if f.SHA256 == "" {
			continue
		}
		if _, ok := seen[f.SHA256]; ok {
			continue
		}
		seen[f.SHA256] = struct{}{}
		if !cas.Has(f.SHA256) {
			missing = append(missing, f.SHA256)
		}
	}
	sort.Strings(missing)
	return missing
}
