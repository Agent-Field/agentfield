// Package workspace implements the workspace-artifacts feature: sealing a local
// folder into a content-addressed manifest, transporting whole-file blobs
// through a local content store, and applying the diff a reasoner produces back
// onto the caller's folder.
//
// The wire format is fixed by docs/design/workspace-artifacts.md. Struct fields
// here are declared in ascending JSON-key order so that json.Marshal already
// emits canonical (sorted-key) output; CanonicalJSON only has to sort the file
// list to produce the exact bytes hashed into the manifest_id.
package workspace

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// ManifestVersion is the only manifest schema version understood by this POC.
const ManifestVersion = 1

// FileEntry describes a single regular file inside a sealed workspace. Blobs are
// whole files keyed by their sha256 (file-level dedup). Fields are ordered
// alphabetically by JSON key so struct marshaling is already canonical.
type FileEntry struct {
	Mode    uint32 `json:"mode"`
	MtimeNS int64  `json:"mtime_ns"`
	Path    string `json:"path"`
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size"`
}

// Manifest is the sealed description of a folder. Files are sorted ascending by
// path in canonical form.
type Manifest struct {
	Files   []FileEntry `json:"files"`
	Version int         `json:"version"`
}

// sortFiles orders entries ascending by path, the canonical ordering.
func (m *Manifest) sortFiles() {
	sort.Slice(m.Files, func(i, j int) bool {
		return m.Files[i].Path < m.Files[j].Path
	})
}

// Lookup returns the entry for a relative path and whether it exists.
func (m *Manifest) Lookup(path string) (FileEntry, bool) {
	for _, f := range m.Files {
		if f.Path == path {
			return f, true
		}
	}
	return FileEntry{}, false
}

// CanonicalJSON returns the manifest encoded with sorted keys, sorted files, and
// no insignificant whitespace. It is the exact byte sequence hashed to derive
// the manifest_id.
func CanonicalJSON(m *Manifest) ([]byte, error) {
	if m == nil {
		return nil, fmt.Errorf("nil manifest")
	}
	// Operate on a shallow copy so callers keep their original slice ordering.
	clone := Manifest{
		Version: m.Version,
		Files:   append([]FileEntry(nil), m.Files...),
	}
	clone.sortFiles()
	// Struct fields are declared in ascending JSON-key order, so json.Marshal
	// (which does not add whitespace) yields canonical output directly.
	return json.Marshal(&clone)
}

// ManifestID returns the hex-encoded sha256 of the manifest's canonical JSON.
func ManifestID(m *Manifest) (string, error) {
	canonical, err := CanonicalJSON(m)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:]), nil
}

// hashBytes returns the hex-encoded sha256 of arbitrary content.
func hashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// HashBytes returns the hex-encoded sha256 of content, the key used to address
// a blob in the content store.
func HashBytes(data []byte) string {
	return hashBytes(data)
}
