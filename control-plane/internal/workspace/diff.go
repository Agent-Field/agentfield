package workspace

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ChangedFile is one added-or-modified file in a workspace diff. Its shape
// matches the result's workspace_diff.changed entries on the wire.
type ChangedFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
	Mode   uint32 `json:"mode"`
}

// Diff is the set of changes a reasoner produced relative to the input manifest.
type Diff struct {
	Changed []ChangedFile `json:"changed"`
	Deleted []string      `json:"deleted"`
}

// Blobs returns the sha256 keys of every changed file, for uploading or
// prefetching the underlying content.
func (d *Diff) Blobs() []string {
	if d == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(d.Changed))
	var out []string
	for _, c := range d.Changed {
		if c.SHA256 == "" {
			continue
		}
		if _, ok := seen[c.SHA256]; ok {
			continue
		}
		seen[c.SHA256] = struct{}{}
		out = append(out, c.SHA256)
	}
	return out
}

// ComputeDiff derives the changes between an input manifest and the current
// state of a folder. It backs testing and mirrors what a node computes after a
// reasoner runs. Blobs for changed files are written into the CAS.
func ComputeDiff(inputManifest *Manifest, dir string, cas *CAS) (*Diff, error) {
	current, err := Seal(dir, cas)
	if err != nil {
		return nil, err
	}
	return DiffManifests(inputManifest, current), nil
}

// DiffManifests compares an input manifest against an output manifest and
// returns the changed and deleted paths. Content identity is decided by sha256.
func DiffManifests(input, output *Manifest) *Diff {
	inputByPath := map[string]FileEntry{}
	if input != nil {
		for _, f := range input.Files {
			inputByPath[f.Path] = f
		}
	}
	outputByPath := map[string]FileEntry{}
	if output != nil {
		for _, f := range output.Files {
			outputByPath[f.Path] = f
		}
	}

	diff := &Diff{}
	for _, f := range outputSorted(output) {
		prev, ok := inputByPath[f.Path]
		if !ok || prev.SHA256 != f.SHA256 {
			diff.Changed = append(diff.Changed, ChangedFile{
				Path:   f.Path,
				SHA256: f.SHA256,
				Size:   f.Size,
				Mode:   f.Mode,
			})
		}
	}
	var deleted []string
	for path := range inputByPath {
		if _, ok := outputByPath[path]; !ok {
			deleted = append(deleted, path)
		}
	}
	sort.Strings(deleted)
	diff.Deleted = deleted
	return diff
}

func outputSorted(output *Manifest) []FileEntry {
	if output == nil {
		return nil
	}
	files := append([]FileEntry(nil), output.Files...)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files
}

// ApplyResult reports the outcome of applying a diff onto a folder.
type ApplyResult struct {
	Written   []string `json:"written"`
	Deleted   []string `json:"deleted"`
	Conflicts []string `json:"conflicts"`
}

// Apply writes a diff's changed and deleted paths onto dir. A path is a conflict
// when the file currently on disk differs from what was sealed (the input
// manifest); conflicts are skipped and listed unless force is set. Blobs for
// changed files must already be present in the CAS.
func Apply(dir string, inputManifest *Manifest, diff *Diff, cas *CAS, force bool) (*ApplyResult, error) {
	if diff == nil {
		return &ApplyResult{}, nil
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve target dir: %w", err)
	}

	result := &ApplyResult{}

	for _, c := range diff.Changed {
		if err := validateRelPath(c.Path); err != nil {
			return nil, err
		}
		sealed, _ := lookupSHA(inputManifest, c.Path)
		current, err := currentSHA(absDir, c.Path)
		if err != nil {
			return nil, err
		}
		if !force && current != sealed {
			result.Conflicts = append(result.Conflicts, c.Path)
			continue
		}
		data, getErr := cas.Get(c.SHA256)
		if getErr != nil {
			return nil, fmt.Errorf("read blob %s for %s: %w", c.SHA256, c.Path, getErr)
		}
		dst := filepath.Join(absDir, filepath.FromSlash(c.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return nil, fmt.Errorf("create dir for %s: %w", c.Path, err)
		}
		mode := os.FileMode(c.Mode).Perm()
		if mode == 0 {
			mode = 0o644
		}
		if err := os.WriteFile(dst, data, mode); err != nil {
			return nil, fmt.Errorf("write %s: %w", c.Path, err)
		}
		result.Written = append(result.Written, c.Path)
	}

	for _, path := range diff.Deleted {
		if err := validateRelPath(path); err != nil {
			return nil, err
		}
		sealed, _ := lookupSHA(inputManifest, path)
		current, err := currentSHA(absDir, path)
		if err != nil {
			return nil, err
		}
		if current == "" {
			// Already gone; nothing to do.
			continue
		}
		if !force && current != sealed {
			result.Conflicts = append(result.Conflicts, path)
			continue
		}
		dst := filepath.Join(absDir, filepath.FromSlash(path))
		if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("delete %s: %w", path, err)
		}
		result.Deleted = append(result.Deleted, path)
	}

	sort.Strings(result.Written)
	sort.Strings(result.Deleted)
	sort.Strings(result.Conflicts)
	return result, nil
}

func lookupSHA(m *Manifest, path string) (string, bool) {
	if m == nil {
		return "", false
	}
	if entry, ok := m.Lookup(path); ok {
		return entry.SHA256, true
	}
	return "", false
}

// currentSHA returns the sha256 of the file at rel under root, or "" if absent.
func currentSHA(root, rel string) (string, error) {
	full := filepath.Join(root, filepath.FromSlash(rel))
	data, err := os.ReadFile(full) //nolint:gosec // path is inside the target workspace root
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read current %s: %w", rel, err)
	}
	return hashBytes(data), nil
}
