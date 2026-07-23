package workspace

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Seal walks dir, honoring .gitignore plus the default excludes, hashes every
// regular file, stores each whole-file blob in the CAS, and returns the sorted
// manifest describing the folder. Symlinks and other non-regular files are
// skipped (out of scope for the POC).
func Seal(dir string, cas *CAS) (*Manifest, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace dir: %w", err)
	}
	info, err := os.Stat(absDir)
	if err != nil {
		return nil, fmt.Errorf("stat workspace dir: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("workspace path %q is not a directory", dir)
	}

	matcher, err := LoadMatcher(absDir)
	if err != nil {
		return nil, fmt.Errorf("load ignore rules: %w", err)
	}

	manifest := &Manifest{Version: ManifestVersion}

	walkErr := filepath.WalkDir(absDir, func(pathName string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if pathName == absDir {
			return nil
		}
		rel, relErr := filepath.Rel(absDir, pathName)
		if relErr != nil {
			return relErr
		}
		relSlash := filepath.ToSlash(rel)

		if d.IsDir() {
			if matcher.Ignored(relSlash, true) {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip symlinks and other irregular entries.
		if !d.Type().IsRegular() {
			return nil
		}
		if matcher.Ignored(relSlash, false) {
			return nil
		}

		fi, statErr := d.Info()
		if statErr != nil {
			return statErr
		}
		data, readErr := os.ReadFile(pathName) //nolint:gosec // path is inside the sealed workspace root
		if readErr != nil {
			return fmt.Errorf("read %s: %w", relSlash, readErr)
		}
		sha, putErr := cas.PutBytes(data)
		if putErr != nil {
			return fmt.Errorf("store blob for %s: %w", relSlash, putErr)
		}
		manifest.Files = append(manifest.Files, FileEntry{
			Path:    relSlash,
			Size:    fi.Size(),
			Mode:    uint32(fi.Mode().Perm()),
			MtimeNS: fi.ModTime().UnixNano(),
			SHA256:  sha,
		})
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}

	manifest.sortFiles()
	return manifest, nil
}

// Materialize writes the manifest's files into dir, copying each blob out of the
// CAS and restoring its recorded permission bits. It creates parent directories
// as needed. This mirrors what a node does when preparing a workspace and backs
// the seal→materialize round-trip test.
func Materialize(m *Manifest, dir string, cas *CAS) error {
	if m == nil {
		return fmt.Errorf("nil manifest")
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("resolve target dir: %w", err)
	}
	if err := os.MkdirAll(absDir, 0o755); err != nil {
		return fmt.Errorf("create target dir: %w", err)
	}
	for _, f := range m.Files {
		if err := validateRelPath(f.Path); err != nil {
			return err
		}
		data, getErr := cas.Get(f.SHA256)
		if getErr != nil {
			return fmt.Errorf("read blob %s for %s: %w", f.SHA256, f.Path, getErr)
		}
		dst := filepath.Join(absDir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", f.Path, err)
		}
		mode := os.FileMode(f.Mode).Perm()
		if mode == 0 {
			mode = 0o644
		}
		if err := os.WriteFile(dst, data, mode); err != nil {
			return fmt.Errorf("write %s: %w", f.Path, err)
		}
	}
	return nil
}

// validateRelPath rejects paths that would escape the workspace root.
func validateRelPath(rel string) error {
	if rel == "" {
		return fmt.Errorf("empty manifest path")
	}
	cleaned := filepath.ToSlash(filepath.Clean(rel))
	if filepath.IsAbs(rel) || cleaned == ".." || len(cleaned) >= 3 && cleaned[:3] == "../" {
		return fmt.Errorf("manifest path %q escapes the workspace root", rel)
	}
	return nil
}
