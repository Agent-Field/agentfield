package packages

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// InspectSource resolves an agent-node source and returns its parsed manifest
// WITHOUT installing it. Nothing is written under the AgentField home
// (~/.agentfield/packages): a local directory is parsed in place, and a Git URL
// is shallow-cloned into a temp directory that is removed before returning. This
// backs `af show-requirements`, letting a user see what environment a node needs
// before committing to an install.
//
// Resolution: an existing local directory wins (so a folder literally named like
// a URL is still read as a path); otherwise a Git URL (optionally carrying an
// @ref and/or a //subdir selector) is cloned and inspected. Anything else is
// treated as a local path so the resulting error names the missing manifest.
func InspectSource(source string) (*PackageMetadata, error) {
	src := strings.TrimSpace(source)
	if src == "" {
		return nil, fmt.Errorf("no source provided")
	}
	if info, err := os.Stat(src); err == nil && info.IsDir() {
		return ParsePackageMetadata(src)
	}
	if IsGitURL(src) {
		return inspectGitSource(src)
	}
	return ParsePackageMetadata(src)
}

// inspectGitSource shallow-clones a Git URL into a temp directory, parses the
// manifest (honoring an @ref / //subdir selector), and removes the clone. The
// temp directory — never ~/.agentfield — is the only thing written to disk.
func inspectGitSource(gitURL string) (*PackageMetadata, error) {
	if err := checkGitAvailable(); err != nil {
		return nil, err
	}
	info, err := ParseGitURL(gitURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Git URL: %w", err)
	}

	gi := &GitInstaller{Subdir: info.Subdir}
	tempDir, err := gi.cloneRepository(info)
	if err != nil {
		return nil, fmt.Errorf("failed to clone repository: %w", err)
	}
	defer func() { _ = os.RemoveAll(tempDir) }()

	dir, err := findManifestDir(tempDir, info.Subdir)
	if err != nil {
		return nil, err
	}
	return ParsePackageMetadata(dir)
}

// findManifestDir returns the directory holding agentfield-package.yaml within
// root. When subdir is set the manifest must live at root/subdir; otherwise the
// tree is walked root-first for the first manifest. Unlike findPackageRoot this
// does not require a startable node — inspection only needs to read the manifest.
func findManifestDir(root, subdir string) (string, error) {
	if strings.TrimSpace(subdir) != "" {
		dir, err := ResolvePackageSubdir(root, subdir)
		if err != nil {
			return "", err
		}
		if !fileExistsAt(dir, "agentfield-package.yaml") {
			return "", fmt.Errorf("agentfield-package.yaml not found in subdirectory %q", subdir)
		}
		return dir, nil
	}

	var found string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && info.Name() == "agentfield-package.yaml" {
			found = filepath.Dir(path)
			return filepath.SkipDir
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if found == "" {
		return "", fmt.Errorf("agentfield-package.yaml not found in the repository")
	}
	return found, nil
}
