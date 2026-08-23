package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// opencodeTarget installs skills where OpenCode discovers them natively.
type opencodeTarget struct{}

func init() { RegisterTarget(opencodeTarget{}) }

func (opencodeTarget) Name() string        { return "opencode" }
func (opencodeTarget) DisplayName() string { return "OpenCode" }
func (opencodeTarget) Method() string      { return "symlink" }

func (opencodeTarget) Detected() bool {
	return commandAvailable("opencode") || dirExists(filepath.Join(homeDir(), ".config", "opencode"))
}

func (opencodeTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".config", "opencode", "skills"), nil
}

func (t opencodeTarget) skillLink(skill Skill) (string, error) {
	root, err := t.TargetPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, skill.Name), nil
}

func (t opencodeTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
	root, err := t.TargetPath()
	if err != nil {
		return InstalledTarget{}, err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return InstalledTarget{}, fmt.Errorf("create %s: %w", root, err)
	}
	link, err := t.skillLink(skill)
	if err != nil {
		return InstalledTarget{}, err
	}
	if info, err := os.Lstat(link); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular() {
			if err := os.RemoveAll(link); err != nil {
				return InstalledTarget{}, fmt.Errorf("remove existing %s: %w", link, err)
			}
		}
	} else if !os.IsNotExist(err) {
		return InstalledTarget{}, fmt.Errorf("inspect %s: %w", link, err)
	}
	if err := os.Symlink(canonicalCurrentDir, link); err != nil {
		return InstalledTarget{}, fmt.Errorf("symlink %s -> %s: %w", link, canonicalCurrentDir, err)
	}
	return InstalledTarget{TargetName: t.Name(), Method: t.Method(), Path: link, Version: skill.Version, InstalledAt: time.Now().UTC()}, nil
}

func (t opencodeTarget) Uninstall() error {
	// Resolve the target root up front so failures (for example, an
	// unavailable home directory) are reported to the caller instead of
	// being silently ignored while iterating over the catalog.
	if _, err := t.TargetPath(); err != nil {
		return err
	}
	for _, s := range Catalog {
		link, err := t.skillLink(s)
		if err != nil {
			return err
		}
		if info, err := os.Lstat(link); err == nil && (info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular()) {
			if err := os.RemoveAll(link); err != nil {
				return fmt.Errorf("remove %s: %w", link, err)
			}
		}
	}
	return nil
}

func (t opencodeTarget) Status() (bool, string, error) {
	link, err := t.skillLink(Catalog[0])
	if err != nil {
		return false, "", err
	}
	info, err := os.Lstat(link)
	if os.IsNotExist(err) {
		return false, "", nil
	}
	if err != nil {
		return false, "", err
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return true, "manual", nil
	}
	dest, err := os.Readlink(link)
	if err != nil {
		return false, "", err
	}
	resolved, err := filepath.EvalSymlinks(dest)
	if err != nil {
		return false, "", err
	}
	return true, filepath.Base(resolved), nil
}
