package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// claudeCodeTarget installs the skill into Claude Code via the
// ~/.claude/skills/<name>/ directory using a symlink to the canonical
// versioned-store location. This is the Anthropic-recommended way: Claude
// Code natively understands SKILL.md + references and the symlink ensures
// updates to the canonical store flow through automatically.
type claudeCodeTarget struct{}

func init() { RegisterTarget(claudeCodeTarget{}) }

func (claudeCodeTarget) Name() string        { return "claude-code" }
func (claudeCodeTarget) DisplayName() string { return "Claude Code" }
func (claudeCodeTarget) Method() string      { return "symlink" }

func (claudeCodeTarget) Detected() bool {
	return dirExists(filepath.Join(homeDir(), ".claude"))
}

func (claudeCodeTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".claude", "skills"), nil
}

func (t claudeCodeTarget) skillLink(skill Skill) (string, error) {
	root, err := t.TargetPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, skill.Name), nil
}

func (t claudeCodeTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

	// Remove any existing entry (regular dir, file, or symlink). Claude Code
	// reads symlinks transparently, so we always replace with a fresh link to
	// the canonical current/ directory.
	if info, err := os.Lstat(link); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular() {
			if err := os.RemoveAll(link); err != nil {
				return InstalledTarget{}, fmt.Errorf("remove existing %s: %w", link, err)
			}
		}
	}

	if err := os.Symlink(canonicalCurrentDir, link); err != nil {
		return InstalledTarget{}, fmt.Errorf("symlink %s -> %s: %w", link, canonicalCurrentDir, err)
	}

	return InstalledTarget{
		TargetName:  t.Name(),
		Method:      t.Method(),
		Path:        link,
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}

func (t claudeCodeTarget) Uninstall() error {
	// Remove every shipped skill's symlink. (Currently a single skill, but the
	// catalog can grow.)
	for _, s := range Catalog {
		link, err := t.skillLink(s)
		if err != nil {
			continue
		}
		if info, err := os.Lstat(link); err == nil {
			if info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular() {
				if err := os.RemoveAll(link); err != nil {
					return fmt.Errorf("remove %s: %w", link, err)
				}
			}
		}
	}
	return nil
}

func (t claudeCodeTarget) Status() (bool, string, error) {
	link, err := t.skillLink(Catalog[0])
	if err != nil {
		return false, "", err
	}
	info, err := os.Lstat(link)
	if err != nil {
		return false, "", nil
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return true, "manual", nil // a regular dir/file lives there — not ours
	}
	dest, err := os.Readlink(link)
	if err != nil {
		return false, "", nil
	}
	// dest looks like .../.agentfield/skills/<name>/<version>
	return true, filepath.Base(dest), nil
}
