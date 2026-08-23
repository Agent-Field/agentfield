package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// codexTarget installs the skill into Codex (OpenAI's coding agent CLI) via
// the cross-tool skills standard Codex adopted in late 2025: a personal skill
// is a directory at ~/.codex/skills/<name>/ containing SKILL.md, discovered at
// startup and auto-activated from its description (Codex's own bundled skills
// live alongside them in ~/.codex/skills/.system/). We symlink that directory
// at the canonical versioned store — exactly like the Claude Code target — so
// updates to the store flow through without rewriting anything Codex owns.
//
// Older af binaries appended a marker block to ~/.codex/AGENTS.override.md, a
// file Codex never reads, so the skill was effectively never installed. Every
// install/update/uninstall now strips that block and removes the file when our
// block was the only thing in it.
type codexTarget struct{}

func init() { RegisterTarget(codexTarget{}) }

func (codexTarget) Name() string        { return "codex" }
func (codexTarget) DisplayName() string { return "Codex (OpenAI)" }
func (codexTarget) Method() string      { return "symlink" }

func (codexTarget) Detected() bool {
	return commandAvailable("codex") || dirExists(filepath.Join(homeDir(), ".codex"))
}

func (codexTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".codex", "skills"), nil
}

// legacyRulesPath is the file older af binaries appended marker blocks to.
// Kept only so the blocks can be cleaned up; nothing is ever written there.
func (codexTarget) legacyRulesPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".codex", "AGENTS.override.md"), nil
}

func (t codexTarget) skillLink(skill Skill) (string, error) {
	root, err := t.TargetPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, skill.Name), nil
}

func (t codexTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

	// Replace whatever is there (stale symlink, copied directory, plain file)
	// with a fresh link at the canonical current/ directory.
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

	// Codex has no ~/.claude/commands equivalent — skills are the whole
	// integration surface, so there is nothing else to link.
	if err := t.removeLegacyMarkerBlock(skill); err != nil {
		return InstalledTarget{}, err
	}

	return InstalledTarget{
		TargetName:  t.Name(),
		Method:      t.Method(),
		Path:        link,
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}

func (t codexTarget) Uninstall() error {
	root, err := t.TargetPath()
	if err != nil {
		return err
	}
	for _, s := range Catalog {
		link := filepath.Join(root, s.Name)
		if info, lstatErr := os.Lstat(link); lstatErr == nil {
			if info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular() {
				if err := os.RemoveAll(link); err != nil {
					return fmt.Errorf("remove %s: %w", link, err)
				}
			}
		}
		if err := t.removeLegacyMarkerBlock(s); err != nil {
			return err
		}
	}
	return nil
}

func (t codexTarget) Status() (bool, string, error) {
	link, err := t.skillLink(Catalog[0])
	if err != nil {
		return false, "", err
	}
	info, err := os.Lstat(link)
	if err != nil {
		return false, "", nil
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return true, "manual", nil // a real dir/file lives there — not ours
	}
	dest, err := os.Readlink(link)
	if err != nil {
		return false, "", nil
	}
	// dest looks like .../.agentfield/skills/<name>/<version>
	return true, filepath.Base(dest), nil
}

// removeLegacyMarkerBlock strips this skill's marker block from
// ~/.codex/AGENTS.override.md, the rules file older af binaries wrote into.
// The file is deleted when nothing but whitespace survives the strip, so a
// file created solely for our block does not linger. Other tools' blocks and
// any user prose are preserved.
func (t codexTarget) removeLegacyMarkerBlock(skill Skill) error {
	path, err := t.legacyRulesPath()
	if err != nil {
		return err
	}
	if err := uninstallMarkerBlock(skill, path); err != nil {
		return fmt.Errorf("remove legacy Codex rules block from %s: %w", path, err)
	}
	// uninstallMarkerBlock drops a file left completely empty; a file left
	// holding only whitespace was equally ours alone.
	data, err := os.ReadFile(path)
	if err != nil {
		return nil // already gone, or not ours to interpret
	}
	if strings.TrimSpace(string(data)) != "" {
		return nil
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove empty %s: %w", path, err)
	}
	return nil
}
