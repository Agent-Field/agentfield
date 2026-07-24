package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

	// Remove command links installed by an older version of this skill that
	// the current version no longer ships (e.g. the retired /agentfield shim
	// — redundant since Claude Code invokes skills directly). Links are
	// matched by their target living under this skill's canonical store
	// directory, so user files and other skills' commands are never touched.
	if err := t.cleanupStaleCommands(skill); err != nil {
		return InstalledTarget{}, fmt.Errorf("clean stale commands: %w", err)
	}

	// Also expose any shipped slash commands (skills/<name>/commands/*.md)
	// at ~/.claude/commands/<file>.md so Claude Code picks them up. This is
	// best-effort: a missing commands dir in the skill is not an error.
	if err := t.installCommands(canonicalCurrentDir); err != nil {
		return InstalledTarget{}, fmt.Errorf("install commands: %w", err)
	}

	return InstalledTarget{
		TargetName:  t.Name(),
		Method:      t.Method(),
		Path:        link,
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}

// cleanupStaleCommands removes ~/.claude/commands/*.md symlinks that point
// under this skill's canonical store directory but whose target no longer
// exists — i.e. commands a previous version shipped and the current one
// dropped. Symlinks with live targets and non-symlink entries are left alone.
func (claudeCodeTarget) cleanupStaleCommands(skill Skill) error {
	root, err := CanonicalRoot()
	if err != nil {
		return nil
	}
	skillStore := filepath.Join(root, skill.Name) + string(filepath.Separator)
	dst := filepath.Join(homeDir(), ".claude", "commands")
	entries, err := os.ReadDir(dst)
	if err != nil {
		// Missing or unreadable destination: nothing to clean. If the path is
		// genuinely blocked, installCommands reports it as the install error.
		return nil
	}
	for _, e := range entries {
		link := filepath.Join(dst, e.Name())
		info, err := os.Lstat(link)
		if err != nil || info.Mode()&os.ModeSymlink == 0 {
			continue
		}
		dest, err := os.Readlink(link)
		if err != nil || !strings.HasPrefix(dest, skillStore) {
			continue
		}
		if _, err := os.Stat(dest); os.IsNotExist(err) {
			if err := os.Remove(link); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove stale command %s: %w", link, err)
			}
		}
	}
	return nil
}

// installCommands symlinks every .md file under <skillDir>/commands/ into
// ~/.claude/commands/. Missing commands dir is a no-op.
func (claudeCodeTarget) installCommands(skillDir string) error {
	src := filepath.Join(skillDir, "commands")
	entries, err := os.ReadDir(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	dst := filepath.Join(homeDir(), ".claude", "commands")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return fmt.Errorf("create %s: %w", dst, err)
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".md" {
			continue
		}
		target := filepath.Join(src, e.Name())
		link := filepath.Join(dst, e.Name())
		if info, err := os.Lstat(link); err == nil {
			if info.Mode()&os.ModeSymlink != 0 || info.Mode().IsRegular() {
				if err := os.Remove(link); err != nil {
					return fmt.Errorf("remove existing %s: %w", link, err)
				}
			}
		}
		if err := os.Symlink(target, link); err != nil {
			return fmt.Errorf("symlink %s -> %s: %w", link, target, err)
		}
	}
	return nil
}

func (t claudeCodeTarget) Uninstall() error {
	// Remove every shipped skill's symlink. (Currently a single skill, but the
	// catalog can grow.)
	root, rootErr := CanonicalRoot()
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
		// Best-effort cleanup of any slash-commands we installed for this skill.
		// Source lives at ~/.agentfield/skills/<name>/current/commands/.
		if rootErr != nil {
			continue
		}
		cmdSrc := filepath.Join(root, s.Name, "current", "commands")
		entries, err := os.ReadDir(cmdSrc)
		if err != nil {
			continue
		}
		cmdDst := filepath.Join(homeDir(), ".claude", "commands")
		for _, e := range entries {
			if e.IsDir() || filepath.Ext(e.Name()) != ".md" {
				continue
			}
			_ = os.Remove(filepath.Join(cmdDst, e.Name()))
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
