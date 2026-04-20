package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// copilotTarget installs the skill into GitHub Copilot CLI via
// ~/.copilot/skills/<name>/ using a symlink to the canonical versioned-store
// location. Copilot CLI natively reads SKILL.md files from that directory and
// the symlink ensures updates to the canonical store flow through
// automatically. This mirrors the claudeCodeTarget approach.
//
// Note on deduplication: Copilot CLI auto-discovers skills from both
// ~/.copilot/skills/ and ~/.claude/skills/ (when the Claude Code config is
// present). If both targets have installed the same skill, Copilot sees it
// twice at discovery time but dedupes internally by skill name. We still
// create the symlink so the integration is explicit; a one-line warning is
// printed to stderr to surface the overlap to the user.
type copilotTarget struct{}

func init() { RegisterTarget(copilotTarget{}) }

func (copilotTarget) Name() string        { return "copilot" }
func (copilotTarget) DisplayName() string { return "GitHub Copilot CLI" }
func (copilotTarget) Method() string      { return "symlink" }

func (copilotTarget) Detected() bool {
	if dirExists(filepath.Join(homeDir(), ".copilot")) {
		return true
	}
	return commandAvailable("copilot")
}

func (copilotTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".copilot", "skills"), nil
}

func (t copilotTarget) skillLink(skill Skill) (string, error) {
	root, err := t.TargetPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, skill.Name), nil
}

func (t copilotTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

	// Warn if the same skill is already installed into ~/.claude/skills/ —
	// Copilot CLI reads from both locations and the duplicate symlinks, while
	// harmless (Copilot dedupes by name), may confuse users grepping for where
	// the skill came from.
	if h := homeDir(); h != "" {
		claudeLink := filepath.Join(h, ".claude", "skills", skill.Name)
		if info, err := os.Lstat(claudeLink); err == nil && info.Mode()&os.ModeSymlink != 0 {
			fmt.Fprintf(os.Stderr, "note: skill %q is also installed into ~/.claude/skills/; Copilot CLI reads both locations and will dedupe by name.\n", skill.Name)
		}
	}

	// Remove any existing entry (regular dir, file, or symlink). Copilot reads
	// symlinks transparently, so we always replace with a fresh link to the
	// canonical current/ directory.
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

func (t copilotTarget) Uninstall() error {
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

func (t copilotTarget) Status() (bool, string, error) {
	link, err := t.skillLink(Catalog[0])
	if err != nil {
		return false, "", err
	}
	info, err := os.Lstat(link)
	if err != nil {
		return false, "", nil
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return true, "manual", nil
	}
	dest, err := os.Readlink(link)
	if err != nil {
		return false, "", nil
	}
	return true, filepath.Base(dest), nil
}
