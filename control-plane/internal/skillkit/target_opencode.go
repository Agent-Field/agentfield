package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// opencodeTarget installs skills where OpenCode discovers them natively: a
// directory at ~/.config/opencode/skills/<name>/, symlinked at the canonical
// versioned store so updates flow through without rewriting anything OpenCode
// owns.
//
// Older af binaries instead appended a marker block to
// ~/.config/opencode/AGENTS.md. Every install/uninstall now strips that block
// so upgrading users are left with the native skill instead of the native
// skill plus stale instructions.
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

// legacyRulesPath is the file older af binaries appended marker blocks to. It
// is derived from an already-resolved skills root, so unlike Codex's variant
// it cannot fail: every caller has proven TargetPath() succeeds before it gets
// here.
//
// Unlike Codex's AGENTS.override.md — a file af created for itself — this one
// is authored by the user and read by OpenCode, so it is only ever read, and
// only rewritten when it still holds a block of ours.
func (opencodeTarget) legacyRulesPath(root string) string {
	return filepath.Join(filepath.Dir(root), "AGENTS.md")
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
	// The native skill is in place; finish the migration off the old
	// AGENTS.md block so the user is not left carrying both.
	//
	// AGENTS.md belongs to the user, and by this point the integration is
	// already live on disk. Failing the install over a file the skill does
	// not need would push the caller down its failure path, which records
	// nothing in state — leaving `af skill list` reporting OpenCode as not
	// installed and every later install exiting non-zero, over a stale block
	// that has nothing to do with whether OpenCode can load the skill. So the
	// cleanup is advisory here and only Uninstall, where the block is the
	// whole point of the call, treats it as fatal.
	if err := t.removeLegacyMarkerBlock(skill, root); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not clean the legacy OpenCode rules block: %v\n", err)
	}
	return InstalledTarget{TargetName: t.Name(), Method: t.Method(), Path: link, Version: skill.Version, InstalledAt: time.Now().UTC()}, nil
}

func (t opencodeTarget) Uninstall() error {
	// Resolve the target root up front so failures (for example, an
	// unavailable home directory) are reported to the caller instead of
	// being silently ignored while iterating over the catalog.
	root, err := t.TargetPath()
	if err != nil {
		return err
	}
	for _, s := range Catalog {
		link := filepath.Join(root, s.Name)
		if info, err := os.Lstat(link); err == nil && (info.Mode()&os.ModeSymlink != 0 || info.IsDir() || info.Mode().IsRegular()) {
			if err := os.RemoveAll(link); err != nil {
				return fmt.Errorf("remove %s: %w", link, err)
			}
		}
		// Machines that never ran an install in between still carry the
		// legacy block; uninstall has to clear it too.
		if err := t.removeLegacyMarkerBlock(s, root); err != nil {
			return err
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
	if !filepath.IsAbs(dest) {
		dest = filepath.Join(filepath.Dir(link), dest)
	}
	base := filepath.Base(dest)
	// Older installations link directly to a version directory, which may
	// have been removed temporarily. Preserve that version from the link name.
	if base != "current" && strings.Count(base, ".") >= 2 && len(base) > 0 && base[0] >= '0' && base[0] <= '9' {
		return true, base, nil
	}
	resolved, err := filepath.EvalSymlinks(dest)
	if err != nil {
		return false, "", err
	}
	return true, filepath.Base(resolved), nil
}

// removeLegacyMarkerBlock strips this skill's marker block from
// ~/.config/opencode/AGENTS.md, the rules file older af binaries wrote into.
//
// That file belongs to the user — OpenCode reads it, and af never created it
// on its own — so the rules are deliberately stricter than the Codex
// equivalent: a file holding no block of ours is not opened for writing at
// all (its bytes and mtime stay exactly as the user left them), and the file
// is deleted only when removing our block is what emptied it. Other tools'
// blocks and any user prose are preserved. A missing file is a no-op; every
// other failure is returned, and the two callers weigh it differently —
// Uninstall propagates it, Install warns (see there).
//
// Every filesystem call goes through the package's reconcile* seams so each
// failure branch below is reachable from a test.
func (t opencodeTarget) removeLegacyMarkerBlock(skill Skill, root string) error {
	path := t.legacyRulesPath(root)
	data, err := reconcileReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read legacy OpenCode rules file %s: %w", path, err)
	}
	if _, ours := findMarkerBlock(string(data), skill); !ours {
		return nil // nothing of ours in there; leave the user's file alone
	}

	cleaned := strings.TrimRight(stripMarkerBlock(string(data), skill), "\n")
	if strings.TrimSpace(cleaned) == "" {
		// Our block was the only thing in it, so the file was ours alone.
		if err := reconcileRemove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove %s: %w", path, err)
		}
		return nil
	}

	perm := os.FileMode(0o644)
	if info, err := os.Stat(path); err == nil {
		perm = info.Mode().Perm()
	}
	tmp := path + ".af-tmp"
	if err := reconcileWriteFile(tmp, []byte(cleaned+"\n"), perm); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := reconcileRename(tmp, path); err != nil {
		return fmt.Errorf("rename into %s: %w", path, err)
	}
	return nil
}
