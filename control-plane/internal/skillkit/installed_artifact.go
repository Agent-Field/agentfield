package skillkit

import (
	"os"
	"path/filepath"
	"strings"
)

// installedArtifactValid reports whether the integration recorded in state for
// (skill, target) is actually present and usable on disk.
//
// The install flow skips a target whose recorded version already matches the
// binary's. That skip is only safe when the recorded artifact is real: state
// files outlive the things they describe. Machines in the wild carry entries
// pointing at deleted temp directories (a test suite that resolved targets
// against the real home), at rules files the user has since cleaned out, and
// at integrations written by an install method this binary no longer uses
// (Codex moved from a rules-file marker block to a native skills symlink).
// Every one of those is a target that would never be repaired, because the
// version matched. Invalid here means the caller reinstalls — the same code
// path --force takes for that one target.
//
// Validation is keyed on the RECORDED method, not the target's current one:
// what is on disk was written by whatever method was in use then. A recorded
// method that no longer matches how this binary installs the target is invalid
// by definition, which is what migrates legacy Codex marker-block entries onto
// the native skills directory.
func installedArtifactValid(skill Skill, target Target, recorded InstalledTarget) bool {
	if recorded.Method != target.Method() {
		return false
	}
	switch recorded.Method {
	case "symlink":
		return symlinkArtifactValid(skill, recorded.Path)
	case "marker-block":
		return markerBlockArtifactValid(skill, recorded.Path)
	case "manual":
		// Nothing was ever written to disk — the user pasted the rules into
		// the agent's own settings UI (Cursor, legacy Windsurf). There is no
		// artifact to inspect, and reinstalling only reprints instructions.
		return true
	default:
		// Unknown method: assume the recording predates this binary and repair.
		return false
	}
}

// symlinkArtifactValid checks that the recorded path is a symlink that
// resolves to something that exists inside this skill's canonical store.
// A link into another skill's store, into a stale temp directory, or a
// dangling link all fail.
func symlinkArtifactValid(skill Skill, path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink == 0 {
		return false
	}
	dest, err := os.Readlink(path)
	if err != nil {
		return false
	}
	if !filepath.IsAbs(dest) {
		dest = filepath.Join(filepath.Dir(path), dest)
	}
	root, err := CanonicalRoot()
	if err != nil {
		return false
	}
	if !pathWithin(filepath.Join(root, skill.Name), dest) {
		return false
	}
	// Following the link must land on something real (the canonical version
	// directory can be deleted out from under an intact link).
	_, err = os.Stat(path)
	return err == nil
}

// markerBlockArtifactValid checks that the recorded rules file still exists,
// still carries this skill's marker block, and that the SKILL.md path the
// block points the agent at is really on disk. A block whose pointer target
// has been deleted teaches the agent to read a file that is not there, which
// is worse than no block at all.
func markerBlockArtifactValid(skill Skill, path string) bool {
	if path == "" {
		return false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	block, ok := findMarkerBlock(string(data), skill)
	if !ok {
		return false
	}
	pointer := markerBlockPointerPath(block, skill)
	if pointer == "" {
		return false
	}
	_, err = os.Stat(pointer)
	return err == nil
}

// findMarkerBlock returns the text of this skill's marker block, from the
// opening marker through the closing one.
func findMarkerBlock(content string, skill Skill) (string, bool) {
	start := strings.Index(content, markerStartPattern(skill))
	if start < 0 {
		return "", false
	}
	end := strings.Index(content[start:], markerEnd(skill))
	if end < 0 {
		return "", false
	}
	return content[start : start+end+len(markerEnd(skill))], true
}

// markerBlockPointerPath extracts the canonical SKILL.md path a rendered
// pointer block sends the agent to (see renderPointerBlock, which writes it
// as its own indented line). Empty when the block carries no such line.
func markerBlockPointerPath(block string, skill Skill) string {
	for _, line := range strings.Split(block, "\n") {
		candidate := strings.TrimSpace(line)
		if candidate == "" || filepath.Base(candidate) != skill.EntryFile {
			continue
		}
		return candidate
	}
	return ""
}

// pathWithin reports whether path is root itself or lives beneath it. The
// comparison is lexical on purpose: both sides are built from the same home
// directory string, and resolving symlinks would make a macOS /var vs
// /private/var difference look like an escape.
func pathWithin(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
