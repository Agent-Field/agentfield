package skillkit

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// installMarkerBlock is the shared install logic used by every file-append
// target (Codex, Gemini, OpenCode, Aider, Windsurf). It:
//
//  1. Ensures the parent directory exists
//  2. Reads the existing file (or starts empty)
//  3. Strips any prior block belonging to THIS skill (regardless of version)
//  4. Appends the freshly rendered pointer block (with the current version)
//  5. Writes the file back atomically
//
// The marker pattern is `<!-- agentfield-skill:<name> v<version> -->` ...
// `<!-- /agentfield-skill:<name> -->` so re-installs replace cleanly and
// other tools (plandb, etc.) can append their own blocks without collision.
func installMarkerBlock(skill Skill, canonicalCurrentDir, targetPath string) (InstalledTarget, error) {
	if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
		return InstalledTarget{}, fmt.Errorf("create parent dir for %s: %w", targetPath, err)
	}

	existing := ""
	if data, err := os.ReadFile(targetPath); err == nil {
		existing = string(data)
	} else if !os.IsNotExist(err) {
		return InstalledTarget{}, fmt.Errorf("read %s: %w", targetPath, err)
	}

	cleaned := stripMarkerBlock(existing, skill)
	cleaned = strings.TrimRight(cleaned, "\n")

	block := renderPointerBlock(skill, canonicalCurrentDir)
	var sb strings.Builder
	if cleaned != "" {
		sb.WriteString(cleaned)
		sb.WriteString("\n\n")
	}
	sb.WriteString(block)
	sb.WriteString("\n")

	tmp := targetPath + ".af-tmp"
	if err := os.WriteFile(tmp, []byte(sb.String()), 0o644); err != nil {
		return InstalledTarget{}, fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, targetPath); err != nil {
		return InstalledTarget{}, fmt.Errorf("rename into %s: %w", targetPath, err)
	}

	return InstalledTarget{
		Method:      "marker-block",
		Path:        targetPath,
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}

// uninstallMarkerBlock strips a skill's marker block from a target file. If
// the file is empty after the strip, it is removed.
func uninstallMarkerBlock(skill Skill, targetPath string) error {
	data, err := os.ReadFile(targetPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read %s: %w", targetPath, err)
	}
	cleaned := strings.TrimRight(stripMarkerBlock(string(data), skill), "\n")
	if cleaned == "" {
		// Don't leave an empty file lying around if it was created solely for our block.
		_ = os.Remove(targetPath)
		return nil
	}
	tmp := targetPath + ".af-tmp"
	if err := os.WriteFile(tmp, []byte(cleaned+"\n"), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	return os.Rename(tmp, targetPath)
}

// stripMarkerBlock removes any agentfield-skill block for the named skill
// from the input string. Tolerates multiple occurrences (defensive).
func stripMarkerBlock(input string, skill Skill) string {
	startNeedle := markerStartPattern(skill)
	endNeedle := markerEnd(skill)

	out := input
	for {
		startIdx := strings.Index(out, startNeedle)
		if startIdx < 0 {
			return out
		}
		endIdx := strings.Index(out[startIdx:], endNeedle)
		if endIdx < 0 {
			// Malformed: opening marker but no close. Drop everything from the
			// opening marker to end-of-file to avoid leaving half a block.
			return strings.TrimRight(out[:startIdx], "\n")
		}
		endIdx += startIdx + len(endNeedle)
		// Trim a single trailing newline after the end marker for cleanliness.
		if endIdx < len(out) && out[endIdx] == '\n' {
			endIdx++
		}
		// Trim trailing whitespace before the start marker too.
		before := strings.TrimRight(out[:startIdx], " \t\n")
		out = before + "\n" + out[endIdx:]
	}
}

// readMarkerVersion scans a target file and returns the version of the skill
// currently installed there, or empty if not present. Used by Status().
func readMarkerVersion(skill Skill, targetPath string) string {
	data, err := os.ReadFile(targetPath)
	if err != nil {
		return ""
	}
	content := string(data)
	pattern := markerStartPattern(skill) // "<!-- agentfield-skill:<name> "
	idx := strings.Index(content, pattern)
	if idx < 0 {
		return ""
	}
	rest := content[idx+len(pattern):]
	// Expect "vX.Y.Z -->" — extract up to space.
	end := strings.IndexAny(rest, " ")
	if end < 0 {
		return ""
	}
	v := strings.TrimPrefix(rest[:end], "v")
	return v
}
