package skillkit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// State is the on-disk record of which skills are installed, at which version,
// and which target integrations are active. Persisted at
// ~/.agentfield/skills/.state.json.
type State struct {
	Version string                       `json:"state_version"`
	Skills  map[string]InstalledSkill    `json:"skills"`
}

// InstalledSkill records the installed-version state of a single skill.
type InstalledSkill struct {
	CurrentVersion    string                       `json:"current_version"`
	InstalledAt       time.Time                    `json:"installed_at"`
	AvailableVersions []string                     `json:"available_versions"`
	Targets           map[string]InstalledTarget   `json:"targets"`
}

// InstalledTarget records one target integration for one skill.
type InstalledTarget struct {
	TargetName  string    `json:"target_name"`  // "claude-code", "codex", ...
	Method      string    `json:"method"`       // "symlink", "marker-block", "manual"
	Path        string    `json:"path"`         // file or directory the integration writes to
	Version     string    `json:"version"`      // version installed at this target
	InstalledAt time.Time `json:"installed_at"`
}

const stateFileVersion = "1"

// CanonicalRoot returns ~/.agentfield/skills/. Honors $AGENTFIELD_HOME if set
// (useful for tests and for users who want a non-default location).
func CanonicalRoot() (string, error) {
	if root := os.Getenv("AGENTFIELD_HOME"); root != "" {
		return filepath.Join(root, "skills"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".agentfield", "skills"), nil
}

func stateFilePath() (string, error) {
	root, err := CanonicalRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, ".state.json"), nil
}

// LoadState reads the state file from disk. If the file does not exist yet,
// returns an empty State so first-install flows just write fresh.
func LoadState() (*State, error) {
	path, err := stateFilePath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errIsNotExist(err) {
		return &State{Version: stateFileVersion, Skills: map[string]InstalledSkill{}}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read state file %s: %w", path, err)
	}
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parse state file %s: %w", path, err)
	}
	if s.Skills == nil {
		s.Skills = map[string]InstalledSkill{}
	}
	if s.Version == "" {
		s.Version = stateFileVersion
	}
	return &s, nil
}

// SaveState writes the state file atomically (write-temp + rename).
func SaveState(s *State) error {
	path, err := stateFilePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create state dir: %w", err)
	}
	s.Version = stateFileVersion
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write state tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename state file: %w", err)
	}
	return nil
}

// SortedTargetNames returns the keys of the Targets map in stable order so
// `af skill list` output is deterministic.
func (i InstalledSkill) SortedTargetNames() []string {
	names := make([]string, 0, len(i.Targets))
	for name := range i.Targets {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// errIsNotExist reports whether err indicates a missing file.
func errIsNotExist(err error) bool {
	return err != nil && os.IsNotExist(err)
}
