package skillkit

import (
	"errors"
	"path/filepath"
)

// codexTarget installs the skill into Codex (OpenAI's coding agent CLI) by
// appending a marker block to ~/.codex/AGENTS.override.md. The block points
// at the canonical SKILL.md so updates flow through automatically.
type codexTarget struct{}

func init() { RegisterTarget(codexTarget{}) }

func (codexTarget) Name() string        { return "codex" }
func (codexTarget) DisplayName() string { return "Codex (OpenAI)" }
func (codexTarget) Method() string      { return "marker-block" }

func (codexTarget) Detected() bool {
	return commandAvailable("codex") || dirExists(filepath.Join(homeDir(), ".codex"))
}

func (codexTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".codex", "AGENTS.override.md"), nil
}

func (t codexTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
	path, err := t.TargetPath()
	if err != nil {
		return InstalledTarget{}, err
	}
	inst, err := installMarkerBlock(skill, canonicalCurrentDir, path)
	if err != nil {
		return InstalledTarget{}, err
	}
	inst.TargetName = t.Name()
	return inst, nil
}

func (t codexTarget) Uninstall() error {
	path, err := t.TargetPath()
	if err != nil {
		return err
	}
	for _, s := range Catalog {
		if err := uninstallMarkerBlock(s, path); err != nil {
			return err
		}
	}
	return nil
}

func (t codexTarget) Status() (bool, string, error) {
	path, err := t.TargetPath()
	if err != nil {
		return false, "", err
	}
	v := readMarkerVersion(Catalog[0], path)
	if v == "" {
		return false, "", nil
	}
	return true, v, nil
}
