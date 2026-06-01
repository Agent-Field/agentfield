package skillkit

import (
	"errors"
	"path/filepath"
)

// opencodeTarget installs into OpenCode by appending a marker block to
// ~/.config/opencode/AGENTS.md.
type opencodeTarget struct{}

func init() { RegisterTarget(opencodeTarget{}) }

func (opencodeTarget) Name() string        { return "opencode" }
func (opencodeTarget) DisplayName() string { return "OpenCode" }
func (opencodeTarget) Method() string      { return "marker-block" }

func (opencodeTarget) Detected() bool {
	return commandAvailable("opencode") || dirExists(filepath.Join(homeDir(), ".config", "opencode"))
}

func (opencodeTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".config", "opencode", "AGENTS.md"), nil
}

func (t opencodeTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

func (t opencodeTarget) Uninstall() error {
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

func (t opencodeTarget) Status() (bool, string, error) {
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
