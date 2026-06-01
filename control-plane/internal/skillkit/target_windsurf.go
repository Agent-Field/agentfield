package skillkit

import (
	"errors"
	"path/filepath"
)

// windsurfTarget installs into Windsurf by appending a marker block to
// ~/.codeium/windsurf/memories/global_rules.md.
type windsurfTarget struct{}

func init() { RegisterTarget(windsurfTarget{}) }

func (windsurfTarget) Name() string        { return "windsurf" }
func (windsurfTarget) DisplayName() string { return "Windsurf" }
func (windsurfTarget) Method() string      { return "marker-block" }

func (windsurfTarget) Detected() bool {
	return dirExists(filepath.Join(homeDir(), ".codeium")) ||
		dirExists(filepath.Join(homeDir(), "Library", "Application Support", "Windsurf"))
}

func (windsurfTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".codeium", "windsurf", "memories", "global_rules.md"), nil
}

func (t windsurfTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

func (t windsurfTarget) Uninstall() error {
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

func (t windsurfTarget) Status() (bool, string, error) {
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
