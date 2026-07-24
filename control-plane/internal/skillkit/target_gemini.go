package skillkit

import (
	"errors"
	"path/filepath"
)

// geminiTarget installs into the Gemini CLI by appending a marker block to
// ~/.gemini/GEMINI.md.
type geminiTarget struct{}

func init() { RegisterTarget(geminiTarget{}) }

func (geminiTarget) Name() string        { return "gemini" }
func (geminiTarget) DisplayName() string { return "Gemini CLI" }
func (geminiTarget) Method() string      { return "marker-block" }

func (geminiTarget) Detected() bool {
	return commandAvailable("gemini") || dirExists(filepath.Join(homeDir(), ".gemini"))
}

func (geminiTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".gemini", "GEMINI.md"), nil
}

func (t geminiTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
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

func (t geminiTarget) Uninstall() error {
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

func (t geminiTarget) Status() (bool, string, error) {
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
