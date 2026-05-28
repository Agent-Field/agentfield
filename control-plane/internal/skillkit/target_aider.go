package skillkit

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// aiderTarget installs into Aider by appending a marker block to
// ~/.aider.conventions.md AND ensuring ~/.aider.conf.yml has a "read:" line
// that loads the conventions file.
type aiderTarget struct{}

func init() { RegisterTarget(aiderTarget{}) }

func (aiderTarget) Name() string        { return "aider" }
func (aiderTarget) DisplayName() string { return "Aider" }
func (aiderTarget) Method() string      { return "marker-block" }

func (aiderTarget) Detected() bool {
	return commandAvailable("aider") ||
		fileExists(filepath.Join(homeDir(), ".aider.conventions.md")) ||
		fileExists(filepath.Join(homeDir(), ".aider.conf.yml"))
}

func (aiderTarget) TargetPath() (string, error) {
	h := homeDir()
	if h == "" {
		return "", errors.New("could not resolve home directory")
	}
	return filepath.Join(h, ".aider.conventions.md"), nil
}

func (t aiderTarget) Install(skill Skill, canonicalCurrentDir string) (InstalledTarget, error) {
	path, err := t.TargetPath()
	if err != nil {
		return InstalledTarget{}, err
	}
	inst, err := installMarkerBlock(skill, canonicalCurrentDir, path)
	if err != nil {
		return InstalledTarget{}, err
	}
	inst.TargetName = t.Name()

	// Make sure ~/.aider.conf.yml references the conventions file. Aider only
	// loads it if explicitly told to.
	confPath := filepath.Join(homeDir(), ".aider.conf.yml")
	readLine := "read: " + path
	if err := ensureLineInFile(confPath, readLine); err != nil {
		return InstalledTarget{}, fmt.Errorf("update aider conf: %w", err)
	}

	return inst, nil
}

func (t aiderTarget) Uninstall() error {
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

func (t aiderTarget) Status() (bool, string, error) {
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

// ensureLineInFile ensures the given line exists in the file at path. Creates
// the file if it doesn't exist. Idempotent — re-runs are no-ops.
func ensureLineInFile(path, line string) error {
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if strings.Contains(string(data), line) {
		return nil
	}
	var sb strings.Builder
	sb.Write(data)
	if len(data) > 0 && !strings.HasSuffix(string(data), "\n") {
		sb.WriteString("\n")
	}
	sb.WriteString(line)
	sb.WriteString("\n")
	return os.WriteFile(path, []byte(sb.String()), 0o644)
}
