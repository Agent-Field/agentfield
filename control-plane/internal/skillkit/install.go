package skillkit

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// InstallOptions controls how a skill is installed across targets.
type InstallOptions struct {
	SkillName string   // canonical skill name; empty = first in catalog
	Version   string   // explicit version; empty = current binary's embedded version
	Targets   []string // explicit target list; empty = use AllDetected/AllRegistered/Selection
	AllDetected bool   // install into every target Detected() reports true
	AllRegistered bool // install into every registered target (even undetected)
	Force     bool    // re-install even if state shows the same version is already present
	DryRun    bool    // print what would happen, don't write
}

// InstallReport summarizes one install operation. The CLI uses this to print
// a clean handoff message.
type InstallReport struct {
	Skill              Skill
	CanonicalDir       string                       // ~/.agentfield/skills/<name>/<version>
	CurrentLink        string                       // ~/.agentfield/skills/<name>/current
	WroteCanonical     bool
	TargetsInstalled   []InstalledTarget
	TargetsSkipped     []SkipReason
	TargetsFailed      []TargetError
}

type SkipReason struct {
	TargetName string
	Reason     string
}

type TargetError struct {
	TargetName string
	Err        error
}

// Install runs an install pass according to opts. It performs the canonical
// write first, switches the `current` symlink, then installs into each
// selected target. Idempotent and safe to re-run.
func Install(opts InstallOptions) (*InstallReport, error) {
	skill, err := resolveSkill(opts.SkillName, opts.Version)
	if err != nil {
		return nil, err
	}

	root, err := CanonicalRoot()
	if err != nil {
		return nil, err
	}

	report := &InstallReport{
		Skill:        skill,
		CanonicalDir: filepath.Join(root, skill.Name, skill.Version),
		CurrentLink:  filepath.Join(root, skill.Name, "current"),
	}

	// 1. Write canonical store (versioned dir + current symlink)
	if !opts.DryRun {
		if err := writeCanonical(skill, report.CanonicalDir); err != nil {
			return nil, fmt.Errorf("write canonical store: %w", err)
		}
		if err := updateCurrentLink(report.CurrentLink, report.CanonicalDir); err != nil {
			return nil, fmt.Errorf("update current symlink: %w", err)
		}
		report.WroteCanonical = true
	}

	// 2. Resolve target selection
	selected, skipped, err := resolveTargets(opts)
	if err != nil {
		return nil, err
	}
	report.TargetsSkipped = append(report.TargetsSkipped, skipped...)

	// 3. Install into each selected target
	state, err := LoadState()
	if err != nil {
		return nil, err
	}
	skillState, ok := state.Skills[skill.Name]
	if !ok {
		skillState = InstalledSkill{
			CurrentVersion:    skill.Version,
			InstalledAt:       time.Now().UTC(),
			AvailableVersions: []string{skill.Version},
			Targets:           map[string]InstalledTarget{},
		}
	} else {
		// Track new version if not seen before
		seen := false
		for _, v := range skillState.AvailableVersions {
			if v == skill.Version {
				seen = true
				break
			}
		}
		if !seen {
			skillState.AvailableVersions = append(skillState.AvailableVersions, skill.Version)
			sort.Strings(skillState.AvailableVersions)
		}
		skillState.CurrentVersion = skill.Version
		if skillState.Targets == nil {
			skillState.Targets = map[string]InstalledTarget{}
		}
	}

	for _, t := range selected {
		// Skip if already at this version and not forced.
		if !opts.Force {
			if existing, ok := skillState.Targets[t.Name()]; ok && existing.Version == skill.Version {
				report.TargetsSkipped = append(report.TargetsSkipped, SkipReason{
					TargetName: t.Name(),
					Reason:     fmt.Sprintf("already installed at v%s (use --force to reinstall)", existing.Version),
				})
				continue
			}
		}

		if opts.DryRun {
			report.TargetsInstalled = append(report.TargetsInstalled, InstalledTarget{
				TargetName: t.Name(),
				Method:     t.Method(),
				Version:    skill.Version,
			})
			continue
		}

		inst, err := t.Install(skill, report.CurrentLink)
		if err != nil {
			report.TargetsFailed = append(report.TargetsFailed, TargetError{TargetName: t.Name(), Err: err})
			continue
		}
		inst.TargetName = t.Name()
		skillState.Targets[t.Name()] = inst
		report.TargetsInstalled = append(report.TargetsInstalled, inst)
	}

	if !opts.DryRun {
		state.Skills[skill.Name] = skillState
		if err := SaveState(state); err != nil {
			return nil, fmt.Errorf("save state: %w", err)
		}
	}

	return report, nil
}

// Uninstall removes a skill from the named targets (or all if empty), and
// optionally drops the canonical store entirely (if RemoveCanonical=true).
type UninstallOptions struct {
	SkillName       string
	Targets         []string
	RemoveCanonical bool
}

func Uninstall(opts UninstallOptions) error {
	skill, err := resolveSkill(opts.SkillName, "")
	if err != nil {
		return err
	}

	state, err := LoadState()
	if err != nil {
		return err
	}
	skillState, ok := state.Skills[skill.Name]
	if !ok {
		return fmt.Errorf("skill %q is not installed", skill.Name)
	}

	targetNames := opts.Targets
	if len(targetNames) == 0 {
		// uninstall from all currently-installed targets
		for name := range skillState.Targets {
			targetNames = append(targetNames, name)
		}
	}
	sort.Strings(targetNames)

	for _, name := range targetNames {
		t, err := TargetByName(name)
		if err != nil {
			return err
		}
		if err := t.Uninstall(); err != nil {
			return fmt.Errorf("uninstall from %s: %w", name, err)
		}
		delete(skillState.Targets, name)
	}

	if len(skillState.Targets) == 0 || opts.RemoveCanonical {
		delete(state.Skills, skill.Name)
		if opts.RemoveCanonical {
			root, err := CanonicalRoot()
			if err == nil {
				_ = os.RemoveAll(filepath.Join(root, skill.Name))
			}
		}
	} else {
		state.Skills[skill.Name] = skillState
	}

	return SaveState(state)
}

// Update is a convenience wrapper that re-installs the skill into every
// target it's currently installed at, using the binary's embedded version.
func Update(skillName string) (*InstallReport, error) {
	state, err := LoadState()
	if err != nil {
		return nil, err
	}
	skill, err := resolveSkill(skillName, "")
	if err != nil {
		return nil, err
	}
	skillState, ok := state.Skills[skill.Name]
	if !ok {
		return nil, fmt.Errorf("skill %q is not installed (run `af skill install` first)", skill.Name)
	}
	var targets []string
	for name := range skillState.Targets {
		targets = append(targets, name)
	}
	sort.Strings(targets)
	return Install(InstallOptions{
		SkillName: skill.Name,
		Targets:   targets,
		Force:     true,
	})
}

// ListInstalled returns the on-disk state for `af skill list`.
func ListInstalled() (*State, error) {
	return LoadState()
}

// ── Internals ────────────────────────────────────────────────────────────

func resolveSkill(name, version string) (Skill, error) {
	if name == "" {
		if len(Catalog) == 0 {
			return Skill{}, fmt.Errorf("no skills registered in this binary")
		}
		name = Catalog[0].Name
	}
	skill, err := CatalogByName(name)
	if err != nil {
		return Skill{}, err
	}
	if version != "" && version != skill.Version {
		return Skill{}, fmt.Errorf("skill %q version %q is not embedded in this binary (binary ships v%s); upgrade the binary or build with the desired version", name, version, skill.Version)
	}
	return skill, nil
}

func resolveTargets(opts InstallOptions) (selected []Target, skipped []SkipReason, err error) {
	if len(opts.Targets) > 0 {
		for _, name := range opts.Targets {
			t, err := TargetByName(name)
			if err != nil {
				return nil, nil, err
			}
			selected = append(selected, t)
		}
		return selected, nil, nil
	}
	if opts.AllRegistered {
		return AllTargets(), nil, nil
	}
	if opts.AllDetected {
		for _, t := range AllTargets() {
			if t.Detected() {
				selected = append(selected, t)
			} else {
				skipped = append(skipped, SkipReason{
					TargetName: t.Name(),
					Reason:     "not detected on this machine (use --all-targets to force)",
				})
			}
		}
		return selected, skipped, nil
	}
	// Default: detected only
	for _, t := range AllTargets() {
		if t.Detected() {
			selected = append(selected, t)
		} else {
			skipped = append(skipped, SkipReason{
				TargetName: t.Name(),
				Reason:     "not detected",
			})
		}
	}
	return selected, skipped, nil
}

func writeCanonical(skill Skill, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	files, err := skill.EnumerateFiles()
	if err != nil {
		return err
	}
	for rel, data := range files {
		dest := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", filepath.Dir(dest), err)
		}
		if err := os.WriteFile(dest, data, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", dest, err)
		}
	}
	return nil
}

func updateCurrentLink(linkPath, targetPath string) error {
	// Remove any existing symlink, file, or directory at the link path
	if info, err := os.Lstat(linkPath); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			if err := os.Remove(linkPath); err != nil {
				return err
			}
		} else if info.IsDir() {
			if err := os.RemoveAll(linkPath); err != nil {
				return err
			}
		} else {
			if err := os.Remove(linkPath); err != nil {
				return err
			}
		}
	}
	// Use a relative symlink so the canonical store is portable across home moves
	rel, err := filepath.Rel(filepath.Dir(linkPath), targetPath)
	if err != nil {
		rel = targetPath
	}
	return os.Symlink(rel, linkPath)
}
