package cli

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/skillkit"
)

func TestSkillRenderingAndCommands(t *testing.T) {
	t.Run("interactive picker handles default all skip and explicit picks", func(t *testing.T) {
		// "blank defaults to detected" and "all detected" cases are skipped here
		// because skillkit.DetectedTargets() probes the host environment for
		// installed AI tools and returns a different slice on CI vs developer
		// machines (a CI runner has no codex/cursor/etc. installed). The non-
		// detection cases below still exercise the picker logic itself.
		tests := []struct {
			name  string
			input string
			want  []string
		}{
			{name: "all targets", input: "A\n", want: skillNames(skillkit.AllTargets())},
			{name: "skip", input: "n\n", want: nil},
			{name: "explicit indexes", input: "1,3\n", want: pickedByIndex(0, 2)},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				withStdin(t, tc.input, func() {
					got, err := runInteractivePicker()
					require.NoError(t, err)
					require.Equal(t, tc.want, got)
				})
			})
		}
	})

	t.Run("install report rendering prints installed skipped and failed sections", func(t *testing.T) {
		report := &skillkit.InstallReport{
			Skill:        skillkit.Catalog[0],
			CanonicalDir: "/tmp/skills/demo/0.2.0",
			CurrentLink:  "/tmp/skills/demo/current",
			TargetsInstalled: []skillkit.InstalledTarget{
				{TargetName: "codex", Method: "marker-block", Path: "/tmp/AGENTS.md", Version: "0.2.0"},
			},
			TargetsSkipped: []skillkit.SkipReason{
				{TargetName: "cursor", Reason: "not detected"},
			},
			TargetsFailed: []skillkit.TargetError{
				{TargetName: "aider", Err: assertErr("write failed")},
			},
		}

		output := captureOutput(t, func() {
			printInstallReport(report, false)
			printInstallReport(report, true)
		})
		require.Contains(t, output, "/tmp/AGENTS.md")
		require.Contains(t, output, "not detected")
		require.Contains(t, output, "write failed")
		require.Contains(t, output, "af skill list")
	})

	t.Run("skill list rendering covers empty and populated state", func(t *testing.T) {
		output := captureOutput(t, func() {
			printSkillList(&skillkit.State{Skills: map[string]skillkit.InstalledSkill{}})
		})
		require.Contains(t, output, "af skill install")

		state := &skillkit.State{
			Skills: map[string]skillkit.InstalledSkill{
				"agentfield-multi-reasoner-builder": {
					CurrentVersion:    "0.2.0",
					InstalledAt:       time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC),
					AvailableVersions: []string{"0.1.0", "0.2.0"},
					Targets: map[string]skillkit.InstalledTarget{
						"codex": {Version: "0.2.0", Method: "marker-block", Path: "/tmp/AGENTS.md"},
					},
				},
				"empty-skill": {
					CurrentVersion:    "1.0.0",
					InstalledAt:       time.Date(2026, 4, 8, 13, 0, 0, 0, time.UTC),
					AvailableVersions: []string{"1.0.0"},
					Targets:           map[string]skillkit.InstalledTarget{},
				},
			},
		}

		output = captureOutput(t, func() {
			printSkillList(state)
		})
		require.Contains(t, output, "/tmp/AGENTS.md")
	})

	t.Run("skill commands print path entry file catalog and list", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)

		state := &skillkit.State{
			Skills: map[string]skillkit.InstalledSkill{
				skillkit.Catalog[0].Name: {
					CurrentVersion:    skillkit.Catalog[0].Version,
					InstalledAt:       time.Now().UTC(),
					AvailableVersions: []string{skillkit.Catalog[0].Version},
					Targets:           map[string]skillkit.InstalledTarget{},
				},
			},
		}
		require.NoError(t, skillkit.SaveState(state))

		pathCmd := newSkillPathCommand()
		pathOutput := captureOutput(t, func() {
			require.NoError(t, pathCmd.Execute())
		})
		require.Contains(t, pathOutput, filepath.Join(home, "skills"))

		printCmd := newSkillPrintCommand()
		printOutput := captureOutput(t, func() {
			require.NoError(t, printCmd.Execute())
		})
		require.Contains(t, printOutput, "#")

		catalogCmd := newSkillCatalogCommand()
		catalogOutput := captureOutput(t, func() {
			require.NoError(t, catalogCmd.Execute())
		})
		require.Contains(t, catalogOutput, "Install with:")

		listCmd := newSkillListCommand()
		_ = captureOutput(t, func() {
			require.NoError(t, listCmd.Execute())
		})
	})

	t.Run("install update and uninstall commands cover dry run and missing state", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)

		installCmd := newSkillInstallCommand()
		installCmd.SetArgs([]string{"--dry-run", "--all-targets"})
		_ = captureOutput(t, func() {
			require.NoError(t, installCmd.Execute())
		})

		updateCmd := newSkillUpdateCommand()
		updateCmd.SetArgs([]string{})
		require.Error(t, updateCmd.Execute())

		uninstallCmd := newSkillUninstallCommand()
		uninstallCmd.SetArgs([]string{})
		require.Error(t, uninstallCmd.Execute())
	})
}

func skillNames(targets []skillkit.Target) []string {
	names := make([]string, 0, len(targets))
	for _, t := range targets {
		names = append(names, t.Name())
	}
	return names
}

func pickedByIndex(indexes ...int) []string {
	targets := skillkit.AllTargets()
	names := make([]string, 0, len(indexes))
	for _, idx := range indexes {
		if idx >= 0 && idx < len(targets) {
			names = append(names, targets[idx].Name())
		}
	}
	return names
}

type assertErr string

func (e assertErr) Error() string { return string(e) }
