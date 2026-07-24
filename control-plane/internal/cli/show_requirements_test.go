package cli

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// Contract (item 3): show-requirements on a manifest with required, optional+default,
// and require_one_of entries reports all three categories in JSON, and creates
// nothing under ~/.agentfield/packages.
func TestShowRequirements_JSONReportsAllCategories(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	packagesDir := filepath.Join(home, "packages")
	require.NoError(t, os.MkdirAll(packagesDir, 0o755))

	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/show-requirements-node", "-o", "json"})

	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})

	var report requirementsReport
	require.NoError(t, json.Unmarshal([]byte(out), &report))

	require.Equal(t, "requirements-demo", report.Node)
	require.Equal(t, "1.2.3", report.Version)

	require.Len(t, report.Required, 1)
	require.Equal(t, "API_KEY", report.Required[0].Name)
	require.Equal(t, "secret", report.Required[0].Type)

	require.Len(t, report.Optional, 1)
	require.Equal(t, "REGION", report.Optional[0].Name)
	require.Equal(t, "us-east-1", report.Optional[0].Default)

	require.Len(t, report.RequireOneOf, 1)
	require.Equal(t, "llm_provider", report.RequireOneOf[0].ID)
	require.Len(t, report.RequireOneOf[0].Options, 2)
	require.Equal(t, "ANTHROPIC_API_KEY", report.RequireOneOf[0].Options[0].Name)
	require.Equal(t, "OPENROUTER_API_KEY", report.RequireOneOf[0].Options[1].Name)

	// Inspection must not install anything under ~/.agentfield/packages.
	entries, err := os.ReadDir(packagesDir)
	require.NoError(t, err)
	require.Empty(t, entries, "show-requirements must not write under ~/.agentfield/packages")
}

// Contract (item 3): the text view names every category and pairs each required
// variable (and group option) with the exact `af secrets set` command.
func TestShowRequirements_TextListsCategoriesAndFixCommands(t *testing.T) {
	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/show-requirements-node"})

	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})

	for _, want := range []string{
		"requirements-demo",
		"Required environment variables:",
		"API_KEY",
		"af secrets set API_KEY --node requirements-demo",
		"At least one of",
		"af secrets set ANTHROPIC_API_KEY --node requirements-demo",
		"af secrets set OPENROUTER_API_KEY --node requirements-demo",
		"Optional environment variables",
		"REGION",
		"us-east-1",
	} {
		require.Contains(t, out, want)
	}
}

func TestShowRequirements_RejectsUnknownOutputFormat(t *testing.T) {
	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/show-requirements-node", "-o", "yaml"})
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true
	require.Error(t, cmd.Execute())
}

func TestShowRequirements_TextNoConfigurationNeeded(t *testing.T) {
	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/show-requirements-bare"})

	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})

	require.Contains(t, out, "This node needs no user configuration.")
	require.Contains(t, out, "Install: af install testdata/show-requirements-bare")
}

func TestShowRequirements_TextUnlabeledRequireOneOfGroup(t *testing.T) {
	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/show-requirements-unlabeled"})

	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})

	require.Contains(t, out, "At least one of")
	require.Contains(t, out, "one of these")
}

func TestShowRequirements_InspectErrorPropagates(t *testing.T) {
	cmd := NewShowRequirementsCommand()
	cmd.SetArgs([]string{"testdata/does-not-exist"})
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true

	require.Error(t, cmd.Execute())
}
