package cli

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVerifyAlias_TopLevelExists pins that `af verify` is reachable as a
// top-level command — not just `af vc verify`. The PR description, demo
// docs, and longstanding muscle memory all use `af verify <file>`. Without
// this alias the command errors with "unknown command".
//
// Regression target: copy-pasting `af verify audit.json` from the docs used
// to fail with "unknown command 'verify'", forcing operators to discover
// the correct path (`af vc verify`) by reading source.
func TestVerifyAlias_TopLevelExists(t *testing.T) {
	cmd := NewVerifyAliasCommand()
	require.NotNil(t, cmd, "alias command must be constructible")
	assert.True(t, strings.HasPrefix(cmd.Use, "verify"),
		"Use should be 'verify <vc-file.json>', got %q", cmd.Use)
	assert.NotEmpty(t, cmd.Short, "alias must surface a Short description in --help")
	assert.NotNil(t, cmd.RunE, "alias must inherit the canonical Run target")
}

// TestVerifyAlias_FlagsMatchVCVerify guards against the alias and the
// canonical command drifting out of sync. Both must accept the same flag
// set so doc snippets work for either path.
func TestVerifyAlias_FlagsMatchVCVerify(t *testing.T) {
	canonical := NewVCVerifyCommand()
	alias := NewVerifyAliasCommand()

	expected := []string{"format", "resolve-web", "did-resolver", "verbose"}
	for _, name := range expected {
		assert.NotNilf(t, canonical.Flag(name),
			"canonical `vc verify` is missing flag --%s — test fixture is stale", name)
		assert.NotNilf(t, alias.Flag(name),
			"alias `verify` must expose --%s like the canonical command", name)
	}

	// Argument arity must also match: both accept exactly one positional
	// arg (the VC file path). If the canonical command grows to a range,
	// the alias should follow.
	require.NotNil(t, canonical.Args)
	require.NotNil(t, alias.Args)
	assert.Error(t, alias.Args(alias, []string{}),
		"alias must reject zero args (file path is required)")
	assert.NoError(t, alias.Args(alias, []string{"some-file.json"}),
		"alias must accept exactly one positional arg")
	assert.Error(t, alias.Args(alias, []string{"a.json", "b.json"}),
		"alias must reject more than one arg")
}

// TestVerifyAlias_RegisteredOnRoot confirms the alias is wired into the
// real RootCmd (not just constructible in isolation). This is the
// integration check that protects against someone removing the
// `RootCmd.AddCommand(NewVerifyAliasCommand())` line in root.go.
func TestVerifyAlias_RegisteredOnRoot(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	root := NewRootCommand(func(cmd *cobra.Command, args []string) {}, VersionInfo{})

	verify, _, err := root.Find([]string{"verify"})
	require.NoError(t, err, "root must resolve `verify` to a command")
	require.NotNil(t, verify)
	assert.True(t, strings.HasPrefix(verify.Use, "verify"),
		"resolved command's Use must start with 'verify'")
}
