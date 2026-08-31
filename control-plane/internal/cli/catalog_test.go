package cli

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRunCatalogJSON(t *testing.T) {
	var stdout bytes.Buffer
	require.NoError(t, runCatalog(&stdout, "json"))

	var entries []map[string]interface{}
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &entries))
	require.GreaterOrEqual(t, len(entries), 5, "catalog must list at least five installable nodes")

	for _, e := range entries {
		require.NotEmpty(t, e["name"], "entry missing name: %v", e)
		require.NotEmpty(t, e["description"], "entry %v missing description", e["name"])
		require.NotEmpty(t, e["source"], "entry %v missing source", e["name"])
	}
}

func TestRunCatalogPrettyEndsWithInstallHint(t *testing.T) {
	var stdout bytes.Buffer
	require.NoError(t, runCatalog(&stdout, "pretty"))
	out := stdout.String()
	require.Contains(t, out, "af install <source>")
	require.Contains(t, out, "swe-planner")
}

func TestRunCatalogRejectsUnknownFormat(t *testing.T) {
	var stdout bytes.Buffer
	err := runCatalog(&stdout, "csv")
	require.Equal(t, 2, ExitCode(err))
}

func TestNewCatalogCommandExecute(t *testing.T) {
	cmd := NewCatalogCommand()
	cmd.SetArgs([]string{"-o", "json"})
	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})
	var entries []map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(out), &entries))
	require.GreaterOrEqual(t, len(entries), 5)
}
