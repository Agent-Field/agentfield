package cli

import (
	"bytes"
	"encoding/json"
	"strings"
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
	require.Contains(t, out, "swe-planner-go")
}

// The SWE fleet ships as exactly one catalog row: the Go node installed from
// the `//go` source selector. A re-added root/Python `swe-planner` entry must
// fail here rather than silently reappear in `af catalog`.
func TestCatalogHasSingleGoSWEEntry(t *testing.T) {
	var sweEntries []nodeCatalogEntry
	for _, e := range nodeCatalog {
		require.NotEqual(t, "swe-planner", e.Name, "SWE fleet must be catalogued only as swe-planner-go")
		if strings.Contains(e.Source, "Agent-Field/SWE-AF") {
			sweEntries = append(sweEntries, e)
		}
	}

	require.Len(t, sweEntries, 1, "exactly one catalog entry may install from Agent-Field/SWE-AF")
	require.Equal(t, "swe-planner-go", sweEntries[0].Name)
	require.True(t, strings.HasSuffix(sweEntries[0].Source, "//go"),
		"SWE entry source must select the go subdirectory, got %q", sweEntries[0].Source)
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
