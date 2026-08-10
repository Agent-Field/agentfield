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
	require.GreaterOrEqual(t, len(entries), 4, "catalog must list at least four installable nodes")

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

// A repo that ships both a Python node and its Go counterpart is offered as
// exactly one row, named for the product rather than the implementation, and
// installed from the bare repo URL so the root manifest's `superseded_by:`
// redirect decides which node lands (and carries an existing install across).
// A second row — a re-added Python entry, or the old implementation-suffixed
// name creeping back — must fail here rather than reappear in `af catalog`.
func TestCatalogOffersConsolidatedNodesOnce(t *testing.T) {
	for _, tc := range []struct {
		repo    string
		want    string
		retired string
	}{
		{repo: "Agent-Field/SWE-AF", want: "swe-planner", retired: "swe-planner-go"},
		{repo: "Agent-Field/pr-af", want: "pr-af", retired: "pr-af-go"},
	} {
		t.Run(tc.want, func(t *testing.T) {
			var entries []nodeCatalogEntry
			for _, e := range nodeCatalog {
				if strings.Contains(e.Source, tc.repo) {
					entries = append(entries, e)
				}
			}

			require.Len(t, entries, 1, "exactly one catalog entry may install from %s", tc.repo)
			require.Equal(t, tc.want, entries[0].Name,
				"the entry is named for the product, not the implementation")
			require.Equal(t, "https://github.com/"+tc.repo, entries[0].Source,
				"source must be the bare repo URL so superseded_by picks the node")
			require.Equal(t, "go", entries[0].Language,
				"the redirect lands on the Go node, so that is what the row advertises")

			for _, e := range nodeCatalog {
				require.NotEqual(t, tc.retired, e.Name,
					"%q is the pre-consolidation name and must not reappear", tc.retired)
			}
		})
	}
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
	require.GreaterOrEqual(t, len(entries), 4)
}
