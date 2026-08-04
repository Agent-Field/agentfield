package cli

import (
	"fmt"
	"io"
	"os"

	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
	"github.com/spf13/cobra"
)

// nodeCatalogEntry describes one installable AgentField agent node.
//
// This is deliberately a hand-vetted, in-binary list so `af catalog` works
// offline and gives a harness a curated set of nodes to install before any
// registry search lands. It is seeded from the desktop app's curated list
// (desktop/src/shared/catalog.ts) — keep the two in sync when adding nodes.
//
// One row per product, sourced at the bare repo URL. A repo that ships more
// than one implementation of the same node says which one it wants installed
// with `superseded_by:` in its root manifest — the redirect that makes
// `af install <repo>` land on the maintained node (SWE-AF and pr-af both point
// their root at `//go`). Naming `//go` here would install that same node, but
// it would skip the redirect, and the redirect is what carries a user who
// already has the superseded node across: it installs the successor first,
// migrates node-scoped secrets, and only then retires the old package. So the
// catalog names the repo and lets the manifest decide.
//
// `name` MUST equal the name the package ends up registered under once the
// install settles — the registry key a harness then passes to `af run`. Note
// that is the name after any `superseded_by:` redirect resolves, which need
// not be the `name:` in the manifest at the source: a successor may
// deliberately take its predecessor's name (an in-place rename), and it may
// live in a subdirectory this list never names. It is often not the repo name
// either (SWE-AF → swe-planner).
type nodeCatalogEntry struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Source      string `json:"source"` // pass to `af install <source>`
	Docs        string `json:"docs"`   // public repo/docs URL
	Language    string `json:"language"`
}

// nodeCatalog is the curated set of installable nodes. Mirrors
// desktop/src/shared/catalog.ts. Docs point at the public GitHub repo (the
// `//<subdir>` source selector stripped, since the docs live at the repo root).
var nodeCatalog = []nodeCatalogEntry{
	{
		Name:        "swe-planner",
		Description: "Autonomous software-engineering fleet: plan, code, test, and ship production-grade PRs — one static binary",
		Source:      "https://github.com/Agent-Field/SWE-AF",
		Docs:        "https://github.com/Agent-Field/SWE-AF",
		Language:    "go",
	},
	{
		Name:        "pr-af",
		Description: "Deep, evidence-backed review of any GitHub pull request — one static binary",
		Source:      "https://github.com/Agent-Field/pr-af",
		Docs:        "https://github.com/Agent-Field/pr-af",
		Language:    "go",
	},
	{
		Name:        "sec-af",
		Description: "Code security auditor: scans repositories and proves exploitability with verdicts and traces",
		Source:      "https://github.com/Agent-Field/sec-af",
		Docs:        "https://github.com/Agent-Field/sec-af",
		Language:    "python",
	},
	{
		Name:        "cloudsecurity-af",
		Description: "Cloud security posture: read-only attack-path scans across AWS, GCP, and Azure",
		Source:      "https://github.com/Agent-Field/cloudsecurity-af",
		Docs:        "https://github.com/Agent-Field/cloudsecurity-af",
		Language:    "python",
	},
}

// NewCatalogCommand builds `af catalog` — the browse step of the golden path,
// letting a harness discover installable nodes before `af install`.
func NewCatalogCommand() *cobra.Command {
	var outputFormat string
	cmd := &cobra.Command{
		Use:   "catalog",
		Short: "Browse installable AgentField agent nodes",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runCatalog(os.Stdout, autoOutputFormat(outputFormat, isOutputTerminal()))
		},
		SilenceUsage: true,
	}
	cmd.Flags().StringVarP(&outputFormat, "output", "o", "", "Output format: pretty, json, yaml")
	return cmd
}

func runCatalog(stdout io.Writer, format string) error {
	switch format {
	case "json", "yaml":
		return writeValue(stdout, nodeCatalog, format)
	case "pretty":
		renderNodeCatalog(stdout)
		return nil
	default:
		return cliExitError{Code: 2, Err: fmt.Errorf("output format must be pretty, json, or yaml")}
	}
}

func renderNodeCatalog(stdout io.Writer) {
	rows := make([][]string, 0, len(nodeCatalog))
	for _, n := range nodeCatalog {
		rows = append(rows, []string{n.Name, n.Language, n.Description, n.Source})
	}
	_, _ = fmt.Fprintln(stdout, ui.Table(
		fmt.Sprintf("Installable agent nodes (%d)", len(rows)),
		[]string{"NODE", "LANG", "DESCRIPTION", "SOURCE"},
		rows,
	))
	_, _ = fmt.Fprintln(stdout)
	_, _ = fmt.Fprintln(stdout, ui.Muted("Install one with:  af install <source>"))
}
