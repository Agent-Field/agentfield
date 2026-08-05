import type { CatalogEntry } from './types'

// Curated list of installable agent nodes, shown in the app's Install view.
//
// This is deliberately a hard-coded list maintained by hand: entries are
// vetted, and the app refuses to install any source that is not in it (the
// renderer only ever passes a catalog *name* over IPC, never a raw source).
// When the marketplace/registry search lands, this file is the seam to
// replace with a remote catalog fetch.
//
// What qualifies: an Agent-Field org repo is installable iff it has an
// `agentfield-package.yaml` manifest — at the repo root, or in a
// subdirectory addressed with the `//<subdir>` source selector.
//
// One row per product, sourced at the bare repo URL. A repo that ships more
// than one implementation of the same node says which one it wants installed
// with `superseded_by:` in its root manifest — the redirect that makes
// `af install <repo>` land on the maintained node (SWE-AF and pr-af both
// point their root at `//go`). Naming `//go` here would install that same
// node, but it would skip the redirect, and the redirect is what carries a
// user who already has the superseded node across: it installs the successor
// first, migrates node-scoped secrets, and only then retires the old package.
// So the catalog names the repo and lets the manifest decide.
//
// `name` MUST equal the name the package ends up REGISTERED under once the
// install settles — that is how the app detects installed state. Note that is
// the name after any `superseded_by:` redirect resolves, which need not be the
// `name:` in the manifest at the source: a successor may deliberately take its
// predecessor's name (an in-place rename), and it may live in a subdirectory
// this list never names. It is often not the repo name either
// (SWE-AF → swe-planner).
export const CATALOG: CatalogEntry[] = [
  {
    name: 'swe-planner',
    description:
      'Software factory — turn any issue into a production-ready pull request, end to end',
    source: 'https://github.com/Agent-Field/SWE-AF',
    language: 'go'
  },
  {
    name: 'pr-af',
    description: 'Code review — deep, evidence-backed review of any GitHub pull request',
    source: 'https://github.com/Agent-Field/pr-af',
    language: 'go'
  },
  {
    name: 'sec-af',
    description:
      'Security auditor — find vulnerabilities in your codebase and prove which ones are exploitable',
    source: 'https://github.com/Agent-Field/sec-af',
    language: 'python'
  },
  {
    name: 'cloudsecurity-af',
    description:
      'Cloud security — map real attack paths across your AWS, GCP, and Azure accounts',
    source: 'https://github.com/Agent-Field/cloudsecurity-af',
    language: 'python'
  }
]

/** Look up a catalog entry by name. Returns undefined for unknown names. */
export function catalogEntry(name: string): CatalogEntry | undefined {
  return CATALOG.find((entry) => entry.name === name)
}
