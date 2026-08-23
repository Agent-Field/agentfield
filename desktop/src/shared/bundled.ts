import type { CatalogEntry } from './types'

// Agent nodes that ship WITH the app rather than being offered as marketplace
// rows. They are provisioned on first launch (see main/bundledAgents.ts) and
// then live in the Agents library like any other installed node.
//
// "Ships with the app" here means fetched on first launch, not baked into the
// installer: the app installs them through the same control-plane install API
// a user-initiated install uses. Nothing about the packaging changes — only
// who decides to press install, and when.
//
// The entries carry the same shape as CATALOG rows on purpose. catalogEntry()
// resolves over both lists, so update / --force reinstall from the Agents view
// keeps working for a bundled node, and the "the renderer only ever passes a
// vetted NAME over IPC" invariant is preserved because both lists are
// hard-coded here in main-process-trusted source.
//
// Sourcing follows the same rule catalog.ts documents at length: name the BARE
// repo URL, never the `//go` subdirectory. All four repos' root manifests carry
// `superseded_by: …//go`, and that redirect is what carries a user who already
// has the older Python node across — it installs the successor, migrates
// node-scoped secrets, and only then retires the predecessor. Naming `//go`
// would land the same node but skip that migration.
//
// As in the catalog, `name` MUST equal the name the package is REGISTERED
// under after the redirect settles (SWE-AF → swe-planner), because that name
// is how the app detects the node is already installed and stops re-provisioning it.
export const BUNDLED_NODES: readonly CatalogEntry[] = [
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
    language: 'go'
  },
  {
    name: 'cloudsecurity-af',
    description:
      'Cloud security — map real attack paths across your AWS, GCP, and Azure accounts',
    source: 'https://github.com/Agent-Field/cloudsecurity-af',
    language: 'go'
  }
]

/** True when this node name ships with the app (never a marketplace row). */
export function isBundled(name: string): boolean {
  return BUNDLED_NODES.some((entry) => entry.name === name)
}

/** Look up a bundled entry by name. Returns undefined for unknown names. */
export function bundledEntry(name: string): CatalogEntry | undefined {
  return BUNDLED_NODES.find((entry) => entry.name === name)
}
