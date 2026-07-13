import type { CatalogEntry } from './types'

// Curated list of installable agent nodes, shown in the app's Install view.
//
// This is deliberately a hard-coded list maintained by hand: entries are
// vetted, and the app refuses to install any source that is not in it (the
// renderer only ever passes a catalog *name* over IPC, never a raw source).
// When the marketplace/registry search lands, this file is the seam to
// replace with a remote catalog fetch.
export const CATALOG: CatalogEntry[] = [
  {
    // `name` MUST equal the node's manifest name (agentfield-package.yaml
    // `name:`), which becomes the registry key after install — it is how the
    // app detects that an entry is already installed. It is NOT the repo name.
    name: 'swe-planner',
    description: 'Autonomous software-engineering fleet: plan, code, test, and ship production-grade PRs',
    source: 'https://github.com/Agent-Field/SWE-AF',
    language: 'python'
  }
]

/** Look up a catalog entry by name. Returns undefined for unknown names. */
export function catalogEntry(name: string): CatalogEntry | undefined {
  return CATALOG.find((entry) => entry.name === name)
}
