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
// subdirectory addressed with the `//<subdir>` source selector (how the Go
// ports living beside their Python originals are installed). When adding an
// entry, `name` MUST equal the manifest's `name:` (the registry key after
// install — how the app detects installed state), which is often NOT the
// repo name (SWE-AF → swe-planner).
export const CATALOG: CatalogEntry[] = [
  {
    name: 'swe-planner',
    description:
      'Autonomous software-engineering fleet: plan, code, test, and ship production-grade PRs',
    source: 'https://github.com/Agent-Field/SWE-AF',
    language: 'python'
  },
  {
    name: 'swe-planner-go',
    description:
      'Go port of the SWE fleet: same planning/execution reasoners, one static binary',
    source: 'https://github.com/Agent-Field/SWE-AF//go',
    language: 'go'
  },
  {
    name: 'pr-af',
    description: 'Turns a plain task description into a draft pull request on GitHub',
    source: 'https://github.com/Agent-Field/pr-af',
    language: 'python'
  },
  {
    name: 'pr-af-go',
    description: 'Go port of the PR review agent: same reasoners, one static binary',
    source: 'https://github.com/Agent-Field/pr-af//go',
    language: 'go'
  },
  {
    name: 'sec-af',
    description:
      'Code security auditor: scans repositories and proves exploitability with verdicts and traces',
    source: 'https://github.com/Agent-Field/sec-af',
    language: 'python'
  },
  {
    name: 'cloudsecurity-af',
    description:
      'Cloud security posture: read-only attack-path scans across AWS, GCP, and Azure',
    source: 'https://github.com/Agent-Field/cloudsecurity-af',
    language: 'python'
  }
]

/** Look up a catalog entry by name. Returns undefined for unknown names. */
export function catalogEntry(name: string): CatalogEntry | undefined {
  return CATALOG.find((entry) => entry.name === name)
}
