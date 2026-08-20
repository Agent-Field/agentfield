import { BUNDLED_NODES } from './bundled'
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
// `af install <repo>` land on the maintained node. Naming the subdirectory
// here would install that same node, but it would skip the redirect, and the
// redirect is what carries a user who already has the superseded node across:
// it installs the successor first, migrates node-scoped secrets, and only then
// retires the old package. So the catalog names the repo and lets the manifest
// decide. shared/bundled.ts follows the identical rule for the nodes that ship
// with the app (all four point their root at `//go`), which is why they are no
// longer rows here: they are provisioned on first launch instead of being
// offered as marketplace cards.
//
// `name` MUST equal the name the package ends up REGISTERED under once the
// install settles — that is how the app detects installed state. Note that is
// the name after any `superseded_by:` redirect resolves, which need not be the
// `name:` in the manifest at the source: a successor may deliberately take its
// predecessor's name (an in-place rename), and it may live in a subdirectory
// this list never names. It is often not the repo name either
// (SWE-AF → swe-planner).
//
// Currently empty: every vetted node ships with the app (shared/bundled.ts).
// The list stays because it is the seam the next marketplace-only node — and
// eventually the remote catalog fetch — lands in.
export const CATALOG: CatalogEntry[] = []

/**
 * Look up an installable entry by name, across the marketplace catalog AND the
 * nodes bundled with the app. Returns undefined for unknown names.
 *
 * Both lists are hard-coded, so widening the lookup does not widen the trust
 * boundary: main/installer.ts still only ever turns a vetted name into a
 * vetted source. Including BUNDLED_NODES is load-bearing — it is what keeps a
 * bundled node installable and `--force` updatable from the Agents view, even
 * though it never appears as a marketplace card.
 */
export function catalogEntry(name: string): CatalogEntry | undefined {
  return [...CATALOG, ...BUNDLED_NODES].find((entry) => entry.name === name)
}
