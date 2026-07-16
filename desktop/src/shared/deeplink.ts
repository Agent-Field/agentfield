// agentfield:// deep links.
//
// The desktop app registers itself as the `agentfield:` protocol handler
// (see main/index.ts and the electron-builder `protocols` config), so other
// surfaces — the macOS menu-bar tray (`af-tray`), docs, the web UI — can open
// the app on a specific view with e.g. `open agentfield://agents`.

export const DEEP_LINK_SCHEME = 'agentfield'

/** The app's views, also the vocabulary of deep-link targets. */
export const VIEWS = [
  'dashboard',
  'agents',
  'activity',
  'install',
  'secrets',
  'settings'
] as const
export type View = (typeof VIEWS)[number]

export function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value)
}

/**
 * Parse an agentfield:// URL into the view it addresses.
 *
 * Returns null for anything that is not an agentfield: URL at all. A bare
 * `agentfield://` and any unknown view fall back to 'dashboard', so links
 * minted by newer (or older) senders still open the app instead of dying.
 */
export function parseDeepLink(url: string): View | null {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return null
  }
  if (parsed.protocol !== `${DEEP_LINK_SCHEME}:`) return null
  // agentfield://agents parses the view as the host; agentfield:agents (no
  // slashes) parses it as an opaque pathname. Accept both spellings.
  const target = (parsed.host || parsed.pathname).replace(/^\/+/, '').split('/')[0].toLowerCase()
  return isView(target) ? target : 'dashboard'
}

/**
 * Find the deep link in a second-instance argv (how Windows/Linux hand the
 * URL to an already-running app). Non-URL args parse to null and are skipped.
 */
export function deepLinkFromArgv(argv: readonly string[]): View | null {
  for (const arg of argv) {
    const view = parseDeepLink(arg)
    if (view) return view
  }
  return null
}
