// Control-plane port selection. The default port 8080 is popular, and users
// running several control planes (or anything else on 8080) need the app to
// get out of the way: prefer the default, then walk forward to the next free
// port, and only fall back to an OS-assigned ephemeral one when a whole run
// of ports is taken — sequential ports keep the URL familiar and stable
// across restarts, an ephemeral port would move on every boot.
//
// No electron imports so everything here is unit-testable under plain vitest.

import { createServer } from 'node:net'

export const DEFAULT_CONTROL_PLANE_PORT = 8080

/** How many ports after the preferred one to try before asking the OS. */
const SCAN_RANGE = 20

export function baseUrlForPort(port: number): string {
  return `http://localhost:${port}`
}

/**
 * True when this process could bind the port on all interfaces — the same
 * bind `af server` will attempt. This also rules out ports squatted by
 * non-HTTP services that a /health probe reports as merely "unreachable".
 */
export function isPortFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = createServer()
    server.unref()
    server.once('error', () => resolve(false))
    server.listen({ port }, () => {
      server.close(() => resolve(true))
    })
  })
}

/** Bind port 0, read what the OS handed out, release it. Null when even that fails. */
export function osAssignedPort(): Promise<number | null> {
  return new Promise((resolve) => {
    const server = createServer()
    server.unref()
    server.once('error', () => resolve(null))
    server.listen({ port: 0 }, () => {
      const address = server.address()
      const port = typeof address === 'object' && address !== null ? address.port : null
      server.close(() => resolve(port))
    })
  })
}

/**
 * The port to start a control plane on: the preferred port when bindable,
 * else the next SCAN_RANGE ports in order, else an OS-assigned ephemeral
 * port. When nothing at all is bindable (out of fds, sandboxed), returns the
 * preferred port — the server start then fails with its own clear message
 * instead of this module inventing one.
 */
export async function pickFreePort(
  preferred: number = DEFAULT_CONTROL_PLANE_PORT,
  probe: (port: number) => Promise<boolean> = isPortFree,
  osAssign: () => Promise<number | null> = osAssignedPort
): Promise<number> {
  for (let offset = 0; offset <= SCAN_RANGE; offset++) {
    const port = preferred + offset
    if (port > 65535) break
    if (await probe(port)) return port
  }
  return (await osAssign()) ?? preferred
}
