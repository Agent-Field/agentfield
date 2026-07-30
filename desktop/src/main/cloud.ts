import type { CloudTestResult, DesktopSettings } from '../shared/types'
import { clearCloudConnection, setCloudConnection } from './connection'

export type { CloudTestResult } from '../shared/types'

function isPrivateHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host === '::1') return true
  const parts = host.split('.').map(Number)
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false
  }
  return (
    parts[0] === 10 ||
    parts[0] === 127 ||
    (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
    (parts[0] === 192 && parts[1] === 168)
  )
}

export function normalizeServerUrl(input: string): string | null {
  const trimmed = input.trim()
  if (!trimmed) return null
  const candidate = /^[a-z][a-z\d+.-]*:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
  try {
    const parsed = new URL(candidate)
    if (!parsed.hostname || parsed.username || parsed.password) return null
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null
    if (parsed.protocol === 'http:' && !isPrivateHost(parsed.hostname)) return null
    return parsed.origin
  } catch {
    return null
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function versionFrom(value: unknown): string | undefined {
  const body = record(value)
  if (!body) return undefined
  for (const key of ['version', 'server_version', 'build_version']) {
    if (typeof body[key] === 'string' && body[key] !== '') return body[key] as string
  }
  const build = record(body.build)
  return build && typeof build.version === 'string' && build.version !== ''
    ? build.version
    : undefined
}

async function jsonBestEffort(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return undefined
  }
}

export async function testCloudConnection(
  url: string,
  apiKey: string,
  deps: { fetchImpl?: typeof fetch } = {}
): Promise<CloudTestResult> {
  const normalized = normalizeServerUrl(url)
  if (!normalized) {
    return {
      ok: false,
      healthy: false,
      authOk: false,
      installApi: false,
      message: 'Enter a valid server URL'
    }
  }
  const fetchImpl = deps.fetchImpl ?? globalThis.fetch
  const headers = { 'X-API-Key': apiKey }

  let health: Response
  try {
    health = await fetchImpl(`${normalized}/health`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000)
    })
  } catch {
    return {
      ok: false,
      healthy: false,
      authOk: false,
      installApi: false,
      message: `Could not reach ${normalized}`
    }
  }

  const healthBody = await jsonBestEffort(health)
  const healthy = health.ok
  let version = versionFrom(healthBody)
  if (!healthy) {
    return {
      ok: false,
      healthy: false,
      authOk: false,
      installApi: false,
      ...(version ? { version } : {}),
      message: `Control plane at ${normalized} is not healthy`
    }
  }

  let authResponse: Response
  try {
    authResponse = await fetchImpl(`${normalized}/api/ui/v1/agents/packages`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000)
    })
  } catch {
    return {
      ok: false,
      healthy: true,
      authOk: false,
      installApi: false,
      ...(version ? { version } : {}),
      message: 'Could not verify API access'
    }
  }
  const authOk = authResponse.ok
  if (!authOk) {
    return {
      ok: false,
      healthy: true,
      authOk: false,
      installApi: false,
      ...(version ? { version } : {}),
      message:
        authResponse.status === 401 || authResponse.status === 403
          ? 'API key rejected'
          : `API request failed with status ${authResponse.status}`
    }
  }

  let installApi = false
  try {
    const installResponse = await fetchImpl(
      `${normalized}/api/ui/v1/agents/packages/install/jobs`,
      { method: 'GET', headers, signal: AbortSignal.timeout(5000) }
    )
    installApi = installResponse.status !== 404
  } catch {
    installApi = false
  }

  if (!version) {
    try {
      const versionResponse = await fetchImpl(`${normalized}/api/v1/version`, {
        method: 'GET',
        headers,
        signal: AbortSignal.timeout(5000)
      })
      if (versionResponse.ok) version = versionFrom(await jsonBestEffort(versionResponse))
    } catch {
      // Version discovery is best-effort.
    }
  }

  return {
    ok: true,
    healthy: true,
    authOk: true,
    installApi,
    ...(version ? { version } : {}),
    message: installApi ? 'Connection successful' : 'Connected; install API unavailable'
  }
}

export function applyConnectionProfile(settings: DesktopSettings): void {
  const normalized = normalizeServerUrl(settings.cloud?.serverUrl ?? '')
  if (settings.cloud?.enabled && normalized) {
    setCloudConnection(normalized, settings.cloud.apiKey || null)
  } else {
    clearCloudConnection()
  }
}
