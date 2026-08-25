import { createHash, randomBytes, randomInt } from 'node:crypto'
import { existsSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { createServer, type Server } from 'node:http'
import { join } from 'node:path'
import { createRailwayApi, type RailwayWorkspace } from './railwayApi'

export type { RailwayWorkspace } from './railwayApi'

export interface RailwayTokens {
  accessToken: string
  refreshToken: string | null
  expiresAt: number
}

export interface TokenCodec {
  encrypt(plain: string): string
  decrypt(blob: string): string
}

export interface RailwayAuthDeps {
  fetchImpl?: typeof fetch
  openUrl?: (url: string) => Promise<void>
  now?: () => number
  codec?: TokenCodec
  storePath?: string
}

const OAUTH_BASE = 'https://backboard.railway.com/oauth'
// AgentField should register its own Railway OAuth client before GA.
const DEFAULT_CLIENT_ID = 'rlwy_oaci_onEklvmksh1hRUiCo7E2zX12'
const SCOPES = 'openid email profile offline_access workspace:admin project:admin'
const CALLBACK_TIMEOUT_MS = 5 * 60 * 1000
const VERIFIER_CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
const identityCodec: TokenCodec = { encrypt: (plain) => plain, decrypt: (blob) => blob }

interface StoredTokens {
  v: 1
  blob: string
}

interface TokenResponse {
  access_token: string
  refresh_token?: string
  expires_in: number
}

interface OAuthErrorBody {
  error?: string
  error_description?: string
}

function clientId(): string {
  return process.env.RAILWAY_OAUTH_CLIENT_ID || DEFAULT_CLIENT_ID
}

function resolvedDeps(deps: RailwayAuthDeps = {}): Required<Pick<RailwayAuthDeps, 'fetchImpl' | 'now' | 'codec' | 'storePath'>> {
  return {
    fetchImpl: deps.fetchImpl ?? fetch,
    now: deps.now ?? Date.now,
    codec: deps.codec ?? identityCodec,
    storePath: deps.storePath ?? join(process.cwd(), 'railway-auth.json'),
  }
}

function base64url(input: Buffer): string {
  return input.toString('base64url')
}

function createVerifier(): string {
  let verifier = ''
  for (let index = 0; index < 128; index += 1) {
    verifier += VERIFIER_CHARACTERS[randomInt(VERIFIER_CHARACTERS.length)]
  }
  return verifier
}

function createChallenge(verifier: string): string {
  return base64url(createHash('sha256').update(verifier, 'ascii').digest())
}

function readTokens(deps: RailwayAuthDeps = {}): RailwayTokens | null {
  const { codec, storePath } = resolvedDeps(deps)
  try {
    const stored = JSON.parse(readFileSync(storePath, 'utf8')) as Partial<StoredTokens>
    if (stored.v !== 1 || typeof stored.blob !== 'string') return null
    const tokens = JSON.parse(codec.decrypt(stored.blob)) as Partial<RailwayTokens>
    if (
      typeof tokens.accessToken !== 'string' ||
      (typeof tokens.refreshToken !== 'string' && tokens.refreshToken !== null) ||
      typeof tokens.expiresAt !== 'number'
    ) return null
    return tokens as RailwayTokens
  } catch {
    return null
  }
}

function saveTokens(tokens: RailwayTokens, deps: RailwayAuthDeps = {}): void {
  const { codec, storePath } = resolvedDeps(deps)
  const stored: StoredTokens = { v: 1, blob: codec.encrypt(JSON.stringify(tokens)) }
  writeFileSync(storePath, JSON.stringify(stored), { encoding: 'utf8', mode: 0o600 })
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]!)
}

function callbackHtml(success: boolean, detail?: string): string {
  const title = success ? "You're signed in" : 'Sign-in failed'
  const message = success
    ? 'Return to AgentField Desktop.'
    : (detail || 'Return to AgentField Desktop and try again.')
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>${title}</title><style>color-scheme:light dark;body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f7f7f5;color:#171714;font:16px system-ui,sans-serif}.card{max-width:34rem;margin:2rem;padding:2rem;border:1px solid #d6b24c;border-radius:14px;background:#fff;box-shadow:0 12px 35px #0002}h1{margin:0 0 .6rem;color:#a06b00;font-size:1.6rem}p{margin:0;line-height:1.5}@media(prefers-color-scheme:dark){body{background:#111214;color:#f4f1e8}.card{background:#1b1c1f;border-color:#8d6b1f}h1{color:#e2b84e}}</style></head><body><main class="card"><h1>${title}</h1><p>${escapeHtml(message)}</p></main></body></html>`
}

function sendHtml(response: import('node:http').ServerResponse, status: number, html: string): void {
  response.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' })
  response.end(html)
}

function closeServer(server: Server): void {
  server.close()
}

async function listenForCode(state: string): Promise<{ code: Promise<string>; redirectUri: string; cancel: () => void }> {
  let resolveCode!: (code: string) => void
  let rejectCode!: (error: Error) => void
  let settled = false
  let timer: NodeJS.Timeout

  const code = new Promise<string>((resolve, reject) => {
    resolveCode = resolve
    rejectCode = reject
  })

  const server = createServer((request, response) => {
    if (!request.url?.startsWith('/callback')) {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
      response.end('Not found')
      return
    }

    const url = new URL(request.url, 'http://127.0.0.1')
    const receivedState = url.searchParams.get('state')
    if (receivedState !== state) {
      const message = 'CSRF validation failed: OAuth state mismatch.'
      sendHtml(response, 400, callbackHtml(false, message))
      if (!settled) {
        settled = true
        clearTimeout(timer)
        rejectCode(new Error(message))
        closeServer(server)
      }
      return
    }

    const oauthError = url.searchParams.get('error')
    if (oauthError) {
      const description = url.searchParams.get('error_description')
      const message = description ? `Railway authorization failed: ${description}` : `Railway authorization failed: ${oauthError}`
      sendHtml(response, 400, callbackHtml(false, message))
      if (!settled) {
        settled = true
        clearTimeout(timer)
        rejectCode(new Error(message))
        closeServer(server)
      }
      return
    }

    const authorizationCode = url.searchParams.get('code')
    if (!authorizationCode) {
      const message = 'Railway authorization failed: callback did not include a code.'
      sendHtml(response, 400, callbackHtml(false, message))
      if (!settled) {
        settled = true
        clearTimeout(timer)
        rejectCode(new Error(message))
        closeServer(server)
      }
      return
    }

    sendHtml(response, 200, callbackHtml(true))
    if (!settled) {
      settled = true
      clearTimeout(timer)
      resolveCode(authorizationCode)
      closeServer(server)
    }
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve())
  })
  const address = server.address()
  if (!address || typeof address === 'string') {
    closeServer(server)
    throw new Error('Could not determine OAuth callback port.')
  }

  timer = setTimeout(() => {
    if (!settled) {
      settled = true
      rejectCode(new Error('Railway sign-in timed out after 5 minutes.'))
      closeServer(server)
    }
  }, CALLBACK_TIMEOUT_MS)
  timer.unref()

  return {
    code,
    redirectUri: `http://127.0.0.1:${address.port}/callback`,
    cancel: () => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        rejectCode(new Error('Railway sign-in was cancelled.'))
        closeServer(server)
      }
    },
  }
}

async function parseTokenResponse(response: Response): Promise<TokenResponse> {
  const body = await response.json().catch(() => ({})) as Partial<TokenResponse> & OAuthErrorBody
  if (!response.ok) {
    const detail = body.error_description || body.error || `${response.status} ${response.statusText}`.trim()
    throw new Error(`Railway token request failed: ${detail}`)
  }
  if (typeof body.access_token !== 'string' || typeof body.expires_in !== 'number') {
    throw new Error('Railway token response was incomplete.')
  }
  return body as TokenResponse
}

async function requestTokens(parameters: URLSearchParams, fetchImpl: typeof fetch): Promise<TokenResponse> {
  const response = await fetchImpl(`${OAUTH_BASE}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: parameters,
  })
  return parseTokenResponse(response)
}

export async function loginWithRailway(deps: RailwayAuthDeps = {}): Promise<{ ok: boolean; message: string }> {
  let cancelCallback: (() => void) | undefined
  try {
    if (!deps.openUrl) throw new Error('No browser opener was configured for Railway sign-in.')
    const verifier = createVerifier()
    const state = base64url(randomBytes(32))
    const callback = await listenForCode(state)
    cancelCallback = callback.cancel
    // Browser openers may wait for the page load; attach a handler immediately so
    // a rejected callback cannot briefly become an unhandled promise rejection.
    void callback.code.catch(() => undefined)
    const authorizeUrl = new URL(`${OAUTH_BASE}/auth`)
    authorizeUrl.search = new URLSearchParams({
      response_type: 'code',
      client_id: clientId(),
      redirect_uri: callback.redirectUri,
      scope: SCOPES,
      code_challenge: createChallenge(verifier),
      code_challenge_method: 'S256',
      state,
      prompt: 'consent',
    }).toString()

    await deps.openUrl(authorizeUrl.toString())
    const code = await callback.code
    const { fetchImpl, now } = resolvedDeps(deps)
    const tokenResponse = await requestTokens(new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: callback.redirectUri,
      client_id: clientId(),
      code_verifier: verifier,
    }), fetchImpl)
    saveTokens({
      accessToken: tokenResponse.access_token,
      refreshToken: tokenResponse.refresh_token ?? null,
      expiresAt: now() + tokenResponse.expires_in * 1000,
    }, deps)
    return { ok: true, message: 'Signed in to Railway.' }
  } catch (error) {
    cancelCallback?.()
    return { ok: false, message: errorMessage(error) }
  }
}

export async function getFreshAccessToken(deps: RailwayAuthDeps = {}): Promise<string | null> {
  const tokens = readTokens(deps)
  if (!tokens) return null
  const { fetchImpl, now } = resolvedDeps(deps)
  if (tokens.expiresAt - now() >= 60_000) return tokens.accessToken
  if (!tokens.refreshToken) return null

  try {
    const refreshed = await requestTokens(new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: tokens.refreshToken,
      client_id: clientId(),
    }), fetchImpl)
    const rotated: RailwayTokens = {
      accessToken: refreshed.access_token,
      refreshToken: refreshed.refresh_token ?? tokens.refreshToken,
      expiresAt: now() + refreshed.expires_in * 1000,
    }
    saveTokens(rotated, deps)
    return rotated.accessToken
  } catch (error) {
    if (errorMessage(error).includes('invalid_grant')) logout(deps)
    return null
  }
}

export function isLoggedIn(deps: RailwayAuthDeps = {}): boolean {
  return readTokens(deps) !== null
}

export function logout(deps: RailwayAuthDeps = {}): void {
  const { storePath } = resolvedDeps(deps)
  try {
    if (existsSync(storePath)) unlinkSync(storePath)
  } catch {
    // Logout is intentionally idempotent; an inaccessible store is treated as logged out.
  }
}

export async function listWorkspaces(accessToken: string, deps: { fetchImpl?: typeof fetch } = {}): Promise<RailwayWorkspace[]> {
  try {
    return await createRailwayApi(accessToken, deps.fetchImpl ?? fetch).listWorkspaces()
  } catch (error) {
    throw new Error(`Failed to list Railway workspaces: ${errorMessage(error)}`)
  }
}

export const railwayAuthTestUtils = {
  createVerifier,
  createChallenge,
}
