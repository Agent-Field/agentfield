const DEFAULT_LOCAL_URL = 'http://localhost:8080'

let localUrl = DEFAULT_LOCAL_URL
let baseUrl = localUrl
let cloudApiKey: string | null = null
// Optional credential for the LOCAL control plane. Normally null: a control
// plane started without AGENTFIELD_API_KEY authenticates loopback callers by
// their peer address alone. Set only when the user runs their local server
// with authentication enabled (see settings.localApiKey).
let localApiKey: string | null = null
let cloudActive = false

export function getBaseUrl(): string {
  return baseUrl
}

export function setLocalPort(port: number): void {
  localUrl = `http://localhost:${port}`
  baseUrl = localUrl
  cloudApiKey = null
  cloudActive = false
}

/** Credential for the local control plane; '' / null means unauthenticated. */
export function setLocalApiKey(key: string | null): void {
  localApiKey = key !== null && key !== '' ? key : null
}

export function setCloudConnection(url: string, key: string | null): void {
  baseUrl = url
  cloudApiKey = key
  cloudActive = true
}

export function clearCloudConnection(): void {
  baseUrl = localUrl
  cloudApiKey = null
  cloudActive = false
}

export function getApiKey(): string | null {
  return cloudActive ? cloudApiKey : localApiKey
}

export function isCloudActive(): boolean {
  return cloudActive
}
