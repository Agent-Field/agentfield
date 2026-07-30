const DEFAULT_LOCAL_URL = 'http://localhost:8080'

let localUrl = DEFAULT_LOCAL_URL
let baseUrl = localUrl
let apiKey: string | null = null
let cloudActive = false

export function getBaseUrl(): string {
  return baseUrl
}

export function setLocalPort(port: number): void {
  localUrl = `http://localhost:${port}`
  baseUrl = localUrl
  apiKey = null
  cloudActive = false
}

export function setCloudConnection(url: string, key: string | null): void {
  baseUrl = url
  apiKey = key
  cloudActive = true
}

export function clearCloudConnection(): void {
  baseUrl = localUrl
  apiKey = null
  cloudActive = false
}

export function getApiKey(): string | null {
  return apiKey
}

export function isCloudActive(): boolean {
  return cloudActive
}
