import { afterEach, describe, expect, it } from 'vitest'
import {
  clearCloudConnection,
  getApiKey,
  getBaseUrl,
  isCloudActive,
  setCloudConnection,
  setLocalApiKey,
  setLocalPort
} from './connection'

afterEach(() => {
  setLocalApiKey(null)
  setLocalPort(8080)
})

describe('connection state', () => {
  it('switches to cloud and restores the last local port', () => {
    setLocalPort(9091)
    setCloudConnection('https://cp.example', 'secret')
    expect(getBaseUrl()).toBe('https://cp.example')
    expect(getApiKey()).toBe('secret')
    expect(isCloudActive()).toBe(true)
    clearCloudConnection()
    expect(getBaseUrl()).toBe('http://localhost:9091')
    expect(getApiKey()).toBeNull()
    expect(isCloudActive()).toBe(false)
  })

  it('sends no key locally until one is configured, and keeps it across ports', () => {
    expect(getApiKey()).toBeNull()
    setLocalApiKey('local-secret')
    expect(getApiKey()).toBe('local-secret')
    setLocalPort(9091)
    expect(getApiKey()).toBe('local-secret')
    setLocalApiKey('')
    expect(getApiKey()).toBeNull()
  })

  it('prefers the cloud key while cloud is active and restores the local one after', () => {
    setLocalApiKey('local-secret')
    setCloudConnection('https://cp.example', 'cloud-key')
    expect(getApiKey()).toBe('cloud-key')
    clearCloudConnection()
    expect(getApiKey()).toBe('local-secret')
  })

  it('setting a local port clears cloud state', () => {
    setCloudConnection('https://cp.example', 'secret')
    setLocalPort(8082)
    expect(getBaseUrl()).toBe('http://localhost:8082')
    expect(getApiKey()).toBeNull()
    expect(isCloudActive()).toBe(false)
  })
})
