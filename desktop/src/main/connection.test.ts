import { afterEach, describe, expect, it } from 'vitest'
import {
  clearCloudConnection,
  getApiKey,
  getBaseUrl,
  isCloudActive,
  setCloudConnection,
  setLocalPort
} from './connection'

afterEach(() => setLocalPort(8080))

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

  it('setting a local port clears cloud state', () => {
    setCloudConnection('https://cp.example', 'secret')
    setLocalPort(8082)
    expect(getBaseUrl()).toBe('http://localhost:8082')
    expect(getApiKey()).toBeNull()
    expect(isCloudActive()).toBe(false)
  })
})
