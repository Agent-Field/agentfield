import { createServer, type Server } from 'node:net'
import { describe, expect, it } from 'vitest'
import { baseUrlForPort, isPortFree, osAssignedPort, pickFreePort } from './ports'

/** Bind an ephemeral port and keep it held for the duration of fn. */
async function withHeldPort(fn: (port: number) => Promise<void>): Promise<void> {
  const server: Server = createServer()
  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject)
    server.listen({ port: 0 }, () => {
      const address = server.address()
      if (typeof address === 'object' && address !== null) resolve(address.port)
      else reject(new Error('no address'))
    })
  })
  try {
    await fn(port)
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()))
  }
}

describe('baseUrlForPort', () => {
  it('builds the localhost URL', () => {
    expect(baseUrlForPort(8080)).toBe('http://localhost:8080')
    expect(baseUrlForPort(9091)).toBe('http://localhost:9091')
  })
})

describe('isPortFree', () => {
  it('is false while something holds the port, true after it lets go', async () => {
    let held = -1
    await withHeldPort(async (port) => {
      held = port
      expect(await isPortFree(port)).toBe(false)
    })
    expect(await isPortFree(held)).toBe(true)
  })
})

describe('osAssignedPort', () => {
  it('hands out a bindable port', async () => {
    const port = await osAssignedPort()
    expect(port).not.toBeNull()
    expect(port).toBeGreaterThan(0)
  })
})

describe('pickFreePort', () => {
  it('returns the preferred port when it is free', async () => {
    const probe = async () => true
    expect(await pickFreePort(8080, probe)).toBe(8080)
  })

  it('walks forward to the next free port', async () => {
    const probe = async (port: number) => port >= 8082
    expect(await pickFreePort(8080, probe)).toBe(8082)
  })

  it('falls back to an OS-assigned port when the whole range is taken', async () => {
    const probe = async () => false
    const osAssign = async () => 51234
    expect(await pickFreePort(8080, probe, osAssign)).toBe(51234)
  })

  it('returns the preferred port when nothing at all is bindable', async () => {
    const probe = async () => false
    const osAssign = async () => null
    expect(await pickFreePort(8080, probe, osAssign)).toBe(8080)
  })

  it('never scans past 65535', async () => {
    const probed: number[] = []
    const probe = async (port: number) => {
      probed.push(port)
      return false
    }
    const osAssign = async () => 51234
    expect(await pickFreePort(65530, probe, osAssign)).toBe(51234)
    expect(probed[probed.length - 1]).toBe(65535)
  })
})
