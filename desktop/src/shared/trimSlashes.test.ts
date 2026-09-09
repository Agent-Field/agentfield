import { describe, it, expect } from 'vitest'
import { stripTrailingSlashes } from './trimSlashes'

describe('stripTrailingSlashes', () => {
  it('strips a single trailing slash', () => {
    expect(stripTrailingSlashes('http://localhost:8080/')).toBe('http://localhost:8080')
  })

  it('strips repeated trailing slashes', () => {
    expect(stripTrailingSlashes('http://localhost:8080///')).toBe('http://localhost:8080')
  })

  it('leaves interior slashes and slash-free input alone', () => {
    expect(stripTrailingSlashes('owner/repo//go')).toBe('owner/repo//go')
    expect(stripTrailingSlashes('owner/repo')).toBe('owner/repo')
  })

  it('handles the empty string and an all-slash string', () => {
    expect(stripTrailingSlashes('')).toBe('')
    expect(stripTrailingSlashes('////')).toBe('')
  })

  it('stays linear on a long slash run', () => {
    const long = `${'/'.repeat(200000)}x`
    const start = Date.now()
    expect(stripTrailingSlashes(long)).toBe(long)
    expect(Date.now() - start).toBeLessThan(1000)
  })
})
