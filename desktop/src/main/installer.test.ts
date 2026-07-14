import { describe, expect, it } from 'vitest'
import { CATALOG } from '../shared/catalog'
import { installCommand } from './installer'

describe('installCommand', () => {
  it('builds a plain install for a catalog entry', () => {
    const cmd = installCommand(CATALOG[0].name)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', CATALOG[0].source])
  })

  it('appends --force for updates (reinstall in place, secrets survive)', () => {
    const cmd = installCommand(CATALOG[0].name, true)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', CATALOG[0].source, '--force'])
  })

  it('refuses names outside the curated catalog', () => {
    expect(installCommand('rm -rf /', true)).toBeNull()
    expect(installCommand('not-in-catalog')).toBeNull()
  })
})
