import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { randomBytes } from 'node:crypto'
import { deflateRawSync } from 'node:zlib'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { extractZip } from './extract-zip.mjs'

// extractZip is what turns the downloaded OpenTofu / Railway provider archives
// into the binaries bundled in the installers. These tests build real .zip
// files byte by byte, so they need no zip tool on the machine and run the real
// extractor of each OS the Desktop CI covers (unzip on macOS, tar.exe on
// Windows).

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

function crc32(buffer) {
  let c = 0xffffffff
  for (const byte of buffer) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

// entries: [{ name, data: Buffer, deflate?: boolean }]
function buildZip(entries) {
  const locals = []
  const centrals = []
  let offset = 0
  for (const { name, data, deflate = false } of entries) {
    const nameBytes = Buffer.from(name)
    const body = deflate ? deflateRawSync(data) : data
    const method = deflate ? 8 : 0
    const crc = crc32(data)

    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4) // version needed
    local.writeUInt16LE(method, 8)
    local.writeUInt16LE(0x21, 12) // date: 1980-01-01
    local.writeUInt32LE(crc, 14)
    local.writeUInt32LE(body.length, 18)
    local.writeUInt32LE(data.length, 22)
    local.writeUInt16LE(nameBytes.length, 26)

    const central = Buffer.alloc(46)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(20, 4) // version made by
    central.writeUInt16LE(20, 6) // version needed
    central.writeUInt16LE(method, 10)
    central.writeUInt16LE(0x21, 14)
    central.writeUInt32LE(crc, 16)
    central.writeUInt32LE(body.length, 20)
    central.writeUInt32LE(data.length, 24)
    central.writeUInt16LE(nameBytes.length, 28)
    central.writeUInt32LE(offset, 42)

    locals.push(local, nameBytes, body)
    centrals.push(central, nameBytes)
    offset += local.length + nameBytes.length + body.length
  }
  const centralBytes = Buffer.concat(centrals)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(entries.length, 8)
  end.writeUInt16LE(entries.length, 10)
  end.writeUInt32LE(centralBytes.length, 12)
  end.writeUInt32LE(offset, 16)
  return Buffer.concat([...locals, centralBytes, end])
}

describe('extractZip', () => {
  let dir

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'agentfield-extract-zip-test-'))
  })

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true })
  })

  it('extracts stored and deflated entries into a directory that does not exist yet', async () => {
    // A compressible 1 MiB payload stands in for a Go binary.
    const binary = Buffer.alloc(1024 * 1024, 'agentfield')
    const license = Buffer.from('MPL-2.0\n')
    const zip = join(dir, 'provider.zip')
    await writeFile(zip, buildZip([
      { name: 'terraform-provider-railway_v0.6.2', data: binary, deflate: true },
      { name: 'LICENSE', data: license }
    ]))

    const out = join(dir, 'nested', 'out')
    await extractZip(zip, out)

    expect((await readdir(out)).sort()).toEqual(['LICENSE', 'terraform-provider-railway_v0.6.2'])
    expect((await readFile(join(out, 'terraform-provider-railway_v0.6.2'))).equals(binary)).toBe(true)
    expect((await readFile(join(out, 'LICENSE'))).equals(license)).toBe(true)
  })

  it('rejects when the archive is cut off mid-entry', async () => {
    const good = buildZip([{ name: 'tofu', data: randomBytes(64 * 1024) }])
    const zip = join(dir, 'truncated.zip')
    await writeFile(zip, good.subarray(0, 4096)) // a partial download

    await expect(extractZip(zip, join(dir, 'out'))).rejects.toThrow()
  })

  it('rejects when an entry fails its CRC check', async () => {
    const good = buildZip([{ name: 'tofu', data: randomBytes(64 * 1024) }])
    good[30 + 4 + 1000] ^= 0xff // flip a byte inside the stored data
    const zip = join(dir, 'corrupt.zip')
    await writeFile(zip, good)

    await expect(extractZip(zip, join(dir, 'out'))).rejects.toThrow()
  })

  it('rejects when the file is not a zip at all', async () => {
    const zip = join(dir, 'html.zip')
    await writeFile(zip, '<html>Not Found</html>')
    await expect(extractZip(zip, join(dir, 'out'))).rejects.toThrow()
  })

  it('rejects when the archive does not exist', async () => {
    await mkdir(join(dir, 'out'))
    await expect(extractZip(join(dir, 'missing.zip'), join(dir, 'out'))).rejects.toThrow()
  })
})
