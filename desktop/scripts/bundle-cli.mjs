// Build the af CLI from the sibling control-plane source and drop it into
// desktop/vendor/, where electron-builder's extraResources picks it up
// (resources/bin/ inside the packaged app). Run before `npm run dist`:
//
//   npm run bundle-cli          # plain build (no embedded web UI)
//   npm run bundle-cli -- full  # embedded web UI + sqlite FTS (needs CGO +
//                               # a prior `npm run build` in web/client)
//
// Release pipelines can skip this script and copy the goreleaser artifact
// for the target platform into vendor/ instead — anything named af/af.exe
// in vendor/ gets bundled.

import { spawnSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopDir = dirname(dirname(fileURLToPath(import.meta.url)))
const controlPlaneDir = join(desktopDir, '..', 'control-plane')
const vendorDir = join(desktopDir, 'vendor')
const output = join(vendorDir, process.platform === 'win32' ? 'af.exe' : 'af')

const full = process.argv.includes('full')
const args = ['build']
if (full) args.push('-tags', 'embedded sqlite_fts5')
args.push('-o', output, './cmd/af')

mkdirSync(vendorDir, { recursive: true })
console.log(`go ${args.join(' ')}  (in ${controlPlaneDir})`)
const result = spawnSync('go', args, { cwd: controlPlaneDir, stdio: 'inherit' })
if (result.error) {
  console.error('failed to run go — is Go installed and on PATH?', result.error.message)
  process.exit(1)
}
process.exit(result.status ?? 1)
