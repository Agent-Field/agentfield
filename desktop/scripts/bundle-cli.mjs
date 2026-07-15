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
//
// The binary is version-stamped exactly like goreleaser's
// (-X main.version/commit/date): AF_CLI_VERSION wins (the release workflow
// passes the tag), then `git describe --tags`, then "dev". An unstamped
// bundle would answer `Version: dev`, and the app's CLI resolution can
// neither gate it against MIN_AF_VERSION nor ever offer an update over it.

import { spawnSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopDir = dirname(dirname(fileURLToPath(import.meta.url)))
const controlPlaneDir = join(desktopDir, '..', 'control-plane')
const vendorDir = join(desktopDir, 'vendor')
const output = join(vendorDir, process.platform === 'win32' ? 'af.exe' : 'af')

function git(...gitArgs) {
  const res = spawnSync('git', gitArgs, { cwd: controlPlaneDir, encoding: 'utf8' })
  return res.status === 0 ? res.stdout.trim() : ''
}

// goreleaser strips the leading v from tags; keep parity so version
// comparisons in the app see the same shape from both install paths.
const version = (process.env.AF_CLI_VERSION || git('describe', '--tags', '--always') || 'dev').replace(/^v/, '')
const commit = git('rev-parse', '--short', 'HEAD') || 'none'
const date = new Date().toISOString()

const full = process.argv.includes('full')
const args = ['build']
if (full) args.push('-tags', 'embedded sqlite_fts5')
args.push('-ldflags', `-s -w -X main.version=${version} -X main.commit=${commit} -X main.date=${date}`)
args.push('-o', output, './cmd/af')

mkdirSync(vendorDir, { recursive: true })
console.log(`go ${args.join(' ')}  (in ${controlPlaneDir})`)
const result = spawnSync('go', args, { cwd: controlPlaneDir, stdio: 'inherit' })
if (result.error) {
  console.error('failed to run go — is Go installed and on PATH?', result.error.message)
  process.exit(1)
}
process.exit(result.status ?? 1)
