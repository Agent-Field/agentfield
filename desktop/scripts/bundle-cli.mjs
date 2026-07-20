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
//
// On macOS this ALSO builds ./cmd/af-tray into vendor/af-tray, stamped with
// the SAME version so the desktop app can provision + install the menu-bar
// companion itself (src/main/tray-companion.ts) — a desktop-app-only install
// gets the tray without ever running the curl installer, and both installers
// converge on ~/.agentfield/bin/af-tray. af-tray carries the systray/CGO
// dependency, so it is built with CGO enabled and only on/for darwin (the app
// has its own in-app tray on Windows/Linux).

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
if (result.status !== 0) process.exit(result.status ?? 1)

// macOS only: build the af-tray menu-bar companion alongside af, stamped
// identically so tray-companion.ts can version-compare bundled vs. managed.
// af-tray needs the systray/CGO dependency (clang ships on macOS runners and
// dev machines); never built for win32/linux — the app has an in-app tray there.
if (process.platform === 'darwin') {
  const trayOut = join(vendorDir, 'af-tray')
  const trayArgs = [
    'build',
    '-ldflags',
    `-s -w -X main.version=${version} -X main.commit=${commit} -X main.date=${date}`,
    '-o',
    trayOut,
    './cmd/af-tray'
  ]
  console.log(`CGO_ENABLED=1 go ${trayArgs.join(' ')}  (in ${controlPlaneDir})`)
  const trayResult = spawnSync('go', trayArgs, {
    cwd: controlPlaneDir,
    stdio: 'inherit',
    env: { ...process.env, CGO_ENABLED: '1' }
  })
  if (trayResult.error) {
    console.error('failed to build af-tray', trayResult.error.message)
    process.exit(1)
  }
  process.exit(trayResult.status ?? 1)
}

process.exit(0)
