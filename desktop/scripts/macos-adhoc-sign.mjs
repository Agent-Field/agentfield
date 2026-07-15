// electron-builder afterPack hook: give unsigned macOS builds a valid ad-hoc
// signature.
//
// When no signing identity is configured, electron-builder skips signing
// entirely and the app ships with only the linker's ad-hoc signature on the
// main binary — no sealed resources. codesign/spctl treat that as an INVALID
// signature ("code has no resources but signature indicates they must be
// present"), which trips Gatekeeper assessments and tooling that validates the
// bundle. A full `codesign --deep --sign -` produces a valid (if still
// unidentified) signature, so a locally built or CI-built app behaves like a
// normal unsigned-developer app instead of a corrupted one.
//
// Identity-signed builds are left alone: their signature already verifies, so
// the hook is a no-op the day real signing/notarization lands.

import { spawnSync } from 'node:child_process'
import { join } from 'node:path'

function codesign(args) {
  return spawnSync('codesign', args, { encoding: 'utf8' })
}

export default async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return

  const appPath = join(
    context.appOutDir,
    `${context.packager.appInfo.productFilename}.app`
  )

  // Valid already (real identity, or a previous ad-hoc pass) → nothing to do.
  if (codesign(['--verify', '--deep', '--strict', appPath]).status === 0) return

  console.log(`  • ad-hoc signing (no identity configured)  file=${appPath}`)
  const sign = codesign(['--force', '--deep', '--sign', '-', appPath])
  if (sign.status !== 0) {
    throw new Error(`ad-hoc codesign failed: ${sign.stderr}`)
  }
  const verify = codesign(['--verify', '--deep', '--strict', appPath])
  if (verify.status !== 0) {
    throw new Error(`ad-hoc signature did not verify: ${verify.stderr}`)
  }
}
