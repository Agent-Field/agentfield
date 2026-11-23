import { spawn } from 'node:child_process'
import { mkdir } from 'node:fs/promises'

function command(program, args) {
  return new Promise((resolveCommand, reject) => {
    const child = spawn(program, args, { stdio: 'inherit' })
    child.on('error', reject)
    child.on('close', (code) => code === 0 ? resolveCommand() : reject(new Error(`${program} exited with ${code}`)))
  })
}

// Extracts a .zip into `directory` (created if missing). Rejects when the
// archive is missing or corrupt, or the extractor is not available; it never
// resolves after a failed extraction.
//
// Windows uses the bsdtar that ships with Windows 10+ (System32\tar.exe), by
// absolute path: Git for Windows puts a GNU tar first on PATH, and GNU tar
// cannot read zip files. This used to shell out to PowerShell's Expand-Archive,
// which broke on the runner image that updated PowerShell 7.6.9 to 7.6.10:
// the step's pwsh exports a PSModulePath holding Microsoft.PowerShell.Archive
// 1.2.7, the child Windows PowerShell 5.1 loaded that copy, and it calls a
// .NET Core only ZipFileExtensions.ExtractToDirectory overload that 5.1 does
// not have. That failure was non-terminating, so powershell.exe still exited 0
// and the caller carried on with an empty directory.
export async function extractZip(zip, directory) {
  await mkdir(directory, { recursive: true })
  if (process.platform === 'win32') {
    const systemRoot = process.env.SystemRoot || process.env.windir || 'C:\\Windows'
    await command(`${systemRoot}\\System32\\tar.exe`, ['-xf', zip, '-C', directory])
  } else {
    await command('unzip', ['-o', zip, '-d', directory])
  }
}
