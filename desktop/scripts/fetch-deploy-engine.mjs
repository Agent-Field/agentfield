import { chmod, copyFile, mkdir, mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const TOFU_VERSION = '1.10.6'
const RAILWAY_VERSION = '0.6.2'
const platform = { win32: 'windows', darwin: 'darwin', linux: 'linux' }[process.platform]
const arch = { x64: 'amd64', arm64: 'arm64' }[process.arch]

if (!platform || !arch) throw new Error(`Unsupported deploy-engine platform: ${process.platform}/${process.arch}`)

const desktopDir = resolve(fileURLToPath(new URL('..', import.meta.url)))
const destination = join(desktopDir, 'vendor', 'deploy-engine')
const providerDir = join(destination, 'providers', 'registry.terraform.io', 'terraform-community-providers', 'railway', RAILWAY_VERSION, `${platform}_${arch}`)
const temporary = await mkdtemp(join(tmpdir(), 'agentfield-deploy-engine-'))

async function download(url, path) {
  console.log(`Downloading ${basename(path)}...`)
  const response = await fetch(url, { redirect: 'follow' })
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${url}`)
  await BunlessWrite(path, new Uint8Array(await response.arrayBuffer()))
}

async function BunlessWrite(path, bytes) {
  const { writeFile } = await import('node:fs/promises')
  await writeFile(path, bytes)
}

function command(program, args) {
  return new Promise((resolveCommand, reject) => {
    const child = spawn(program, args, { stdio: 'inherit' })
    child.on('error', reject)
    child.on('close', (code) => code === 0 ? resolveCommand() : reject(new Error(`${program} exited with ${code}`)))
  })
}

async function extract(zip, directory) {
  await mkdir(directory, { recursive: true })
  if (process.platform === 'win32') {
    const escapedZip = zip.replaceAll("'", "''")
    const escapedDir = directory.replaceAll("'", "''")
    await command('powershell.exe', ['-NoProfile', '-Command', `Expand-Archive -LiteralPath '${escapedZip}' -DestinationPath '${escapedDir}' -Force`])
  } else {
    await command('unzip', ['-o', zip, '-d', directory])
  }
}

try {
  const tofuZip = join(temporary, 'tofu.zip')
  const providerZip = join(temporary, 'provider.zip')
  const tofuExtract = join(temporary, 'tofu')
  const providerExtract = join(temporary, 'provider')
  await Promise.all([
    download(`https://github.com/opentofu/opentofu/releases/download/v${TOFU_VERSION}/tofu_${TOFU_VERSION}_${platform}_${arch}.zip`, tofuZip),
    download(`https://github.com/terraform-community-providers/terraform-provider-railway/releases/download/v${RAILWAY_VERSION}/terraform-provider-railway_${RAILWAY_VERSION}_${platform}_${arch}.zip`, providerZip)
  ])
  await Promise.all([extract(tofuZip, tofuExtract), extract(providerZip, providerExtract)])
  await mkdir(providerDir, { recursive: true })
  const tofuName = process.platform === 'win32' ? 'tofu.exe' : 'tofu'
  const providerName = process.platform === 'win32' ? `terraform-provider-railway_v${RAILWAY_VERSION}.exe` : `terraform-provider-railway_v${RAILWAY_VERSION}`
  const providerFiles = await readdir(providerExtract)
  const extractedProvider = providerFiles.find((name) => name.startsWith('terraform-provider-railway'))
  if (!extractedProvider) throw new Error('Provider archive did not contain the Railway provider binary')
  await mkdir(destination, { recursive: true })
  await rm(join(destination, tofuName), { force: true })
  await rm(join(providerDir, providerName), { force: true })
  await copyFile(join(tofuExtract, tofuName), join(destination, tofuName))
  await copyFile(join(providerExtract, extractedProvider), join(providerDir, providerName))
  if (process.platform !== 'win32') await Promise.all([chmod(join(destination, tofuName), 0o755), chmod(join(providerDir, providerName), 0o755)])
  console.log(`Deploy engine ${TOFU_VERSION} and Railway provider ${RAILWAY_VERSION} installed in ${destination}`)
} finally {
  await rm(temporary, { recursive: true, force: true })
}
