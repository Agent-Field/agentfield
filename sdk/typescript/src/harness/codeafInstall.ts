import { createHash, randomUUID } from 'node:crypto';
import fs, { type promises } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { codeafDownloadResponse } from './codeafDownload.js';
import { CODEAF_BASE_URL, CODEAF_MAX_DOWNLOAD_BYTES, CODEAF_SHA256, CODEAF_VERSION } from './codeafRelease.js';

type Environment = Readonly<Record<string, string | undefined>>;

export function agentfieldHome(env: Environment = process.env): string {
  return env.AGENTFIELD_HOME || path.join(os.homedir(), '.agentfield');
}

export function codeafStateHome(env: Environment): string {
  const preferred = path.join(agentfieldHome(env), 'codeaf');
  try {
    fs.mkdirSync(preferred, { recursive: true });
  } catch {
    // CodeAF reports state write failures in its result.
  }
  return preferred;
}

export function codeafManagedPaths(env: Environment = process.env): string[] {
  const name = process.platform === 'win32' ? 'codeaf.exe' : 'codeaf';
  return [
    path.join(agentfieldHome(env), 'bin', name),
  ];
}

function versionMarker(binary: string): string {
  return path.join(path.dirname(binary), '.codeaf.version');
}

export function isCodeafInstalled(binary: string): boolean {
  try {
    if (!fs.statSync(binary).isFile()) {
      return false;
    }
    fs.accessSync(binary, process.platform === 'win32' ? fs.constants.F_OK : fs.constants.X_OK);
    return fs.readFileSync(versionMarker(binary), 'utf8').trim() === CODEAF_VERSION;
  } catch {
    return false;
  }
}

export function findManagedCodeaf(env: Environment = process.env): string | undefined {
  return codeafManagedPaths(env).find(isCodeafInstalled);
}

export function isTruthy(value: string | undefined): boolean {
  return /^(1|true|yes|on)$/i.test(value?.trim() ?? '');
}

export function skipCodeafInstall(env: Environment = process.env): boolean {
  return isTruthy(env.AGENTFIELD_SKIP_CODEAF) || isTruthy(env.AGENTFIELD_SKIP_AFORGE);
}

export function codeafAsset(platform: string = process.platform, arch: string = process.arch): string | undefined {
  const releaseOS = platform === 'win32' ? 'windows' : platform;
  const releaseArch = arch === 'x64' ? 'amd64' : arch;
  const asset = `codeaf-${releaseOS}-${releaseArch}${platform === 'win32' ? '.exe' : ''}`;
  return Object.hasOwn(CODEAF_SHA256, asset) ? asset : undefined;
}

async function download(url: URL, file: promises.FileHandle, expectedHash: string): Promise<void> {
  const response = await codeafDownloadResponse(url);
  try {
    if (Number(response.headers['content-length']) > CODEAF_MAX_DOWNLOAD_BYTES) {
      throw new Error('CodeAF download exceeds the 128 MiB size limit');
    }
    const hash = createHash('sha256');
    let size = 0;
    for await (const chunk of response) {
      const data = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += data.length;
      if (size > CODEAF_MAX_DOWNLOAD_BYTES) {
        throw new Error('CodeAF download exceeds the 128 MiB size limit');
      }
      hash.update(data);
      await file.writeFile(data);
    }
    if (hash.digest('hex') !== expectedHash) {
      throw new Error('CodeAF download SHA-256 mismatch');
    }
  } finally {
    response.destroy();
  }
}

async function installCodeaf(): Promise<string> {
  const asset = codeafAsset();
  if (!asset) {
    throw new Error(`CodeAF is not installable on ${process.platform}/${process.arch}`);
  }

  if (process.env.HTTPS_PROXY || process.env.https_proxy) {
    throw new Error('CodeAF download skipped: HTTPS_PROXY is set and the TypeScript SDK does not use proxies; run `af codeaf ensure`');
  }
  const [binary] = codeafManagedPaths();
  const temporary = path.join(path.dirname(binary), `.codeaf-${randomUUID()}.tmp`);
  let file: promises.FileHandle;
  try {
    await fs.promises.mkdir(path.dirname(binary), { recursive: true });
    file = await fs.promises.open(temporary, 'wx', 0o600);
  } catch (error) {
    throw new Error(`cannot install CodeAF into ${path.dirname(binary)}: ${error instanceof Error ? error.message : String(error)}`);
  }
  console.info(`downloading CodeAF ${CODEAF_VERSION} (~70 MB) to ${binary}`);
  try {
    const base = (process.env.AGENTFIELD_CODEAF_BASE_URL || CODEAF_BASE_URL).replace(/\/+$/, '');
    await download(new URL(`${base}/${asset}`), file, CODEAF_SHA256[asset]);
    await file.chmod(0o755);
    await file.close();
    try {
      await fs.promises.rename(temporary, binary);
    } catch (error) {
      if (process.platform === 'win32' && isCodeafInstalled(binary)) {
        return binary;
      }
      throw error;
    }
    await fs.promises.writeFile(versionMarker(binary), `${CODEAF_VERSION}\n`);
    return binary;
  } finally {
    await file.close().catch(() => undefined);
    await fs.promises.rm(temporary, { force: true });
  }
}

let installation: Promise<string> | undefined;
let nextInstallAttempt = 0;

/** Share in-flight installs and throttle retries; construction and offline doctor never install. */
export async function ensureCodeafInstalled(): Promise<string | undefined> {
  const installed = findManagedCodeaf();
  if (installed) {
    return installed;
  }
  if (skipCodeafInstall()) {
    return undefined;
  }
  if (installation) return installation;
  if (Date.now() < nextInstallAttempt) return undefined;
  nextInstallAttempt = Date.now() + 60_000;
  installation = installCodeaf().finally(() => { installation = undefined; });
  return installation;
}
