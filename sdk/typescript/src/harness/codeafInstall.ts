import { createHash, randomUUID } from 'node:crypto';
import fs, { type promises } from 'node:fs';
import http, { type IncomingMessage } from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { CODEAF_BASE_URL, CODEAF_MAX_DOWNLOAD_BYTES, CODEAF_SHA256, CODEAF_VERSION } from './codeafRelease.js';

type Environment = Readonly<Record<string, string | undefined>>;

export function agentfieldHome(env: Environment = process.env): string {
  return env.AGENTFIELD_HOME || path.join(os.homedir(), '.agentfield');
}

export function codeafManagedPaths(env: Environment = process.env): string[] {
  const name = process.platform === 'win32' ? 'codeaf.exe' : 'codeaf';
  return [
    path.join(agentfieldHome(env), 'bin', name),
    path.join(os.tmpdir(), 'agentfield-codeaf', CODEAF_VERSION, name),
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

async function downloadResponse(url: URL, redirects = 0): Promise<IncomingMessage> {
  const client = url.protocol === 'https:' ? https : url.protocol === 'http:' ? http : undefined;
  if (!client) {
    throw new Error(`Unsupported CodeAF download protocol: ${url.protocol}`);
  }
  const response = await new Promise<IncomingMessage>((resolve, reject) => {
    const request = client.get(url, resolve);
    request.on('error', reject);
    request.setTimeout(30_000, () => request.destroy(new Error('CodeAF download timed out')));
  });
  if ([301, 302, 303, 307, 308].includes(response.statusCode ?? 0) && response.headers.location) {
    response.resume();
    if (redirects >= 5) {
      throw new Error('Too many CodeAF download redirects');
    }
    return downloadResponse(new URL(response.headers.location, url), redirects + 1);
  }
  if (response.statusCode !== 200) {
    response.resume();
    throw new Error(`CodeAF download failed: HTTP ${response.statusCode}`);
  }
  return response;
}

async function download(url: URL, file: promises.FileHandle, expectedHash: string): Promise<void> {
  const response = await downloadResponse(url);
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

  let target: { binary: string; temporary: string; file: promises.FileHandle } | undefined;
  for (const binary of codeafManagedPaths()) {
    const temporary = path.join(path.dirname(binary), `.codeaf-${randomUUID()}.tmp`);
    try {
      await fs.promises.mkdir(path.dirname(binary), { recursive: true });
      const file = await fs.promises.open(temporary, 'wx', 0o600);
      target = { binary, temporary, file };
      break;
    } catch (error) {
      if (binary === codeafManagedPaths()[1]) {
        throw error;
      }
    }
  }
  const { binary, temporary, file } = target!;
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

/** Download at most once per process; construction and offline doctor never install. */
export async function ensureCodeafInstalled(): Promise<string | undefined> {
  const installed = findManagedCodeaf();
  if (installed) {
    return installed;
  }
  if (skipCodeafInstall()) {
    return undefined;
  }
  installation ??= installCodeaf();
  return installation;
}
