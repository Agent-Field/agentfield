import { createHash } from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const bytes = Buffer.from('#!/bin/sh\nprintf \'%s\\n\' \'{"ok":true,"stop":"done","answer":"installed answer"}\'\n');
const digest = createHash('sha256').update(bytes).digest('hex');
let directory: string;
let server: http.Server;
let requests: string[];
let handler: (request: http.IncomingMessage, response: http.ServerResponse) => void;
let limit: number | undefined;
const originalPlatform = Object.getOwnPropertyDescriptor(process, 'platform')!;

beforeEach(async () => {
  vi.resetModules();
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-codeaf-install-'));
  vi.stubEnv('AGENTFIELD_HOME', path.join(directory, 'home'));
  vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
  vi.stubEnv('AGENTFIELD_SKIP_AFORGE', '');
  for (const name of ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy']) {
    vi.stubEnv(name, undefined);
  }
  vi.spyOn(os, 'tmpdir').mockReturnValue(path.join(directory, 'temp'));
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  requests = [];
  limit = undefined;
  handler = (_request, response) => response.end(bytes);
  server = http.createServer((request, response) => {
    requests.push(request.url!);
    handler(request, response);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;
  vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `http://127.0.0.1:${port}/v0.8.2///`);
  // Inject only at the module boundary in tests. Production has no hash override.
  vi.doMock('../src/harness/codeafRelease.js', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../src/harness/codeafRelease.js')>();
    return {
      ...actual,
      CODEAF_MAX_DOWNLOAD_BYTES: limit ?? actual.CODEAF_MAX_DOWNLOAD_BYTES,
      CODEAF_SHA256: Object.fromEntries(Object.keys(actual.CODEAF_SHA256).map((asset) => [asset, digest])),
    };
  });
});

afterEach(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  Object.defineProperty(process, 'platform', originalPlatform);
  vi.restoreAllMocks();
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.doUnmock('../src/harness/codeafRelease.js');
  fs.rmSync(directory, { recursive: true, force: true });
});

describe('V6 pinned CodeAF installation', () => {
  it.each(['mkdir', 'write', 'non-error'])('C1 reports %s failures without temp entries', async (failure) => {
    const install = await import('../src/harness/codeafInstall.js');
    if (failure === 'mkdir') fs.writeFileSync(process.env.AGENTFIELD_HOME!, 'not a directory');
    else vi.spyOn(fs.promises, 'open').mockRejectedValue(failure === 'write' ? new Error('read-only home') : 'read-only home');
    await expect(install.ensureCodeafInstalled()).rejects.toThrow('cannot install CodeAF into');
    expect(requests).toEqual([]);
    expect(fs.existsSync(path.join(directory, 'temp'))).toBe(false);
  });

  it('embeds exactly the six pinned raw release hashes', async () => {
    const release = await vi.importActual<typeof import('../src/harness/codeafRelease.js')>('../src/harness/codeafRelease.js');
    expect(release.CODEAF_VERSION).toBe('v0.8.2');
    expect(release.CODEAF_BASE_URL).toBe('https://github.com/Agent-Field/CodeAF/releases/download/v0.8.2');
    expect(release.CODEAF_MAX_DOWNLOAD_BYTES).toBe(128 * 1024 * 1024);
    expect(release.CODEAF_SHA256).toEqual({
      'codeaf-darwin-amd64': 'a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7',
      'codeaf-darwin-arm64': 'e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1',
      'codeaf-linux-amd64': '681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c',
      'codeaf-linux-arm64': 'c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78',
      'codeaf-windows-amd64.exe': 'ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d',
      'codeaf-windows-arm64.exe': 'bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80',
    });
  });

  it.each([
    ['darwin', 'x64', 'codeaf-darwin-amd64'],
    ['darwin', 'arm64', 'codeaf-darwin-arm64'],
    ['linux', 'x64', 'codeaf-linux-amd64'],
    ['linux', 'arm64', 'codeaf-linux-arm64'],
    ['win32', 'x64', 'codeaf-windows-amd64.exe'],
    ['win32', 'arm64', 'codeaf-windows-arm64.exe'],
    ['freebsd', 'x64', undefined],
    ['linux', 'ia32', undefined],
  ])('selects the pinned asset for %s/%s', async (platform, arch, expected) => {
    const { codeafAsset } = await import('../src/harness/codeafInstall.js');
    expect(codeafAsset(platform, arch)).toBe(expected);
  });

  it('downloads once for concurrent callers, verifies bytes, and writes the marker last', async () => {
    const install = await import('../src/harness/codeafInstall.js');
    const [binary] = install.codeafManagedPaths();
    const marker = path.join(path.dirname(binary), '.codeaf.version');
    handler = (_request, response) => {
      expect(fs.existsSync(binary)).toBe(false);
      expect(fs.existsSync(marker)).toBe(false);
      response.end(bytes);
    };
    const writeFile = fs.promises.writeFile.bind(fs.promises);
    vi.spyOn(fs.promises, 'writeFile').mockImplementation(async (name, ...args) => {
      if (String(name) === marker) {
        expect(fs.readFileSync(binary)).toEqual(bytes);
        if (process.platform !== 'win32') expect(fs.statSync(binary).mode & 0o777).toBe(0o755);
      }
      return writeFile(name, ...args);
    });
    expect(await Promise.all(Array.from({ length: 16 }, () => install.ensureCodeafInstalled())))
      .toEqual(Array(16).fill(binary));
    expect(requests).toEqual([`/v0.8.2/${install.codeafAsset()}`]);
    expect(fs.readFileSync(marker, 'utf8')).toBe('v0.8.2\n');
    expect(fs.readdirSync(path.dirname(binary)).sort()).toEqual(['.codeaf.version', path.basename(binary)]);
    expect(install.findManagedCodeaf()).toBe(binary);
    await expect(install.ensureCodeafInstalled()).resolves.toBe(binary);
    expect(console.info).toHaveBeenCalledExactlyOnceWith(`downloading CodeAF v0.8.2 (~70 MB) to ${binary}`);
  });

  it('follows release redirects to the object download', async () => {
    handler = (request, response) => {
      if (!request.url?.startsWith('/objects/')) {
        response.writeHead(302, { location: '/objects/codeaf' });
        response.end();
      } else {
        response.end(bytes);
      }
    };
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    const binary = await ensureCodeafInstalled();
    expect(fs.readFileSync(binary!)).toEqual(bytes);
    expect(requests).toHaveLength(2);
    expect(requests[1]).toBe('/objects/codeaf');
  });

  it.each(['checksum', 'declared-size', 'stream-size', 'http', 'redirect-loop', 'connection'])
    ('rejects %s failures, cleans up, and throttles downloads for 60 seconds', async (kind) => {
      vi.useFakeTimers({ toFake: ['Date'] });
      if (kind === 'stream-size') limit = 16;
      handler = (request, response) => {
        if (kind === 'checksum') response.end('tampered');
        if (kind === 'declared-size') {
          response.setHeader('content-length', String(128 * 1024 * 1024 + 1));
          response.end(bytes);
        }
        if (kind === 'stream-size') {
          response.write(bytes);
          response.end(bytes);
        }
        if (kind === 'http') response.writeHead(404).end();
        if (kind === 'redirect-loop') response.writeHead(302, { location: request.url }).end();
        if (kind === 'connection') request.socket.destroy();
      };
      const install = await import('../src/harness/codeafInstall.js');
      const errors = { checksum: 'SHA-256 mismatch', 'declared-size': '128 MiB', 'stream-size': '128 MiB',
        http: 'HTTP 404', 'redirect-loop': 'Too many', connection: 'socket hang up' };
      await expect(install.ensureCodeafInstalled()).rejects.toThrow(errors[kind]);
      const count = requests.length;
      await expect(install.ensureCodeafInstalled()).resolves.toBeUndefined();
      expect(requests).toHaveLength(count);
      const [binary] = install.codeafManagedPaths();
      expect(install.isCodeafInstalled(binary)).toBe(false);
      expect(fs.readdirSync(path.dirname(binary))).toEqual([]);
      handler = (_request, response) => response.end(bytes);
      vi.setSystemTime(Date.now() + 59_999);
      await expect(install.ensureCodeafInstalled()).resolves.toBeUndefined();
      expect(requests).toHaveLength(count);
      vi.setSystemTime(Date.now() + 1);
      // A stream-size fixture's cap deliberately remains below the fixture size.
      if (kind === 'stream-size') await expect(install.ensureCodeafInstalled()).rejects.toThrow('128 MiB');
      else await expect(install.ensureCodeafInstalled()).resolves.toBe(binary);
      expect(requests).toHaveLength(count + 1);
    });

  it('rechecks a repaired managed install immediately after a failed download', async () => {
    handler = (_request, response) => response.writeHead(503).end();
    const install = await import('../src/harness/codeafInstall.js');
    await expect(install.ensureCodeafInstalled()).rejects.toThrow('HTTP 503');
    const [binary] = install.codeafManagedPaths();
    fs.writeFileSync(binary, bytes, { mode: 0o755 });
    fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), 'v0.8.2\n');
    await expect(install.ensureCodeafInstalled()).resolves.toBe(binary);
    expect(requests).toHaveLength(1);
  });

  it.each(['AGENTFIELD_SKIP_CODEAF', 'AGENTFIELD_SKIP_AFORGE'])('honors %s truthy spellings', async (key) => {
    const install = await import('../src/harness/codeafInstall.js');
    for (const value of ['1', 'TRUE', 'Yes', 'on', ' ON ']) {
      vi.stubEnv(key, value);
      await expect(install.ensureCodeafInstalled()).resolves.toBeUndefined();
    }
    expect(requests).toEqual([]);
    expect(console.info).not.toHaveBeenCalled();
  });

  it('does not download on an unsupported platform', async () => {
    Object.defineProperty(process, 'platform', { value: 'freebsd', configurable: true });
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    await expect(ensureCodeafInstalled()).rejects.toThrow('not installable');
    expect(requests).toEqual([]);
  });

  it('requires a regular executable plus an exact, trimmed version marker', async () => {
    const install = await import('../src/harness/codeafInstall.js');
    const [binary] = install.codeafManagedPaths();
    fs.mkdirSync(path.dirname(binary), { recursive: true });
    fs.mkdirSync(binary);
    fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), 'v0.8.2\n');
    expect(install.isCodeafInstalled(binary)).toBe(false);
    fs.rmdirSync(binary);
    fs.writeFileSync(binary, bytes, { mode: 0o600 });
    if (process.platform !== 'win32') expect(install.isCodeafInstalled(binary)).toBe(false);
    fs.chmodSync(binary, 0o755);
    expect(install.isCodeafInstalled(binary)).toBe(true);
    fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), 'v0.8.1');
    expect(install.isCodeafInstalled(binary)).toBe(false);
    fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), 'v0.8.2');
    expect(install.findManagedCodeaf()).toBe(binary);
  });

  it.each([true, false])('handles Windows replacement failures with matching marker=%s', async (matching) => {
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    const install = await import('../src/harness/codeafInstall.js');
    const [binary] = install.codeafManagedPaths();
    expect(binary).toMatch(/codeaf\.exe$/);
    vi.spyOn(fs.promises, 'rename').mockImplementation(async () => {
      fs.writeFileSync(binary, bytes);
      fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), matching ? 'v0.8.2\n' : 'v0.8.1\n');
      throw new Error('binary is busy');
    });
    if (matching) await expect(install.ensureCodeafInstalled()).resolves.toBe(binary);
    else await expect(install.ensureCodeafInstalled()).rejects.toThrow('binary is busy');
    expect(fs.readdirSync(path.dirname(binary)).sort()).toEqual(['.codeaf.version', 'codeaf.exe']);
  });

  it('uses ~/.agentfield by default and rejects unsupported URL protocols', async () => {
    vi.stubEnv('AGENTFIELD_HOME', '');
    vi.spyOn(os, 'homedir').mockReturnValue(path.join(directory, 'user'));
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', 'file:///somewhere');
    const install = await import('../src/harness/codeafInstall.js');
    expect(install.agentfieldHome()).toBe(path.join(directory, 'user', '.agentfield'));
    await expect(install.ensureCodeafInstalled()).rejects.toThrow('Unsupported CodeAF download protocol');
    expect(requests).toEqual([]);
  });
});
