import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import type { Duplex } from 'node:stream';
import tls from 'node:tls';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const bytes = Buffer.from('#!/bin/sh\nprintf proxy-codeaf\n');
const digest = createHash('sha256').update(bytes).digest('hex');
// Self-signed test-only credentials, generated per run so no private key is
// committed; production TLS verification stays enabled.
function selfSignedCredentials(): { cert: Buffer; key: Buffer } | null {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-codeaf-tls-'));
  const result = spawnSync('openssl', [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
    '-subj', '/CN=codeaf-origin.invalid',
    '-addext', 'subjectAltName=DNS:codeaf-origin.invalid,DNS:localhost,IP:127.0.0.1',
    '-keyout', path.join(dir, 'key.pem'), '-out', path.join(dir, 'cert.pem'),
  ], { stdio: 'ignore' });
  if (result.status !== 0) return null;
  return {
    cert: fs.readFileSync(path.join(dir, 'cert.pem')),
    key: fs.readFileSync(path.join(dir, 'key.pem')),
  };
}
const credentials = selfSignedCredentials();
const cert = credentials?.cert ?? Buffer.alloc(0);
const key = credentials?.key ?? Buffer.alloc(0);
let directory: string;
let origin: https.Server;
let proxy: http.Server;
let originPort: number;
let proxyURL: string;
let httpRequests: Array<{ url: string; host: string | undefined; authorization: string | undefined }>;
let tunnels: string[];
let tunnelAuth: Array<string | undefined>;
let originRequests: string[];
let originAuth: Array<string | undefined>;
let sockets: Set<Duplex>;
let connectStatus: number;
let corrupt: boolean;

async function listen(server: http.Server | https.Server): Promise<number> {
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return (server.address() as net.AddressInfo).port;
}

beforeEach(async () => {
  vi.resetModules();
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-codeaf-proxy-'));
  vi.stubEnv('AGENTFIELD_HOME', path.join(directory, 'home'));
  vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
  vi.stubEnv('AGENTFIELD_SKIP_AFORGE', '');
  for (const name of ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy']) {
    vi.stubEnv(name, undefined);
  }
  vi.spyOn(os, 'tmpdir').mockReturnValue(path.join(directory, 'temp'));
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  vi.doMock('../src/harness/codeafRelease.js', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../src/harness/codeafRelease.js')>();
    return { ...actual, CODEAF_SHA256: Object.fromEntries(Object.keys(actual.CODEAF_SHA256).map((asset) => [asset, digest])) };
  });
  // Trust only this fixture's certificate while still checking its hostname.
  const connect = tls.connect.bind(tls);
  vi.spyOn(tls, 'connect').mockImplementation(((options: tls.ConnectionOptions) => connect({ ...options, ca: cert })) as typeof tls.connect);
  httpRequests = [];
  tunnels = [];
  tunnelAuth = [];
  originRequests = [];
  originAuth = [];
  sockets = new Set();
  connectStatus = 200;
  corrupt = false;
  origin = https.createServer({ key, cert }, (request, response) => {
    originRequests.push(request.url!);
    originAuth.push(request.headers['proxy-authorization']);
    if (request.url?.startsWith('/objects/')) response.end(corrupt ? 'corrupt' : bytes);
    else response.writeHead(302, { location: '/objects/codeaf' }).end();
  });
  originPort = await listen(origin);
  proxy = http.createServer((request, response) => {
    httpRequests.push({ url: request.url!, host: request.headers.host, authorization: request.headers['proxy-authorization'] });
    if (request.url?.includes('/redirect/')) {
      response.writeHead(302, { location: `https://codeaf-origin.invalid:${originPort}/release/codeaf` }).end();
    } else response.end(corrupt ? 'corrupt' : bytes);
  });
  proxy.on('connect', (request, socket, head) => {
    tunnels.push(request.url!);
    tunnelAuth.push(request.headers['proxy-authorization']);
    if (connectStatus !== 200) {
      socket.end(`HTTP/1.1 ${connectStatus} Proxy refused\r\n\r\n`);
      return;
    }
    const upstream = net.connect(originPort, '127.0.0.1', () => {
      socket.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      if (head.length) upstream.write(head);
      socket.pipe(upstream).pipe(socket);
    });
    sockets.add(upstream);
    upstream.on('error', () => socket.destroy());
    socket.on('error', () => upstream.destroy());
    socket.on('close', () => upstream.destroy());
  });
  for (const server of [origin, proxy]) server.on('connection', (socket) => sockets.add(socket));
  proxyURL = `http://127.0.0.1:${await listen(proxy)}`;
});

afterEach(async () => {
  for (const socket of sockets) socket.destroy();
  await Promise.all([origin, proxy].map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.doUnmock('../src/harness/codeafRelease.js');
  fs.rmSync(directory, { recursive: true, force: true });
});

describe.skipIf(credentials === null)('A4 CodeAF downloads through environment proxies', () => {
  it.each(['HTTP_PROXY', 'http_proxy'])('forwards absolute HTTP URLs via %s with proxy authentication', async (name) => {
    vi.stubEnv(name, proxyURL.replace('http://', 'http://test:fixture-password@'));
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', 'http://codeaf-origin.invalid/release');
    const install = await import('../src/harness/codeafInstall.js');
    const binary = await install.ensureCodeafInstalled();
    expect(fs.readFileSync(binary!)).toEqual(bytes);
    expect(httpRequests).toEqual([{
      url: `http://codeaf-origin.invalid/release/${install.codeafAsset()}`, host: 'codeaf-origin.invalid',
      authorization: `Basic ${Buffer.from('test:fixture-password').toString('base64')}`,
    }]);
  });

  it.each(['HTTPS_PROXY', 'https_proxy'])('tunnels HTTPS through %s and follows redirects', async (name) => {
    vi.stubEnv(name, proxyURL.replace('http://', 'http://test:fixture-password@'));
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `https://codeaf-origin.invalid:${originPort}/release`);
    const install = await import('../src/harness/codeafInstall.js');
    const binary = await install.ensureCodeafInstalled();
    expect(fs.readFileSync(binary!)).toEqual(bytes);
    expect(tunnels).toEqual(Array(2).fill(`codeaf-origin.invalid:${originPort}`));
    expect(tunnelAuth).toEqual(Array(2).fill(`Basic ${Buffer.from('test:fixture-password').toString('base64')}`));
    expect(originRequests).toEqual([`/release/${install.codeafAsset()}`, '/objects/codeaf']);
    expect(originAuth).toEqual([undefined, undefined]);
    expect(httpRequests).toEqual([]);
  });

  it('reselects the proxy when an HTTP origin redirects to HTTPS', async () => {
    vi.stubEnv('HTTP_PROXY', proxyURL);
    vi.stubEnv('HTTPS_PROXY', proxyURL);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', 'http://codeaf-origin.invalid/redirect');
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    expect(fs.readFileSync((await ensureCodeafInstalled())!)).toEqual(bytes);
    expect(httpRequests).toHaveLength(1);
    expect(tunnels).toHaveLength(2);
  });

  it('prefers lowercase proxy settings when both forms are present', async () => {
    vi.stubEnv('HTTP_PROXY', 'http://unreachable-proxy.invalid');
    vi.stubEnv('http_proxy', proxyURL);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', 'http://codeaf-origin.invalid/release');
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    expect(fs.readFileSync((await ensureCodeafInstalled())!)).toEqual(bytes);
    expect(httpRequests).toHaveLength(1);
  });

  it.each(['NO_PROXY', 'no_proxy'])('bypasses the HTTPS proxy for matching %s host and port', async (name) => {
    vi.stubEnv('HTTPS_PROXY', proxyURL);
    vi.stubEnv(name, `other.invalid, 127.0.0.1:${originPort}`);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `https://127.0.0.1:${originPort}/release`);
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    expect(fs.readFileSync((await ensureCodeafInstalled())!)).toEqual(bytes);
    expect(tunnels).toEqual([]);
    expect(originRequests).toHaveLength(2);
  });

  it.each(['127.0.0.1', '.0.0.1', '*.0.0.1', '*'])('bypasses the HTTP proxy for NO_PROXY=%s', async (noProxy) => {
    const direct = http.createServer((_request, response) => response.end(bytes));
    try {
      const port = await listen(direct);
      vi.stubEnv('HTTP_PROXY', proxyURL);
      vi.stubEnv('NO_PROXY', noProxy);
      vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `http://127.0.0.1:${port}/release`);
      const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
      expect(fs.readFileSync((await ensureCodeafInstalled())!)).toEqual(bytes);
      expect(httpRequests).toEqual([]);
    } finally {
      direct.closeAllConnections();
      await new Promise<void>((resolve) => direct.close(() => resolve()));
    }
  });

  it('uses the proxy when NO_PROXY has a different port or only a partial hostname', async () => {
    vi.stubEnv('HTTP_PROXY', proxyURL);
    vi.stubEnv('NO_PROXY', 'codeaf-origin.invalid:1,origin.invalid');
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', 'http://codeaf-origin.invalid/release');
    const { ensureCodeafInstalled } = await import('../src/harness/codeafInstall.js');
    await ensureCodeafInstalled();
    expect(httpRequests).toHaveLength(1);
  });

  it.each(['http', 'https'])('rejects hash mismatches over the %s proxy and cleans up', async (protocol) => {
    corrupt = true;
    vi.stubEnv(protocol === 'http' ? 'HTTP_PROXY' : 'HTTPS_PROXY', proxyURL);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `${protocol}://codeaf-origin.invalid${protocol === 'https' ? `:${originPort}` : ''}/release`);
    const install = await import('../src/harness/codeafInstall.js');
    await expect(install.ensureCodeafInstalled()).rejects.toThrow('SHA-256 mismatch');
    const [binary] = install.codeafManagedPaths();
    expect(fs.readdirSync(path.dirname(binary))).toEqual([]);
  });

  it('reports a refused CONNECT without publishing an install', async () => {
    connectStatus = 407;
    vi.stubEnv('HTTPS_PROXY', proxyURL);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `https://codeaf-origin.invalid:${originPort}/release`);
    const install = await import('../src/harness/codeafInstall.js');
    await expect(install.ensureCodeafInstalled()).rejects.toThrow('407');
    expect(originRequests).toEqual([]);
    expect(install.findManagedCodeaf()).toBeUndefined();
  });

  it('verifies the origin hostname inside the TLS tunnel', async () => {
    vi.stubEnv('HTTPS_PROXY', proxyURL);
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `https://wrong-host.invalid:${originPort}/release`);
    const install = await import('../src/harness/codeafInstall.js');
    await expect(install.ensureCodeafInstalled()).rejects.toThrow(/Hostname\/IP does not match/);
    expect(tunnels).toEqual([`wrong-host.invalid:${originPort}`]);
    expect(originRequests).toEqual([]);
    expect(install.findManagedCodeaf()).toBeUndefined();
  });
});
