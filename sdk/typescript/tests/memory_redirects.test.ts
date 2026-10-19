import { createServer, type IncomingHttpHeaders, type Server } from 'node:http';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryClient } from '../src/memory/MemoryClient.js';
import { MemoryEventClient } from '../src/memory/MemoryEventClient.js';

const servers: Server[] = [];

async function listen(server: Server) {
  servers.push(server);
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Expected TCP address');
  return `http://127.0.0.1:${address.port}`;
}

afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  await Promise.all(servers.splice(0).map(server => new Promise<void>((resolve, reject) => {
    server.close(error => error ? reject(error) : resolve());
    server.closeAllConnections();
  })));
});

// SPEC-7: exercise axios over real loopback HTTP, including a different port
// on the same host (a different origin). No transport or redirect mocks.
describe.each(['memory', 'history'] as const)('%s refuses redirects', (kind) => {
  async function expectRedirectError(origin: string, headers: Record<string, string>, requestHeaders?: Record<string, string>) {
    if (kind === 'memory') {
      await expect(new MemoryClient(origin, headers).set('key', 'private-data', {
        scope: 'global', headers: requestHeaders
      })).rejects.toMatchObject({ isAxiosError: true, response: { status: 307 } });
    } else {
      const log = vi.spyOn(console, 'error').mockImplementation(() => {});
      // History retains its existing catch/log/empty-result contract. The
      // underlying axios request reports the 307 as an error in both clients.
      await expect(new MemoryEventClient(origin, headers).history({ headers: requestHeaders })).resolves.toEqual([]);
      expect(log).toHaveBeenCalledWith(expect.stringContaining('Request failed with status code 307'));
    }
  }

  it.each([false, true])('refuses cross-origin 307 redirects (lowercase=%s)', async (lowercase) => {
    const destinationHeaders: IncomingHttpHeaders[] = [];
    const destination = await listen(createServer((req, res) => {
      destinationHeaders.push(req.headers);
      res.end('[]');
    }));
    const originHeaders: IncomingHttpHeaders[] = [];
    const origin = await listen(createServer((req, res) => {
      originHeaders.push(req.headers);
      req.resume();
      res.writeHead(307, { Location: `${destination}/redirected` });
      res.end();
    }));
    const entries = Object.entries({
      'X-API-Key': 'secret-key', Authorization: 'Bearer secret', Cookie: 'session=secret',
      'X-Agent-Node-ID': 'agent-beta', 'X-Caller-Agent-ID': 'agent-beta'
    });
    const headers = Object.fromEntries(entries.map(([key, value]) => [lowercase ? key.toLowerCase() : key, value]));

    await expectRedirectError(origin, headers);
    expect(originHeaders).toHaveLength(1);
    expect(originHeaders[0]).toMatchObject({
      'x-api-key': 'secret-key', authorization: 'Bearer secret', cookie: 'session=secret',
      'x-agent-node-id': 'agent-beta', 'x-caller-agent-id': 'agent-beta'
    });
    expect(destinationHeaders).toEqual([]);
  });

  it.each(['default', 'request'] as const)('refuses relative redirects with a %s Host override', async (override) => {
    const destinationRequests: string[] = [];
    const destination = await listen(createServer((req, res) => {
      destinationRequests.push(req.url ?? '');
      res.end('[]');
    }));
    const originHeaders: IncomingHttpHeaders[] = [];
    const origin = await listen(createServer((req, res) => {
      originHeaders.push(req.headers);
      req.resume();
      res.writeHead(307, { Location: '/redirected' });
      res.end();
    }));
    const host = { Host: new URL(destination).host };
    await expectRedirectError(origin, {
      'X-API-Key': 'secret-key', ...(override === 'default' ? host : {})
    }, override === 'request' ? host : undefined);
    expect(originHeaders).toHaveLength(1);
    expect(originHeaders[0]).toMatchObject({ host: new URL(destination).host, 'x-api-key': 'secret-key' });
    expect(destinationRequests).toEqual([]);
  });

  it.each([false, true])('refuses same-origin 307 redirects (absolute=%s)', async (absolute) => {
    const requests: string[] = [];
    const origin = await listen(createServer((req, res) => {
      requests.push(req.url ?? '');
      req.resume();
      res.writeHead(307, { Location: absolute ? `${origin}/redirected` : '/redirected' });
      res.end();
    }));
    await expectRedirectError(origin, { 'X-API-Key': 'secret-key' });
    expect(requests).toHaveLength(1);
    expect(requests[0]).not.toBe('/redirected');
  });

  it('surfaces a 307 through a loopback HTTP proxy without a second request', async () => {
    const originRequests: string[] = [];
    const origin = await listen(createServer((req, res) => {
      originRequests.push(req.url ?? '');
      res.end('[]');
    }));
    const proxyRequests: Array<{ url?: string; headers: IncomingHttpHeaders }> = [];
    const proxy = await listen(createServer((req, res) => {
      proxyRequests.push({ url: req.url, headers: req.headers });
      req.resume();
      res.writeHead(307, { Location: `${origin}/redirected` });
      res.end();
    }));
    vi.stubEnv('http_proxy', proxy);
    vi.stubEnv('HTTP_PROXY', proxy);
    vi.stubEnv('no_proxy', '');
    vi.stubEnv('NO_PROXY', '');
    await expectRedirectError(origin, { 'X-API-Key': 'secret-key' });
    expect(proxyRequests).toHaveLength(1);
    expect(proxyRequests[0].url).toContain(`${origin}/api/v1/memory/`);
    expect(proxyRequests[0].headers['x-api-key']).toBe('secret-key');
    expect(originRequests).toEqual([]);
  });

  it.each(['Host', 'host', 'hOsT'])('accepts equivalent default-port %s authorities', (name) => {
    const headers = { [name]: 'localhost:80' };
    expect(() => kind === 'memory'
      ? new MemoryClient('http://localhost', headers)
      : new MemoryEventClient('http://localhost', headers)).not.toThrow();
  });
});
