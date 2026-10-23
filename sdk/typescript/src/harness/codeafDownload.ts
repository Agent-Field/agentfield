import http, { type IncomingMessage } from 'node:http';
import https from 'node:https';
import net from 'node:net';
import type { Duplex } from 'node:stream';
import tls from 'node:tls';

function proxyFor(url: URL): URL | undefined {
  const hostname = url.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  const port = url.port || (url.protocol === 'https:' ? '443' : '80');
  const noProxy = process.env.no_proxy || process.env.NO_PROXY || '';
  for (const entry of noProxy.toLowerCase().split(/[\s,]+/).filter(Boolean)) {
    if (entry === '*') return undefined;
    const match = entry.match(/^(\[[^\]]+\]|[^:]+)(?::(\d+))?$/);
    if (!match || (match[2] && match[2] !== port)) continue;
    const host = match[1].replace(/^\*?\./, '').replace(/^\[|\]$/g, '');
    if (hostname === host || hostname.endsWith(`.${host}`)) return undefined;
  }
  const value = url.protocol === 'https:'
    ? process.env.https_proxy || process.env.HTTPS_PROXY
    : process.env.http_proxy || process.env.HTTP_PROXY;
  if (!value) return undefined;
  const proxy = new URL(value);
  if (proxy.protocol !== 'http:' && proxy.protocol !== 'https:') {
    throw new Error(`Unsupported CodeAF proxy protocol: ${proxy.protocol}`);
  }
  return proxy;
}

function proxyOptions(proxy: URL): http.RequestOptions {
  const headers: http.OutgoingHttpHeaders = {};
  if (proxy.username || proxy.password) {
    const credentials = `${decodeURIComponent(proxy.username)}:${decodeURIComponent(proxy.password)}`;
    headers['Proxy-Authorization'] = `Basic ${Buffer.from(credentials).toString('base64')}`;
  }
  return { protocol: proxy.protocol, hostname: proxy.hostname.replace(/^\[|\]$/g, ''), port: proxy.port || undefined, headers };
}

function tunnel(url: URL, proxy: URL): Promise<Duplex> {
  return new Promise((resolve, reject) => {
    const client = proxy.protocol === 'https:' ? https : http;
    const authority = `${url.hostname}:${url.port || '443'}`;
    const options = proxyOptions(proxy);
    const request = client.request({ ...options, method: 'CONNECT', path: authority,
      headers: { ...options.headers, Host: authority }, agent: false });
    request.on('error', reject);
    request.setTimeout(30_000, () => request.destroy(new Error('CodeAF proxy connection timed out')));
    request.on('connect', (response, socket, head) => {
      request.setTimeout(0);
      if (response.statusCode !== 200) {
        socket.destroy();
        reject(new Error(`CodeAF proxy CONNECT failed: HTTP ${response.statusCode}`));
        return;
      }
      if (head.length) socket.unshift(head);
      resolve(socket);
    });
    request.end();
  });
}

async function requestResponse(url: URL): Promise<IncomingMessage> {
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new Error(`Unsupported CodeAF download protocol: ${url.protocol}`);
  }
  const proxy = proxyFor(url);
  let agent: https.Agent | undefined;
  if (proxy && url.protocol === 'https:') {
    const socket = await tunnel(url, proxy);
    const hostname = url.hostname.replace(/^\[|\]$/g, '');
    agent = new https.Agent({ keepAlive: false });
    agent.createConnection = () => tls.connect({ socket, host: hostname,
      servername: net.isIP(hostname) ? undefined : hostname });
  }
  return new Promise<IncomingMessage>((resolve, reject) => {
    const onResponse = (response: IncomingMessage): void => {
      response.once('close', () => agent?.destroy());
      resolve(response);
    };
    const options = proxy ? proxyOptions(proxy) : undefined;
    const request = proxy && url.protocol === 'http:'
      ? (proxy.protocol === 'https:' ? https : http).get({ ...options, path: url.href,
        headers: { ...options?.headers, Host: url.host } }, onResponse)
      : (url.protocol === 'https:' ? https : http).get(url, { agent }, onResponse);
    request.on('error', (error) => {
      agent?.destroy();
      reject(error);
    });
    request.setTimeout(30_000, () => request.destroy(new Error('CodeAF download timed out')));
  });
}

export async function codeafDownloadResponse(url: URL, redirects = 0): Promise<IncomingMessage> {
  const response = await requestResponse(url);
  if ([301, 302, 303, 307, 308].includes(response.statusCode ?? 0) && response.headers.location) {
    response.resume();
    if (redirects >= 5) throw new Error('Too many CodeAF download redirects');
    return codeafDownloadResponse(new URL(response.headers.location, url), redirects + 1);
  }
  if (response.statusCode !== 200) {
    response.resume();
    throw new Error(`CodeAF download failed: HTTP ${response.statusCode}`);
  }
  return response;
}
