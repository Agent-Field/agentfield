import http, { type IncomingMessage } from 'node:http';
import https from 'node:https';

function requestResponse(url: URL): Promise<IncomingMessage> {
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new Error(`Unsupported CodeAF download protocol: ${url.protocol}`);
  }
  return new Promise((resolve, reject) => {
    const request = (url.protocol === 'https:' ? https : http).get(url, resolve);
    request.on('error', reject);
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
