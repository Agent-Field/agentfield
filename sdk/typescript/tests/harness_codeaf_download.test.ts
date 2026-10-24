import { EventEmitter } from 'node:events';
import type { ClientRequest, IncomingMessage } from 'node:http';
import https from 'node:https';
import { PassThrough } from 'node:stream';
import { afterEach, expect, it, vi } from 'vitest';
import { codeafDownloadResponse } from '../src/harness/codeafDownload.js';

afterEach(() => vi.restoreAllMocks());

it('uses https.get for a release URL and keeps the 30 second socket timeout', async () => {
  const request = new EventEmitter() as ClientRequest;
  request.setTimeout = vi.fn().mockReturnValue(request);
  const get = vi.spyOn(https, 'get').mockReturnValue(request);
  const url = new URL('https://releases.invalid/codeaf');
  const pending = codeafDownloadResponse(url);
  const response = Object.assign(new PassThrough(), { statusCode: 200, headers: {} }) as IncomingMessage;
  const callback = get.mock.calls[0].at(-1) as (response: IncomingMessage) => void;
  callback(response);
  await expect(pending).resolves.toBe(response);
  expect(get).toHaveBeenCalledWith(url, expect.any(Function));
  expect(request.setTimeout).toHaveBeenCalledWith(30_000, expect.any(Function));
  response.destroy();
});

it('destroys a stalled HTTPS request with an actionable download timeout', async () => {
  const request = new EventEmitter() as ClientRequest;
  request.setTimeout = vi.fn().mockReturnValue(request);
  request.destroy = vi.fn((error) => {
    request.emit('error', error);
    return request;
  });
  vi.spyOn(https, 'get').mockReturnValue(request);
  const pending = codeafDownloadResponse(new URL('https://releases.invalid/codeaf'));
  const timeout = vi.mocked(request.setTimeout).mock.calls[0][1]!;
  timeout();
  await expect(pending).rejects.toThrow('CodeAF download timed out');
  expect(request.destroy).toHaveBeenCalledOnce();
});
