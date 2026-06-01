import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { File, createMultimodalResponse } from '../src/index.js';

describe('multimodal helpers', () => {
  let tempDir: string | null = null;

  afterEach(async () => {
    vi.restoreAllMocks();
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it('embeds local files as data URLs', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-'));
    const filePath = join(tempDir, 'sample.txt');
    await writeFile(filePath, 'hello multimodal', 'utf8');

    const file = await File.fromFile(filePath);

    expect(file.file.mimeType).toBe('text/plain');
    expect(file.file.url.startsWith('data:text/plain;base64,')).toBe(true);
  });

  it('saves URL-based multimodal outputs by downloading them', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-'));
    const outputDir = join(tempDir, 'out');
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const href = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (href.endsWith('/image.png')) {
        return new Response(Uint8Array.from([1, 2, 3]), { status: 200 });
      }
      if (href.endsWith('/audio.wav')) {
        return new Response(Uint8Array.from([4, 5, 6]), { status: 200 });
      }
      if (href.endsWith('/artifact.bin')) {
        return new Response(Uint8Array.from([7, 8, 9]), { status: 200 });
      }
      return new Response('not found', { status: 404 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const response = createMultimodalResponse({
      images: [{ url: 'https://example.com/image.png' }],
      audio: { url: 'https://example.com/audio.wav', format: 'wav' },
      files: [{ url: 'https://example.com/artifact.bin', filename: 'artifact.bin' }],
    }, 'saved text');

    const saved = await response.save(outputDir, 'case');

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(await readFile(saved.image_0)).toEqual(Buffer.from([1, 2, 3]));
    expect(await readFile(saved.audio)).toEqual(Buffer.from([4, 5, 6]));
    expect(await readFile(saved.file_0)).toEqual(Buffer.from([7, 8, 9]));
    expect(await readFile(saved.text, 'utf8')).toBe('saved text');
  });
});
