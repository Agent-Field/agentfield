import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  File,
  Video,
  createMultimodalResponse,
  videoFromBase64,
  videoFromBuffer,
  videoFromFile,
  videoFromUrl,
} from '../src/index.js';

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

  it('embeds local videos as video_url data URLs', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-'));
    const filePath = join(tempDir, 'sample.mp4');
    await writeFile(filePath, Buffer.from('\x00\x00\x00\x18ftypmp42'));

    const video = await Video.fromFile(filePath);

    expect(video.type).toBe('video_url');
    expect(video.videoUrl.url.startsWith('data:video/mp4;base64,')).toBe(true);
  });

  it('detects video MIME types for generic file content', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-'));
    const filePath = join(tempDir, 'sample.mov');
    await writeFile(filePath, Buffer.from('mov-data'));

    const file = await File.fromFile(filePath);

    expect(file.file.mimeType).toBe('video/quicktime');
    expect(file.file.url.startsWith('data:video/quicktime;base64,')).toBe(true);
  });

  it('creates videos from URLs, buffers, and base64 data', async () => {
    const urlVideo = Video.fromUrl('https://example.com/video.mp4');
    const bufferVideo = await Video.fromBuffer(Buffer.from('video-data'), 'video/webm');
    const base64Video = await Video.fromBase64('dmllby1kYXRh', 'video/ogg');

    expect(urlVideo.videoUrl.url).toBe('https://example.com/video.mp4');
    expect(bufferVideo.videoUrl.url).toBe('data:video/webm;base64,dmlkZW8tZGF0YQ==');
    expect(base64Video.videoUrl.url).toBe('data:video/ogg;base64,dmllby1kYXRh');
  });

  it('creates videos through convenience factory functions', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-'));
    const filePath = join(tempDir, 'factory.webm');
    await writeFile(filePath, Buffer.from('webm-data'));

    const fileVideo = await videoFromFile(filePath);
    const urlVideo = videoFromUrl('https://example.com/factory.mp4');
    const bufferVideo = await videoFromBuffer(Uint8Array.from([1, 2, 3]), 'video/mp4');
    const base64Video = await videoFromBase64('AQID', 'video/mp4');

    expect(fileVideo.videoUrl.url.startsWith('data:video/webm;base64,')).toBe(true);
    expect(urlVideo.videoUrl.url).toBe('https://example.com/factory.mp4');
    expect(bufferVideo.videoUrl.url).toBe('data:video/mp4;base64,AQID');
    expect(base64Video.videoUrl.url).toBe('data:video/mp4;base64,AQID');
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
