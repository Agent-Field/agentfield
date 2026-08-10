import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  Audio,
  File,
  Image,
  Text,
  audioFromFile,
  audioFromUrl,
  fileFromPath,
  imageFromBuffer,
  imageFromFile,
  text,
} from '../src/index.js';

describe('multimodal helpers comprehensive coverage', () => {
  let tempDir: string | null = null;

  async function writeFixture(filename: string, contents: Uint8Array): Promise<string> {
    tempDir ??= await mkdtemp(join(tmpdir(), 'agentfield-multimodal-comprehensive-'));
    const filePath = join(tempDir, filename);
    await writeFile(filePath, contents);
    return filePath;
  }

  afterEach(async () => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it('constructs text content without changing its value', () => {
    const direct = new Text('  hello\nworld  ');
    const fromHelper = text('');

    expect(direct).toBeInstanceOf(Text);
    expect(direct).toEqual({ type: 'text', text: '  hello\nworld  ' });
    expect(fromHelper).toEqual({ type: 'text', text: '' });
  });

  it('imageFromFile encodes image bytes with a case-insensitive MIME type', async () => {
    const bytes = Uint8Array.from([0x89, 0x50, 0x4e, 0x47]);
    const filePath = await writeFixture('sample.PNG', bytes);

    const image = await imageFromFile(filePath, 'auto');

    expect(image).toBeInstanceOf(Image);
    expect(image.type).toBe('image_url');
    expect(image.imageUrl).toEqual({
      url: `data:image/png;base64,${Buffer.from(bytes).toString('base64')}`,
      detail: 'auto',
    });
  });

  it('Image.fromFile falls back to JPEG and high detail for an unknown extension', async () => {
    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const filePath = await writeFixture('sample.unknown', bytes);

    const image = await Image.fromFile(filePath);

    expect(image.imageUrl).toEqual({
      url: `data:image/jpeg;base64,${Buffer.from(bytes).toString('base64')}`,
      detail: 'high',
    });
  });

  it('creates images from buffers, base64, and URLs', async () => {
    const bufferImage = await imageFromBuffer(Uint8Array.from([1, 2, 3]), 'image/webp', 'low');
    const base64Image = await Image.fromBase64('AQID');
    const urlImage = Image.fromUrl('https://example.com/image.png', 'auto');

    expect(bufferImage.imageUrl).toEqual({
      url: 'data:image/webp;base64,AQID',
      detail: 'low',
    });
    expect(base64Image.imageUrl).toEqual({
      url: 'data:image/jpeg;base64,AQID',
      detail: 'high',
    });
    expect(urlImage.imageUrl).toEqual({
      url: 'https://example.com/image.png',
      detail: 'auto',
    });
  });

  it('Audio.fromBuffer creates input_audio content and honors the requested format', async () => {
    const bytes = Uint8Array.from([4, 5, 6]);

    const audio = await Audio.fromBuffer(bytes, 'flac');

    expect(audio).toBeInstanceOf(Audio);
    expect(audio.type).toBe('input_audio');
    expect(audio.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'flac',
    });
  });

  it('Audio.fromBase64 preserves data and defaults to WAV', async () => {
    const audio = await Audio.fromBase64('BAUG');

    expect(audio.audio).toEqual({ data: 'BAUG', format: 'wav' });
  });

  it('audioFromFile round-trips bytes and detects an uppercase extension', async () => {
    const bytes = Uint8Array.from([0x49, 0x44, 0x33, 0x04]);
    const filePath = await writeFixture('sample.MP3', bytes);

    const audio = await audioFromFile(filePath);

    expect(audio).toBeInstanceOf(Audio);
    expect(audio.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'mp3',
    });
  });

  it('Audio.fromFile uses WAV for unknown extensions unless a format is supplied', async () => {
    const bytes = Uint8Array.from([7, 8, 9]);
    const filePath = await writeFixture('sample.aac', bytes);

    const fallback = await Audio.fromFile(filePath);
    const explicit = await Audio.fromFile(filePath, 'ogg');

    expect(fallback.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'wav',
    });
    expect(explicit.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'ogg',
    });
  });

  it('audioFromUrl fetches audio bytes and converts them to base64', async () => {
    const bytes = Uint8Array.from([10, 11, 12]);
    const fetchMock = vi.fn(async () => new Response(bytes, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const audio = await audioFromUrl('https://example.com/audio.ogg', 'ogg');

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith('https://example.com/audio.ogg');
    expect(audio.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'ogg',
    });
  });

  it('Audio.fromUrl defaults to WAV', async () => {
    const bytes = Uint8Array.from([10, 11, 12]);
    const fetchMock = vi.fn(async () => new Response(bytes, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const audio = await Audio.fromUrl('https://example.com/audio.wav');

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith('https://example.com/audio.wav');
    expect(audio.audio).toEqual({
      data: Buffer.from(bytes).toString('base64'),
      format: 'wav',
    });
  });

  it('Audio.fromUrl reports unsuccessful HTTP responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 503, statusText: 'Service Unavailable' }))
    );

    await expect(Audio.fromUrl('https://example.com/unavailable.wav')).rejects.toThrow(
      'Failed to fetch audio from URL: 503 Service Unavailable'
    );
  });

  it('Audio.fromUrl normalizes fetch TypeErrors and preserves other failures', async () => {
    const otherFailure = new Error('connection closed');
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('fetch failed'))
      .mockRejectedValueOnce(otherFailure);
    vi.stubGlobal('fetch', fetchMock);

    await expect(Audio.fromUrl('https://example.com/network.wav')).rejects.toThrow(
      'URL download requires a fetch-compatible environment'
    );
    await expect(Audio.fromUrl('https://example.com/error.wav')).rejects.toBe(otherFailure);
  });

  it.each([
    ['report.pdf', 'application/pdf'],
    ['report.DOCX', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    ['image.png', 'image/png'],
    ['audio.MP3', 'audio/mpeg'],
    ['README', 'application/octet-stream'],
  ])('infers %s as %s when creating generic file content', async (filename, mimeType) => {
    const bytes = Uint8Array.from([13, 14, 15]);
    const filePath = await writeFixture(filename, bytes);

    const file = await fileFromPath(filePath);

    expect(file).toBeInstanceOf(File);
    expect(file.type).toBe('file');
    expect(file.file).toEqual({
      url: `data:${mimeType};base64,${Buffer.from(bytes).toString('base64')}`,
      mimeType,
    });
  });
});
