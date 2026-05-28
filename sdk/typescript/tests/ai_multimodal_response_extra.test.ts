import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  Audio,
  File,
  Image,
  MultimodalResponse,
  Text,
  audioFromBase64,
  audioFromBuffer,
  audioFromFile,
  audioFromUrl,
  createMultimodalResponse,
  fileFromBase64,
  fileFromBuffer,
  fileFromPath,
  fileFromUrl,
  imageFromBase64,
  imageFromBuffer,
  imageFromFile,
  imageFromUrl,
  text
} from '../src/index.js';

describe('multimodal helpers extra coverage', () => {
  let tempDir: string | null = null;

  afterEach(async () => {
    vi.restoreAllMocks();
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it('creates text, image, audio, and file content from exported helpers', async () => {
    const txt = text('hello');
    expect(txt).toBeInstanceOf(Text);
    expect(txt.type).toBe('text');
    expect(txt.text).toBe('hello');

    const imageBuffer = await imageFromBuffer(Uint8Array.from([1, 2, 3]), 'image/png', 'low');
    expect(imageBuffer).toBeInstanceOf(Image);
    expect(imageBuffer.imageUrl).toEqual({
      url: `data:image/png;base64,${Buffer.from([1, 2, 3]).toString('base64')}`,
      detail: 'low'
    });

    const imageB64 = await imageFromBase64('YWJj', undefined, 'auto');
    expect(imageB64.imageUrl).toEqual({
      url: 'data:image/jpeg;base64,YWJj',
      detail: 'auto'
    });

    const imageUrl = imageFromUrl('https://example.com/image.webp');
    expect(imageUrl.imageUrl.url).toBe('https://example.com/image.webp');
    expect(imageUrl.imageUrl.detail).toBe('high');

    const audioBuffer = await audioFromBuffer(Buffer.from([4, 5]), 'mp3');
    expect(audioBuffer).toBeInstanceOf(Audio);
    expect(audioBuffer.audio).toEqual({
      data: Buffer.from([4, 5]).toString('base64'),
      format: 'mp3'
    });

    const audioB64 = await audioFromBase64('ZGF0YQ==', 'ogg');
    expect(audioB64.audio).toEqual({ data: 'ZGF0YQ==', format: 'ogg' });

    const fileBuffer = await fileFromBuffer(Uint8Array.from([9, 8]), 'application/pdf');
    expect(fileBuffer).toBeInstanceOf(File);
    expect(fileBuffer.file).toEqual({
      url: `data:application/pdf;base64,${Buffer.from([9, 8]).toString('base64')}`,
      mimeType: 'application/pdf'
    });

    const fileB64 = await fileFromBase64('Zm9v', 'text/plain');
    expect(fileB64.file).toEqual({
      url: 'data:text/plain;base64,Zm9v',
      mimeType: 'text/plain'
    });

    const fileUrl = fileFromUrl('https://example.com/report.txt', 'text/plain');
    expect(fileUrl.file).toEqual({
      url: 'https://example.com/report.txt',
      mimeType: 'text/plain'
    });
  });

  it('detects MIME types and formats from local file extensions', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-multimodal-extra-'));
    const imagePath = join(tempDir, 'diagram.PNG');
    const audioPath = join(tempDir, 'clip.mp3');
    const docPath = join(tempDir, 'report.unknown');
    await writeFile(imagePath, Buffer.from([1, 2, 3]));
    await writeFile(audioPath, Buffer.from([4, 5, 6]));
    await writeFile(docPath, Buffer.from([7, 8, 9]));

    const image = await Image.fromFile(imagePath);
    expect(image.imageUrl.url).toBe(`data:image/png;base64,${Buffer.from([1, 2, 3]).toString('base64')}`);
    expect(image.imageUrl.detail).toBe('high');

    const audio = await audioFromFile(audioPath);
    expect(audio.audio).toEqual({
      data: Buffer.from([4, 5, 6]).toString('base64'),
      format: 'mp3'
    });

    const file = await fileFromPath(docPath);
    expect(file.file.mimeType).toBe('application/octet-stream');
    expect(file.file.url).toBe(`data:application/octet-stream;base64,${Buffer.from([7, 8, 9]).toString('base64')}`);

    const imageViaHelper = await imageFromFile(imagePath, 'auto');
    expect(imageViaHelper.imageUrl.detail).toBe('auto');
  });

  it('downloads audio from URLs and normalizes fetch failures', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(Uint8Array.from([10, 11]), { status: 200 }))
      .mockResolvedValueOnce(new Response('nope', { status: 404, statusText: 'Not Found' }))
      .mockRejectedValueOnce(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    const success = await audioFromUrl('https://example.com/audio.wav', 'wav');
    expect(success.audio).toEqual({
      data: Buffer.from([10, 11]).toString('base64'),
      format: 'wav'
    });

    await expect(audioFromUrl('https://example.com/missing.wav', 'wav')).rejects.toThrow(
      'Failed to fetch audio from URL: 404 Not Found'
    );

    await expect(audioFromUrl('https://example.com/network.wav', 'wav')).rejects.toThrow(
      'URL download requires a fetch-compatible environment'
    );
  });
});

describe('MultimodalResponse extra coverage', () => {
  let tempDir: string | null = null;

  afterEach(async () => {
    vi.restoreAllMocks();
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it('reports content presence and preserves metadata', () => {
    const raw = { provider: 'mock' };
    const response = new MultimodalResponse(
      'hello world',
      { data: 'YQ==', format: 'mp3' },
      [{ url: 'https://example.com/image.png' }],
      [{ url: 'https://example.com/file.bin', filename: 'file.bin' }],
      raw,
      0.12,
      { promptTokens: 1, completionTokens: 2, totalTokens: 3 }
    );

    expect(response.text).toBe('hello world');
    expect(response.audio).toEqual({ data: 'YQ==', format: 'mp3' });
    expect(response.images).toHaveLength(1);
    expect(response.files).toHaveLength(1);
    expect(response.hasAudio()).toBe(true);
    expect(response.hasImage()).toBe(true);
    expect(response.hasFile()).toBe(true);
    expect(response.isMultimodal()).toBe(true);
    expect(response.rawResponse).toBe(raw);
    expect(response.costUsd).toBe(0.12);
    expect(response.usage).toEqual({ promptTokens: 1, completionTokens: 2, totalTokens: 3 });
    expect(response.getAudio()).toEqual({ data: 'YQ==', format: 'mp3' });
    expect(response.getImages()).toEqual([{ url: 'https://example.com/image.png' }]);
    expect(response.getFiles()).toEqual([{ url: 'https://example.com/file.bin', filename: 'file.bin' }]);
    expect(response.toString()).toBe('hello world');
    expect(response.toJSON()).toEqual({
      text: 'hello world',
      audio: { data: 'YQ==', format: 'mp3' },
      images: [{ url: 'https://example.com/image.png' }],
      files: [{ url: 'https://example.com/file.bin', filename: 'file.bin' }],
      _debug: 'MultimodalResponse(audio=mp3, images=1, files=1)'
    });
  });

  it('saves data-url and base64 assets to disk and returns paths', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-response-extra-'));
    const outputDir = join(tempDir, 'nested');
    const response = new MultimodalResponse(
      'saved text',
      { data: Buffer.from([1, 2]).toString('base64'), format: 'wav' },
      [{ url: `data:image/png;base64,${Buffer.from([3, 4]).toString('base64')}` }],
      [{ data: Buffer.from([5, 6]).toString('base64'), filename: 'report.bin' }]
    );

    const saved = await response.save(outputDir, 'case');

    expect(await readFile(saved.audio)).toEqual(Buffer.from([1, 2]));
    expect(await readFile(saved.image_0)).toEqual(Buffer.from([3, 4]));
    expect(await readFile(saved.file_0)).toEqual(Buffer.from([5, 6]));
    expect(await readFile(saved.text, 'utf8')).toBe('saved text');
    expect(saved.image_0.endsWith('case_image_0.png')).toBe(true);
    expect(saved.audio.endsWith('case_audio.wav')).toBe(true);
    expect(saved.file_0.endsWith('report.bin')).toBe(true);
  });

  it('throws clear errors for invalid or missing multimodal data', async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'agentfield-response-extra-'));
    const response = new MultimodalResponse();

    await expect(response.saveImage({ url: 'data:image/png;base64' }, join(tempDir, 'image.png'))).rejects.toThrow(
      'Invalid data URL'
    );
    await expect(response.saveImage({}, join(tempDir, 'image.png'))).rejects.toThrow('No image data available');
    await expect(response.saveAudio({ format: 'wav' }, join(tempDir, 'audio.wav'))).rejects.toThrow(
      'No audio data available'
    );
    await expect(response.saveFile({}, join(tempDir, 'file.bin'))).rejects.toThrow('No file data available');
  });

  it('creates multimodal responses from heterogeneous raw payload shapes', () => {
    const raw = {
      images: [
        'https://example.com/one.png',
        { image_url: { url: `data:image/png;base64,${Buffer.from('img').toString('base64')}` } },
        { b64_json: Buffer.from('two').toString('base64'), revised_prompt: 'revise' }
      ],
      audio: {
        data: Buffer.from('audio').toString('base64'),
        format: 'mp3',
        url: 'https://example.com/audio.mp3'
      },
      files: [
        { url: 'https://example.com/file.txt', mimeType: 'text/plain', filename: 'file.txt' },
        { data: Buffer.from('blob').toString('base64'), filename: 'blob.bin' }
      ]
    };

    const response = createMultimodalResponse(raw, 'body');

    expect(response.text).toBe('body');
    expect(response.images).toEqual([
      { url: 'https://example.com/one.png' },
      {
        url: `data:image/png;base64,${Buffer.from('img').toString('base64')}`,
        b64Json: Buffer.from('img').toString('base64')
      },
      { b64Json: Buffer.from('two').toString('base64'), revisedPrompt: 'revise' }
    ]);
    expect(response.audio).toEqual({
      data: Buffer.from('audio').toString('base64'),
      format: 'mp3',
      url: 'https://example.com/audio.mp3'
    });
    expect(response.files).toEqual([
      { url: 'https://example.com/file.txt', mimeType: 'text/plain', filename: 'file.txt' },
      { data: Buffer.from('blob').toString('base64'), filename: 'blob.bin' }
    ]);
  });

  it('extracts alternate image and audio structures with defaults', () => {
    const imageOnly = createMultimodalResponse(
      { image_url: { url: 'https://example.com/direct.jpg' } },
      'img'
    );
    expect(imageOnly.images).toEqual([{ url: 'https://example.com/direct.jpg' }]);

    const audioOnly = createMultimodalResponse(
      { input_audio: { data: Buffer.from('voice').toString('base64') } },
      'voice'
    );
    expect(audioOnly.audio).toEqual({
      data: Buffer.from('voice').toString('base64'),
      format: 'wav'
    });
  });
});
