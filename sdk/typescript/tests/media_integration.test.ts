/**
 * Integration tests for the Media Generation milestone (#470).
 *
 * Verifies MediaRouter, OpenRouterMediaProvider, SSRF protection,
 * and error typing — all without live API calls.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MediaRouter, MediaProviderError } from '../src/ai/MediaProvider.js';
import type { MediaProvider, MediaResponse } from '../src/ai/MediaProvider.js';
import { OpenRouterMediaProvider } from '../src/ai/OpenRouterMediaProvider.js';

// ── Test helpers ────────────────────────────────────────────────────────

function makeStubProvider(
  name: string,
  modalities: string[] = ['image', 'audio', 'video']
): MediaProvider {
  const empty: MediaResponse = {
    text: '',
    images: [],
    audio: null,
    files: [],
    videos: [],
    rawResponse: null,
  };
  return {
    name,
    supportedModalities: modalities,
    generateImage: vi.fn().mockResolvedValue(empty),
    generateAudio: vi.fn().mockResolvedValue(empty),
    generateVideo: vi.fn().mockResolvedValue(empty),
  };
}

// ── 1. MediaRouter integration ──���───────────────────────────────────────

describe('Integration: MediaRouter', () => {
  it('registers and resolves single provider by prefix', () => {
    const router = new MediaRouter();
    const prov = makeStubProvider('fal');
    router.register('fal-ai/', prov);
    expect(router.resolve('fal-ai/flux/dev', 'image')).toBe(prov);
  });

  it('longest prefix wins over shorter prefix', () => {
    const router = new MediaRouter();
    const generic = makeStubProvider('generic');
    const specific = makeStubProvider('specific');

    router.register('openrouter/', generic);
    router.register('openrouter/google/', specific);

    expect(router.resolve('openrouter/google/veo-3', 'video')).toBe(specific);
    expect(router.resolve('openrouter/openai/model', 'image')).toBe(generic);
  });

  it('empty prefix acts as catch-all', () => {
    const router = new MediaRouter();
    const fallback = makeStubProvider('fallback');
    router.register('', fallback);
    expect(router.resolve('dall-e-3', 'image')).toBe(fallback);
  });

  it('respects capability filter', () => {
    const router = new MediaRouter();
    const imageOnly = makeStubProvider('imgOnly', ['image']);
    router.register('img/', imageOnly);

    expect(router.resolve('img/model', 'image')).toBe(imageOnly);
    expect(() => router.resolve('img/model', 'video')).toThrow(MediaProviderError);
  });

  it('throws MediaProviderError when no match', () => {
    const router = new MediaRouter();
    expect(() => router.resolve('unknown/model', 'video')).toThrow(MediaProviderError);
    expect(() => router.resolve('unknown/model', 'video')).toThrow(
      "No provider for model 'unknown/model' with 'video' capability"
    );
  });

  it('routes multiple providers correctly', () => {
    const router = new MediaRouter();
    const fal = makeStubProvider('fal', ['image', 'video']);
    const or_ = makeStubProvider('openrouter', ['image', 'audio', 'video']);
    const litellm = makeStubProvider('litellm', ['image', 'audio']);

    router.register('fal-ai/', fal);
    router.register('openrouter/', or_);
    router.register('', litellm);

    expect(router.resolve('fal-ai/flux/dev', 'image')).toBe(fal);
    expect(router.resolve('openrouter/google/veo-3', 'video')).toBe(or_);
    expect(router.resolve('dall-e-3', 'image')).toBe(litellm);
    expect(router.resolve('tts-1', 'audio')).toBe(litellm);
  });
});

// ── 2. OpenRouterMediaProvider integration ───────────────────────────────

describe('Integration: OpenRouterMediaProvider', () => {
  const originalFetch = globalThis.fetch;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe('generateVideo full lifecycle', () => {
    it('submit → poll (pending) → poll (completed) → download', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
      const videoBytes = Buffer.from('fake-video-data');

      // Submit
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-42' }),
      });
      // Poll 1: pending
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-42', status: 'pending' }),
      });
      // Poll 2: completed
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'job-42',
          status: 'completed',
          unsigned_url: 'https://cdn.example.com/video.mp4',
          cost_usd: 0.10,
        }),
      });
      // Download
      mockFetch.mockResolvedValueOnce({
        ok: true,
        arrayBuffer: async () =>
          videoBytes.buffer.slice(
            videoBytes.byteOffset,
            videoBytes.byteOffset + videoBytes.byteLength
          ),
      });

      const resp = await provider.generateVideo({
        prompt: 'A sunset timelapse',
        model: 'openrouter/google/veo-3',
        pollInterval: 1,
      });

      expect(resp.videos).toHaveLength(1);
      expect(resp.videos[0].url).toBe('https://cdn.example.com/video.mp4');
      expect(resp.videos[0].data).toBe(videoBytes.toString('base64'));
      expect(resp.videos[0].costUsd).toBe(0.10);

      // 4 calls: submit, poll*2, download
      expect(mockFetch).toHaveBeenCalledTimes(4);

      // Verify submit body has model with prefix stripped
      const submitBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(submitBody.model).toBe('google/veo-3');
      expect(submitBody.prompt).toBe('A sunset timelapse');
    });

    it('throws on submit failure', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 402,
        text: async () => 'Insufficient credits',
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1 })
      ).rejects.toThrow('Video submit failed');
    });

    it('throws on generation failure status', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-fail' }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-fail', status: 'failed', error: 'policy' }),
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1 })
      ).rejects.toThrow('Video generation failed');
    });

    it('throws on timeout', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-slow' }),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ id: 'job-slow', status: 'pending' }),
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1, timeout: 10 })
      ).rejects.toThrow('timed out');
    });
  });

  describe('generateImage', () => {
    it('sends correct payload and extracts images', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: [
                  { type: 'text', text: 'Your cat image' },
                  { type: 'image_url', image_url: { url: 'https://cdn.example.com/cat.png' } },
                ],
              },
            },
          ],
        }),
      });

      const resp = await provider.generateImage({
        prompt: 'a cat',
        model: 'openrouter/openai/gpt-image-1',
      });

      expect(resp.text).toBe('Your cat image');
      expect(resp.images).toHaveLength(1);
      expect(resp.images[0].url).toBe('https://cdn.example.com/cat.png');

      // Verify model prefix stripped
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.model).toBe('openai/gpt-image-1');
    });

    it('handles base64 data URL images', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: [
                  {
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,iVBOR' },
                  },
                ],
              },
            },
          ],
        }),
      });

      const resp = await provider.generateImage({ prompt: 'test' });
      expect(resp.images[0].b64Json).toBe('iVBOR');
    });

    it('throws MediaProviderError on failure', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        text: async () => 'Rate limited',
      });

      await expect(
        provider.generateImage({ prompt: 'test' })
      ).rejects.toThrow(MediaProviderError);
    });
  });

  describe('generateAudio', () => {
    it('parses SSE stream into audio output', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      const sseLines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"audio":{"data":"AAAA"}}}]}\n\n',
        'data: {"choices":[{"delta":{"audio":{"data":"BBBB"}}}]}\n\n',
        'data: [DONE]\n\n',
      ];
      const encoder = new TextEncoder();
      let callIndex = 0;

      const mockReader = {
        read: vi.fn().mockImplementation(async () => {
          if (callIndex < sseLines.length) {
            const chunk = encoder.encode(sseLines[callIndex]);
            callIndex++;
            return { done: false, value: chunk };
          }
          return { done: true, value: undefined };
        }),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader },
      });

      const resp = await provider.generateAudio({ text: 'say hello', voice: 'nova' });

      expect(resp.text).toBe('Hello');
      expect(resp.audio).not.toBeNull();
      expect(resp.audio!.data).toBe('AAAABBBB');
      expect(resp.audio!.format).toBe('wav');
    });

    it('custom format is respected', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      const sseLines = [
        'data: {"choices":[{"delta":{"audio":{"data":"X"}}}]}\n\n',
        'data: [DONE]\n\n',
      ];
      const encoder = new TextEncoder();
      let callIndex = 0;

      const mockReader = {
        read: vi.fn().mockImplementation(async () => {
          if (callIndex < sseLines.length) {
            const chunk = encoder.encode(sseLines[callIndex]);
            callIndex++;
            return { done: false, value: chunk };
          }
          return { done: true, value: undefined };
        }),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader },
      });

      const resp = await provider.generateAudio({
        text: 'test',
        format: 'mp3',
      });

      expect(resp.audio!.format).toBe('mp3');

      // Verify format was sent in payload
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.audio.format).toBe('mp3');
    });

    it('throws MediaProviderError on HTTP failure', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Internal error',
      });

      await expect(
        provider.generateAudio({ text: 'test' })
      ).rejects.toThrow(MediaProviderError);
    });
  });
});

// ─��� 3. SSRF protection ────────────���─────────────────────────────────────

describe('Integration: SSRF protection', () => {
  const originalFetch = globalThis.fetch;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  const ssrfCases = [
    { label: 'non-HTTPS URL', url: 'http://example.com/video.mp4', match: 'non-HTTPS' },
    { label: 'localhost', url: 'https://localhost/video.mp4', match: 'localhost' },
    { label: '127.0.0.1', url: 'https://127.0.0.1/video.mp4', match: 'localhost' },
    { label: '[::1]', url: 'https://[::1]/video.mp4', match: 'localhost' },
    { label: '10.0.0.1 (private)', url: 'https://10.0.0.1/video.mp4', match: 'private IP' },
    { label: '172.16.0.1 (private)', url: 'https://172.16.0.1/video.mp4', match: 'private IP' },
    { label: '192.168.1.1 (private)', url: 'https://192.168.1.1/video.mp4', match: 'private IP' },
    { label: '169.254.0.1 (link-local)', url: 'https://169.254.0.1/video.mp4', match: 'private IP' },
  ];

  for (const { label, url, match } of ssrfCases) {
    it(`rejects ${label}`, async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      // Submit
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-ssrf' }),
      });
      // Poll -> completed with unsafe URL
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'job-ssrf',
          status: 'completed',
          unsigned_url: url,
        }),
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1 })
      ).rejects.toThrow(match);
    });
  }
});

// ── 4. Error typing ──���────────────────────────��─────────────────────────

describe('Integration: MediaProviderError typing', () => {
  it('is an instance of Error', () => {
    const err = new MediaProviderError('test');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(MediaProviderError);
    expect(err.name).toBe('MediaProviderError');
  });

  it('carries structured context', () => {
    const err = new MediaProviderError('fail', {
      provider: 'openrouter',
      model: 'google/veo-3',
      endpoint: '/api/v1/videos',
    });
    expect(err.provider).toBe('openrouter');
    expect(err.model).toBe('google/veo-3');
    expect(err.endpoint).toBe('/api/v1/videos');
    expect(err.message).toBe('fail');
  });

  it('works with cause option', () => {
    const cause = new Error('root cause');
    const err = new MediaProviderError('wrapped', { cause });
    expect(err.cause).toBe(cause);
  });

  it('thrown by provider on API failure has correct type', async () => {
    const originalFetch = globalThis.fetch;
    const mockFetch = vi.fn();
    globalThis.fetch = mockFetch;

    try {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      });

      try {
        await provider.generateImage({ prompt: 'test' });
        expect.fail('Should have thrown');
      } catch (e) {
        expect(e).toBeInstanceOf(MediaProviderError);
        const err = e as MediaProviderError;
        expect(err.provider).toBe('openrouter');
        expect(err.endpoint).toContain('chat/completions');
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
