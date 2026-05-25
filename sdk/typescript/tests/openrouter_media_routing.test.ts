/**
 * Coverage for new OpenRouter routing / param-translation logic in
 * OpenRouterMediaProvider:
 *   - fetchModelMeta (HTTP success/cache/error/exception paths)
 *   - generateImage: imageUrls multi-part content, imageConfig snake_case mapping
 *   - generateVideo: imageUrl, frame_images snake_case, input_references, extra
 *   - generateAudio: /audio/speech path with speed + extra
 */
import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  vi,
} from 'vitest';
import { OpenRouterMediaProvider } from '../src/ai/OpenRouterMediaProvider.js';

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('OpenRouterMediaProvider: model metadata fetch', () => {
  it('caches metadata after first GET', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: {
          id: 'openai/gpt-audio-mini',
          architecture: {
            output_modalities: ['text', 'audio'],
            input_modalities: ['text'],
          },
        },
      }),
    });
    // For both audio calls below — they hit chat-completions SSE.
    const sseBody = {
      getReader: () => {
        let done = false;
        return {
          read: async () => {
            if (done) return { done: true, value: undefined };
            done = true;
            const enc = new TextEncoder();
            return { done: false, value: enc.encode('data: [DONE]\n\n') };
          },
        };
      },
    };
    mockFetch.mockResolvedValueOnce({ ok: true, body: sseBody });
    mockFetch.mockResolvedValueOnce({ ok: true, body: sseBody });

    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    await provider.generateAudio({
      text: 'a', model: 'openrouter/openai/gpt-audio-mini', format: 'mp3',
    });
    await provider.generateAudio({
      text: 'b', model: 'openrouter/openai/gpt-audio-mini', format: 'mp3',
    });
    // 1 metadata GET + 2 audio POSTs = 3 calls
    expect(mockFetch).toHaveBeenCalledTimes(3);
    const firstUrl = String(mockFetch.mock.calls[0][0]);
    expect(firstUrl).toContain('/models/openai/gpt-audio-mini/endpoints');
  });

  it('falls back to /audio/speech when metadata GET returns 500', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => 'audio/mpeg' },
      arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      text: async () => '',
    });

    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    const resp = await provider.generateAudio({
      text: 'x', model: 'unknown/tts', format: 'mp3',
    });
    expect(resp.audio).not.toBeNull();
    expect(String(mockFetch.mock.calls[1][0])).toContain('/audio/speech');
  });

  it('falls back to /audio/speech when metadata GET throws', async () => {
    mockFetch.mockRejectedValueOnce(new Error('network'));
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => 'audio/mpeg' },
      arrayBuffer: async () => new Uint8Array([1]).buffer,
      text: async () => '',
    });
    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    await provider.generateAudio({ text: 'x', model: 'unknown/y', format: 'mp3' });
    expect(String(mockFetch.mock.calls[1][0])).toContain('/audio/speech');
  });
});

describe('OpenRouterMediaProvider.generateVideo: param translation', () => {
  it('maps imageUrl, frameImages, inputReferences, extra to OpenRouter schema', async () => {
    // 1) submit, 2) poll → completed, 3) download
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 'jobZ' }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'jobZ',
        status: 'completed',
        unsigned_urls: ['https://cdn.example.com/v.mp4'],
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      arrayBuffer: async () => new Uint8Array([0x66, 0x66]).buffer,
    });

    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    await provider.generateVideo({
      prompt: 'x',
      model: 'openrouter/google/veo-3.1-lite',
      duration: 4,
      imageUrl: 'https://example.com/seed.jpg',
      frameImages: [
        { imageUrl: { url: 'https://x/first.jpg' }, frameType: 'first_frame' },
        { imageUrl: { url: 'https://x/last.jpg' }, frameType: 'last_frame' },
      ],
      inputReferences: [{ imageUrl: { url: 'https://x/ref.jpg' } }],
      extra: { personGeneration: 'allow_all' },
      pollInterval: 1,
      timeout: 10_000,
    });

    const submitBody = JSON.parse(mockFetch.mock.calls[0][1].body as string);
    expect(submitBody.image_url).toBe('https://example.com/seed.jpg');
    expect(submitBody.frame_images).toEqual([
      {
        type: 'image_url',
        image_url: { url: 'https://x/first.jpg' },
        frame_type: 'first_frame',
      },
      {
        type: 'image_url',
        image_url: { url: 'https://x/last.jpg' },
        frame_type: 'last_frame',
      },
    ]);
    expect(submitBody.input_references).toEqual([
      { type: 'image_url', image_url: { url: 'https://x/ref.jpg' } },
    ]);
    expect(submitBody.personGeneration).toBe('allow_all');
  });
});

describe('OpenRouterMediaProvider.generateImage: imageUrls + imageConfig', () => {
  it('builds multi-part user content and snake_cases imageConfig', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: null,
              images: [
                {
                  type: 'image_url',
                  image_url: { url: 'data:image/png;base64,QUJD' },
                },
              ],
            },
          },
        ],
      }),
    });

    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    const resp = await provider.generateImage({
      prompt: 'fox in watercolor',
      model: 'openrouter/x-ai/grok-imagine-image-quality',
      imageUrls: ['https://r/1.png', 'https://r/2.png'],
      imageConfig: {
        aspectRatio: '16:9',
        imageSize: '1024x576',
        strength: 0.6,
        style: 'painterly',
        rgbColors: [[255, 0, 0]],
        backgroundRgbColor: [0, 0, 0],
        superResolutionReferences: ['https://r/sr.png'],
        fontInputs: [{ fontUrl: 'https://f.com/x.ttf', text: 'hi' }],
      },
      extra: { quality_boost: true },
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string);
    // Multi-part content
    expect(body.messages[0].content).toEqual([
      { type: 'text', text: 'fox in watercolor' },
      { type: 'image_url', image_url: { url: 'https://r/1.png' } },
      { type: 'image_url', image_url: { url: 'https://r/2.png' } },
    ]);
    // imageConfig snake_case translation
    expect(body.image_config).toEqual({
      aspect_ratio: '16:9',
      image_size: '1024x576',
      strength: 0.6,
      style: 'painterly',
      rgb_colors: [[255, 0, 0]],
      background_rgb_color: [0, 0, 0],
      super_resolution_references: ['https://r/sr.png'],
      font_inputs: [{ font_url: 'https://f.com/x.ttf', text: 'hi' }],
    });
    expect(body.quality_boost).toBe(true);
    expect(body.modalities).toEqual(['image']);
    // Returned image base64 captured from data: URL
    expect(resp.images[0].b64Json).toBe('QUJD');
  });
});

describe('OpenRouterMediaProvider.generateAudio: /audio/speech extras', () => {
  it('passes speed and extra through to the speech body', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => 'audio/mpeg' },
      arrayBuffer: async () => new Uint8Array([1, 2]).buffer,
      text: async () => '',
    });

    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    provider.seedModelMeta('openai/gpt-4o-mini-tts', ['speech'], ['text']);
    await provider.generateAudio({
      text: 'bonjour',
      model: 'openai/gpt-4o-mini-tts',
      voice: 'alloy',
      format: 'mp3',
      speed: 1.5,
      extra: { language: 'fr-FR' },
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string);
    expect(body.speed).toBe(1.5);
    expect(body.language).toBe('fr-FR');
    expect(body.response_format).toBe('mp3');
    expect(body.voice).toBe('alloy');
  });

  it('throws MediaProviderError when /audio/speech returns 401', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => 'unauthorized',
    });
    const provider = new OpenRouterMediaProvider({ apiKey: 'k' });
    provider.seedModelMeta('hexgrad/kokoro-82m', ['speech'], ['text']);
    await expect(
      provider.generateAudio({
        text: 'x', model: 'hexgrad/kokoro-82m', voice: 'af_bella', format: 'wav',
      })
    ).rejects.toThrow('Audio generation failed');
  });
});
