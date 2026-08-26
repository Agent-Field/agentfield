/**
 * Chat-completions audio format support (#584).
 *
 * OpenRouter's chat-completions audio modality only ever delivers pcm16:
 * OpenAI rejects a non-pcm16 `audio.format` while streaming, and the OpenRouter
 * gateway rejects an audio completion that is not streamed at all. pcm16 — and
 * wav, which is pcm16 re-wrapped into a RIFF/WAVE container client-side — are
 * the only formats this route can serve; anything else must fail locally,
 * before a request is spent on a 400.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenRouterMediaProvider } from '../src/ai/OpenRouterMediaProvider.js';
import { MediaProviderError } from '../src/ai/MediaProvider.js';

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** Provider routed to chat-completions (gpt-audio family), metadata pre-seeded. */
function chatAudioProvider(): OpenRouterMediaProvider {
  const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
  provider.seedModelMeta('openai/gpt-audio-mini', ['text', 'audio'], ['text']);
  return provider;
}

/** A fetch result whose body streams the given chunks, like an SSE response. */
function streamingResponse(chunks: string[]) {
  const encoder = new TextEncoder();
  let i = 0;
  const reader = {
    read: vi.fn().mockImplementation(async () => {
      if (i < chunks.length) return { done: false, value: encoder.encode(chunks[i++]) };
      return { done: true, value: undefined };
    }),
  };
  return { ok: true, body: { getReader: () => reader } };
}

function sentBody(call = 0): Record<string, any> {
  return JSON.parse(mockFetch.mock.calls[call][1].body);
}

describe('generateAudio format support (#584)', () => {
  it('wires wav as streamed pcm16 and returns a RIFF/WAVE container', async () => {
    mockFetch.mockResolvedValueOnce(
      streamingResponse([
        'data: {"choices":[{"delta":{"audio":{"data":"AAAAAAAA"}}}]}\n\n',
        'data: [DONE]\n\n',
      ])
    );

    const resp = await chatAudioProvider().generateAudio({
      text: 'hello',
      model: 'openai/gpt-audio-mini',
      voice: 'nova',
      format: 'wav',
    });

    const body = sentBody();
    expect(body.stream).toBe(true);
    expect(body.audio.format).toBe('pcm16');

    expect(resp.audio!.format).toBe('wav');
    const wav = Buffer.from(resp.audio!.data!, 'base64');
    expect(wav.subarray(0, 4).toString()).toBe('RIFF');
    expect(wav.subarray(8, 12).toString()).toBe('WAVE');
  });

  it('keeps pcm16 on the streaming path with the caller voice', async () => {
    mockFetch.mockResolvedValueOnce(
      streamingResponse([
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        'data: {"choices":[{"delta":{"audio":{"data":"AAAA"}}}]}\n\n',
        'data: [DONE]\n\n',
      ])
    );

    const resp = await chatAudioProvider().generateAudio({
      text: 'hello',
      model: 'openai/gpt-audio-mini',
      voice: 'nova',
      format: 'pcm16',
    });

    const body = sentBody();
    expect(body.stream).toBe(true);
    expect(body.audio).toEqual({ voice: 'nova', format: 'pcm16' });

    expect(resp.text).toBe('Hi');
    expect(resp.audio!.data).toBe('AAAA');
    expect(resp.audio!.format).toBe('pcm16');
  });

  for (const format of ['mp3', 'flac', 'opus']) {
    it(`refuses ${format} before issuing a request`, async () => {
      const err: unknown = await chatAudioProvider()
        .generateAudio({
          text: 'hello',
          model: 'openai/gpt-audio-mini',
          voice: 'nova',
          format,
        })
        .then(
          () => new Error('expected generateAudio to reject'),
          (e: unknown) => e
        );

      expect(err).toBeInstanceOf(MediaProviderError);
      // The message must name what was asked for and what is actually served.
      const message = (err as Error).message;
      expect(message).toContain(`"${format}" is not available`);
      expect(message).toContain('pcm16');
      expect(message).toContain('wav');
      expect(mockFetch).not.toHaveBeenCalled();
    });
  }
});

describe('generateAudio SSE body decoding', () => {
  /** A real Response over a ReadableStream, so production reads res.body. */
  function streamedResponse(chunks: Uint8Array[]): Response {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    });
    return new Response(stream, { status: 200 });
  }

  it('reassembles a multi-byte character split across body chunks', async () => {
    // Transcript text whose non-ASCII characters straddle chunk boundaries: a
    // per-chunk decode would emit U+FFFD where the byte sequence was cut.
    const spoken = 'répétez après moi 🎧';
    const sse =
      `data: ${JSON.stringify({ choices: [{ delta: { content: spoken, audio: { data: 'AAAA' } } }] })}\n\n` +
      'data: [DONE]\n\n';
    const bytes = new TextEncoder().encode(sse);
    // Slice into 7-byte chunks — small enough that the 2-byte é and the 4-byte
    // emoji are guaranteed to be cut mid-sequence.
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < bytes.length; i += 7) chunks.push(bytes.slice(i, i + 7));
    expect(chunks.length).toBeGreaterThan(1);

    mockFetch.mockResolvedValueOnce(streamedResponse(chunks));

    const resp = await chatAudioProvider().generateAudio({
      text: 'hello',
      model: 'openai/gpt-audio-mini',
      format: 'pcm16',
    });

    expect(resp.text).toBe(spoken);
    expect(resp.text).not.toContain('�');
    expect(resp.audio!.data).toBe('AAAA');
  });
});
