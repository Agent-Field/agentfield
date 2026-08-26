/**
 * Chat-completions audio transport selection.
 *
 * OpenRouter only streams pcm16 audio deltas — asking for mp3/flac/opus with
 * stream=true is rejected upstream — so every other wire format must be
 * requested with stream=false and read from the single JSON body (#584).
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

/** Provider routed to chat-completions (gpt-audio family). */
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

describe('generateAudio transport selection (#584)', () => {
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

  it('keeps pcm16 on the streaming path', async () => {
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
      format: 'pcm16',
    });

    expect(sentBody().stream).toBe(true);
    expect(resp.text).toBe('Hi');
    expect(resp.audio!.data).toBe('AAAA');
    expect(resp.audio!.format).toBe('pcm16');
  });

  for (const format of ['mp3', 'flac', 'opus']) {
    it(`requests ${format} without streaming and parses the JSON completion`, async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: async () =>
          JSON.stringify({
            choices: [
              {
                message: {
                  role: 'assistant',
                  content: null,
                  audio: { data: `audio-${format}`, transcript: 'spoken words' },
                },
              },
            ],
          }),
      });

      const resp = await chatAudioProvider().generateAudio({
        text: 'hello',
        model: 'openai/gpt-audio-mini',
        voice: 'nova',
        format,
      });

      const body = sentBody();
      expect(body.stream).toBe(false);
      expect(body.audio.format).toBe(format);

      expect(resp.audio!.data).toBe(`audio-${format}`);
      expect(resp.audio!.format).toBe(format);
      expect(resp.text).toBe('spoken words');
    });
  }

  it('falls back to message content when the completion has no transcript', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        JSON.stringify({
          choices: [{ message: { content: 'plain text', audio: { data: 'YQ==' } } }],
        }),
    });

    const resp = await chatAudioProvider().generateAudio({
      text: 'hello',
      model: 'openai/gpt-audio-mini',
      format: 'mp3',
    });

    expect(resp.text).toBe('plain text');
    expect(resp.audio!.data).toBe('YQ==');
  });

  it('surfaces an upstream failure with status and body', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      text: async () => '{"error":{"message":"audio.format does not support mp3"}}',
    });

    await expect(
      chatAudioProvider().generateAudio({
        text: 'hello',
        model: 'openai/gpt-audio-mini',
        format: 'mp3',
      })
    ).rejects.toThrow(/400.*audio\.format does not support mp3/);
  });

  it('reports a non-JSON body as a parse failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '<html>gateway timeout</html>',
    });

    await expect(
      chatAudioProvider().generateAudio({
        text: 'hello',
        model: 'openai/gpt-audio-mini',
        format: 'flac',
      })
    ).rejects.toThrow('was not valid JSON');
  });

  it('returns an empty response when the completion carries no audio', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => JSON.stringify({ choices: [] }),
    });

    const resp = await chatAudioProvider().generateAudio({
      text: 'hello',
      model: 'openai/gpt-audio-mini',
      format: 'opus',
    });

    expect(resp.audio).toBeNull();
    expect(resp.text).toBe('');
  });

  it('refuses a body whose declared Content-Length exceeds the cap', async () => {
    const text = vi.fn();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: (h: string) => (h === 'content-length' ? String(200 * 1024 * 1024) : null) },
      text,
    });

    await expect(
      chatAudioProvider().generateAudio({
        text: 'hello',
        model: 'openai/gpt-audio-mini',
        format: 'mp3',
      })
    ).rejects.toThrow(MediaProviderError);
    expect(text).not.toHaveBeenCalled(); // rejected before buffering the body
  });

  it('stops reading an undeclared body once it runs past the cap', async () => {
    // The chunk reports its size without allocating 120 MB: the guard must fire
    // on the byte count, before any of the body is buffered.
    const chunk = { byteLength: 120 * 1024 * 1024 };
    let reads = 0;
    const reader = {
      read: vi.fn().mockImplementation(async () => {
        reads++;
        return { done: false, value: chunk };
      }),
      cancel: vi.fn(),
    };
    mockFetch.mockResolvedValueOnce({ ok: true, body: { getReader: () => reader } });

    await expect(
      chatAudioProvider().generateAudio({
        text: 'hello',
        model: 'openai/gpt-audio-mini',
        format: 'mp3',
      })
    ).rejects.toThrow(/exceeds the \d+ byte limit/);
    expect(reads).toBe(1); // stopped on the first oversized chunk, not drained
    expect(reader.cancel).toHaveBeenCalled();
  });
});
