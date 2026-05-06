/**
 * OpenRouter-backed MediaProvider implementation.
 * Supports video generation (async job), image generation, and audio generation via SSE.
 */

import type {
  MediaProvider,
  MediaResponse,
  VideoRequest,
  ImageRequest,
  AudioRequest,
} from './MediaProvider.js';
import { MediaProviderError } from './MediaProvider.js';

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

const DEFAULT_POLL_INTERVAL = 30_000; // 30s
const DEFAULT_TIMEOUT = 600_000; // 10min

const API_TIMEOUT = 30_000; // 30s for API calls
const DOWNLOAD_TIMEOUT = 120_000; // 120s for video download

const MAX_CONSECUTIVE_PARSE_ERRORS = 50;

/** Module-level WeakMap to keep API key off the instance (CR-03). */
const apiKeyStore = new WeakMap<OpenRouterMediaProvider, string>();

function emptyMediaResponse(raw: unknown): MediaResponse {
  return { text: '', images: [], audio: null, files: [], videos: [], rawResponse: raw };
}

function stripPrefix(model: string): string {
  return model.startsWith('openrouter/') ? model.slice('openrouter/'.length) : model;
}

/**
 * Validate a URL is safe to download from (CR-02 — SSRF protection).
 * Rejects non-https, localhost, and private/reserved IP ranges.
 */
function assertSafeUrl(urlStr: string): void {
  let parsed: URL;
  try {
    parsed = new URL(urlStr);
  } catch {
    throw new MediaProviderError(`Invalid download URL: ${urlStr}`);
  }

  if (parsed.protocol !== 'https:') {
    throw new MediaProviderError(
      `Refusing to download from non-HTTPS URL: ${urlStr}`
    );
  }

  const host = parsed.hostname.toLowerCase();

  // Block localhost variants
  if (
    host === 'localhost' ||
    host === '127.0.0.1' ||
    host === '[::1]' ||
    host === '::1' ||
    host === '0.0.0.0'
  ) {
    throw new MediaProviderError(`Refusing to download from localhost: ${urlStr}`);
  }

  // Block private IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x)
  const ipv4Match = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (ipv4Match) {
    const [, a, b] = ipv4Match.map(Number);
    if (
      a === 10 ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 169 && b === 254) ||
      a === 0
    ) {
      throw new MediaProviderError(
        `Refusing to download from private IP: ${urlStr}`
      );
    }
  }
}

export interface OpenRouterMediaProviderOptions {
  apiKey?: string;
  baseUrl?: string;
}

export class OpenRouterMediaProvider implements MediaProvider {
  readonly name = 'openrouter';
  readonly supportedModalities = ['image', 'audio', 'video'];

  private readonly baseUrl: string;

  constructor(options: OpenRouterMediaProviderOptions = {}) {
    const key = options.apiKey ?? process.env.OPENROUTER_API_KEY ?? '';
    this.baseUrl = options.baseUrl ?? OPENROUTER_BASE;
    if (!key) {
      throw new MediaProviderError('OpenRouter API key required: pass apiKey or set OPENROUTER_API_KEY', {
        provider: 'openrouter',
      });
    }
    apiKeyStore.set(this, key);
  }

  /** Prevent API key from leaking via JSON.stringify (CR-03). */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      supportedModalities: this.supportedModalities,
      baseUrl: this.baseUrl,
    };
  }

  // ── Video ──────────────────────────────────────────────────────────

  async generateVideo(request: VideoRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'google/veo-3');
    const pollInterval = request.pollInterval ?? DEFAULT_POLL_INTERVAL;
    const timeout = request.timeout ?? DEFAULT_TIMEOUT;

    // Build request body
    const body: Record<string, unknown> = {
      model,
      prompt: request.prompt,
    };
    if (request.duration != null) body.duration = request.duration;
    if (request.resolution) body.resolution = request.resolution;
    if (request.aspectRatio) body.aspect_ratio = request.aspectRatio;
    if (request.generateAudio != null) body.generate_audio = request.generateAudio;
    if (request.seed != null) body.seed = request.seed;
    if (request.frameImages) body.frame_images = request.frameImages;
    if (request.inputReferences) body.input_references = request.inputReferences;

    const submitEndpoint = `${this.baseUrl}/videos`;

    // Submit job
    const submitRes = await this.post(submitEndpoint, body);
    if (!submitRes.ok) {
      throw new MediaProviderError(
        `Video submit failed [model=${model}] [endpoint=${submitEndpoint}]: ${submitRes.status} ${await submitRes.text()}`,
        { provider: 'openrouter', model, endpoint: submitEndpoint }
      );
    }
    const submitData = (await submitRes.json()) as Record<string, unknown>;
    const jobId = submitData.id as string;
    if (!jobId) {
      throw new MediaProviderError('No job id returned from video submit', {
        provider: 'openrouter',
        model,
        endpoint: submitEndpoint,
      });
    }

    // Poll until done (WR-01: check deadline AFTER sleep, use Math.min for sleep)
    const deadline = Date.now() + timeout;
    let jobData: Record<string, unknown> = {};
    const pollEndpoint = `${this.baseUrl}/videos/${jobId}`;

    while (true) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await sleep(Math.min(pollInterval, remaining));
      if (Date.now() >= deadline) break;

      const pollRes = await this.get(pollEndpoint);
      if (!pollRes.ok) {
        throw new MediaProviderError(
          `Video poll failed [model=${model}] [endpoint=${pollEndpoint}]: ${pollRes.status} ${await pollRes.text()}`,
          { provider: 'openrouter', model, endpoint: pollEndpoint }
        );
      }
      jobData = (await pollRes.json()) as Record<string, unknown>;
      const status = jobData.status as string | undefined;
      if (status === 'completed') break;
      if (status === 'failed' || status === 'error') {
        throw new MediaProviderError(
          `Video generation failed [model=${model}]: ${JSON.stringify(jobData)}`,
          { provider: 'openrouter', model }
        );
      }
    }

    if ((jobData.status as string) !== 'completed') {
      throw new MediaProviderError(
        `Video generation timed out [model=${model}] after ${timeout}ms`,
        { provider: 'openrouter', model }
      );
    }

    // Extract video URL
    const unsignedUrl = jobData.unsigned_url as string | undefined;
    const signedUrl = jobData.url as string | undefined;
    const videoUrl = unsignedUrl ?? signedUrl;

    // Download video bytes if URL available (CR-02: validate URL, redirect: 'error')
    let videoData: string | undefined;
    if (videoUrl) {
      assertSafeUrl(videoUrl);
      const dlRes = await fetch(videoUrl, {
        signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT),
        redirect: 'error',
      });
      if (dlRes.ok) {
        const buf = Buffer.from(await dlRes.arrayBuffer());
        videoData = buf.toString('base64');
      }
    }

    const resp = emptyMediaResponse(jobData);
    resp.videos.push({
      url: videoUrl,
      data: videoData,
      mimeType: 'video/mp4',
      duration: request.duration,
      resolution: request.resolution,
      aspectRatio: request.aspectRatio,
      hasAudio: request.generateAudio,
      costUsd: jobData.cost_usd as number | undefined,
    });
    return resp;
  }

  // ── Image ──────────────────────────────────────────────────────────

  async generateImage(request: ImageRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'openai/gpt-image-1');

    const messages: unknown[] = [{ role: 'user', content: request.prompt }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['image', 'text'],
    };
    if (request.size) body.size = request.size;
    if (request.quality) body.quality = request.quality;
    if (request.imageConfig) body.image_config = request.imageConfig;

    const endpoint = `${this.baseUrl}/chat/completions`;
    const res = await this.post(endpoint, body);
    if (!res.ok) {
      throw new MediaProviderError(
        `Image generation failed [model=${model}] [endpoint=${endpoint}]: ${res.status} ${await res.text()}`,
        { provider: 'openrouter', model, endpoint }
      );
    }
    const data = (await res.json()) as Record<string, unknown>;
    const resp = emptyMediaResponse(data);

    // Extract images from choices
    const choices = data.choices as Array<Record<string, unknown>> | undefined;
    if (choices) {
      for (const choice of choices) {
        const msg = choice.message as Record<string, unknown> | undefined;
        if (!msg) continue;
        // Text
        if (typeof msg.content === 'string') {
          resp.text += msg.content;
        }
        // Content array (multimodal)
        if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            const p = part as Record<string, unknown>;
            if (p.type === 'text') {
              resp.text += p.text as string;
            } else if (p.type === 'image_url') {
              const imgUrl = p.image_url as Record<string, unknown> | undefined;
              const url = imgUrl?.url as string | undefined;
              if (url?.startsWith('data:')) {
                const b64 = url.split(',', 2)[1];
                resp.images.push({ url, b64Json: b64 });
              } else if (url) {
                resp.images.push({ url });
              }
            }
          }
        }
      }
    }

    return resp;
  }

  // ── Audio ──────────────────────────────────────────────────────────

  async generateAudio(request: AudioRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'openai/gpt-4o-mini-tts');

    const messages: unknown[] = [{ role: 'user', content: request.text }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['text', 'audio'],
      stream: true,
      audio: {
        voice: request.voice ?? 'alloy',
        format: request.format ?? 'wav',
      },
    };

    const endpoint = `${this.baseUrl}/chat/completions`;
    const res = await this.post(endpoint, body);
    if (!res.ok) {
      throw new MediaProviderError(
        `Audio generation failed [model=${model}] [endpoint=${endpoint}]: ${res.status} ${await res.text()}`,
        { provider: 'openrouter', model, endpoint }
      );
    }

    // Parse SSE stream and collect audio chunks
    const audioChunks: string[] = [];
    let textContent = '';
    const reader = res.body?.getReader();
    if (!reader) {
      throw new MediaProviderError('No response body stream available', {
        provider: 'openrouter',
        model,
        endpoint,
      });
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let consecutiveParseErrors = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // Keep last incomplete line in buffer
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') continue;

        try {
          const chunk = JSON.parse(payload) as Record<string, unknown>;
          consecutiveParseErrors = 0; // reset on success
          const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
          if (!choices) continue;
          for (const choice of choices) {
            const delta = choice.delta as Record<string, unknown> | undefined;
            if (!delta) continue;
            if (typeof delta.content === 'string') {
              textContent += delta.content;
            }
            const audioDelta = delta.audio as Record<string, unknown> | undefined;
            if (audioDelta?.data) {
              audioChunks.push(audioDelta.data as string);
            }
          }
        } catch {
          consecutiveParseErrors++;
          if (consecutiveParseErrors > MAX_CONSECUTIVE_PARSE_ERRORS) {
            throw new MediaProviderError(
              `Too many consecutive SSE parse errors (>${MAX_CONSECUTIVE_PARSE_ERRORS}) [model=${model}]`,
              { provider: 'openrouter', model, endpoint }
            );
          }
        }
      }
    }

    // WR-02: Process remaining buffer after reader loop ends
    if (buffer.trim()) {
      const remaining = buffer.trim();
      if (remaining.startsWith('data:')) {
        const payload = remaining.slice(5).trim();
        if (payload && payload !== '[DONE]') {
          try {
            const chunk = JSON.parse(payload) as Record<string, unknown>;
            const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
            if (choices) {
              for (const choice of choices) {
                const delta = choice.delta as Record<string, unknown> | undefined;
                if (!delta) continue;
                if (typeof delta.content === 'string') {
                  textContent += delta.content;
                }
                const audioDelta = delta.audio as Record<string, unknown> | undefined;
                if (audioDelta?.data) {
                  audioChunks.push(audioDelta.data as string);
                }
              }
            }
          } catch {
            // final chunk malformed — ignore
          }
        }
      }
    }

    const resp = emptyMediaResponse(null);
    resp.text = textContent;
    if (audioChunks.length > 0) {
      resp.audio = {
        data: audioChunks.join(''),
        format: request.format ?? 'wav',
      };
    }
    return resp;
  }

  // ── Helpers ────────────────────────────────────────────────────────

  private post(url: string, body: unknown): Promise<Response> {
    const key = apiKeyStore.get(this);
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(API_TIMEOUT),
    });
  }

  private get(url: string): Promise<Response> {
    const key = apiKeyStore.get(this);
    return fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${key}`,
      },
      signal: AbortSignal.timeout(API_TIMEOUT),
    });
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
