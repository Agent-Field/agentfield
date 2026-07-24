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
import { mergeOpenRouterAttributionHeaders } from './openrouterAttribution.js';

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

const DEFAULT_POLL_INTERVAL = 30_000; // 30s
const DEFAULT_TIMEOUT = 600_000; // 10min

const API_TIMEOUT = 30_000; // 30s for API calls
const DOWNLOAD_TIMEOUT = 120_000; // 120s for video download

const MAX_CONSECUTIVE_PARSE_ERRORS = 50;
const DEFAULT_IMAGE_MODEL = 'google/gemini-3.1-flash-image-preview';
const DEFAULT_TTS_MODEL = 'hexgrad/kokoro-82m';

/** Module-level WeakMap to keep API key off the instance (CR-03). */
const apiKeyStore = new WeakMap<OpenRouterMediaProvider, string>();

/** Per-instance cache of model metadata (output_modalities, input_modalities). */
const modelMetaStore = new WeakMap<
  OpenRouterMediaProvider,
  Map<string, { outputModalities: string[]; inputModalities: string[] }>
>();

function emptyMediaResponse(raw: unknown): MediaResponse {
  return { text: '', images: [], audio: null, files: [], videos: [], rawResponse: raw };
}

function stripPrefix(model: string): string {
  return model.startsWith('openrouter/') ? model.slice('openrouter/'.length) : model;
}

function defaultVoiceForModel(model: string): string {
  return stripPrefix(model) === 'hexgrad/kokoro-82m' ? 'af_alloy' : 'alloy';
}

/**
 * Wrap raw little-endian PCM16 mono bytes in a WAV (RIFF) container.
 * OpenRouter's TTS endpoints emit PCM at 24 kHz; default to that.
 */
function wrapPcm16AsWav(pcm: Uint8Array, sampleRate = 24000): Uint8Array {
  const channels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const dataSize = pcm.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  // RIFF header
  view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // "RIFF"
  view.setUint32(4, 36 + dataSize, true);
  view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // "WAVE"
  // fmt chunk
  view.setUint8(12, 0x66); view.setUint8(13, 0x6d); view.setUint8(14, 0x74); view.setUint8(15, 0x20); // "fmt "
  view.setUint32(16, 16, true);                 // PCM chunk size
  view.setUint16(20, 1, true);                  // PCM format
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  // data chunk
  view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // "data"
  view.setUint32(40, dataSize, true);
  new Uint8Array(buffer, 44).set(pcm);
  return new Uint8Array(buffer);
}

function bytesToBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64');
}

function base64ToBytes(b64: string): Uint8Array {
  return new Uint8Array(Buffer.from(b64, 'base64'));
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
  openRouterSiteUrl?: string;
  openRouterAppName?: string;
  openRouterHeaders?: Record<string, string>;
}

export class OpenRouterMediaProvider implements MediaProvider {
  readonly name = 'openrouter';
  readonly supportedModalities = ['image', 'audio', 'video'];

  private readonly baseUrl: string;
  private readonly attributionHeaders: Record<string, string>;

  constructor(options: OpenRouterMediaProviderOptions = {}) {
    const key = options.apiKey ?? process.env.OPENROUTER_API_KEY ?? '';
    this.baseUrl = options.baseUrl ?? OPENROUTER_BASE;
    this.attributionHeaders = mergeOpenRouterAttributionHeaders(options.openRouterHeaders, {
      siteUrl: options.openRouterSiteUrl,
      appName: options.openRouterAppName,
    });
    if (!key) {
      throw new MediaProviderError('OpenRouter API key required: pass apiKey or set OPENROUTER_API_KEY', {
        provider: 'openrouter',
      });
    }
    apiKeyStore.set(this, key);
    modelMetaStore.set(this, new Map());
  }

  /**
   * Seed the metadata cache for a model. Useful when running against test
   * servers that don't expose `GET /models/{id}/endpoints`, or when callers
   * already know the routing they want.
   *
   * Output modalities follow OpenRouter's convention — `["speech"]` for
   * TTS-only (Kokoro etc.), `["text","audio"]` for chat-audio (gpt-audio
   * family), `["video"]`, `["image"]`, etc.
   */
  seedModelMeta(model: string, outputModalities: string[], inputModalities: string[] = []): void {
    const stripped = stripPrefix(model);
    const cache = modelMetaStore.get(this)!;
    cache.set(stripped, {
      outputModalities: [...outputModalities],
      inputModalities: [...inputModalities],
    });
  }

  /**
   * Fetch + cache OpenRouter model metadata so we can route requests to the
   * right endpoint. On any error returns an empty meta object so callers can
   * fall back to defaults.
   */
  private async fetchModelMeta(model: string): Promise<{
    outputModalities: string[];
    inputModalities: string[];
  }> {
    const stripped = stripPrefix(model);
    const cache = modelMetaStore.get(this)!;
    const cached = cache.get(stripped);
    if (cached) return cached;

    const url = `${this.baseUrl}/models/${stripped}/endpoints`;
    try {
      const res = await this.get(url);
      if (!res.ok) {
        const meta = { outputModalities: [], inputModalities: [] };
        cache.set(stripped, meta);
        return meta;
      }
      const data = (await res.json()) as { data?: { architecture?: { output_modalities?: string[]; input_modalities?: string[] } } };
      const arch = data?.data?.architecture ?? {};
      const meta = {
        outputModalities: arch.output_modalities ?? [],
        inputModalities: arch.input_modalities ?? [],
      };
      cache.set(stripped, meta);
      return meta;
    } catch {
      const meta = { outputModalities: [], inputModalities: [] };
      cache.set(stripped, meta);
      return meta;
    }
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
    if (request.imageUrl) body.image_url = request.imageUrl;
    if (request.frameImages) {
      // Convert TS camelCase to OpenRouter snake_case.
      body.frame_images = request.frameImages.map((fi) => ({
        type: fi.type ?? 'image_url',
        image_url: fi.imageUrl,
        ...(fi.frameType ? { frame_type: fi.frameType } : {}),
      }));
    }
    if (request.inputReferences) {
      body.input_references = request.inputReferences.map((ref) => ({
        type: ref.type ?? 'image_url',
        image_url: ref.imageUrl,
      }));
    }
    if (request.extra) Object.assign(body, request.extra);

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

    // Extract video URL. OpenRouter returns either an array `unsigned_urls`
    // (current API) or a single `unsigned_url` / `url` for legacy responses.
    const unsignedUrls = jobData.unsigned_urls as string[] | undefined;
    const unsignedUrl = jobData.unsigned_url as string | undefined;
    const signedUrl = jobData.url as string | undefined;
    const videoUrl = unsignedUrls?.[0] ?? unsignedUrl ?? signedUrl;

    // Download video bytes if URL available (CR-02: validate URL, redirect: 'error')
    let videoData: string | undefined;
    if (videoUrl) {
      assertSafeUrl(videoUrl);
      // OpenRouter's "unsigned" URLs are served from openrouter.ai itself and
      // require the same Bearer auth as the API; non-openrouter hosts (CDN)
      // accept the URL bare.
      const downloadHeaders: Record<string, string> = {};
      try {
        const host = new URL(videoUrl).hostname.toLowerCase();
        if (host === 'openrouter.ai' || host.endsWith('.openrouter.ai')) {
          const key = apiKeyStore.get(this);
          if (key) downloadHeaders.Authorization = `Bearer ${key}`;
        }
      } catch {
        /* non-URL — leave headers empty */
      }

      const dlRes = await fetch(videoUrl, {
        headers: mergeOpenRouterAttributionHeaders({
          ...this.attributionHeaders,
          ...downloadHeaders,
        }),
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
    const model = stripPrefix(request.model ?? DEFAULT_IMAGE_MODEL);

    // Request only image output — works for both image-only models (e.g.
    // x-ai/grok-imagine-image-quality) and dual-output models. Image-only
    // models return 404 when "text" is also requested.
    let userContent: unknown = request.prompt;
    if (request.imageUrls && request.imageUrls.length > 0) {
      // Multi-modal content array — text + reference images.
      userContent = [
        { type: 'text', text: request.prompt },
        ...request.imageUrls.map((url) => ({
          type: 'image_url',
          image_url: { url },
        })),
      ];
    }
    const messages: unknown[] = [{ role: 'user', content: userContent }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['image'],
    };
    if (request.size) body.size = request.size;
    if (request.quality) body.quality = request.quality;
    if (request.imageConfig) {
      // Convert camelCase keys to OpenRouter snake_case.
      const ic = request.imageConfig;
      const out: Record<string, unknown> = {};
      if (ic.aspectRatio) out.aspect_ratio = ic.aspectRatio;
      if (ic.imageSize) out.image_size = ic.imageSize;
      if (ic.strength != null) out.strength = ic.strength;
      if (ic.style) out.style = ic.style;
      if (ic.rgbColors) out.rgb_colors = ic.rgbColors;
      if (ic.backgroundRgbColor) out.background_rgb_color = ic.backgroundRgbColor;
      if (ic.superResolutionReferences) out.super_resolution_references = ic.superResolutionReferences;
      if (ic.fontInputs) {
        out.font_inputs = ic.fontInputs.map((fi) => ({
          font_url: fi.fontUrl,
          text: fi.text,
        }));
      }
      body.image_config = out;
    }
    if (request.extra) Object.assign(body, request.extra);

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

    // Extract images from choices. OpenRouter places images either inline in
    // `message.content` as multimodal parts (gpt-image-1 style) or in a
    // dedicated `message.images` array (gemini-*-image, grok-imagine style
    // where `content` is null).
    const pushImageFromUrl = (url: string | undefined) => {
      if (!url) return;
      if (url.startsWith('data:')) {
        const b64 = url.split(',', 2)[1];
        resp.images.push({ url, b64Json: b64 });
      } else {
        resp.images.push({ url });
      }
    };
    const choices = data.choices as Array<Record<string, unknown>> | undefined;
    if (choices) {
      for (const choice of choices) {
        const msg = choice.message as Record<string, unknown> | undefined;
        if (!msg) continue;
        // Text
        if (typeof msg.content === 'string') {
          resp.text += msg.content;
        }
        // Content array (gpt-image-1 multimodal style)
        if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            const p = part as Record<string, unknown>;
            if (p.type === 'text') {
              resp.text += p.text as string;
            } else if (p.type === 'image_url') {
              const imgUrl = p.image_url as Record<string, unknown> | undefined;
              pushImageFromUrl(imgUrl?.url as string | undefined);
            }
          }
        }
        // Dedicated images array (gemini-*-image, grok-imagine — content is null)
        const images = msg.images as Array<Record<string, unknown>> | undefined;
        if (Array.isArray(images)) {
          for (const img of images) {
            const imgUrl = img.image_url as Record<string, unknown> | undefined;
            pushImageFromUrl(imgUrl?.url as string | undefined);
          }
        }
      }
    }

    return resp;
  }

  // ── Audio ──────────────────────────────────────────────────────────

  async generateAudio(request: AudioRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? DEFAULT_TTS_MODEL);
    const voice = request.voice ?? defaultVoiceForModel(model);
    const requestedFormat = request.format ?? 'wav';

    // Route based on model capability.
    //   output_modalities=["speech"]  → POST /audio/speech (OpenAI-compat TTS,
    //                                    e.g. hexgrad/kokoro-82m)
    //   contains "audio"              → chat-completions SSE w/ audio modality
    //                                    (e.g. openai/gpt-audio*)
    //   unknown                       → try /audio/speech (broader-compat).
    const meta = await this.fetchModelMeta(model);
    const outMods = meta.outputModalities;
    const useSpeechEndpoint =
      outMods.includes('speech') ||
      (outMods.length === 0) ||
      !outMods.includes('audio');

    if (useSpeechEndpoint) {
      return this.generateAudioViaSpeechEndpoint(
        model,
        request.text,
        voice,
        requestedFormat,
        request
      );
    }

    // Chat-completions audio modality: openai/gpt-audio family. Streaming on
    // OpenAI is locked to pcm16 — wire that and re-wrap to user's format below.
    const wireFormat = requestedFormat === 'wav' ? 'pcm16' : requestedFormat;
    const messages: unknown[] = [{ role: 'user', content: request.text }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['text', 'audio'],
      stream: true,
      audio: {
        voice,
        format: wireFormat,
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
      let b64 = audioChunks.join('');
      // SSE chunks decode independently — concatenate raw bytes for cleaner output.
      try {
        const parts = audioChunks.map(base64ToBytes);
        const total = parts.reduce((n, p) => n + p.byteLength, 0);
        const merged = new Uint8Array(total);
        let off = 0;
        for (const p of parts) { merged.set(p, off); off += p.byteLength; }
        b64 = bytesToBase64(merged);
        if (requestedFormat === 'wav') {
          b64 = bytesToBase64(wrapPcm16AsWav(merged));
        }
      } catch {
        /* fall back to concatenated base64 strings */
      }
      resp.audio = {
        data: b64,
        format: requestedFormat,
      };
    }
    return resp;
  }

  /**
   * Call OpenRouter's OpenAI-compatible TTS endpoint (`POST /audio/speech`).
   * Returns raw bytes for the requested format; wraps PCM → WAV when needed.
   */
  private async generateAudioViaSpeechEndpoint(
    model: string,
    text: string,
    voice: string,
    requestedFormat: string,
    request?: AudioRequest
  ): Promise<MediaResponse> {
    // Map requested format → upstream response_format. Kokoro etc. only
    // emit pcm/mp3; we wrap pcm into WAV ourselves when caller asked for wav.
    const wireFormat =
      requestedFormat === 'wav' || requestedFormat === 'pcm' || requestedFormat === 'pcm16'
        ? 'pcm'
        : requestedFormat;

    const endpoint = `${this.baseUrl}/audio/speech`;
    const body: Record<string, unknown> = {
      model,
      input: text,
      voice,
      response_format: wireFormat,
    };
    if (request?.speed != null) body.speed = request.speed;
    if (request?.extra) Object.assign(body, request.extra);
    const res = await this.post(endpoint, body);
    if (!res.ok) {
      throw new MediaProviderError(
        `Audio generation failed [model=${model}] [endpoint=${endpoint}]: ${res.status} ${await res.text()}`,
        { provider: 'openrouter', model, endpoint }
      );
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    const finalBytes = requestedFormat === 'wav' ? wrapPcm16AsWav(buf) : buf;
    const resp = emptyMediaResponse({
      endpoint: 'audio/speech',
      model,
      mime_type: res.headers.get('content-type') ?? '',
    });
    resp.text = text;
    resp.audio = {
      data: bytesToBase64(finalBytes),
      format: requestedFormat,
    };
    return resp;
  }

  // ── Helpers ────────────────────────────────────────────────────────

  private post(url: string, body: unknown): Promise<Response> {
    const key = apiKeyStore.get(this);
    return fetch(url, {
      method: 'POST',
      headers: mergeOpenRouterAttributionHeaders({
        ...this.attributionHeaders,
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`,
      }),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(API_TIMEOUT),
    });
  }

  private get(url: string): Promise<Response> {
    const key = apiKeyStore.get(this);
    return fetch(url, {
      method: 'GET',
      headers: mergeOpenRouterAttributionHeaders({
        ...this.attributionHeaders,
        Authorization: `Bearer ${key}`,
      }),
      signal: AbortSignal.timeout(API_TIMEOUT),
    });
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
