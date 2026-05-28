/**
 * MediaProvider interface and MediaRouter for multi-provider media generation.
 * Mirrors the Python SDK's MediaProvider abstraction.
 */

/** Custom error class for media provider failures with structured context. */
export class MediaProviderError extends Error {
  readonly provider?: string;
  readonly model?: string;
  readonly endpoint?: string;

  constructor(
    message: string,
    options?: { provider?: string; model?: string; endpoint?: string; cause?: unknown }
  ) {
    super(message, options?.cause ? { cause: options.cause } : undefined);
    this.name = 'MediaProviderError';
    this.provider = options?.provider;
    this.model = options?.model;
    this.endpoint = options?.endpoint;
  }
}

/** Frame guidance for image-to-video models (e.g. Veo). */
export interface VideoFrameImage {
  /** Image content type — usually "image_url". */
  type?: string;
  /** Image URL or `data:` URL. */
  imageUrl: { url: string };
  /** Which frame this image controls. */
  frameType?: 'first_frame' | 'last_frame';
}

/** Reference image for style / subject guidance (Veo "reference-to-video"). */
export interface VideoInputReference {
  type?: string;
  imageUrl: { url: string };
}

export interface VideoRequest {
  prompt: string;
  model?: string;
  /** Duration in seconds (model-dependent — typically 4, 6, or 8). */
  duration?: number;
  resolution?: '480p' | '720p' | '1080p' | '1K' | '2K' | '4K';
  aspectRatio?: '16:9' | '9:16' | '1:1' | '4:3' | '3:4' | '21:9' | '9:21';
  /** Toggle synchronized audio track (when model supports it). */
  generateAudio?: boolean;
  seed?: number;
  /** Single input image for image-to-video (legacy convenience field). */
  imageUrl?: string;
  /** Per-frame guidance — first_frame / last_frame. Takes precedence over `imageUrl`. */
  frameImages?: VideoFrameImage[];
  /** Reference images for style/subject guidance. */
  inputReferences?: VideoInputReference[];
  /** Model-specific passthrough parameters (e.g. Veo's `personGeneration`). */
  extra?: Record<string, unknown>;
  pollInterval?: number; // ms, default 30000
  timeout?: number; // ms, default 600000
}

export interface ImageRequest {
  prompt: string;
  model?: string;
  size?: string;
  quality?: string;
  /** Reference / source image(s) for image+text→image models (e.g. grok-imagine). */
  imageUrls?: string[];
  imageConfig?: {
    aspectRatio?: string;
    imageSize?: string;
    /** Image-to-image blend strength (model-dependent, 0–1). */
    strength?: number;
    /** Style hint — Recraft V3 etc. */
    style?: string;
    /** RGB color palette — array of [r,g,b]. */
    rgbColors?: number[][];
    /** Background color hint as [r,g,b]. */
    backgroundRgbColor?: number[];
    superResolutionReferences?: string[];
    fontInputs?: Array<{ fontUrl: string; text: string }>;
  };
  /** Model-specific passthrough parameters. */
  extra?: Record<string, unknown>;
}

export interface AudioRequest {
  text: string;
  model?: string;
  voice?: string;
  format?: string;
  /** Playback speed multiplier (OpenAI TTS only — other models ignore). */
  speed?: number;
  /** Model-specific passthrough parameters. */
  extra?: Record<string, unknown>;
}

export interface MediaResponse {
  text: string;
  images: Array<{ url?: string; b64Json?: string; revisedPrompt?: string }>;
  audio: { data?: string; format: string; url?: string } | null;
  files: Array<{ url?: string; data?: string; mimeType?: string; filename?: string }>;
  videos: Array<{
    url?: string;
    data?: string;
    mimeType?: string;
    filename?: string;
    duration?: number;
    resolution?: string;
    aspectRatio?: string;
    hasAudio?: boolean;
    costUsd?: number;
  }>;
  rawResponse: unknown;
}

export interface MediaProvider {
  readonly name: string;
  readonly supportedModalities: string[];
  generateImage(request: ImageRequest): Promise<MediaResponse>;
  generateAudio(request: AudioRequest): Promise<MediaResponse>;
  generateVideo(request: VideoRequest): Promise<MediaResponse>;
}

/**
 * Prefix-based media provider router.
 * Resolves model strings to providers by longest-prefix match.
 */
export class MediaRouter {
  private providers: Array<{ prefix: string; provider: MediaProvider }> = [];

  register(prefix: string, provider: MediaProvider): void {
    this.providers.push({ prefix, provider });
    // Sort longest prefix first for greedy matching
    this.providers.sort((a, b) => b.prefix.length - a.prefix.length);
  }

  resolve(model: string, capability: string): MediaProvider {
    for (const { prefix, provider } of this.providers) {
      if (model.startsWith(prefix) && provider.supportedModalities.includes(capability)) {
        return provider;
      }
    }
    throw new MediaProviderError(
      `No provider for model '${model}' with '${capability}' capability`,
      { model }
    );
  }
}
