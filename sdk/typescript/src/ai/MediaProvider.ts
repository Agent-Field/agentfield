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

export interface VideoRequest {
  prompt: string;
  model?: string;
  duration?: number;
  resolution?: '480p' | '720p' | '1080p' | '1K' | '2K' | '4K';
  aspectRatio?: '16:9' | '9:16' | '1:1' | '4:3' | '3:4' | '21:9' | '9:21';
  generateAudio?: boolean;
  seed?: number;
  frameImages?: Array<{ type: string; imageUrl: { url: string }; frameType?: string }>;
  inputReferences?: Array<{ type: string; imageUrl: { url: string } }>;
  pollInterval?: number; // ms, default 30000
  timeout?: number; // ms, default 600000
}

export interface ImageRequest {
  prompt: string;
  model?: string;
  size?: string;
  quality?: string;
  imageConfig?: {
    aspectRatio?: string;
    imageSize?: string;
    superResolutionReferences?: string[];
    fontInputs?: Array<{ fontUrl: string; text: string }>;
  };
}

export interface AudioRequest {
  text: string;
  model?: string;
  voice?: string;
  format?: string;
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
