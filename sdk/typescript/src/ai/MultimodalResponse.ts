/**
 * Multimodal response classes for handling LLM multimodal outputs.
 * Provides seamless integration with audio, image, and file outputs while maintaining backward compatibility.
 */

import { promises as fs } from 'fs';
import * as path from 'path';

/**
 * Represents image output from LLM with convenient access methods.
 */
export interface ImageOutput {
  /** URL to image */
  url?: string;
  /** Base64-encoded image data */
  b64Json?: string;
  /** Revised prompt used for generation */
  revisedPrompt?: string;
}

/**
 * Represents audio output from LLM with convenient access methods.
 */
export interface AudioOutput {
  /** Base64-encoded audio data */
  data?: string;
  /** Audio format (wav, mp3, etc.) */
  format: string;
  /** URL to audio file if available */
  url?: string;
}

/**
 * Represents generic file output from LLM.
 */
export interface FileOutput {
  /** URL to file */
  url?: string;
  /** Base64-encoded file data */
  data?: string;
  /** MIME type of file */
  mimeType?: string;
  /** Suggested filename */
  filename?: string;
}

/**
 * Enhanced response object that provides seamless access to multimodal content
 * while maintaining backward compatibility with string responses.
 */
export class MultimodalResponse {
  private _text: string;
  private _audio: AudioOutput | null;
  private _images: ImageOutput[];
  private _files: FileOutput[];
  private _rawResponse: unknown;
  private _costUsd: number | null;
  private _usage: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };

  constructor(
    text: string = '',
    audio: AudioOutput | null = null,
    images: ImageOutput[] = [],
    files: FileOutput[] = [],
    rawResponse: unknown = null,
    costUsd: number | null = null,
    usage: { promptTokens?: number; completionTokens?: number; totalTokens?: number } = {}
  ) {
    this._text = text;
    this._audio = audio;
    this._images = images;
    this._files = files;
    this._rawResponse = rawResponse;
    this._costUsd = costUsd;
    this._usage = usage;
  }

  /**
   * Get text content.
   */
  get text(): string {
    return this._text;
  }

  /**
   * Get audio output if available.
   */
  get audio(): AudioOutput | null {
    return this._audio;
  }

  /**
   * Get list of image outputs.
   */
  get images(): ImageOutput[] {
    return this._images;
  }

  /**
   * Get list of file outputs.
   */
  get files(): FileOutput[] {
    return this._files;
  }

  /**
   * Check if response contains audio.
   */
  hasAudio(): boolean {
    return this._audio !== null;
  }

  /**
   * Check if response contains images.
   */
  hasImage(): boolean {
    return this._images.length > 0;
  }

  /**
   * Check if response contains files.
   */
  hasFile(): boolean {
    return this._files.length > 0;
  }

  /**
   * Check if response contains any multimodal content.
   */
  isMultimodal(): boolean {
    return this.hasAudio() || this.hasImage() || this.hasFile();
  }

  /**
   * Get the raw LLM response object.
   */
  get rawResponse(): unknown {
    return this._rawResponse;
  }

  /**
   * Estimated cost of this LLM call in USD, if available.
   */
  get costUsd(): number | null {
    return this._costUsd;
  }

  /**
   * Token usage breakdown.
   */
  get usage(): { promptTokens?: number; completionTokens?: number; totalTokens?: number } {
    return this._usage;
  }

  /**
   * Get all images.
   * Alias for images property for API consistency.
   */
  getImages(): ImageOutput[] {
    return this._images;
  }

  /**
   * Get audio.
   * Alias for audio property for API consistency.
   */
  getAudio(): AudioOutput | null {
    return this._audio;
  }

  /**
   * Get all files.
   * Alias for files property for API consistency.
   */
  getFiles(): FileOutput[] {
    return this._files;
  }

  /**
   * Get raw image bytes from an ImageOutput.
   */
  private decodeBase64(data: string): Uint8Array {
    return new Uint8Array(Buffer.from(data, 'base64'));
  }

  private async getUrlBytes(sourceUrl: string): Promise<Uint8Array> {
    if (sourceUrl.startsWith('data:')) {
      const base64Data = sourceUrl.split(',', 2)[1];
      if (!base64Data) {
        throw new Error('Invalid data URL');
      }
      return this.decodeBase64(base64Data);
    }

    const response = await fetch(sourceUrl);
    if (!response.ok) {
      throw new Error(`Failed to download multimodal asset: ${response.status} ${response.statusText}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  }

  private async getImageBytes(image: ImageOutput): Promise<Uint8Array> {
    if (image.b64Json) {
      return this.decodeBase64(image.b64Json);
    } else if (image.url) {
      return this.getUrlBytes(image.url);
    }
    throw new Error('No image data available');
  }

  /**
   * Get raw audio bytes from an AudioOutput.
   */
  private async getAudioBytes(audio: AudioOutput): Promise<Uint8Array> {
    if (audio.data) {
      return this.decodeBase64(audio.data);
    }
    if (audio.url) {
      return this.getUrlBytes(audio.url);
    }
    throw new Error('No audio data available');
  }

  /**
   * Get raw file bytes from a FileOutput.
   */
  private async getFileBytes(file: FileOutput): Promise<Uint8Array> {
    if (file.data) {
      return this.decodeBase64(file.data);
    }
    if (file.url) {
      return this.getUrlBytes(file.url);
    }
    throw new Error('No file data available');
  }

  /**
   * Save a single image to file.
   */
  async saveImage(image: ImageOutput, imagePath: string): Promise<void> {
    const bytes = await this.getImageBytes(image);
    await fs.mkdir(path.dirname(imagePath), { recursive: true });
    await fs.writeFile(imagePath, bytes);
  }

  /**
   * Save a single audio to file.
   */
  async saveAudio(audio: AudioOutput, audioPath: string): Promise<void> {
    const bytes = await this.getAudioBytes(audio);
    await fs.mkdir(path.dirname(audioPath), { recursive: true });
    await fs.writeFile(audioPath, bytes);
  }

  /**
   * Save a single file to disk.
   */
  async saveFile(file: FileOutput, filePath: string): Promise<void> {
    const bytes = await this.getFileBytes(file);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, bytes);
  }

  /**
   * Save all multimodal content to a directory.
   * Returns a dict mapping content type to saved file paths.
   */
  async save(outputDir: string, prefix: string = 'output'): Promise<Record<string, string>> {
    const savedFiles: Record<string, string> = {};

    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    // Save audio
    if (this._audio) {
      const audioPath = path.join(outputDir, `${prefix}_audio.${this._audio.format}`);
      await this.saveAudio(this._audio, audioPath);
      savedFiles['audio'] = audioPath;
    }

    // Save images
    for (let i = 0; i < this._images.length; i++) {
      const image = this._images[i];
      // Determine extension from URL or default to png
      let ext = 'png';
      if (image.url) {
        const urlExt = path.extname(image.url).slice(1);
        if (urlExt) ext = urlExt;
      }
      const imagePath = path.join(outputDir, `${prefix}_image_${i}.${ext}`);
      await this.saveImage(image, imagePath);
      savedFiles[`image_${i}`] = imagePath;
    }

    // Save files
    for (let i = 0; i < this._files.length; i++) {
      const file = this._files[i];
      const filename = file.filename || `${prefix}_file_${i}`;
      const filePath = path.join(outputDir, filename);
      await this.saveFile(file, filePath);
      savedFiles[`file_${i}`] = filePath;
    }

    // Save text content
    if (this._text) {
      const textPath = path.join(outputDir, `${prefix}_text.txt`);
      await fs.writeFile(textPath, this._text, 'utf-8');
      savedFiles['text'] = textPath;
    }

    return savedFiles;
  }

  /**
   * String representation for backward compatibility.
   */
  toString(): string {
    return this._text;
  }

  /**
   * Developer-friendly representation.
   */
  toJSON(): object {
    const parts: string[] = [];
    if (this._audio) parts.push(`audio=${this._audio.format}`);
    if (this._images.length > 0) parts.push(`images=${this._images.length}`);
    if (this._files.length > 0) parts.push(`files=${this._files.length}`);
    return {
      text: this._text,
      audio: this._audio,
      images: this._images,
      files: this._files,
      _debug: `MultimodalResponse(${parts.join(', ')})`
    };
  }
}

/**
 * Create a MultimodalResponse from raw LLM response data.
 * Handles multiple formats: OpenRouter, OpenAI, and generic patterns.
 */
export function createMultimodalResponse(
  rawResponse: unknown,
  text: string = ''
): MultimodalResponse {
  let audio: AudioOutput | null = null;
  let images: ImageOutput[] = [];
  let files: FileOutput[] = [];

  // Extract images from response
  const extractedImages = extractImages(rawResponse);
  if (extractedImages.length > 0) {
    images = extractedImages;
  }

  // Extract audio from response
  const extractedAudio = extractAudio(rawResponse);
  if (extractedAudio) {
    audio = extractedAudio;
  }

  // Extract files from response
  const extractedFiles = extractFiles(rawResponse);
  if (extractedFiles.length > 0) {
    files = extractedFiles;
  }

  return new MultimodalResponse(text, audio, images, files, rawResponse);
}

/**
 * Extract images from various data structures.
 */
function extractImages(data: unknown): ImageOutput[] {
  const images: ImageOutput[] = [];

  if (!data) return images;

  // Handle array of images
  if (Array.isArray(data)) {
    for (const item of data) {
      const extracted = extractSingleImage(item);
      if (extracted) images.push(extracted);
    }
    return images;
  }

  // Handle object with image data
  if (typeof data === 'object') {
    const obj = data as Record<string, unknown>;

    // Check for images array
    if (Array.isArray(obj.images)) {
      for (const img of obj.images) {
        const extracted = extractSingleImage(img);
        if (extracted) images.push(extracted);
      }
    }

    // Check for image_url structure (OpenRouter pattern)
    if (obj.image_url) {
      const extracted = extractSingleImage(obj.image_url);
      if (extracted) images.push(extracted);
    }

    // Check for direct url/b64_json
    if (obj.url || obj.b64_json || obj.b64Json) {
      const extracted = extractSingleImage(obj);
      if (extracted) images.push(extracted);
    }
  }

  return images;
}

/**
 * Extract a single image from data.
 */
function extractSingleImage(data: unknown): ImageOutput | null {
  if (!data) return null;

  if (typeof data === 'string') {
    // Direct URL string
    if (data.startsWith('http') || data.startsWith('data:')) {
      return { url: data };
    }
    return null;
  }

  if (typeof data === 'object') {
    const obj = data as Record<string, unknown>;

    // Handle image_url nested structure
    if (obj.image_url) {
      const imageUrl = obj.image_url as Record<string, unknown>;
      const url = imageUrl.url as string | undefined;
      if (url) {
        // Handle data URLs
        if (url.startsWith('data:image')) {
          const base64Data = url.split(',', 2)[1];
          return { url, b64Json: base64Data || undefined };
        }
        return { url };
      }
    }

    // Direct url/b64_json
    const url = (obj.url || obj.image_url) as string | undefined;
    const b64Json = (obj.b64_json || obj.b64Json) as string | undefined;
    const revisedPrompt = (obj.revised_prompt || obj.revisedPrompt) as string | undefined;

    if (url || b64Json) {
      return { url, b64Json, revisedPrompt };
    }
  }

  return null;
}

/**
 * Extract audio from response data.
 */
function extractAudio(data: unknown): AudioOutput | null {
  if (!data || typeof data !== 'object') return null;

  const obj = data as Record<string, unknown>;

  // Check for audio object
  if (obj.audio && typeof obj.audio === 'object') {
    const audio = obj.audio as Record<string, unknown>;
    return {
      data: audio.data as string | undefined,
      format: (audio.format as string) || 'wav',
      url: audio.url as string | undefined
    };
  }

  // Check for input_audio structure
  if (obj.input_audio && typeof obj.input_audio === 'object') {
    const inputAudio = obj.input_audio as Record<string, unknown>;
    return {
      data: inputAudio.data as string | undefined,
      format: (inputAudio.format as string) || 'wav'
    };
  }

  return null;
}

/**
 * Extract files from response data.
 */
function extractFiles(data: unknown): FileOutput[] {
  const files: FileOutput[] = [];

  if (!data) return files;

  if (typeof data === 'object') {
    const obj = data as Record<string, unknown>;

    // Check for files array
    if (Array.isArray(obj.files)) {
      for (const file of obj.files) {
        const extracted = extractSingleFile(file);
        if (extracted) files.push(extracted);
      }
    }
  }

  return files;
}

/**
 * Extract a single file from data.
 */
function extractSingleFile(data: unknown): FileOutput | null {
  if (!data || typeof data !== 'object') return null;

  const obj = data as Record<string, unknown>;

  const url = obj.url as string | undefined;
  const data_b64 = obj.data as string | undefined;
  const mimeType = (obj.mime_type || obj.mimeType) as string | undefined;
  const filename = obj.filename as string | undefined;

  if (url || data_b64) {
    return { url, data: data_b64, mimeType, filename };
  }

  return null;
}
