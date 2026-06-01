/**
 * Multimodal content helpers for AI prompts.
 * Provides Image, Audio, and File classes with factory methods for creating
 * multimodal content from various sources (files, URLs, buffers, base64).
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// MIME type mappings for common image formats
const IMAGE_MIME_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
};

// MIME type mappings for common audio formats
const AUDIO_MIME_TYPES: Record<string, string> = {
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

/**
 * Represents text content in a multimodal prompt.
 */
export class Text {
  readonly type: 'text' = 'text';
  readonly text: string;

  constructor(text: string) {
    this.text = text;
  }
}

/**
 * Represents image content in a multimodal prompt.
 */
export class Image {
  readonly type: 'image_url' = 'image_url';
  readonly imageUrl: { url: string; detail?: 'low' | 'high' | 'auto' };

  private constructor(imageUrl: { url: string; detail?: 'low' | 'high' | 'auto' }) {
    this.imageUrl = imageUrl;
  }

  /**
   * Create Image from a local file by converting to base64 data URL.
   */
  static async fromFile(
    filePath: string,
    detail: 'low' | 'high' | 'auto' = 'high'
  ): Promise<Image> {
    const absolutePath = resolve(filePath);

    // Read file and encode to base64
    const buffer = await readFile(absolutePath);
    const base64Data = buffer.toString('base64');

    // Determine MIME type from extension
    const ext = getExtension(absolutePath).toLowerCase();
    const mimeType = IMAGE_MIME_TYPES[ext] || 'image/jpeg';

    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new Image({ url: dataUrl, detail });
  }

  /**
   * Create Image from a URL.
   */
  static fromUrl(
    url: string,
    detail: 'low' | 'high' | 'auto' = 'high'
  ): Image {
    return new Image({ url, detail });
  }

  /**
   * Create Image from a buffer.
   */
  static async fromBuffer(
    buffer: Buffer | Uint8Array,
    mimeType: string = 'image/jpeg',
    detail: 'low' | 'high' | 'auto' = 'high'
  ): Promise<Image> {
    const base64Data = Buffer.from(buffer).toString('base64');
    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new Image({ url: dataUrl, detail });
  }

  /**
   * Create Image from a base64 string.
   */
  static async fromBase64(
    base64Data: string,
    mimeType: string = 'image/jpeg',
    detail: 'low' | 'high' | 'auto' = 'high'
  ): Promise<Image> {
    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new Image({ url: dataUrl, detail });
  }
}

/**
 * Represents audio content in a multimodal prompt.
 */
export class Audio {
  readonly type: 'input_audio' = 'input_audio';
  readonly audio: { data: string; format: string };

  private constructor(audio: { data: string; format: string }) {
    this.audio = audio;
  }

  /**
   * Create Audio from a local file by converting to base64.
   */
  static async fromFile(
    filePath: string,
    format?: 'wav' | 'mp3' | 'flac' | 'ogg'
  ): Promise<Audio> {
    const absolutePath = resolve(filePath);

    // Auto-detect format from extension if not provided
    const ext = getExtension(absolutePath).toLowerCase().replace('.', '');
    const audioFormat =
      format ||
      (['wav', 'mp3', 'flac', 'ogg'].includes(ext) ? (ext as 'wav' | 'mp3' | 'flac' | 'ogg') : 'wav');

    // Read file and encode to base64
    const buffer = await readFile(absolutePath);
    const base64Data = buffer.toString('base64');

    return new Audio({ data: base64Data, format: audioFormat });
  }

  /**
   * Create Audio from a URL (downloads and converts to base64).
   */
  static async fromUrl(
    url: string,
    format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
  ): Promise<Audio> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio from URL: ${response.status} ${response.statusText}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      const base64Data = Buffer.from(arrayBuffer).toString('base64');
      return new Audio({ data: base64Data, format });
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('URL download requires a fetch-compatible environment');
      }
      throw error;
    }
  }

  /**
   * Create Audio from a buffer.
   */
  static async fromBuffer(
    buffer: Buffer | Uint8Array,
    format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
  ): Promise<Audio> {
    const base64Data = Buffer.from(buffer).toString('base64');
    return new Audio({ data: base64Data, format });
  }

  /**
   * Create Audio from a base64 string.
   */
  static async fromBase64(
    base64Data: string,
    format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
  ): Promise<Audio> {
    return new Audio({ data: base64Data, format });
  }
}

/**
 * Represents a generic file content in a multimodal prompt.
 */
export class File {
  readonly type: 'file' = 'file';
  readonly file: { url: string; mimeType?: string };

  private constructor(file: { url: string; mimeType?: string }) {
    this.file = file;
  }

  /**
   * Create File from a local file path.
   */
  static async fromFile(filePath: string, mimeType?: string): Promise<File> {
    const absolutePath = resolve(filePath);

    // Auto-detect MIME type if not provided
    const detectedMimeType =
      mimeType ||
      guessMimeType(absolutePath) ||
      'application/octet-stream';

    const buffer = await readFile(absolutePath);
    const base64Data = buffer.toString('base64');
    const dataUrl = `data:${detectedMimeType};base64,${base64Data}`;
    return new File({ url: dataUrl, mimeType: detectedMimeType });
  }

  /**
   * Create File from a URL.
   */
  static fromUrl(url: string, mimeType?: string): File {
    return new File({ url, mimeType });
  }

  /**
   * Create File from a buffer.
   */
  static async fromBuffer(buffer: Buffer | Uint8Array, mimeType: string): Promise<File> {
    const base64Data = Buffer.from(buffer).toString('base64');
    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new File({ url: dataUrl, mimeType });
  }

  /**
   * Create File from a base64 string.
   */
  static async fromBase64(base64Data: string, mimeType: string): Promise<File> {
    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new File({ url: dataUrl, mimeType });
  }
}

// Utility functions

/**
 * Get the file extension from a path.
 */
function getExtension(filePath: string): string {
  const lastDot = filePath.lastIndexOf('.');
  if (lastDot === -1) {
    return '';
  }
  return filePath.slice(lastDot);
}

/**
 * Guess MIME type from file extension.
 */
function guessMimeType(filePath: string): string | null {
  const ext = getExtension(filePath).toLowerCase();

  // Check image types
  if (ext in IMAGE_MIME_TYPES) {
    return IMAGE_MIME_TYPES[ext];
  }

  // Check audio types
  if (ext in AUDIO_MIME_TYPES) {
    return AUDIO_MIME_TYPES[ext];
  }

  // Common document types
  const documentMimeTypes: Record<string, string> = {
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.html': 'text/html',
    '.json': 'application/json',
    '.xml': 'application/xml',
    '.zip': 'application/zip',
  };

  return documentMimeTypes[ext] || null;
}

// Convenience factory functions

/**
 * Create text content.
 */
export function text(content: string): Text {
  return new Text(content);
}

/**
 * Create image content from a local file.
 */
export async function imageFromFile(
  filePath: string,
  detail: 'low' | 'high' | 'auto' = 'high'
): Promise<Image> {
  return Image.fromFile(filePath, detail);
}

/**
 * Create image content from a URL.
 */
export function imageFromUrl(
  url: string,
  detail: 'low' | 'high' | 'auto' = 'high'
): Image {
  return Image.fromUrl(url, detail);
}

/**
 * Create image content from a buffer.
 */
export async function imageFromBuffer(
  buffer: Buffer | Uint8Array,
  mimeType: string = 'image/jpeg',
  detail: 'low' | 'high' | 'auto' = 'high'
): Promise<Image> {
  return Image.fromBuffer(buffer, mimeType, detail);
}

/**
 * Create image content from a base64 string.
 */
export async function imageFromBase64(
  base64Data: string,
  mimeType?: string,
  detail: 'low' | 'high' | 'auto' = 'high'
): Promise<Image> {
  return Image.fromBase64(base64Data, mimeType, detail);
}

/**
 * Create audio content from a local file.
 */
export async function audioFromFile(
  filePath: string,
  format?: 'wav' | 'mp3' | 'flac' | 'ogg'
): Promise<Audio> {
  return Audio.fromFile(filePath, format);
}

/**
 * Create audio content from a URL.
 */
export async function audioFromUrl(
  url: string,
  format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
): Promise<Audio> {
  return Audio.fromUrl(url, format);
}

/**
 * Create audio content from a buffer.
 */
export async function audioFromBuffer(
  buffer: Buffer | Uint8Array,
  format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
): Promise<Audio> {
  return Audio.fromBuffer(buffer, format);
}

/**
 * Create audio content from a base64 string.
 */
export async function audioFromBase64(
  base64Data: string,
  format: 'wav' | 'mp3' | 'flac' | 'ogg' = 'wav'
): Promise<Audio> {
  return Audio.fromBase64(base64Data, format);
}

/**
 * Create file content from a local file.
 */
export async function fileFromPath(filePath: string, mimeType?: string): Promise<File> {
  return File.fromFile(filePath, mimeType);
}

/**
 * Create file content from a URL.
 */
export function fileFromUrl(url: string, mimeType?: string): File {
  return File.fromUrl(url, mimeType);
}

/**
 * Create file content from a buffer.
 */
export async function fileFromBuffer(buffer: Buffer | Uint8Array, mimeType: string): Promise<File> {
  return File.fromBuffer(buffer, mimeType);
}

/**
 * Create file content from a base64 string.
 */
export async function fileFromBase64(base64Data: string, mimeType: string): Promise<File> {
  return File.fromBase64(base64Data, mimeType);
}

// Type for all multimodal content types
export type MultimodalContent = Text | Image | Audio | File;
