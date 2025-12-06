import {
  embed,
  embedMany,
  generateObject,
  generateText,
  streamText
} from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createMistral } from '@ai-sdk/mistral';
import { createGroq } from '@ai-sdk/groq';
import { createXai } from '@ai-sdk/xai';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { createCohere } from '@ai-sdk/cohere';
import type { z } from 'zod';
import type { AIConfig } from '../types/agent.js';
import { StatelessRateLimiter } from './RateLimiter.js';

export type ZodSchema<T> = z.Schema<T, z.ZodTypeDef, any>;

/**
 * Attempts to repair malformed JSON text from model responses.
 * Handles common issues like markdown code blocks, trailing commas, etc.
 */
function repairJsonText(text: string): string | null {
  let cleaned = text.trim();

  // Remove markdown code blocks (```json ... ``` or ``` ... ```)
  const codeBlockMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    cleaned = codeBlockMatch[1].trim();
  }

  // Try to extract JSON object/array if there's extra text
  const jsonMatch = cleaned.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (jsonMatch) {
    cleaned = jsonMatch[1];
  }

  // Remove trailing commas before } or ]
  cleaned = cleaned.replace(/,(\s*[}\]])/g, '$1');

  // Try to parse to verify it's valid
  try {
    JSON.parse(cleaned);
    return cleaned;
  } catch {
    return null;
  }
}

export interface AIRequestOptions {
  system?: string;
  schema?: ZodSchema<any>;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  provider?: AIConfig['provider'];
  /**
   * Mode for structured output generation.
   * - 'auto': Let the provider choose (default in ai-sdk, uses tool calling)
   * - 'json': Use JSON mode (more compatible across providers/models)
   * - 'tool': Force tool calling mode
   */
  mode?: 'auto' | 'json' | 'tool';
}

export type AIStream = AsyncIterable<string>;

export interface AIEmbeddingOptions {
  model?: string;
  provider?: AIConfig['provider'];
}

export class AIClient {
  private readonly config: AIConfig;
  private rateLimiter?: StatelessRateLimiter;

  constructor(config: AIConfig = {}) {
    this.config = {
      enableRateLimitRetry: true,
      rateLimitMaxRetries: 20,
      rateLimitBaseDelay: 1.0,
      rateLimitMaxDelay: 300.0,
      rateLimitJitterFactor: 0.25,
      rateLimitCircuitBreakerThreshold: 10,
      rateLimitCircuitBreakerTimeout: 300,
      ...config
    };
  }

  async generate<T>(prompt: string, options: AIRequestOptions & { schema: ZodSchema<T> }): Promise<T>;
  async generate(prompt: string, options?: AIRequestOptions): Promise<string>;
  async generate<T = any>(prompt: string, options: AIRequestOptions = {}): Promise<T | string> {
    const model = this.buildModel(options);

    if (options.schema) {
      const schema = options.schema;
      // Default to 'json' mode for better compatibility across providers
      // 'auto' mode uses tool calling which some models/providers don't support well
      const mode = options.mode ?? 'json';
      const call = async () =>
        generateObject({
          model: model,
          prompt,
          output: 'object',
          mode,
          system: options.system,
          temperature: options.temperature ?? this.config.temperature,
          maxOutputTokens: options.maxTokens ?? this.config.maxTokens,
          schema,
          experimental_repairText: async ({ text }) => repairJsonText(text)
        });

      const response = await this.withRateLimitRetry(call);
      return response.object as T;
    }

    const call = async () =>
      generateText({
        model: model,
        prompt,
        system: options.system,
        temperature: options.temperature ?? this.config.temperature,
        maxOutputTokens: options.maxTokens ?? this.config.maxTokens
      });

    const response = await this.withRateLimitRetry(call);
    return (response).text as string;
  }

  async stream(prompt: string, options: AIRequestOptions = {}): Promise<AIStream> {
    const model = this.buildModel(options);
    const streamResult = streamText({
      model: model,
      prompt,
      system: options.system,
      temperature: options.temperature ?? this.config.temperature,
      maxOutputTokens: options.maxTokens ?? this.config.maxTokens
    });

    return streamResult.textStream;
  }

  async embed(value: string, options: AIEmbeddingOptions = {}) {
    const model = this.buildEmbeddingModel(options);
    const result = await this.withRateLimitRetry(() =>
      embed({
        model: model,
        value
      })
    );
    return (result).embedding as number[];
  }

  async embedMany(values: string[], options: AIEmbeddingOptions = {}) {
    const model = this.buildEmbeddingModel(options);
    const result = await this.withRateLimitRetry(() =>
      embedMany({
        model: model,
        values
      })
    );
    return (result).embeddings as number[][];
  }

  private buildModel(options: AIRequestOptions) {
    const provider = options.provider ?? this.config.provider ?? 'openai';
    const modelName = options.model ?? this.config.model ?? 'gpt-4o';

    switch (provider) {
      case 'anthropic': {
        const anthropic = createAnthropic({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return anthropic(modelName);
      }

      case 'google': {
        const google = createGoogleGenerativeAI({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return google(modelName);
      }

      case 'mistral': {
        const mistral = createMistral({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return mistral(modelName);
      }

      case 'groq': {
        const groq = createGroq({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return groq(modelName);
      }

      case 'xai': {
        const xai = createXai({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return xai(modelName);
      }

      case 'deepseek': {
        const deepseek = createDeepSeek({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return deepseek(modelName);
      }

      case 'cohere': {
        const cohere = createCohere({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return cohere(modelName);
      }

      case 'openrouter': {
        // OpenRouter is OpenAI-compatible
        const openrouter = createOpenAI({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl ?? 'https://openrouter.ai/api/v1'
        });
        return openrouter(modelName);
      }

      case 'ollama': {
        // Ollama is OpenAI-compatible
        const ollama = createOpenAI({
          apiKey: this.config.apiKey ?? 'ollama', // Ollama doesn't need real key
          baseURL: this.config.baseUrl ?? 'http://localhost:11434/v1'
        });
        return ollama(modelName);
      }

      case 'openai':
      default: {
        const openai = createOpenAI({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return openai(modelName);
      }
    }
  }

  private buildEmbeddingModel(options: AIEmbeddingOptions) {
    const provider = options.provider ?? this.config.provider ?? 'openai';
    const modelName = options.model ?? this.config.embeddingModel ?? 'text-embedding-3-small';

    // Providers without embedding support
    const noEmbeddingProviders = ['anthropic', 'xai', 'deepseek', 'groq'];
    if (noEmbeddingProviders.includes(provider)) {
      throw new Error(`Embedding generation is not supported for ${provider} provider`);
    }

    switch (provider) {
      case 'google': {
        const google = createGoogleGenerativeAI({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return google.textEmbeddingModel(modelName);
      }

      case 'mistral': {
        const mistral = createMistral({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return mistral.textEmbeddingModel(modelName);
      }

      case 'cohere': {
        const cohere = createCohere({
          apiKey: this.config.apiKey,
          baseURL: this.config.baseUrl
        });
        return cohere.textEmbeddingModel(modelName);
      }

      case 'openai':
      case 'openrouter':
      case 'ollama':
      default: {
        const openai = createOpenAI({
          apiKey: this.config.apiKey ?? (provider === 'ollama' ? 'ollama' : undefined),
          baseURL:
            this.config.baseUrl ??
            (provider === 'openrouter'
              ? 'https://openrouter.ai/api/v1'
              : provider === 'ollama'
                ? 'http://localhost:11434/v1'
                : undefined)
        });
        return openai.embedding(modelName);
      }
    }
  }

  private getRateLimiter() {
    if (!this.rateLimiter) {
      this.rateLimiter = new StatelessRateLimiter({
        maxRetries: this.config.rateLimitMaxRetries,
        baseDelay: this.config.rateLimitBaseDelay,
        maxDelay: this.config.rateLimitMaxDelay,
        jitterFactor: this.config.rateLimitJitterFactor,
        circuitBreakerThreshold: this.config.rateLimitCircuitBreakerThreshold,
        circuitBreakerTimeout: this.config.rateLimitCircuitBreakerTimeout
      });
    }
    return this.rateLimiter;
  }

  private withRateLimitRetry<T>(fn: () => Promise<T>): Promise<T> {
    if (this.config.enableRateLimitRetry === false) {
      return fn();
    }
    return this.getRateLimiter().executeWithRetry(fn);
  }
}
