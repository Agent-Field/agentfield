import type { ExecutionTimelineResponse } from '../types/executionTimeline';
import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

// In-memory cache for timeline data to reduce API calls
let timelineCache: {
  data: ExecutionTimelineResponse | null;
  timestamp: number;
  ttl: number;
} = {
  data: null,
  timestamp: 0,
  ttl: 300000 // 5 minutes to match backend cache
};

/**
 * Enhanced fetch wrapper with error handling and timeout support
 */
async function fetchWrapper<T>(url: string, options?: RequestInit & { timeout?: number }): Promise<T> {
  const { timeout = 8000, ...fetchOptions } = options || {};

  const headers = new Headers(fetchOptions.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }

  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        message: 'Request failed with status ' + response.status
      }));

      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json() as Promise<T>;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }

    throw error;
  }
}

/**
 * Check if cache is valid
 */
function isCacheValid(): boolean {
  return timelineCache.data !== null &&
         (Date.now() - timelineCache.timestamp) < timelineCache.ttl;
}

/**
 * Get execution timeline data with intelligent caching
 * GET /api/ui/v1/executions/timeline
 */
export async function getExecutionTimeline(forceRefresh: boolean = false): Promise<ExecutionTimelineResponse> {
  // Return cached data if valid and not forcing refresh
  if (!forceRefresh && isCacheValid() && timelineCache.data) {
    return timelineCache.data;
  }

  const data = await fetchWrapper<ExecutionTimelineResponse>('/executions/timeline', {
    timeout: 10000 // 10 second timeout for timeline data (larger dataset)
  });

  // Update cache
  timelineCache = {
    data,
    timestamp: Date.now(),
    ttl: timelineCache.ttl
  };

  return data;
}

/**
 * Get execution timeline with retry logic and intelligent caching
 */
export async function getExecutionTimelineWithRetry(
  maxRetries: number = 2,
  baseDelayMs: number = 1000,
  forceRefresh: boolean = false
): Promise<ExecutionTimelineResponse> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await getExecutionTimeline(forceRefresh);
    } catch (error) {
      lastError = error as Error;

      // Don't retry on last attempt
      if (attempt === maxRetries) {
        throw lastError;
      }

      // Calculate delay with exponential backoff
      const delay = baseDelayMs * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

/**
 * Clear the timeline cache
 */
export function clearTimelineCache(): void {
  timelineCache = {
    data: null,
    timestamp: 0,
    ttl: timelineCache.ttl
  };
}

/**
 * Get current cache status
 */
export function getTimelineCacheStatus(): {
  hasData: boolean;
  isValid: boolean;
  age: number;
  ttl: number;
} {
  return {
    hasData: timelineCache.data !== null,
    isValid: isCacheValid(),
    age: timelineCache.data ? Date.now() - timelineCache.timestamp : 0,
    ttl: timelineCache.ttl
  };
}

/**
 * Check if timeline data is fresh based on cache timestamp
 */
export function isTimelineDataFresh(
  data: ExecutionTimelineResponse,
  maxAgeMs: number = 300000 // 5 minutes default
): boolean {
  if (!data.cache_timestamp) return false;

  const cacheTime = new Date(data.cache_timestamp).getTime();
  const now = Date.now();
  const age = now - cacheTime;

  return age < maxAgeMs;
}

/**
 * Get cache age in milliseconds for timeline data
 */
export function getTimelineCacheAge(data: ExecutionTimelineResponse): number {
  if (!data.cache_timestamp) return Infinity;

  const cacheTime = new Date(data.cache_timestamp).getTime();
  return Date.now() - cacheTime;
}
