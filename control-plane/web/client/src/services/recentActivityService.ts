import type { RecentActivityResponse } from '../types/recentActivity';
import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

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
 * Get recent activity data
 * GET /api/ui/v1/executions/recent
 */
export async function getRecentActivity(): Promise<RecentActivityResponse> {
  return fetchWrapper<RecentActivityResponse>('/executions/recent', {
    timeout: 6000 // 6 second timeout for recent activity
  });
}

/**
 * Get recent activity with retry logic
 */
export async function getRecentActivityWithRetry(
  maxRetries: number = 2,
  baseDelayMs: number = 1000
): Promise<RecentActivityResponse> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await getRecentActivity();
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
