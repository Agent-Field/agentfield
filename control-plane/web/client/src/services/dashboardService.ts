import type { DashboardSummary, EnhancedDashboardResponse } from '../types/dashboard';
import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

/**
 * Enhanced fetch wrapper with error handling, retry logic, and timeout support
 * Following the pattern from api.ts
 */
async function fetchWrapper<T>(url: string, options?: RequestInit & { timeout?: number }): Promise<T> {
  const { timeout = 10000, ...fetchOptions } = options || {};

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
 * Retry wrapper for dashboard operations with exponential backoff
 */
async function retryOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
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
 * Get dashboard summary data
 * GET /api/ui/v1/dashboard/summary
 */
export async function getDashboardSummary(): Promise<DashboardSummary> {
  return retryOperation(() =>
    fetchWrapper<DashboardSummary>('/dashboard/summary', {
      timeout: 8000 // 8 second timeout for dashboard data
    })
  );
}

/**
 * Get dashboard summary with manual retry control
 */
export async function getDashboardSummaryWithRetry(
  maxRetries: number = 3,
  baseDelayMs: number = 1000
): Promise<DashboardSummary> {
  return retryOperation(() =>
    fetchWrapper<DashboardSummary>('/dashboard/summary', {
      timeout: 8000
    }),
    maxRetries,
    baseDelayMs
  );
}

/**
 * Parameters for fetching enhanced dashboard data
 */
export interface EnhancedDashboardParams {
  preset?: '1h' | '24h' | '7d' | '30d' | 'custom';
  startTime?: string; // RFC3339 format
  endTime?: string;   // RFC3339 format
  compare?: boolean;
}

/**
 * Build query string from dashboard parameters
 */
function buildDashboardQueryString(params: EnhancedDashboardParams): string {
  const queryParams = new URLSearchParams();

  if (params.preset) {
    queryParams.set('preset', params.preset);
  }
  if (params.preset === 'custom' && params.startTime && params.endTime) {
    queryParams.set('start_time', params.startTime);
    queryParams.set('end_time', params.endTime);
  }
  if (params.compare) {
    queryParams.set('compare', 'true');
  }

  const queryString = queryParams.toString();
  return queryString ? `?${queryString}` : '';
}

/**
 * Get enhanced dashboard summary data
 * GET /api/ui/v1/dashboard/enhanced
 *
 * @param params - Optional parameters for time range and comparison
 */
export async function getEnhancedDashboardSummary(
  params: EnhancedDashboardParams = {}
): Promise<EnhancedDashboardResponse> {
  const queryString = buildDashboardQueryString(params);

  return retryOperation(() =>
    fetchWrapper<EnhancedDashboardResponse>(`/dashboard/enhanced${queryString}`, {
      timeout: 15000 // Slightly longer timeout for larger time ranges
    })
  );
}
