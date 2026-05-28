import { getGlobalApiKey } from './api';

const API_BASE = '/api/v1';

export class ObservabilityWebhookApiError extends Error {
  public status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ObservabilityWebhookApiError';
    this.status = status;
  }
}

// Types
export interface ObservabilityWebhookConfig {
  id: string;
  url: string;
  has_secret: boolean;
  headers?: Record<string, string>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface ObservabilityWebhookConfigResponse {
  configured: boolean;
  config?: ObservabilityWebhookConfig;
}

export interface ObservabilityWebhookRequest {
  url: string;
  secret?: string;
  headers?: Record<string, string>;
  enabled?: boolean;
}

export interface ObservabilityForwarderStatus {
  enabled: boolean;
  webhook_url?: string;
  queue_depth: number;
  events_forwarded: number;
  events_dropped: number;
  dead_letter_count: number;
  last_forwarded_at?: string;
  last_error?: string;
}

export interface ObservabilityDeadLetterEntry {
  id: number;
  event_type: string;
  event_source: string;
  event_timestamp: string;
  payload: string;
  error_message: string;
  retry_count: number;
  created_at: string;
}

export interface ObservabilityDeadLetterListResponse {
  entries: ObservabilityDeadLetterEntry[];
  total_count: number;
}

export interface ObservabilityRedriveResponse {
  success: boolean;
  message: string;
  processed: number;
  failed: number;
}

export interface SetWebhookResponse {
  success: boolean;
  message: string;
  config: ObservabilityWebhookConfig;
}

export interface DeleteWebhookResponse {
  success: boolean;
  message: string;
}

// Helper functions
const addAuthHeaders = (options: RequestInit = {}): RequestInit => {
  const headers = new Headers(options.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }
  headers.set('Content-Type', 'application/json');
  return { ...options, headers };
};

const fetchWithTimeout = async (url: string, options: RequestInit & { timeout?: number } = {}) => {
  const { timeout = 10000, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...addAuthHeaders(fetchOptions),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ObservabilityWebhookApiError(`Request timeout after ${timeout}ms`, 408);
    }
    throw error;
  }
};

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const errorText = await response.text();
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

    try {
      const errorData = JSON.parse(errorText);
      errorMessage = errorData.error || errorData.message || errorMessage;
    } catch {
      if (errorText) {
        errorMessage = errorText;
      }
    }

    throw new ObservabilityWebhookApiError(errorMessage, response.status);
  }

  return response.json();
};

// API Functions

/**
 * Get the current observability webhook configuration
 */
export const getObservabilityWebhook = async (): Promise<ObservabilityWebhookConfigResponse> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook`);
  return handleResponse<ObservabilityWebhookConfigResponse>(response);
};

/**
 * Set or update the observability webhook configuration
 */
export const setObservabilityWebhook = async (
  config: ObservabilityWebhookRequest
): Promise<SetWebhookResponse> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
  return handleResponse<SetWebhookResponse>(response);
};

/**
 * Delete the observability webhook configuration
 */
export const deleteObservabilityWebhook = async (): Promise<DeleteWebhookResponse> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook`, {
    method: 'DELETE',
  });
  return handleResponse<DeleteWebhookResponse>(response);
};

/**
 * Get the observability forwarder status
 */
export const getObservabilityWebhookStatus = async (): Promise<ObservabilityForwarderStatus> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook/status`);
  return handleResponse<ObservabilityForwarderStatus>(response);
};

/**
 * Get dead letter queue entries
 */
export const getDeadLetterQueue = async (
  limit = 100,
  offset = 0
): Promise<ObservabilityDeadLetterListResponse> => {
  const response = await fetchWithTimeout(
    `${API_BASE}/settings/observability-webhook/dlq?limit=${limit}&offset=${offset}`
  );
  return handleResponse<ObservabilityDeadLetterListResponse>(response);
};

/**
 * Trigger redrive of dead letter queue
 */
export const redriveDeadLetterQueue = async (): Promise<ObservabilityRedriveResponse> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook/redrive`, {
    method: 'POST',
    timeout: 60000, // Longer timeout for redrive operation
  });
  return handleResponse<ObservabilityRedriveResponse>(response);
};

/**
 * Clear all entries from the dead letter queue
 */
export const clearDeadLetterQueue = async (): Promise<{ success: boolean; message: string }> => {
  const response = await fetchWithTimeout(`${API_BASE}/settings/observability-webhook/dlq`, {
    method: 'DELETE',
  });
  return handleResponse<{ success: boolean; message: string }>(response);
};
