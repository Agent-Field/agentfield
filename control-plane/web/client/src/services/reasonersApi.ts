import type { ReasonersResponse, ReasonerWithNode, ReasonerFilters } from '../types/reasoners';
import type {
  ExecutionRequest,
  ExecutionResponse,
  ExecutionHistory,
  PerformanceMetrics,
  ExecutionTemplate,
  AsyncExecuteResponse,
  ExecutionStatusResponse
} from '../types/execution';
import { getGlobalApiKey } from './api';

const API_BASE_URL = '/api/ui/v1';
const withAuthHeaders = (headers?: HeadersInit) => {
  const merged = new Headers(headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    merged.set('X-API-Key', apiKey);
  }
  return merged;
};

export class ReasonersApiError extends Error {
  public status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ReasonersApiError';
    this.status = status;
  }
}

export const reasonersApi = {
  /**
   * Fetch all reasoners with optional filters
   */
  getAllReasoners: async (filters: ReasonerFilters = {}): Promise<ReasonersResponse> => {
    const params = new URLSearchParams();

    if (filters.status && filters.status !== 'all') {
      params.append('status', filters.status);
    }
    if (filters.search) {
      params.append('search', filters.search);
    }
    if (filters.limit) {
      params.append('limit', filters.limit.toString());
    }
    if (filters.offset) {
      params.append('offset', filters.offset.toString());
    }

    const url = `${API_BASE_URL}/reasoners/all${params.toString() ? `?${params.toString()}` : ''}`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        throw new ReasonersApiError(
          `Failed to fetch reasoners: ${response.statusText}`,
          response.status
        );
      }

      const data: ReasonersResponse = await response.json();

      // Validate and ensure proper structure
      const validatedData: ReasonersResponse = {
        reasoners: Array.isArray(data.reasoners) ? data.reasoners : [],
        total: typeof data.total === 'number' ? data.total : 0,
        online_count: typeof data.online_count === 'number' ? data.online_count : 0,
        offline_count: typeof data.offline_count === 'number' ? data.offline_count : 0,
        nodes_count: typeof data.nodes_count === 'number' ? data.nodes_count : 0,
      };

      return validatedData;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Fetch details for a specific reasoner
   */
  getReasonerDetails: async (reasonerId: string): Promise<ReasonerWithNode> => {
    const url = `${API_BASE_URL}/reasoners/${encodeURIComponent(reasonerId)}/details`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        if (response.status === 404) {
          throw new ReasonersApiError('Reasoner not found', 404);
        }
        throw new ReasonersApiError(
          `Failed to fetch reasoner details: ${response.statusText}`,
          response.status
        );
      }

      const data: ReasonerWithNode = await response.json();
      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Execute a reasoner with given input data (synchronous - waits for completion)
   * Enhanced version with proper request/response types and validation
   */
  executeReasoner: async (reasonerId: string, request: ExecutionRequest): Promise<ExecutionResponse> => {
    const url = `/api/v1/execute/${encodeURIComponent(reasonerId)}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: withAuthHeaders({
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new ReasonersApiError(
          `Failed to execute reasoner: ${response.statusText} - ${errorText}`,
          response.status
        );
      }

      const data: ExecutionResponse = await response.json();

      // Validate response structure
      if (!data || typeof data !== 'object') {
        console.error('Invalid response structure:', data);
        throw new ReasonersApiError('Invalid response format from server');
      }

      // Log response for debugging if it seems malformed
      if (!data.result && !data.error_message && data.status !== 'succeeded') {
        console.warn('Response missing both result and error_message fields:', data);
      }

      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }

      // Enhanced error logging for debugging
      console.error('Reasoner execution error:', {
        reasonerId,
        request,
        error: error instanceof Error ? error.message : error
      });

      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Execute a reasoner asynchronously (returns immediately with execution_id)
   * Use this when you need the execution_id immediately for navigation/tracking
   */
  executeReasonerAsync: async (reasonerId: string, request: ExecutionRequest): Promise<AsyncExecuteResponse> => {
    const url = `/api/v1/execute/async/${encodeURIComponent(reasonerId)}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: withAuthHeaders({
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new ReasonersApiError(
          `Failed to execute reasoner async: ${response.statusText} - ${errorText}`,
          response.status
        );
      }

      const data: AsyncExecuteResponse = await response.json();

      // Validate response structure
      if (!data || typeof data !== 'object' || !data.execution_id) {
        console.error('Invalid async response structure:', data);
        throw new ReasonersApiError('Invalid async response format from server');
      }

      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }

      // Enhanced error logging for debugging
      console.error('Reasoner async execution error:', {
        reasonerId,
        request,
        error: error instanceof Error ? error.message : error
      });

      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Get execution status by execution_id
   * Use this to poll for execution completion after starting with executeReasonerAsync
   */
  getExecutionStatus: async (executionId: string): Promise<ExecutionStatusResponse> => {
    const url = `/api/v1/executions/${encodeURIComponent(executionId)}`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        if (response.status === 404) {
          throw new ReasonersApiError('Execution not found', 404);
        }
        throw new ReasonersApiError(
          `Failed to fetch execution status: ${response.statusText}`,
          response.status
        );
      }

      const data: ExecutionStatusResponse = await response.json();

      // Validate response structure
      if (!data || typeof data !== 'object' || !data.execution_id) {
        console.error('Invalid execution status response structure:', data);
        throw new ReasonersApiError('Invalid execution status response format from server');
      }

      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }

      console.error('Execution status fetch error:', {
        executionId,
        error: error instanceof Error ? error.message : error
      });

      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Get performance metrics for a specific reasoner
   */
  getPerformanceMetrics: async (reasonerId: string): Promise<PerformanceMetrics> => {
    const url = `${API_BASE_URL}/reasoners/${encodeURIComponent(reasonerId)}/metrics`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        throw new ReasonersApiError(
          `Failed to fetch performance metrics: ${response.statusText}`,
          response.status
        );
      }

      const data: PerformanceMetrics = await response.json();
      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Get execution history for a specific reasoner
   */
  getExecutionHistory: async (
    reasonerId: string,
    page: number = 1,
    limit: number = 20
  ): Promise<ExecutionHistory> => {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });

    const url = `${API_BASE_URL}/reasoners/${encodeURIComponent(reasonerId)}/executions?${params.toString()}`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        throw new ReasonersApiError(
          `Failed to fetch execution history: ${response.statusText}`,
          response.status
        );
      }

      const data: ExecutionHistory = await response.json();
      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Get saved execution templates for a reasoner
   */
  getExecutionTemplates: async (reasonerId: string): Promise<ExecutionTemplate[]> => {
    const url = `${API_BASE_URL}/reasoners/${encodeURIComponent(reasonerId)}/templates`;

    try {
      const response = await fetch(url, { headers: withAuthHeaders() });

      if (!response.ok) {
        throw new ReasonersApiError(
          `Failed to fetch execution templates: ${response.statusText}`,
          response.status
        );
      }

      const data: ExecutionTemplate[] = await response.json();
      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Save an execution template
   */
  saveExecutionTemplate: async (
    reasonerId: string,
    template: Omit<ExecutionTemplate, 'id' | 'created_at'>
  ): Promise<ExecutionTemplate> => {
    const url = `${API_BASE_URL}/reasoners/${encodeURIComponent(reasonerId)}/templates`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: withAuthHeaders({
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify(template),
      });

      if (!response.ok) {
        throw new ReasonersApiError(
          `Failed to save execution template: ${response.statusText}`,
          response.status
        );
      }

      const data: ExecutionTemplate = await response.json();
      return data;
    } catch (error) {
      if (error instanceof ReasonersApiError) {
        throw error;
      }
      throw new ReasonersApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  /**
   * Create an SSE connection for real-time reasoner events
   */
  createEventStream: (
    onEvent: (event: any) => void,
    onError?: (error: Error) => void,
    onConnect?: () => void
  ): EventSource => {
    const apiKey = getGlobalApiKey();
    const url = apiKey
      ? `${API_BASE_URL}/reasoners/events?api_key=${encodeURIComponent(apiKey)}`
      : `${API_BASE_URL}/reasoners/events`;
    console.log('🔄 Attempting to create SSE connection to:', url);
    const eventSource = new EventSource(url);

    eventSource.onopen = () => {
      console.log('✅ Reasoner SSE connection opened successfully');
      onConnect?.();
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📡 Received reasoner event:', data);
        onEvent(data);
      } catch (error) {
        console.error('❌ Failed to parse SSE event data:', error, 'Raw data:', event.data);
        onError?.(new Error('Failed to parse event data'));
      }
    };

    eventSource.onerror = (error) => {
      console.error('❌ Reasoner SSE connection error:', error);
      console.error('❌ EventSource readyState:', eventSource.readyState);
      console.error('❌ EventSource url:', eventSource.url);
      onError?.(new Error(`SSE connection error - readyState: ${eventSource.readyState}`));
    };

    return eventSource;
  },

  /**
   * Close an SSE connection
   */
  closeEventStream: (eventSource: EventSource): void => {
    if (eventSource) {
      eventSource.close();
      console.log('🔌 Reasoner SSE connection closed');
    }
  }
};
