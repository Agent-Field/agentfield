import { useState, useEffect, useCallback } from 'react';
import { getGlobalApiKey } from '../../../services/api';

interface CachedNodeDetails {
  data: NodeDetails;
  timestamp: number;
}

const NODE_DETAILS_CACHE = new Map<string, CachedNodeDetails>();
const CACHE_TTL_MS = 60_000; // 1 minute cache to prevent repeated fetches when reopen sidebar

interface NodeDetails {
  input?: any;
  output?: any;
  error_message?: string;
  cost?: number;
  memory_updates?: any[];
  performance_metrics?: {
    response_time_ms: number;
    tokens_used?: number;
  };
}

interface UseNodeDetailsReturn {
  nodeDetails: NodeDetails | undefined;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useNodeDetails(executionId?: string): UseNodeDetailsReturn {
  const [nodeDetails, setNodeDetails] = useState<NodeDetails | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchNodeDetails = useCallback(async (forceRefresh: boolean = false) => {
    if (!executionId) {
      setNodeDetails(undefined);
      setLoading(false);
      setError(null);
      return;
    }

    if (!forceRefresh) {
      const cached = NODE_DETAILS_CACHE.get(executionId);
      if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
        setNodeDetails(cached.data);
        setLoading(false);
        setError(null);
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      // Fetch execution details from the API (using same endpoint as ExecutionDetailPage)
      console.log(`🔍 SIDEBAR DEBUG: Fetching execution details for ${executionId}`);
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';
      const headers: HeadersInit = {};
      const apiKey = getGlobalApiKey();
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      const response = await fetch(`${API_BASE_URL}/executions/${executionId}/details`, { headers });

      if (!response.ok) {
        console.error(`🔍 SIDEBAR DEBUG: API response not ok: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to fetch execution details: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`🔍 SIDEBAR DEBUG: Raw API response:`, data);

      // Transform the API response to match our NodeDetails interface
      // The API returns input_data/output_data, not input/output
      const details: NodeDetails = {
        input: data.input_data || data.input,
        output: data.output_data || data.output,
        error_message: data.error_message,
        cost: data.cost,
        memory_updates: data.memory_updates || [],
        performance_metrics: data.performance_metrics ? {
          response_time_ms: data.performance_metrics.response_time_ms || data.duration_ms || 0,
          tokens_used: data.performance_metrics.tokens_used
        } : undefined
      };

      console.log(`🔍 SIDEBAR DEBUG: Transformed details:`, details);
      console.log(`🔍 SIDEBAR DEBUG: Input available:`, !!details.input);
      console.log(`🔍 SIDEBAR DEBUG: Output available:`, !!details.output);

      setNodeDetails(details);
      NODE_DETAILS_CACHE.set(executionId, {
        data: details,
        timestamp: Date.now(),
      });
    } catch (err) {
      console.error('🔍 SIDEBAR DEBUG: Error fetching node details:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch execution details');
      setNodeDetails(undefined);
    } finally {
      setLoading(false);
    }
  }, [executionId]);

  const refetch = useCallback(() => {
    fetchNodeDetails(true);
  }, [fetchNodeDetails]);

  useEffect(() => {
    fetchNodeDetails();
  }, [fetchNodeDetails]);

  return {
    nodeDetails,
    loading,
    error,
    refetch
  };
}
