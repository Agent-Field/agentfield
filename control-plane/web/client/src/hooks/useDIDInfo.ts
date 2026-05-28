import { useState, useEffect, useCallback } from 'react';
import type { AgentDIDInfo, DIDStatusSummary } from '../types/did';
import { getAgentDIDInfo, getDIDStatusSummary } from '../services/didApi';

/**
 * Hook for managing DID information for an agent node
 */
export function useDIDInfo(nodeId: string) {
  const [didInfo, setDIDInfo] = useState<AgentDIDInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDIDInfo = useCallback(async () => {
    if (!nodeId) return;

    try {
      setLoading(true);
      setError(null);
      const info = await getAgentDIDInfo(nodeId);
      setDIDInfo(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch DID info');
      setDIDInfo(null);
    } finally {
      setLoading(false);
    }
  }, [nodeId]);

  useEffect(() => {
    fetchDIDInfo();
  }, [fetchDIDInfo]);

  return {
    didInfo,
    loading,
    error,
    refetch: fetchDIDInfo
  };
}

/**
 * Hook for getting DID status summary (lightweight for UI indicators)
 */
export function useDIDStatus(nodeId: string) {
  const [status, setStatus] = useState<DIDStatusSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!nodeId) return;

    try {
      setLoading(true);
      setError(null);
      const statusInfo = await getDIDStatusSummary(nodeId);
      setStatus(statusInfo);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch DID status');
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, [nodeId]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return {
    status,
    loading,
    error,
    refetch: fetchStatus
  };
}

/**
 * Hook for managing multiple node DID statuses (for nodes list page)
 */
export function useMultipleDIDStatuses(nodeIds: string[]) {
  const [statuses, setStatuses] = useState<Record<string, DIDStatusSummary>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatuses = useCallback(async () => {
    if (nodeIds.length === 0) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch statuses in parallel
      const statusPromises = nodeIds.map(async (nodeId) => {
        try {
          const status = await getDIDStatusSummary(nodeId);
          return { nodeId, status };
        } catch (err) {
          console.warn(`Failed to fetch DID status for node ${nodeId}:`, err);
          return {
            nodeId,
            status: {
              has_did: false,
              did_status: 'inactive' as const,
              reasoner_count: 0,
              skill_count: 0,
              last_updated: ''
            }
          };
        }
      });

      const results = await Promise.all(statusPromises);
      const statusMap = results.reduce((acc, { nodeId, status }) => {
        acc[nodeId] = status;
        return acc;
      }, {} as Record<string, DIDStatusSummary>);

      setStatuses(statusMap);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch DID statuses');
    } finally {
      setLoading(false);
    }
  }, [nodeIds]);

  useEffect(() => {
    fetchStatuses();
  }, [fetchStatuses]);

  return {
    statuses,
    loading,
    error,
    refetch: fetchStatuses,
    getStatus: (nodeId: string) => statuses[nodeId] || null
  };
}

/**
 * Hook for DID operations (register, resolve, etc.)
 */
export function useDIDOperations() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeOperation = useCallback(async <T>(
    operation: () => Promise<T>
  ): Promise<T | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await operation();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Operation failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    executeOperation,
    clearError: () => setError(null)
  };
}

/**
 * Hook for real-time DID updates (if SSE is available)
 */
export function useDIDUpdates(nodeId: string, onUpdate?: (didInfo: AgentDIDInfo) => void) {
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    // For now, we'll use polling. In the future, this could be replaced with SSE
    const interval = setInterval(async () => {
      try {
        const didInfo = await getAgentDIDInfo(nodeId);
        setLastUpdate(new Date());
        onUpdate?.(didInfo);
      } catch (err) {
        console.warn('Failed to fetch DID updates:', err);
      }
    }, 30000); // Poll every 30 seconds

    return () => clearInterval(interval);
  }, [nodeId, onUpdate]);

  return {
    lastUpdate
  };
}
