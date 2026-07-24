import { useState, useCallback, useEffect, useMemo } from 'react';
import type {
  VCStatusSummary,
  WorkflowVCChainResponse,
  VCVerificationResponse,
  AuditTrailEntry,
  VCStatusData
} from '../types/did';
import {
  getVCStatusSummary,
  getWorkflowVCChain,
  verifyVC,
  getWorkflowAuditTrail,
  getExecutionVCStatus,
  getWorkflowVCStatuses
} from '../services/vcApi';

/**
 * Hook for managing VC verification operations
 */
export function useVCVerification() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] = useState<VCVerificationResponse | null>(null);

  const verifyVCDocument = useCallback(async (vcDocument: unknown) => {
    try {
      setLoading(true);
      setError(null);
      const result = await verifyVC(vcDocument);
      setVerificationResult(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Verification failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setVerificationResult(null);
    setError(null);
  }, []);

  return {
    loading,
    error,
    verificationResult,
    verifyVCDocument,
    clearResults
  };
}

/**
 * Hook for getting VC status summary for workflows
 */
export function useVCStatus(workflowId: string) {
  const [status, setStatus] = useState<VCStatusSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!workflowId) {
      setStatus(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const statusInfo = await getVCStatusSummary(workflowId);
      setStatus(statusInfo);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch VC status');
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  // Auto-fetch on mount and when workflowId changes
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
 * Hook for managing workflow VC chains
 */
export function useWorkflowVCChain(workflowId: string) {
  const [vcChain, setVCChain] = useState<WorkflowVCChainResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVCChain = useCallback(async () => {
    if (!workflowId) return;

    try {
      setLoading(true);
      setError(null);
      const chain = await getWorkflowVCChain(workflowId);
      setVCChain(chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch VC chain');
      setVCChain(null);
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    fetchVCChain();
  }, [fetchVCChain]);

  return {
    vcChain,
    loading,
    error,
    refetch: fetchVCChain
  };
}

/**
 * Hook for managing audit trails
 */
export function useAuditTrail(workflowId: string) {
  const [auditTrail, setAuditTrail] = useState<AuditTrailEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAuditTrail = useCallback(async () => {
    if (!workflowId) return;

    try {
      setLoading(true);
      setError(null);
      const trail = await getWorkflowAuditTrail(workflowId);
      setAuditTrail(trail);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch audit trail');
      setAuditTrail([]);
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    fetchAuditTrail();
  }, [fetchAuditTrail]);

  return {
    auditTrail,
    loading,
    error,
    refetch: fetchAuditTrail
  };
}

/**
 * Hook for managing execution VC status
 */
export function useExecutionVCStatus(executionId: string) {
  const [vcStatus, setVCStatus] = useState<VCStatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVCStatus = useCallback(async () => {
    if (!executionId) return;

    try {
      setLoading(true);
      setError(null);
      const status = await getExecutionVCStatus(executionId);
      setVCStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch execution VC status');
      setVCStatus(null);
    } finally {
      setLoading(false);
    }
  }, [executionId]);

  useEffect(() => {
    fetchVCStatus();
  }, [fetchVCStatus]);

  return {
    vcStatus,
    loading,
    error,
    refetch: fetchVCStatus
  };
}

/**
 * Hook for managing multiple workflow VC statuses (for workflows list page)
 */
export function useWorkflowVCStatuses(workflowIds: string[]) {
  const [statuses, setStatuses] = useState<Record<string, VCStatusSummary>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uniqueWorkflowIds = useMemo(
    () => Array.from(new Set((workflowIds || []).filter((id): id is string => Boolean(id)))),
    [workflowIds]
  );

  const fetchStatuses = useCallback(async (targetIds?: string[]) => {
    const idsToFetch = (targetIds && targetIds.length > 0 ? targetIds : uniqueWorkflowIds).filter(Boolean);
    if (idsToFetch.length === 0) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await getWorkflowVCStatuses(idsToFetch);
      setStatuses((prev) => ({ ...prev, ...result }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch VC statuses');
    } finally {
      setLoading(false);
    }
  }, [uniqueWorkflowIds]);

  useEffect(() => {
    const missingIds = uniqueWorkflowIds.filter((id) => !statuses[id]);
    if (missingIds.length > 0) {
      fetchStatuses(missingIds);
    }
  }, [uniqueWorkflowIds, statuses, fetchStatuses]);

  return {
    statuses,
    loading,
    error,
    refetch: () => fetchStatuses(uniqueWorkflowIds),
    getStatus: (workflowId: string) => statuses[workflowId] || null
  };
}

/**
 * Hook for VC export operations
 */
export function useVCExport() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportProgress, setExportProgress] = useState<{
    current: number;
    total: number;
    status: string;
  } | null>(null);

  const executeExport = useCallback(async <T>(
    exportOperation: () => Promise<T>,
    progressCallback?: (current: number, total: number, status: string) => void
  ): Promise<T | null> => {
    try {
      setLoading(true);
      setError(null);
      setExportProgress(null);

      // If progress callback is provided, simulate progress updates
      if (progressCallback) {
        progressCallback(0, 100, 'Starting export...');
        setExportProgress({ current: 0, total: 100, status: 'Starting export...' });
      }

      const result = await exportOperation();

      if (progressCallback) {
        progressCallback(100, 100, 'Export completed');
        setExportProgress({ current: 100, total: 100, status: 'Export completed' });
      }

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Export failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
      // Clear progress after a delay
      setTimeout(() => setExportProgress(null), 2000);
    }
  }, []);

  return {
    loading,
    error,
    exportProgress,
    executeExport,
    clearError: () => setError(null)
  };
}

/**
 * Hook for real-time VC updates
 */
export function useVCUpdates(workflowId: string, onUpdate?: (vcChain: WorkflowVCChainResponse) => void) {
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!workflowId) return;

    // Poll for VC updates every 15 seconds
    const interval = setInterval(async () => {
      try {
        const vcChain = await getWorkflowVCChain(workflowId);
        setLastUpdate(new Date());
        onUpdate?.(vcChain);
      } catch (err) {
        console.warn('Failed to fetch VC updates:', err);
      }
    }, 15000);

    return () => clearInterval(interval);
  }, [workflowId, onUpdate]);

  return {
    lastUpdate
  };
}
