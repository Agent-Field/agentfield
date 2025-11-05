import { useState, useCallback, useRef } from 'react';
import type {
  MCPServerActionResponse,
  MCPServerAction,
  MCPServerHealthForUI
} from '../types/haxen';
import {
  startMCPServer,
  stopMCPServer,
  restartMCPServer,
  bulkMCPServerAction
} from '../services/api';

/**
 * Server operation state
 */
interface ServerOperation {
  serverId: string;
  action: MCPServerAction;
  status: 'pending' | 'success' | 'error';
  error?: string;
  timestamp: Date;
}

/**
 * MCP servers management state
 */
interface MCPServersState {
  /** Currently running operations */
  operations: ServerOperation[];
  /** Whether any operation is in progress */
  loading: boolean;
  /** Last error that occurred */
  error: string | null;
}

/**
 * Options for server operations
 */
interface ServerOperationOptions {
  /** Callback for operation completion */
  onComplete?: (result: MCPServerActionResponse) => void;
  /** Callback for operation error */
  onError?: (error: string, serverId: string, action: MCPServerAction) => void;
  /** Whether to show optimistic updates */
  optimistic?: boolean;
}

/**
 * Custom hook for managing MCP server operations (start, stop, restart)
 * with support for bulk operations and optimistic updates
 * 
 * @param nodeId - The node ID for server operations
 * @returns Object containing server management state and functions
 */
export function useMCPServers(nodeId: string | null) {
  const [state, setState] = useState<MCPServersState>({
    operations: [],
    loading: false,
    error: null
  });

  const mountedRef = useRef(true);

  /**
   * Add operation to state
   */
  const addOperation = useCallback((serverId: string, action: MCPServerAction) => {
    const operation: ServerOperation = {
      serverId,
      action,
      status: 'pending',
      timestamp: new Date()
    };

    setState(prev => ({
      ...prev,
      operations: [...prev.operations, operation],
      loading: true,
      error: null
    }));

    return operation;
  }, []);

  /**
   * Update operation status
   */
  const updateOperation = useCallback((
    serverId: string,
    action: MCPServerAction,
    status: 'success' | 'error',
    error?: string
  ) => {
    setState(prev => {
      const updatedOperations = prev.operations.map(op => {
        if (op.serverId === serverId && op.action === action) {
          return { ...op, status, error };
        }
        return op;
      });

      const hasLoading = updatedOperations.some(op => op.status === 'pending');

      return {
        ...prev,
        operations: updatedOperations,
        loading: hasLoading,
        error: status === 'error' ? error || 'Operation failed' : prev.error
      };
    });
  }, []);

  /**
   * Remove completed operations older than 30 seconds
   */
  const cleanupOperations = useCallback(() => {
    const cutoff = new Date(Date.now() - 30000); // 30 seconds ago
    
    setState(prev => ({
      ...prev,
      operations: prev.operations.filter(op => 
        op.status === 'pending' || op.timestamp > cutoff
      )
    }));
  }, []);

  /**
   * Execute server action with error handling and state management
   */
  const executeServerAction = useCallback(async (
    serverId: string,
    action: MCPServerAction,
    options: ServerOperationOptions = {}
  ): Promise<MCPServerActionResponse | null> => {
    if (!nodeId || !mountedRef.current) return null;

    const { onComplete, onError, optimistic = true } = options;

    // Add operation to state if optimistic
    if (optimistic) {
      addOperation(serverId, action);
    }

    try {
      let response: MCPServerActionResponse;

      switch (action) {
        case 'start':
          response = await startMCPServer(nodeId, serverId);
          break;
        case 'stop':
          response = await stopMCPServer(nodeId, serverId);
          break;
        case 'restart':
          response = await restartMCPServer(nodeId, serverId);
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }

      if (!mountedRef.current) return null;

      if (optimistic) {
        updateOperation(serverId, action, 'success');
      }

      onComplete?.(response);
      
      // Cleanup after a delay
      setTimeout(cleanupOperations, 5000);

      return response;
    } catch (error) {
      if (!mountedRef.current) return null;

      const errorMessage = error instanceof Error ? error.message : 'Operation failed';
      
      if (optimistic) {
        updateOperation(serverId, action, 'error', errorMessage);
      }

      onError?.(errorMessage, serverId, action);
      
      // Cleanup after a delay
      setTimeout(cleanupOperations, 10000);

      return null;
    }
  }, [nodeId, addOperation, updateOperation, cleanupOperations]);

  /**
   * Start a server
   */
  const startServer = useCallback((
    serverId: string,
    options?: ServerOperationOptions
  ) => {
    return executeServerAction(serverId, 'start', options);
  }, [executeServerAction]);

  /**
   * Stop a server
   */
  const stopServer = useCallback((
    serverId: string,
    options?: ServerOperationOptions
  ) => {
    return executeServerAction(serverId, 'stop', options);
  }, [executeServerAction]);

  /**
   * Restart a server
   */
  const restartServer = useCallback((
    serverId: string,
    options?: ServerOperationOptions
  ) => {
    return executeServerAction(serverId, 'restart', options);
  }, [executeServerAction]);

  /**
   * Execute bulk server actions
   */
  const executeBulkAction = useCallback(async (
    serverIds: string[],
    action: MCPServerAction,
    options: ServerOperationOptions = {}
  ): Promise<MCPServerActionResponse[]> => {
    if (!nodeId || !mountedRef.current || serverIds.length === 0) return [];

    const { onComplete, onError, optimistic = true } = options;

    // Add operations to state if optimistic
    if (optimistic) {
      serverIds.forEach(serverId => addOperation(serverId, action));
    }

    try {
      const responses = await bulkMCPServerAction(nodeId, serverIds, action);

      if (!mountedRef.current) return [];

      // Update operation status for each server
      if (optimistic) {
        responses.forEach((response, index) => {
          const alias = response.server_alias ?? serverIds[index] ?? null;
          if (!alias) return;
          const status = response.success ? 'success' : 'error';
          const error = response.success ? undefined : response.error_details?.message;
          updateOperation(alias, action, status, error);
        });
      }

      // Call completion callback for successful operations
      responses.forEach((response, index) => {
        const alias = response.server_alias ?? serverIds[index] ?? '';
        if (response.success) {
          onComplete?.(response);
        } else {
          onError?.(
            response.error_details?.message || 'Operation failed',
            alias,
            action
          );
        }
      });

      // Cleanup after a delay
      setTimeout(cleanupOperations, 5000);

      return responses;
    } catch (error) {
      if (!mountedRef.current) return [];

      const errorMessage = error instanceof Error ? error.message : 'Bulk operation failed';

      // Update all operations as failed if optimistic
      if (optimistic) {
        serverIds.forEach(serverId => {
          updateOperation(serverId, action, 'error', errorMessage);
        });
      }

      // Call error callback for all servers
      serverIds.forEach(serverId => {
        onError?.(errorMessage, serverId, action);
      });

      // Cleanup after a delay
      setTimeout(cleanupOperations, 10000);

      return [];
    }
  }, [nodeId, addOperation, updateOperation, cleanupOperations]);

  /**
   * Start multiple servers
   */
  const startServers = useCallback((
    serverIds: string[],
    options?: ServerOperationOptions
  ) => {
    return executeBulkAction(serverIds, 'start', options);
  }, [executeBulkAction]);

  /**
   * Stop multiple servers
   */
  const stopServers = useCallback((
    serverIds: string[],
    options?: ServerOperationOptions
  ) => {
    return executeBulkAction(serverIds, 'stop', options);
  }, [executeBulkAction]);

  /**
   * Restart multiple servers
   */
  const restartServers = useCallback((
    serverIds: string[],
    options?: ServerOperationOptions
  ) => {
    return executeBulkAction(serverIds, 'restart', options);
  }, [executeBulkAction]);

  /**
   * Start all servers of a specific status
   */
  const startServersByStatus = useCallback((
    servers: MCPServerHealthForUI[],
    status: 'stopped' | 'error',
    options?: ServerOperationOptions
  ) => {
    const serverIds = servers
      .filter(server => server.status === status)
      .map(server => server.alias);
    
    return startServers(serverIds, options);
  }, [startServers]);

  /**
   * Restart all running servers
   */
  const restartAllRunning = useCallback((
    servers: MCPServerHealthForUI[],
    options?: ServerOperationOptions
  ) => {
    const serverIds = servers
      .filter(server => server.status === 'running')
      .map(server => server.alias);
    
    return restartServers(serverIds, options);
  }, [restartServers]);

  /**
   * Stop all running servers
   */
  const stopAllRunning = useCallback((
    servers: MCPServerHealthForUI[],
    options?: ServerOperationOptions
  ) => {
    const serverIds = servers
      .filter(server => server.status === 'running')
      .map(server => server.alias);
    
    return stopServers(serverIds, options);
  }, [stopServers]);

  /**
   * Clear all operations
   */
  const clearOperations = useCallback(() => {
    setState(prev => ({
      ...prev,
      operations: [],
      loading: false,
      error: null
    }));
  }, []);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  /**
   * Get operations for a specific server
   */
  const getServerOperations = useCallback((serverId: string) => {
    return state.operations.filter(op => op.serverId === serverId);
  }, [state.operations]);

  /**
   * Check if server has pending operations
   */
  const isServerBusy = useCallback((serverId: string) => {
    return state.operations.some(op => 
      op.serverId === serverId && op.status === 'pending'
    );
  }, [state.operations]);

  /**
   * Get pending operations count
   */
  const getPendingCount = useCallback(() => {
    return state.operations.filter(op => op.status === 'pending').length;
  }, [state.operations]);

  /**
   * Get recent operations (last 10)
   */
  const getRecentOperations = useCallback(() => {
    return [...state.operations]
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10);
  }, [state.operations]);

  // Cleanup on unmount
  useState(() => {
    return () => {
      mountedRef.current = false;
    };
  });

  return {
    // State
    ...state,
    
    // Single server operations
    startServer,
    stopServer,
    restartServer,
    
    // Bulk operations
    startServers,
    stopServers,
    restartServers,
    
    // Convenience operations
    startServersByStatus,
    restartAllRunning,
    stopAllRunning,
    
    // State management
    clearOperations,
    clearError,
    
    // Utility functions
    getServerOperations,
    isServerBusy,
    getPendingCount,
    getRecentOperations,
    
    // Computed properties
    hasOperations: state.operations.length > 0,
    hasPendingOperations: state.operations.some(op => op.status === 'pending'),
    hasErrors: state.operations.some(op => op.status === 'error') || !!state.error
  };
}

/**
 * Simplified hook for basic server operations
 */
export function useMCPServersSimple(nodeId: string | null) {
  const {
    startServer,
    stopServer,
    restartServer,
    loading,
    error,
    isServerBusy
  } = useMCPServers(nodeId);

  return {
    startServer,
    stopServer,
    restartServer,
    loading,
    error,
    isServerBusy
  };
}
