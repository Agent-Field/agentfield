import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  MCPTool,
  MCPToolsResponse,
  MCPToolTestResponse
} from '../types/agentfield';
import { getMCPTools, testMCPTool } from '../services/api';
import { validateToolParameters } from '../utils/mcpUtils';

/**
 * Tool execution state
 */
interface ToolExecution {
  id: string;
  toolName: string;
  parameters: Record<string, any>;
  status: 'pending' | 'success' | 'error';
  result?: any;
  error?: string;
  executionTime?: number;
  timestamp: Date;
}

/**
 * Tool discovery and testing state
 */
interface MCPToolsState {
  /** Available tools for the current server */
  tools: MCPTool[];
  /** Whether tools are currently loading */
  loading: boolean;
  /** Last error that occurred */
  error: string | null;
  /** Tool execution history */
  executions: ToolExecution[];
  /** Whether any execution is in progress */
  executing: boolean;
  /** Timestamp of last tools fetch */
  lastFetch: Date | null;
}

/**
 * Options for tool testing
 */
interface ToolTestOptions {
  /** Timeout for tool execution in milliseconds */
  timeoutMs?: number;
  /** Whether to validate parameters before execution */
  validateParams?: boolean;
  /** Callback for execution completion */
  onComplete?: (result: MCPToolTestResponse) => void;
  /** Callback for execution error */
  onError?: (error: string) => void;
  /** Whether to add to execution history */
  addToHistory?: boolean;
}

/**
 * Custom hook for MCP tool exploration and testing
 * 
 * @param nodeId - The node ID
 * @param serverId - The server ID/alias
 * @returns Object containing tools state and testing functions
 */
export function useMCPTools(
  nodeId: string | null,
  serverId: string | null
) {
  const [state, setState] = useState<MCPToolsState>({
    tools: [],
    loading: false,
    error: null,
    executions: [],
    executing: false,
    lastFetch: null
  });

  const mountedRef = useRef(true);
  const executionCounterRef = useRef(0);

  /**
   * Generate unique execution ID
   */
  const generateExecutionId = useCallback(() => {
    return `exec_${Date.now()}_${++executionCounterRef.current}`;
  }, []);

  /**
   * Add execution to history
   */
  const addExecution = useCallback((execution: Omit<ToolExecution, 'id' | 'timestamp'>) => {
    const newExecution: ToolExecution = {
      ...execution,
      id: generateExecutionId(),
      timestamp: new Date()
    };

    setState(prev => ({
      ...prev,
      executions: [newExecution, ...prev.executions.slice(0, 49)], // Keep last 50
      executing: execution.status === 'pending'
    }));

    return newExecution.id;
  }, [generateExecutionId]);

  /**
   * Update execution status
   */
  const updateExecution = useCallback((
    executionId: string,
    updates: Partial<Pick<ToolExecution, 'status' | 'result' | 'error' | 'executionTime'>>
  ) => {
    setState(prev => {
      const updatedExecutions = prev.executions.map(exec => {
        if (exec.id === executionId) {
          return { ...exec, ...updates };
        }
        return exec;
      });

      const hasExecuting = updatedExecutions.some(exec => exec.status === 'pending');

      return {
        ...prev,
        executions: updatedExecutions,
        executing: hasExecuting
      };
    });
  }, []);

  /**
   * Fetch available tools for the server
   */
  const fetchTools = useCallback(async () => {
    if (!nodeId || !serverId || !mountedRef.current) return;

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response: MCPToolsResponse = await getMCPTools(nodeId, serverId);
      
      if (!mountedRef.current) return;

      setState(prev => ({
        ...prev,
        tools: response.tools || [],
        loading: false,
        error: null,
        lastFetch: new Date()
      }));
    } catch (error) {
      if (!mountedRef.current) return;

      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch tools';
      setState(prev => ({
        ...prev,
        tools: [],
        loading: false,
        error: errorMessage
      }));
    }
  }, [nodeId, serverId]);

  /**
   * Execute a tool with parameters
   */
  const executeTool = useCallback(async (
    toolName: string,
    parameters: Record<string, any> = {},
    options: ToolTestOptions = {}
  ): Promise<MCPToolTestResponse | null> => {
    if (!nodeId || !serverId || !mountedRef.current) return null;

    const {
      timeoutMs = 30000,
      validateParams = true,
      onComplete,
      onError,
      addToHistory = true
    } = options;

    // Find tool definition
    const tool = state.tools.find(t => t.name === toolName);
    if (!tool) {
      const error = `Tool '${toolName}' not found`;
      onError?.(error);
      return null;
    }

    // Validate parameters if requested
    if (validateParams && tool.input_schema) {
      const validation = validateToolParameters(parameters, tool.input_schema);
      if (!validation.valid) {
        const error = `Parameter validation failed: ${validation.errors.join(', ')}`;
        onError?.(error);
        return null;
      }
    }

    // Add to execution history
    let executionId: string | null = null;
    if (addToHistory) {
      executionId = addExecution({
        toolName,
        parameters,
        status: 'pending'
      });
    }

    try {
      const response = await testMCPTool(nodeId, serverId, toolName, parameters, timeoutMs);
      
      if (!mountedRef.current) return null;

      // Update execution history
      if (executionId) {
        updateExecution(executionId, {
          status: response.success ? 'success' : 'error',
          result: response.result,
          error: response.error,
          executionTime: response.execution_time_ms
        });
      }

      onComplete?.(response);
      return response;
    } catch (error) {
      if (!mountedRef.current) return null;

      const errorMessage = error instanceof Error ? error.message : 'Tool execution failed';
      
      // Update execution history
      if (executionId) {
        updateExecution(executionId, {
          status: 'error',
          error: errorMessage
        });
      }

      onError?.(errorMessage);
      return null;
    }
  }, [nodeId, serverId, state.tools, addExecution, updateExecution]);

  /**
   * Test tool with sample parameters
   */
  const testToolWithSample = useCallback(async (
    toolName: string,
    options?: ToolTestOptions
  ) => {
    const tool = state.tools.find(t => t.name === toolName);
    if (!tool || !tool.input_schema) return null;

    // Generate sample parameters from schema
    const sampleParams = generateSampleParameters(tool.input_schema);
    return executeTool(toolName, sampleParams, options);
  }, [state.tools, executeTool]);

  /**
   * Re-execute a previous execution
   */
  const reExecute = useCallback(async (
    executionId: string,
    options?: ToolTestOptions
  ) => {
    const execution = state.executions.find(exec => exec.id === executionId);
    if (!execution) return null;

    return executeTool(execution.toolName, execution.parameters, options);
  }, [state.executions, executeTool]);

  /**
   * Clear execution history
   */
  const clearExecutions = useCallback(() => {
    setState(prev => ({
      ...prev,
      executions: [],
      executing: false
    }));
  }, []);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  /**
   * Get executions for a specific tool
   */
  const getToolExecutions = useCallback((toolName: string) => {
    return state.executions.filter(exec => exec.toolName === toolName);
  }, [state.executions]);

  /**
   * Get successful executions
   */
  const getSuccessfulExecutions = useCallback(() => {
    return state.executions.filter(exec => exec.status === 'success');
  }, [state.executions]);

  /**
   * Get failed executions
   */
  const getFailedExecutions = useCallback(() => {
    return state.executions.filter(exec => exec.status === 'error');
  }, [state.executions]);

  /**
   * Get execution statistics
   */
  const getExecutionStats = useCallback(() => {
    const total = state.executions.length;
    const successful = state.executions.filter(exec => exec.status === 'success').length;
    const failed = state.executions.filter(exec => exec.status === 'error').length;
    const pending = state.executions.filter(exec => exec.status === 'pending').length;

    const avgExecutionTime = state.executions
      .filter(exec => exec.executionTime !== undefined)
      .reduce((sum, exec, _, arr) => sum + (exec.executionTime! / arr.length), 0);

    return {
      total,
      successful,
      failed,
      pending,
      successRate: total > 0 ? successful / total : 0,
      avgExecutionTime: avgExecutionTime || 0
    };
  }, [state.executions]);

  /**
   * Search tools by name or description
   */
  const searchTools = useCallback((query: string) => {
    if (!query.trim()) return state.tools;

    const lowerQuery = query.toLowerCase();
    return state.tools.filter(tool =>
      tool.name.toLowerCase().includes(lowerQuery) ||
      (tool.description?.toLowerCase().includes(lowerQuery) ?? false)
    );
  }, [state.tools]);

  /**
   * Get tool by name
   */
  const getTool = useCallback((toolName: string) => {
    return state.tools.find(tool => tool.name === toolName);
  }, [state.tools]);

  // Fetch tools when nodeId or serverId changes
  useEffect(() => {
    if (nodeId && serverId) {
      fetchTools();
    } else {
      // Clear state when no nodeId or serverId
      setState({
        tools: [],
        loading: false,
        error: null,
        executions: [],
        executing: false,
        lastFetch: null
      });
    }
  }, [nodeId, serverId, fetchTools]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return {
    // State
    ...state,
    
    // Tool management
    fetchTools,
    refreshTools: fetchTools,
    
    // Tool execution
    executeTool,
    testToolWithSample,
    reExecute,
    
    // History management
    clearExecutions,
    clearError,
    
    // Utility functions
    getToolExecutions,
    getSuccessfulExecutions,
    getFailedExecutions,
    getExecutionStats,
    searchTools,
    getTool,
    
    // Computed properties
    hasTools: state.tools.length > 0,
    hasExecutions: state.executions.length > 0,
    toolCount: state.tools.length,
    executionCount: state.executions.length
  };
}

/**
 * Generate sample parameters from JSON schema
 */
function generateSampleParameters(schema: any): Record<string, any> {
  const params: Record<string, any> = {};
  
  if (!schema || !schema.properties) return params;

  for (const [key, fieldSchema] of Object.entries(schema.properties as Record<string, any>)) {
    if (fieldSchema.type === 'string') {
      params[key] = fieldSchema.example || fieldSchema.default || `sample_${key}`;
    } else if (fieldSchema.type === 'number') {
      params[key] = fieldSchema.example || fieldSchema.default || 42;
    } else if (fieldSchema.type === 'boolean') {
      params[key] = fieldSchema.example !== undefined ? fieldSchema.example : 
                   fieldSchema.default !== undefined ? fieldSchema.default : true;
    } else if (fieldSchema.type === 'array') {
      params[key] = fieldSchema.example || fieldSchema.default || [];
    } else if (fieldSchema.type === 'object') {
      params[key] = fieldSchema.example || fieldSchema.default || {};
    }
  }

  return params;
}

/**
 * Simplified hook for basic tool testing
 */
export function useMCPToolsSimple(nodeId: string | null, serverId: string | null) {
  const {
    tools,
    loading,
    error,
    executeTool,
    hasTools
  } = useMCPTools(nodeId, serverId);

  return {
    tools,
    loading,
    error,
    executeTool,
    hasTools
  };
}
