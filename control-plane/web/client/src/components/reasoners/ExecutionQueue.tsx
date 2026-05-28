import { useRef, useState, forwardRef, useImperativeHandle, type KeyboardEvent, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import {
  InProgress,
  Close,
  Copy,
  Time,
  CheckmarkFilled,
  ErrorFilled,
  Analytics,
  Launch
} from '@/components/ui/icon-bridge';
import { reasonersApi } from '../../services/reasonersApi';
import type { ExecutionRequest } from '../../types/execution';

export interface QueuedExecution {
  id: string;
  execution_id?: string; // Backend execution ID for navigation
  run_id?: string;
  workflow_id?: string;
  input: unknown;
  status: 'queued' | 'running' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  duration?: number;
  result?: unknown;
  error?: string;
  inputSummary: string;
  page_available?: boolean; // Whether the execution details page is available
  checking_availability?: boolean; // Whether we're currently checking page availability
}

interface ExecutionQueueProps {
  reasonerId: string;
  maxConcurrent?: number;
  onExecutionComplete?: (execution: QueuedExecution) => void;
  onExecutionSelect?: (execution: QueuedExecution | null) => void;
}

export interface ExecutionQueueRef {
  addExecution: (input: unknown) => string;
}

export const ExecutionQueue = forwardRef<ExecutionQueueRef, ExecutionQueueProps>(({
  reasonerId,
  maxConcurrent = 5,
  onExecutionComplete,
  onExecutionSelect
}, ref) => {
  const [executions, setExecutions] = useState<QueuedExecution[]>([]);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const navigate = useNavigate();
  const addExecutionRef = useRef<(input: unknown) => string>(() => "");

  // Get active executions (queued or running)
  const activeExecutions = executions.filter(e =>
    e.status === 'queued' || e.status === 'running'
  );

  // Get recent completed executions (last 5)
  const recentExecutions = executions
    .filter(e => e.status === 'completed' || e.status === 'failed')
    .sort((a, b) => (b.endTime?.getTime() || 0) - (a.endTime?.getTime() || 0))
    .slice(0, 5);

  const cancelExecution = (executionId: string) => {
    setExecutions(prev =>
      prev.map(exec =>
        exec.id === executionId && (exec.status === 'queued' || exec.status === 'running')
          ? { ...exec, status: 'failed', error: 'Cancelled by user', endTime: new Date() }
          : exec
      )
    );

    // Start next queued execution if any
    processQueue();
  };

  const executeReasoner = async (executionId: string, input: unknown) => {
    try {
      // Update status to running
      setExecutions(prev =>
        prev.map(exec =>
          exec.id === executionId
            ? { ...exec, status: 'running' as const }
            : exec
        )
      );

      // Create execution request
      const request: ExecutionRequest = {
        input: input || {},
        context: {
          session_id: `session_${Date.now()}`,
        },
      };

      // Use async API to get execution_id immediately
      const asyncResponse = await reasonersApi.executeReasonerAsync(reasonerId, request);

      // Immediately capture the backend execution_id for navigation
      setExecutions(prev =>
        prev.map(exec =>
          exec.id === executionId
            ? {
                ...exec,
                execution_id: asyncResponse.execution_id,
                run_id: asyncResponse.run_id,
                workflow_id: asyncResponse.workflow_id
              }
            : exec
        )
      );

      // Start checking if the execution details page is available
      checkPageAvailability(executionId, asyncResponse.execution_id);

      // Poll for execution completion
      const pollForCompletion = async () => {
        const maxAttempts = 300; // 5 minutes with 1-second intervals
        let attempts = 0;

        while (attempts < maxAttempts) {
          try {
            const statusResponse = await reasonersApi.getExecutionStatus(asyncResponse.execution_id);

            if (statusResponse.status === 'completed') {
              const endTime = new Date();
              const execution = executions.find(e => e.id === executionId);
              const duration = statusResponse.duration || (execution ? endTime.getTime() - execution.startTime.getTime() : 0);

              // Update with success
              setExecutions(prev =>
                prev.map(exec =>
                  exec.id === executionId
                    ? {
                        ...exec,
                        status: 'completed' as const,
                        result: statusResponse.result,
                        endTime,
                        duration
                      }
                    : exec
                )
              );

              // Notify parent component
              const completedExecution = executions.find(e => e.id === executionId);
              if (completedExecution && onExecutionComplete) {
                onExecutionComplete({
                  ...completedExecution,
                  status: 'completed',
                  execution_id: asyncResponse.execution_id,
                  run_id: asyncResponse.run_id ?? completedExecution.run_id,
                  workflow_id: asyncResponse.workflow_id ?? completedExecution.workflow_id,
                  result: statusResponse.result,
                  endTime,
                  duration
                });
              }
              return;
            } else if (statusResponse.status === 'failed') {
              const endTime = new Date();
              const execution = executions.find(e => e.id === executionId);
              const duration = statusResponse.duration || (execution ? endTime.getTime() - execution.startTime.getTime() : 0);

              // Update with error
              setExecutions(prev =>
                prev.map(exec =>
                  exec.id === executionId
                    ? {
                        ...exec,
                        status: 'failed' as const,
                        error: statusResponse.error || 'Execution failed',
                        endTime,
                        duration
                      }
                    : exec
                )
              );
              return;
            }

            // Still running, wait and try again
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
          } catch (error) {
            console.error('Error polling execution status:', error);
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
          }
        }

        // Timeout reached
        const endTime = new Date();
        const execution = executions.find(e => e.id === executionId);
        const duration = execution ? endTime.getTime() - execution.startTime.getTime() : 0;

        setExecutions(prev =>
          prev.map(exec =>
            exec.id === executionId
              ? {
                  ...exec,
                  status: 'failed' as const,
                  error: 'Execution timeout',
                  endTime,
                  duration
                }
              : exec
          )
        );
      };

      // Start polling in background
      pollForCompletion();

    } catch (error) {
      const endTime = new Date();
      const execution = executions.find(e => e.id === executionId);
      const duration = execution ? endTime.getTime() - execution.startTime.getTime() : 0;

      // Update with error
      setExecutions(prev =>
        prev.map(exec =>
          exec.id === executionId
            ? {
                ...exec,
                status: 'failed' as const,
                error: error instanceof Error ? error.message : 'Execution failed',
                endTime,
                duration
              }
            : exec
        )
      );
    } finally {
      // Process queue for next execution
      processQueue();
    }
  };

  const processQueue = () => {
    const runningCount = executions.filter(e => e.status === 'running').length;
    const queuedExecutions = executions.filter(e => e.status === 'queued');

    if (runningCount < maxConcurrent && queuedExecutions.length > 0) {
      const nextExecution = queuedExecutions[0];
      executeReasoner(nextExecution.id, nextExecution.input);
    }
  };

  const checkPageAvailability = async (executionId: string, backendExecutionId: string, retryCount = 0) => {
    const maxRetries = 10; // Maximum 20 seconds of checking (10 retries * 2 seconds)

    try {
      // Mark as checking availability
      setExecutions(prev =>
        prev.map(exec =>
          exec.id === executionId
            ? { ...exec, checking_availability: true }
            : exec
        )
      );

      // Check if execution details are available by trying to fetch execution status
      await reasonersApi.getExecutionStatus(backendExecutionId);

      // If we can fetch the status, the page should be available
      setExecutions(prev =>
        prev.map(exec =>
          exec.id === executionId
            ? {
                ...exec,
                page_available: true,
                checking_availability: false
              }
            : exec
        )
      );
    } catch {
      // If we can't fetch the status yet, the page might not be available
      if (retryCount < maxRetries) {
        // Wait a bit and try again
        setTimeout(() => {
          checkPageAvailability(executionId, backendExecutionId, retryCount + 1);
        }, 2000); // Check again in 2 seconds
      } else {
        // Max retries reached, mark as unavailable
        setExecutions(prev =>
          prev.map(exec =>
            exec.id === executionId
              ? {
                  ...exec,
                  page_available: false,
                  checking_availability: false
                }
              : exec
          )
        );
        console.warn(`Page availability check failed for execution ${backendExecutionId} after ${maxRetries} retries`);
      }
    }
  };

  const createInputSummary = (input: unknown): string => {
    if (!input || typeof input !== 'object') {
      return String(input || 'Empty');
    }

    // Extract key-value pairs for summary
    const entries = Object.entries(input);
    if (entries.length === 0) return 'Empty object';

    if (entries.length === 1) {
      const [key, value] = entries[0];
      return `${key}: ${String(value).slice(0, 20)}${String(value).length > 20 ? '...' : ''}`;
    }

    return `${entries.length} parameters`;
  };


  const copyExecutionData = (execution: QueuedExecution) => {
    const data = {
      input: execution.input,
      result: execution.result,
      duration: execution.duration,
      status: execution.status
    };
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  };

  const handleNavigateToExecution = (execution: QueuedExecution) => {
    if (execution.execution_id) {
      navigate(`/executions/${execution.execution_id}`);
    }
  };

  const handleExecutionCardClick = (
    execution: QueuedExecution,
    event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>
  ) => {
    // Prevent navigation if clicking on action buttons
    if ((event.target as HTMLElement).closest('button')) {
      return;
    }

    // If execution has backend execution_id and page is available, navigate to details page
    if (execution.execution_id && execution.page_available) {
      handleNavigateToExecution(execution);
    } else {
      // Otherwise, use existing selection behavior for inline viewing
      const isCurrentlySelected = selectedExecution === execution.id;
      const newSelection = isCurrentlySelected ? null : execution.id;
      setSelectedExecution(newSelection);

      if (onExecutionSelect) {
        onExecutionSelect(isCurrentlySelected ? null : execution);
      }
    }
  };

  const handleViewDetailsClick = (event: MouseEvent<HTMLElement>, execution: QueuedExecution) => {
    event.stopPropagation();
    if (execution.execution_id && execution.page_available) {
      handleNavigateToExecution(execution);
    }
  };

  const getStatusIcon = (status: QueuedExecution['status']) => {
    switch (status) {
      case 'queued':
        return <Time className="h-4 w-4 text-yellow-500" />;
      case 'running':
        return <InProgress className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckmarkFilled className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <ErrorFilled className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusColor = (status: QueuedExecution['status']) => {
    switch (status) {
      case 'queued':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
    }
  };

  addExecutionRef.current = (input: unknown) => {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const inputSummary = createInputSummary(input);
    const nextStatus = activeExecutions.length >= maxConcurrent ? 'queued' : 'running';
    const newExecution: QueuedExecution = {
      id: executionId,
      input,
      status: nextStatus,
      startTime: new Date(),
      inputSummary,
    };

    setExecutions(prev => [newExecution, ...prev]);

    if (nextStatus === 'running') {
      executeReasoner(executionId, input);
    }

    return executionId;
  };

  useImperativeHandle(ref, () => ({
    addExecution(input: unknown) {
      return addExecutionRef.current(input);
    }
  }), []);

  if (activeExecutions.length === 0 && recentExecutions.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Active Executions */}
      {activeExecutions.length > 0 && (
        <Card className="bg-card border border-border rounded-lg shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <InProgress className="h-4 w-4 text-blue-500" />
              Active Executions ({activeExecutions.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {activeExecutions.map((execution) => (
              <div
                key={execution.id}
                className={`flex items-center justify-between p-3 rounded-lg border bg-card transition-colors ${
                  execution.execution_id
                    ? 'hover:bg-accent/50 cursor-pointer hover:border-primary/30'
                    : ''
                }`}
                onClick={execution.execution_id ? (e) => handleExecutionCardClick(execution, e) : undefined}
                role={execution.execution_id ? "button" : undefined}
                tabIndex={execution.execution_id ? 0 : undefined}
                aria-label={execution.execution_id ? `Navigate to execution ${execution.execution_id}` : undefined}
                onKeyDown={execution.execution_id ? (e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleExecutionCardClick(execution, e);
                  }
                } : undefined}
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(execution.status)}
                  <div className="flex-1">
                    <div className="font-medium text-sm flex items-center gap-2">
                      {execution.inputSummary}
                      {execution.execution_id && (
                        <span className="text-xs text-primary/70 bg-primary/10 px-1.5 py-0.5 rounded-md">
                          navigable
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {execution.status === 'running'
                        ? `Running ${Math.floor((Date.now() - execution.startTime.getTime()) / 1000)}s...`
                        : 'Queued...'
                      }
                      {execution.execution_id && (
                        <span className="ml-2 text-primary/60">
                          ID: {execution.execution_id.slice(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={getStatusColor(execution.status)}>
                    {execution.status}
                  </Badge>
                  {execution.execution_id && (
                    <>
                      {execution.checking_availability ? (
                        <div className="h-8 w-8 flex items-center justify-center">
                          <InProgress className="h-4 w-4 text-blue-500 animate-spin" />
                        </div>
                      ) : execution.page_available ? (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => handleViewDetailsClick(e, execution)}
                          className="h-8 w-8 p-0"
                          aria-label="View execution details"
                          title="View execution details"
                        >
                          <Launch className="h-4 w-4" />
                        </Button>
                      ) : (
                        <div className="h-8 w-8 flex items-center justify-center">
                          <Time className="h-4 w-4 text-yellow-500" />
                        </div>
                      )}
                    </>
                  )}
                  {(execution.status === 'queued' || execution.status === 'running') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => cancelExecution(execution.id)}
                      className="h-8 w-8 p-0"
                      aria-label="Cancel execution"
                      title="Cancel execution"
                    >
                      <Close className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Recent Executions */}
      {recentExecutions.length > 0 && (
        <Card className="bg-card border border-border rounded-lg shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Analytics className="h-4 w-4 text-muted-foreground" />
              Recent Executions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentExecutions.map((execution) => (
              <div
                key={execution.id}
                className={`flex items-center justify-between p-3 rounded-lg border bg-card transition-colors ${
                  execution.execution_id && execution.page_available
                    ? 'hover:bg-accent/50 cursor-pointer hover:border-primary/30'
                    : execution.execution_id
                      ? 'hover:bg-accent/20 cursor-default'
                      : 'hover:bg-accent/30 cursor-pointer'
                } ${
                  selectedExecution === execution.id ? 'ring-2 ring-primary/20 bg-accent/30' : ''
                }`}
                onClick={(e) => handleExecutionCardClick(execution, e)}
                role="button"
                tabIndex={0}
                aria-label={
                  execution.execution_id && execution.page_available
                    ? `Navigate to execution ${execution.execution_id}`
                    : execution.execution_id && execution.checking_availability
                      ? `Checking availability for execution ${execution.execution_id}`
                      : execution.execution_id
                        ? `Execution ${execution.execution_id} page not ready yet`
                        : `View execution details`
                }
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleExecutionCardClick(execution, e);
                  }
                }}
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(execution.status)}
                  <div className="flex-1">
                    <div className="font-medium text-sm flex items-center gap-2">
                      {execution.inputSummary}
                      {execution.execution_id && (
                        <span className={`text-xs px-1.5 py-0.5 rounded-md ${
                          execution.checking_availability
                            ? 'text-blue-700 bg-blue-100'
                            : execution.page_available
                              ? 'text-primary/70 bg-primary/10'
                              : 'text-yellow-700 bg-yellow-100'
                        }`}>
                          {execution.checking_availability
                            ? 'checking...'
                            : execution.page_available
                              ? 'navigable'
                              : 'preparing...'}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {execution.duration}ms • {execution.endTime?.toLocaleTimeString()}
                      {execution.execution_id && (
                        <span className="ml-2 text-primary/60">
                          ID: {execution.execution_id.slice(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={getStatusColor(execution.status)}>
                    {execution.status}
                  </Badge>
                  {execution.execution_id && (
                    <>
                      {execution.checking_availability ? (
                        <div className="h-8 w-8 flex items-center justify-center">
                          <InProgress className="h-4 w-4 text-blue-500 animate-spin" />
                        </div>
                      ) : execution.page_available ? (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => handleViewDetailsClick(e, execution)}
                          className="h-8 w-8 p-0"
                          aria-label="View execution details"
                          title="View execution details"
                        >
                          <Launch className="h-4 w-4" />
                        </Button>
                      ) : (
                        <div className="h-8 w-8 flex items-center justify-center">
                          <Time className="h-4 w-4 text-yellow-500" />
                        </div>
                      )}
                    </>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyExecutionData(execution);
                    }}
                    className="h-8 w-8 p-0"
                    aria-label="Copy execution data"
                    title="Copy execution data"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
});

ExecutionQueue.displayName = 'ExecutionQueue';
